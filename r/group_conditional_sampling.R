# Group-level conditional sampling with ARF/FORDE.
#
# For one group at a time, condition on *all* features outside the group and
# draw the whole group jointly. This is what makes the resulting quantity a
# group conditional importance: the group is sampled as a block, so the
# within-group dependence structure is produced by the sampler rather than
# destroyed by independent per-feature draws.
#
# Usage: Rscript r/group_conditional_sampling.R <plan.json>
#
# The plan is written by python/run_group_cfi.py and contains paths to the
# train/test CSVs, the feature list, the group map, the seed and the number of
# replicates. Only training rows are used to fit ARF/FORDE; test rows are only
# ever used as evidence.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop('usage: Rscript r/group_conditional_sampling.R <plan.json>')
plan_path <- args[1]
script <- sub('^--file=', '', commandArgs(trailingOnly = FALSE)[
  grepl('^--file=', commandArgs(trailingOnly = FALSE))][1])
root <- normalizePath(file.path(dirname(script), '..'))
setwd(root)
.libPaths(c(file.path(root, 'RLibrary'), .libPaths()))
suppressPackageStartupMessages({
  library(jsonlite); library(arf); library(data.table)
})
plan <- fromJSON(plan_path)
setDTthreads(1L)

train <- fread(plan$train_csv, data.table = FALSE)
test <- fread(plan$test_csv, data.table = FALSE)
features <- plan$features
groups <- plan$groups
repeats <- as.integer(plan$repeats)
pool <- if (is.null(plan$pool)) 1L else as.integer(plan$pool)
if (is.na(pool) || pool < 1L) stop('pool must be a positive integer')
proposals <- repeats * pool
seed <- as.integer(plan$seed)
task <- plan$task
out_dir <- plan$out_dir
# arf 0.2.5 truncates every numeric leaf distribution at [min, max]:
# 'no' = unbounded (old behaviour, can emit negatives), 'global' = empirical
# data range, 'local' = per-leaf empirical range. epsilon widens the bounds by
# epsilon/2 * range on each side.
finite_bounds <- if (is.null(plan$finite_bounds)) 'no' else as.character(plan$finite_bounds)
if (!finite_bounds %in% c('no', 'local', 'global'))
  stop('finite_bounds must be one of no/local/global')
epsilon <- if (is.null(plan$epsilon)) 0 else as.numeric(plan$epsilon)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Constant columns carry no density information and cannot be conditioned on.
active <- features[vapply(train, function(x) length(unique(x)) > 1L, logical(1))]
cat(sprintf('[ARF] task=%s seed=%d: %d train, %d test, %d/%d active features\n',
            task, seed, nrow(train), nrow(test), length(active), length(features)))

set.seed(seed)
arf_obj <- adversarial_rf(train[, active, drop = FALSE],
                          num_trees = as.integer(plan$arf_num_trees),
                          min_node_size = as.integer(plan$arf_min_node_size),
                          parallel = FALSE, seed = seed)
psi <- forde(arf_obj, train[, active, drop = FALSE],
             finite_bounds = finite_bounds, epsilon = epsilon, parallel = FALSE)

# Unseen categorical evidence is silently ignored by arf 0.2.5, which would
# return draws from the wrong conditional. Reject those rows explicitly.
discrete <- unique(c(setdiff(features, active),
                     psi$meta[family == 'multinom', variable]))

diagnostics <- list()
chunks <- split(seq_len(nrow(test)), ceiling(seq_len(nrow(test)) / 100))

for (group in names(groups)) {
  members <- groups[[group]]
  inside <- intersect(members, active)
  outside <- setdiff(active, members)
  if (!length(inside)) {
    # A group made only of constant columns cannot be resampled; emit its
    # constant values so the downstream loss is exactly the baseline.
    draws <- data.frame(test_position = rep(seq_len(nrow(test)) - 1L, each = proposals),
                        replicate = rep(seq_len(proposals), nrow(test)))
    for (feature in members) draws[[feature]] <- train[[feature]][1]
    fwrite(draws, file.path(out_dir, sprintf('draws_%s_%d_%s.csv', task, seed, group)))
    cat(sprintf('[group %s] no active feature; constant draws written\n', group))
    next
  }
  supported <- rep(TRUE, nrow(test))
  for (col in intersect(outside, discrete))
    supported <- supported & test[[col]] %in% unique(train[[col]])
  cat(sprintf('[group %s] %d features sampled jointly, %d/%d rows supported\n',
              group, length(inside), sum(supported), length(supported)))

  collected <- vector('list', length(chunks))
  for (k in seq_along(chunks)) {
    rows <- chunks[[k]]
    positions <- which(supported[rows])
    if (!length(positions)) next
    rows <- rows[positions]
    evidence <- test[rows, outside, drop = FALSE]
    syn <- forge(psi, n_synth = proposals, evidence = evidence,
                 evidence_row_mode = 'separate', nomatch = 'na',
                 round = TRUE, parallel = FALSE, verbose = FALSE)
    if (nrow(syn) != length(rows) * proposals)
      stop(sprintf('forge returned %d rows, expected %d', nrow(syn), length(rows) * proposals))
    # Guard the R/Python contract: replicates contiguous, evidence preserved.
    expected <- evidence[rep(seq_along(rows), each = proposals), , drop = FALSE]
    for (col in outside) {
      valid <- is.finite(syn[[col]])
      if (any(abs(syn[[col]][valid] - expected[[col]][valid]) > 1e-7))
        stop(paste('forge evidence order/value mismatch:', col))
    }
    # R rows are 1-based; the Python side indexes test rows from 0.
    frame <- data.frame(test_position = rep(rows - 1L, each = proposals),
                        replicate = rep(seq_len(proposals), length(rows)))
    for (feature in inside) frame[[feature]] <- as.numeric(syn[[feature]])
    collected[[k]] <- frame
    for (feature in inside) {
      values <- frame[[feature]]
      diagnostics[[length(diagnostics) + 1L]] <- data.frame(
        task = task, seed = seed, group = group, feature = feature,
        n_draws = length(values), n_missing = sum(!is.finite(values)),
        n_negative = sum(values < 0, na.rm = TRUE),
        n_outside_train_range = sum(values < min(train[[feature]]) |
                                      values > max(train[[feature]]), na.rm = TRUE),
        train_min = min(train[[feature]]), train_max = max(train[[feature]]),
        n_unsupported_rows = sum(!supported))
    }
  }
  draws <- rbindlist(collected[!vapply(collected, is.null, logical(1))])
  # Rows without support keep no draw at all; the Python side reads this as a
  # missing conditional sample rather than silently falling back to marginal.
  fwrite(draws, file.path(out_dir, sprintf('draws_%s_%d_%s.csv', task, seed, group)))
  cat(sprintf('[group %s] %d draws written\n', group, nrow(draws)))
}

fwrite(rbindlist(diagnostics),
       file.path(out_dir, sprintf('diagnostics_%s_%d.csv', task, seed)))
write_json(list(task = task, seed = seed, repeats = repeats, pool = pool,
                train_rows = nrow(train), test_rows = nrow(test),
                n_features = length(features), active_features = active,
                arf_version = as.character(packageVersion('arf')),
                r_version = R.version.string,
                discriminator_accuracy = arf_obj$acc,
                finite_bounds = finite_bounds, epsilon = epsilon,
                nomatch = 'na', round = 'train decimals',
                conditioning = 'all features outside the group, sampled jointly'),
           file.path(out_dir, sprintf('arf_meta_%s_%d.json', task, seed)),
           auto_unbox = TRUE, pretty = TRUE)
cat('[ARF] group conditional sampling complete\n')
