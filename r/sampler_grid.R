# Evaluate ARF/FORDE sampler settings on a fixed feature list.
#
# Each combo refits ARF/FORDE on the same training rows and draws every listed
# feature conditionally on all other active features (single-feature CFI). The
# Python driver (python/run_sampler_grid.py) writes the plan and scores the
# draws for domain validity, support, marginals and cross-feature constraints.
#
# Usage: Rscript r/sampler_grid.R <plan.json>
#
# Plan keys: root, task, seed, repeats, pool, features, train_csv, test_csv,
# out_dir, combos = [{name, finite_bounds, epsilon, num_trees, min_node_size}].
# ``pool`` draws repeats*pool proposals per evidence row; the Python driver
# keeps the first repeats proposals that satisfy the enforced constraints.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop('usage: Rscript r/sampler_grid.R <plan.json>')
script <- sub('^--file=', '', commandArgs(trailingOnly = FALSE)[
  grepl('^--file=', commandArgs(trailingOnly = FALSE))][1])
root <- normalizePath(file.path(dirname(script), '..'))
setwd(root)
.libPaths(c(file.path(root, 'RLibrary'), .libPaths()))
suppressPackageStartupMessages({library(jsonlite); library(arf); library(data.table)})
plan <- fromJSON(args[1], simplifyVector = FALSE)
setDTthreads(1L)

train <- fread(plan$train_csv, data.table = FALSE)
test <- fread(plan$test_csv, data.table = FALSE)
columns <- colnames(train)
sampled <- unlist(plan$features, use.names = FALSE)
if (!all(sampled %in% columns))
  stop(paste('sampled features missing from CSV:', setdiff(sampled, columns)))
repeats <- as.integer(plan$repeats)
pool <- if (is.null(plan$pool)) 1L else as.integer(plan$pool)
if (is.na(pool) || pool < 1L) stop('pool must be a positive integer')
proposals <- repeats * pool
seed <- as.integer(plan$seed)
out_dir <- plan$out_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

active <- columns[vapply(train, function(x) length(unique(x)) > 1L, logical(1))]
chunks <- split(seq_len(nrow(test)), ceiling(seq_len(nrow(test)) / 100))
cat(sprintf('[grid] task=%s: %d train, %d test, %d active, %d sampled, %d combos\n',
            plan$task, nrow(train), nrow(test), length(active), length(sampled),
            length(plan$combos)))

run_combo <- function(combo) {
  name <- as.character(combo$name)
  bounds <- as.character(combo$finite_bounds)
  if (!bounds %in% c('no', 'local', 'global'))
    stop(sprintf('combo %s: finite_bounds must be no/local/global', name))
  epsilon <- as.numeric(combo$epsilon)
  start <- Sys.time()
  set.seed(seed)
  arf_obj <- adversarial_rf(train[, active, drop = FALSE],
                            num_trees = as.integer(combo$num_trees),
                            min_node_size = as.integer(combo$min_node_size),
                            parallel = FALSE, seed = seed)
  psi <- forde(arf_obj, train[, active, drop = FALSE], finite_bounds = bounds,
               epsilon = epsilon, parallel = FALSE)
  discrete <- unique(c(setdiff(columns, active),
                       psi$meta[family == 'multinom', variable]))
  fit_seconds <- as.numeric(difftime(Sys.time(), start, units = 'secs'))
  cat(sprintf('[grid %s] ARF/FORDE fit in %.0fs (discriminator acc %.3f)\n',
              name, fit_seconds, arf_obj$acc))

  collected <- list()
  diagnostics <- list()
  for (feature in sampled) {
    outside <- setdiff(active, feature)
    supported <- rep(TRUE, nrow(test))
    for (col in intersect(outside, discrete))
      supported <- supported & test[[col]] %in% unique(train[[col]])
    pieces <- list()
    for (rows in chunks) {
      positions <- which(supported[rows])
      if (!length(positions)) next
      rows <- rows[positions]
      evidence <- test[rows, outside, drop = FALSE]
      syn <- forge(psi, n_synth = proposals, evidence = evidence,
                   evidence_row_mode = 'separate', nomatch = 'na',
                   round = TRUE, parallel = FALSE, verbose = FALSE)
      if (nrow(syn) != length(rows) * proposals)
        stop(sprintf('forge returned %d rows, expected %d', nrow(syn),
                     length(rows) * proposals))
      expected <- evidence[rep(seq_along(rows), each = proposals), , drop = FALSE]
      for (col in outside) {
        valid <- is.finite(syn[[col]])
        if (any(abs(syn[[col]][valid] - expected[[col]][valid]) > 1e-7))
          stop(paste('forge evidence order/value mismatch:', col))
      }
      pieces[[length(pieces) + 1L]] <- data.frame(
        test_position = rep(rows - 1L, each = proposals),
        replicate = rep(seq_len(proposals), length(rows)),
        value = as.numeric(syn[[feature]]))
    }
    part <- if (length(pieces)) rbindlist(pieces) else
      data.table(test_position = integer(0), replicate = integer(0),
                 value = numeric(0))
    collected[[feature]] <- part
    diagnostics[[length(diagnostics) + 1L]] <- data.frame(
      combo = name, feature = feature, n_draws = nrow(part),
      n_missing = sum(!is.finite(part$value)),
      n_unsupported_rows = sum(!supported))
    cat(sprintf('[grid %s] %s: %d draws, %d missing, %d unsupported rows\n',
                name, feature, nrow(part), sum(!is.finite(part$value)),
                sum(!supported)))
  }
  draws <- rbindlist(collected, idcol = 'feature')
  fwrite(draws, file.path(out_dir, sprintf('grid_draws_%s_%s.csv', plan$task, name)))
  fwrite(rbindlist(diagnostics),
         file.path(out_dir, sprintf('grid_diag_%s_%s.csv', plan$task, name)))
  write_json(list(combo = name, task = plan$task, seed = seed, repeats = repeats,
                  pool = pool, sampled_features = sampled,
                  finite_bounds = bounds, epsilon = epsilon,
                  num_trees = as.integer(combo$num_trees),
                  min_node_size = as.integer(combo$min_node_size),
                  train_rows = nrow(train), test_rows = nrow(test),
                  discriminator_accuracy = arf_obj$acc,
                  fit_seconds = fit_seconds,
                  arf_version = as.character(packageVersion('arf')),
                  r_version = R.version.string),
             file.path(out_dir, sprintf('grid_meta_%s_%s.json', plan$task, name)),
             auto_unbox = TRUE, pretty = TRUE)
  cat(sprintf('[grid %s] %d draws written\n', name, nrow(draws)))
}

for (combo in plan$combos) run_combo(combo)
cat('[grid] complete\n')
