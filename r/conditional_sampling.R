# Deterministic ARF sampling. Only training data fit ARF/FORDE.
args <- commandArgs(trailingOnly = FALSE)
script <- sub('^--file=', '', args[grepl('^--file=', args)][1])
root <- normalizePath(file.path(dirname(script), '..'))
setwd(root)
.libPaths(c(file.path(root, 'RLibrary'), .libPaths()))
suppressPackageStartupMessages({library(arf); library(data.table); library(jsonlite); library(foreach)})
setDTthreads(1L)
registerDoSEQ()
cfg <- fromJSON('config.json')
info <- fromJSON('results/intermediate/split_r.json')
set.seed(cfg$seed)
full <- fread('results/intermediate/full_data.csv', data.table=FALSE)
features <- info$feature_names
train <- full[info$train_indices[info$arf_train_positions + 1L] + 1L, features, drop=FALSE]
test <- full[info$test_indices[info$importance_test_positions + 1L] + 1L, features, drop=FALSE]
# Constant columns contribute no predictive variation and need not enter density estimation.
active <- features[vapply(train, function(x) length(unique(x)) > 1L, logical(1))]
cat(sprintf('[ARF] %d train, %d held-out evaluation rows, %d active features\n', nrow(train), nrow(test), length(active)))
arf_obj <- adversarial_rf(train[,active,drop=FALSE], num_trees=as.integer(cfg$arf_num_trees),
                          min_node_size=as.integer(cfg$arf_min_node_size), parallel=FALSE,
                          seed=as.integer(cfg$seed))
# Unbounded outer leaves avoid artificial zero density beyond empirical extrema.
psi <- forde(arf_obj, train[,active,drop=FALSE], finite_bounds='no', parallel=FALSE)
saveRDS(list(arf=arf_obj, psi=psi, config=cfg, train_positions=info$arf_train_positions),
        'results/models/arf_model.rds')
repeats <- cfg$conditional_repeats
n <- nrow(test)
diagnostics <- list()

sample_feature <- function(feature, conditioning, seed) {
  set.seed(seed)
  if (!(feature %in% active)) return(rep(train[[feature]][1], n * repeats))
  # arf 0.2.5 may silently ignore an unseen categorical evidence value.
  # Reject out-of-support evidence before calling FORGE.
  supported <- rep(TRUE, n)
  discrete <- c(setdiff(features, active), psi$meta[family == 'multinom', variable])
  for (col in intersect(conditioning, discrete))
    supported <- supported & test[[col]] %in% unique(train[[col]])
  conditioning <- intersect(conditioning, active)
  chunks <- split(seq_len(n), ceiling(seq_len(n) / 100))
  values <- lapply(chunks, function(rows) {
    output <- rep(NA_real_, length(rows)*repeats)
    positions <- which(supported[rows])
    if (!length(positions)) return(output)
    rows <- rows[positions]
    if (length(conditioning) == 0) {
      syn <- forge(psi, n_synth=length(rows)*repeats, parallel=FALSE)
    } else {
      evidence <- test[rows,conditioning,drop=FALSE]
      syn <- forge(psi, n_synth=repeats, evidence=evidence, evidence_row_mode='separate',
                   nomatch='na', round=TRUE, parallel=FALSE, verbose=FALSE)
      stopifnot(nrow(syn) == length(rows)*repeats)
      # Guard the R/Python contract: row-major order with replicates contiguous.
      expected <- evidence[rep(seq_along(rows), each=repeats),,drop=FALSE]
      for (col in conditioning) {
        valid <- is.finite(syn[[col]])
        if (any(abs(syn[[col]][valid] - expected[[col]][valid]) > 1e-7))
          stop(paste('forge evidence order/value mismatch:', col))
      }
    }
    output[unlist(lapply(positions, function(i) ((i-1)*repeats+1):(i*repeats)))] <- as.numeric(syn[[feature]])
    output
  })
  unlist(values, use.names=FALSE)
}

record <- function(feature, conditioning_set, values) {
  diagnostics[[length(diagnostics)+1L]] <<- data.frame(
    feature=feature, conditioning_set=conditioning_set,
    n_draws=length(values), n_missing=sum(!is.finite(values)),
    n_negative=sum(values < 0, na.rm=TRUE),
    n_outside_train_range=sum(values < min(train[[feature]]) | values > max(train[[feature]]), na.rm=TRUE))
}

result <- data.frame(test_position=rep(info$importance_test_positions,each=repeats),
                     replicate=rep(seq_len(repeats),n))
feature_values <- parallel::mclapply(seq_along(features), function(j) {
  feature <- features[j]
  values <- sample_feature(feature,setdiff(features,feature),cfg$seed+j)
  cat(sprintf('[CFI %d/%d] %s: %d missing draws\n',j,length(features),feature,sum(!is.finite(values))))
  values
}, mc.cores=cfg$threads, mc.set.seed=FALSE)
for (j in seq_along(features)) {
  if (inherits(feature_values[[j]], 'try-error')) stop(feature_values[[j]])
  result[[features[j]]] <- feature_values[[j]]
  record(features[j],'All other features',feature_values[[j]])
}
fwrite(result,'results/intermediate/X_cond_resampled.csv')

# Fixed questions, not selected by test CFI significance.
top <- c('svcscan.nservices','svcscan.process_services','handles.avg_handles_per_proc',
         'handles.nthread','callbacks.ncallbacks','malfind.commitCharge','pslist.avg_threads')
if (!is.null(cfg$case_features)) top <- unlist(cfg$case_features, use.names=FALSE)
if (!length(top) || length(top) > 7L || anyDuplicated(top) || !all(top %in% features))
  stop('case_features must contain 1 to 7 distinct input features')
cases <- result[,c('test_position','replicate')]
for (j in seq_along(top)) {
  feature <- top[j]
  group <- sub('\\..*','',feature)
  same <- features[sub('\\..*','',features) == group]
  sets <- list('cond=same_group'=setdiff(same,feature),'cond=other_groups'=setdiff(features,same))
  cases[[paste0(feature,'|cond=all')]] <- result[[feature]]
  for (k in seq_along(sets)) {
    values <- sample_feature(feature,sets[[k]],cfg$seed+1000L+j*10L+k)
    cases[[paste0(feature,'|',names(sets)[k])]] <- values
    record(feature,names(sets)[k],values)
    cat(sprintf('[Case] %s %s: %d missing\n',feature,names(sets)[k],sum(!is.finite(values))))
  }
}
fwrite(cases,'results/intermediate/X_case_studies.csv')
fwrite(rbindlist(diagnostics),'results/tables/sampling_diagnostics.csv')
write_json(list(seed=cfg$seed,train_rows=nrow(train),test_rows=n,repeats=repeats,
                finite_bounds='no',nomatch='na',parallel_sampling=TRUE,seed_strategy="explicit per-feature seed; no nested RNG parallelism",
                arf_version=as.character(packageVersion('arf')),r_version=R.version.string,
                discriminator_accuracy=arf_obj$acc,
                active_features=active), 'results/intermediate/arf_metadata.json',auto_unbox=TRUE,pretty=TRUE)
cat('[ARF] Complete\n')
