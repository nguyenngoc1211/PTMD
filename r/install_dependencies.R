# Install a minimal, project-local R environment. Requires network and a C++ compiler.
args <- commandArgs(trailingOnly=FALSE)
script <- sub('^--file=', '', args[grepl('^--file=', args)][1])
root <- normalizePath(file.path(dirname(script), '..'))
lib <- file.path(root,'RLibrary')
dir.create(lib,showWarnings=FALSE,recursive=TRUE)
.libPaths(c(lib,.libPaths()))
packages <- c('data.table','ranger','foreach','jsonlite','truncnorm','stringr','remotes')
missing <- packages[!vapply(packages,requireNamespace,quietly=TRUE,FUN.VALUE=logical(1))]
if (length(missing)) install.packages(missing,lib=lib,repos='https://cloud.r-project.org')
commit <- 'f08c562b76144f652db8b3dbcdd156cced0c1291'
installed_commit <- if (requireNamespace('arf',quietly=TRUE)) packageDescription('arf')$RemoteSha else NULL
if (!identical(installed_commit,commit))
  remotes::install_github(paste0('bips-hb/arf@',commit),lib=lib,upgrade='never')
cat('Project R dependencies ready in',lib,'\n')
