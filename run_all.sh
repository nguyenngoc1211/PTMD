#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment missing. Follow README.md or set PYTHON to an executable." >&2
  exit 1
fi
export R_LIBS_USER="$PWD/RLibrary${R_LIBS_USER:+:$R_LIBS_USER}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
"$PYTHON" -c 'import numpy, pandas, sklearn, scipy, xgboost, matplotlib, seaborn, joblib'
Rscript -e 'stopifnot(requireNamespace("arf",quietly=TRUE), requireNamespace("data.table",quietly=TRUE), requireNamespace("jsonlite",quietly=TRUE), requireNamespace("foreach",quietly=TRUE))'
"$PYTHON" python/preprocess.py
"$PYTHON" python/train_xgboost.py
"$PYTHON" python/export_for_carfi.py
"$PYTHON" python/marginal_importance.py
Rscript r/conditional_sampling.R
"$PYTHON" python/compute_cfi.py
"$PYTHON" python/analyze_results.py
"$PYTHON" python/generate_report.py
"$PYTHON" python/verify_results.py
echo 'Pipeline, report and verification complete.'
