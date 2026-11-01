#!/usr/bin/env bash
# Chain a heavy job behind a running one so two memory-hungry runs never overlap
# on this 4,9 GB machine. usage: chain_after.sh <pid-to-wait> <name> <python-args...>
set -uo pipefail
cd "$(dirname "$0")/.."

PID="${1:?usage: chain_after.sh <pid-to-wait> <out-name> <argv...>}"
NAME="${2:?missing out-name}"
shift 2
LOG="results/group_experiments/${NAME}.log"

while kill -0 "$PID" 2>/dev/null; do sleep 60; done

avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ "${avail:-0}" -lt 1500 ]; then
  echo "[chain] $(date -Is) SKIP: only ${avail}MB available" >>"$LOG"
  exit 1
fi
echo "[chain] $(date -Is) pid=$PID finished, avail=${avail}MB -> starting: $*" >>"$LOG"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python "$@" >>"$LOG" 2>&1
rc=$?
echo "[chain] $(date -Is) finished rc=$rc out-name=${NAME}" >>"$LOG"
exit "$rc"
