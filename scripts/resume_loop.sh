#!/usr/bin/env bash
# Keep one checkpointed cell running across in-VM deaths.
#
# This cannot help when the WSL VM itself is recycled: everything inside the
# distro dies with it, including this loop. What it does cover is a Python
# process killed from inside a live VM (OOM killer, watchdog, a bug), which
# would otherwise leave a 15-hour cell idle until a human noticed. --checkpoint
# makes each relaunch resume from the last completed block, so the cost of an
# interruption is one control block, not the cell.
#
# Usage: resume_loop.sh <progress_csv> <command...>
# Aborts if an attempt adds no rows: a relaunch that makes no progress is a
# reproducible failure, and looping on it burns the night.
set -uo pipefail

PROGRESS_CSV=$1
shift
MAX_ATTEMPTS="${RESUME_MAX_ATTEMPTS:-8}"
GATE_MB="${RESUME_GATE_MB:-1500}"
GATE_WAIT_S="${RESUME_GATE_WAIT_S:-1200}"

rows() { [ -f "$PROGRESS_CSV" ] && wc -l < "$PROGRESS_CSV" || echo 0; }

available_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

wait_for_ram() {
  local deadline=$((SECONDS + GATE_WAIT_S))
  while [ "$(available_mb)" -lt "$GATE_MB" ] && [ "$SECONDS" -lt "$deadline" ]; do
    echo "[resume] avail=$(available_mb)MB < ${GATE_MB}MB, waiting"
    sleep 60
  done
  [ "$(available_mb)" -ge "$GATE_MB" ]
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  before=$(rows)
  echo "[resume] attempt $attempt/$MAX_ATTEMPTS rows=$before $(date +%FT%T)"
  if ! wait_for_ram; then
    echo "[resume] RAM gate not met after ${GATE_WAIT_S}s, giving up"
    exit 1
  fi
  "$@"
  rc=$?
  after=$(rows)
  if [ "$rc" -eq 0 ]; then
    echo "[resume] finished rc=0 rows=$after $(date +%FT%T)"
    exit 0
  fi
  echo "[resume] rc=$rc rows=$before->$after $(date +%FT%T)"
  if [ "$after" -le "$before" ]; then
    echo "[resume] no new rows this attempt, aborting instead of looping"
    exit 1
  fi
  sleep 30
done
echo "[resume] exhausted $MAX_ATTEMPTS attempts"
exit 1
