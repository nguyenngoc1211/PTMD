#!/usr/bin/env bash
# Two-tier RAM guard for long experiment runs on this box.
#
# WARN tier (default 1200MB): log one line, throttled. Nothing is signalled,
# because a SIGTERM to a Python ablation driver destroys a multi-hour run and
# frees nothing that was about to OOM.
# KILL tier (default 300MB): SIGTERM only the processes matching
# WATCHDOG_KILL_PATTERN. The default pattern is the R sampler child, so the ARF
# driver can record the seed in blocked_seeds.csv and move on. Broadening the
# pattern to the Python drivers is a deliberate choice made per run, not here.
set -u

THRESHOLD_MB="${WATCHDOG_THRESHOLD_MB:-300}"
WARN_MB="${WATCHDOG_WARN_MB:-1200}"
INTERVAL_S="${WATCHDOG_INTERVAL_S:-5}"
WARN_EVERY_S="${WATCHDOG_WARN_EVERY_S:-120}"
KILL_PATTERN="${WATCHDOG_KILL_PATTERN:-exec/R --no-echo}"

available_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

top_consumer() {
  ps -eo rss,comm --sort=-rss 2>/dev/null | awk 'NR==2{printf "%s %dMB", $2, $1/1024}'
}

last_warn=0
echo "[watchdog] start kill=${THRESHOLD_MB}MB warn=${WARN_MB}MB interval=${INTERVAL_S}s pattern='${KILL_PATTERN}'"
while true; do
  avail=$(available_mb)
  now=$(date +%s)
  if [ -z "${avail:-}" ]; then sleep "$INTERVAL_S"; continue; fi

  if [ "$avail" -lt "$THRESHOLD_MB" ]; then
    for pid in $(pgrep -f "$KILL_PATTERN"); do
      rss_kb=$(awk '/VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
      kill -TERM "$pid" 2>/dev/null && \
        echo "[watchdog] CRITICAL avail=${avail}MB -> SIGTERM pid=$pid rss=$((rss_kb / 1024))MB top=$(top_consumer)"
    done
    last_warn=0
  elif [ "$avail" -lt "$WARN_MB" ] && [ $((now - last_warn)) -ge "$WARN_EVERY_S" ]; then
    echo "[watchdog] WARN avail=${avail}MB top=$(top_consumer) running=$(pgrep -fc 'run_group_' 2>/dev/null || echo 0)"
    last_warn=$now
  fi
  sleep "$INTERVAL_S"
done
