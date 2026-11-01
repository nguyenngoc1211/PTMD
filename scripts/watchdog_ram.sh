#!/usr/bin/env bash
# Bảo vệ máy khỏi OOM khi R sampler (ARF) chạy: SIGTERM đúng process R,
# để driver Python ghi seed đó vào blocked_seeds.csv và đi tiếp seed sau.
# Không đụng driver Python, không đụng process nào khác.
set -u

THRESHOLD_MB="${WATCHDOG_THRESHOLD_MB:-300}"
INTERVAL_S="${WATCHDOG_INTERVAL_S:-5}"
PATTERN='exec/R --no-echo'

available_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

echo "[watchdog] start threshold=${THRESHOLD_MB}MB interval=${INTERVAL_S}s"
while true; do
  avail=$(available_mb)
  if [ -n "${avail:-}" ] && [ "$avail" -lt "$THRESHOLD_MB" ]; then
    for pid in $(pgrep -f "$PATTERN"); do
      rss_kb=$(awk '/VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
      kill -TERM "$pid" 2>/dev/null && \
        echo "[watchdog] avail=${avail}MB -> SIGTERM R pid=$pid rss=$((rss_kb / 1024))MB"
    done
  fi
  sleep "$INTERVAL_S"
done
