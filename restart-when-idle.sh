#!/bin/bash
# Restart sse-map-proxy once it has been idle for ~60s (no in-flight SSE streams).
# Safe to launch anytime: it waits for a quiet window, so it never cuts a live reply.
LOG=/var/log/sse-map-proxy.log
SETTLE=12         # 12 * 5s = 60s of continuous idle required
base_req=$(grep -ac "] REQ " "$LOG" 2>/dev/null || echo 0)
base_done=$(grep -ac "] done=" "$LOG" 2>/dev/null || echo 0)
idle=0
while :; do
  sleep 5
  req=$(grep -ac "] REQ " "$LOG" 2>/dev/null || echo 0)
  done=$(grep -ac "] done=" "$LOG" 2>/dev/null || echo 0)
  if [ "$req" -lt "$base_req" ] || [ "$done" -lt "$base_done" ]; then
    base_req=$req; base_done=$done; idle=0; continue
  fi
  active=$(( req - base_req - (done - base_done) ))
  if [ "$active" -le 0 ]; then idle=$((idle + 1)); else idle=0; fi
  if [ "$idle" -ge "$SETTLE" ]; then
    systemctl restart sse-map-proxy
    echo "$(date '+%Y-%m-%d %H:%M:%S') restarted after idle"
    exit 0
  fi
done
