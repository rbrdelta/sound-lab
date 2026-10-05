#!/usr/bin/env bash
# Stops this Runpod pod after IDLE_MINUTES consecutive minutes with GPU utilisation at or below
# IDLE_UTIL percent. Started by bootstrap.sh; safe to run by hand:
#   IDLE_MINUTES=20 nohup bash runpod/idle_watchdog.sh > logs/watchdog.log 2>&1 &
# To pause it for a long CPU-only step: touch /tmp/watchdog.hold (remove to resume).
#
# UNVERIFIED (no pod yet): that `runpodctl` is present on the pod and that RUNPOD_POD_ID and an
# API key are injected into the pod environment. If `runpodctl stop pod` fails, the fallback
# kills PID 1, which ends the container -- check the Runpod console that billing actually stopped
# the first time it fires. A STOPPED pod still bills for its volume disk; TERMINATE removes it.
set -uo pipefail

IDLE_MINUTES="${IDLE_MINUTES:-20}"
IDLE_UTIL="${IDLE_UTIL:-5}"           # percent; model loading from disk can sit near 0, see below
INTERVAL="${INTERVAL:-60}"            # seconds between checks
GRACE_MINUTES="${GRACE_MINUTES:-15}"  # no shutdown in the first N minutes after start

log() { echo "$(date -u +%FT%TZ) $*"; }

busy() {
  # GPU busy if utilisation is above threshold OR any process holds GPU memory and is computing.
  local util
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -1)
  [[ -z "$util" ]] && { log "nvidia-smi unavailable"; return 0; }   # fail safe: never stop blind
  (( util > IDLE_UTIL ))
}

stop_pod() {
  log "idle for ${IDLE_MINUTES} min -- stopping pod ${RUNPOD_POD_ID:-?}"
  if command -v runpodctl >/dev/null && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    runpodctl stop pod "$RUNPOD_POD_ID" && exit 0
    log "runpodctl stop failed"
  fi
  log "fallback: ending container"
  kill -TERM 1
}

start=$(date +%s)
idle=0
log "watchdog: stop after ${IDLE_MINUTES} idle min (util <= ${IDLE_UTIL}%), grace ${GRACE_MINUTES} min"
while true; do
  sleep "$INTERVAL"
  if [[ -e /tmp/watchdog.hold ]]; then idle=0; continue; fi
  if busy; then
    idle=0
  else
    idle=$((idle + INTERVAL))
  fi
  elapsed=$(( $(date +%s) - start ))
  if (( elapsed >= GRACE_MINUTES * 60 && idle >= IDLE_MINUTES * 60 )); then
    stop_pod
  fi
done
