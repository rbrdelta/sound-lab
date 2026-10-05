#!/usr/bin/env bash
# CPU-only checks for sound-lab. Uses sound-lab/.venv if present, else python3.
set -euo pipefail
cd "$(dirname "$0")"
PY=python3
[[ -x .venv/bin/python ]] && PY=.venv/bin/python
"$PY" -m pytest -q tests
"$PY" simulate_clips.py --sims 500 --accuracies 0.67 0.9 >/dev/null && echo "simulate_clips.py: runs"
for f in runpod/*.sh; do bash -n "$f"; done && echo "runpod scripts: syntax ok"
