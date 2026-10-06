#!/usr/bin/env bash
# Run ONCE on a fresh GPU pod, from the repo's sound-lab/ directory:
#   bash runpod/bootstrap.sh
# Installs pinned deps, starts the idle watchdog, prints a sanity line. Does not download models
# (the first extract.py run does that, into HF_HOME on the persistent volume).
# Verified 2026-10-05 on an RTX 4090 secure pod, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# (CUDA 12.8 driver 570, /workspace persistent volume).
set -euo pipefail
cd "$(dirname "$0")/.."

# SSH sessions don't inherit the container's env, so pull the pod id + API key the watchdog needs
# (runpodctl stop) from PID 1. Verified 2026-10-05: without this RUNPOD_POD_ID is empty over SSH.
if [[ -z "${RUNPOD_POD_ID:-}" && -r /proc/1/environ ]]; then
  set -a; eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=')"; set +a
fi

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"   # model weights survive pod restarts
mkdir -p "$HF_HOME" runs logs

# Ubuntu 24.04 images refuse system-wide pip installs (PEP 668), so install into a venv.
# With the EU-RO-1 network volume (2026-10-06): the pinned wheels live in /workspace/wheels and the
# venv goes on the pod's LOCAL disk, linked as .venv. Measured: a venv ON the network volume took
# 33 min to install and 3 min to import (many small files over the network); from saved wheels to
# local disk, 79 s. Without the volume, fall back to downloading into .venv as before.
if [[ -d /workspace/wheels ]]; then
  VENV=/root/venv
  python3 -m venv "$VENV"
  W=(--no-index --find-links /workspace/wheels)
  "$VENV/bin/pip" install -q "${W[@]}" pip
  "$VENV/bin/pip" install -q "${W[@]}" torch==2.9.1
  "$VENV/bin/pip" install -q "${W[@]}" -r requirements-pod.txt
  rm -rf .venv && ln -s "$VENV" .venv
  PY=.venv/bin/python
else
  [[ -x .venv/bin/python ]] || python3 -m venv .venv
  PY=.venv/bin/python
  "$PY" -m pip install --upgrade pip
  # CUDA 12.8 wheels; change cu128 if the pod's driver is older (check: nvidia-smi, top right).
  "$PY" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
  "$PY" -m pip install -r requirements-pod.txt
fi

"$PY" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__)
from transformers import Qwen2AudioForConditionalGeneration, MusicFlamingoForConditionalGeneration  # noqa
PY

# CPU tests also run here, so the pod is checked against the same suite as the laptop.
bash verify.sh

# Idle watchdog: stops the pod after IDLE_MINUTES of no GPU activity (default 20).
nohup bash runpod/idle_watchdog.sh > logs/watchdog.log 2>&1 &
echo "watchdog started (pid $!), log: logs/watchdog.log"
echo "HF_HOME=$HF_HOME  -- export this in your shell before running extract.py"
