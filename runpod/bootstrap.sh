#!/usr/bin/env bash
# Run ONCE on a fresh GPU pod, from the repo's sound-lab/ directory:
#   bash runpod/bootstrap.sh
# Installs pinned deps, starts the idle watchdog, prints a sanity line. Does not download models
# (the first extract.py run does that, into HF_HOME on the persistent volume).
# UNVERIFIED: written without a pod. Assumes a Runpod PyTorch image with CUDA 12.x, python3,
# nvidia-smi, and /workspace as the persistent volume.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"   # model weights survive pod restarts
mkdir -p "$HF_HOME" runs logs

python3 -m pip install --upgrade pip
# CUDA 12.8 wheels; change cu128 if the pod's driver is older (check: nvidia-smi, top right).
python3 -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -r requirements-pod.txt

python3 - <<'PY'
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
