#!/usr/bin/env bash
set -euo pipefail

# Many would like to run nanochat on a single GPU or a tiny cluster.
# This script is the "Best ChatGPT clone that a crappy single GPU can buy".
# It is designed to run in ~1 hour on a single 3080 GPU with ~10GB of VRAM.
#
# Adds: GPU metrics logging via nvidia-smi (background) with per-phase CSV files.

# -----------------------------------------------------------------------------
# 0) Basics

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Disable wandb prompts unless explicitly enabled
if [ -z "${WANDB_RUN:-}" ]; then
  export WANDB_MODE=disabled
  export WANDB_SILENT=true
  export WANDB_DISABLED=true
fi

# -----------------------------------------------------------------------------
# 1) GPU metrics logging (nvidia-smi)

command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi not found. Install NVIDIA drivers/tools or run on a host with GPU access."
  exit 1
}

METRICS_DIR="$NANOCHAT_BASE_DIR/metrics"
mkdir -p "$METRICS_DIR"

# Query fields (add/remove as you like)
SMI_QUERY="timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu,memory.used,memory.total"
SMI_FORMAT="csv"
SMI_INTERVAL_SECONDS="${SMI_INTERVAL_SECONDS:-1}"  # sampling interval

GPU_LOG_PID=""

start_gpu_log() {
  local phase="$1"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local out="$METRICS_DIR/gpu_${phase}_${ts}.csv"

  echo "=== [metrics] START phase=$phase -> $out"
  # Line-buffer output to ensure continuous writes
  stdbuf -oL nvidia-smi --query-gpu="$SMI_QUERY" --format="$SMI_FORMAT" -l "$SMI_INTERVAL_SECONDS" \
    > "$out" 2> "${out}.err" &
  GPU_LOG_PID=$!

  # Metadata for reproducibility
  {
    echo "phase=$phase"
    echo "started_at=$(date -Is)"
    echo "interval_seconds=$SMI_INTERVAL_SECONDS"
    echo "command=nvidia-smi --query-gpu=$SMI_QUERY --format=$SMI_FORMAT -l $SMI_INTERVAL_SECONDS"
    echo ""
    echo "[nvidia-smi -L]"
    nvidia-smi -L || true
    echo ""
    echo "[gpu name/driver/bus]"
    nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv || true
  } > "${out%.csv}.meta.txt"
}

stop_gpu_log() {
  if [ -n "${GPU_LOG_PID}" ] && kill -0 "${GPU_LOG_PID}" 2>/dev/null; then
    echo "=== [metrics] STOP pid=$GPU_LOG_PID"
    kill "${GPU_LOG_PID}" || true
    wait "${GPU_LOG_PID}" 2>/dev/null || true
  fi
  GPU_LOG_PID=""
}

cleanup() {
  stop_gpu_log
}
trap cleanup EXIT INT TERM

# Convenience: run a command with phase logging
run_phase() {
  local phase="$1"
  shift
  start_gpu_log "$phase"
  "$@"
  stop_gpu_log
}

# -----------------------------------------------------------------------------
# 2) Python venv setup with uv

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
# shellcheck disable=SC1091
source .venv/bin/activate

# -----------------------------------------------------------------------------
# 3) Install Rust / Cargo (if not already installed)

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

# Prefer Rustup's standard env file if present; otherwise, warn but continue.
if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
else
  echo "Warning: $HOME/.cargo/env not found; ensure Cargo is on your PATH (e.g., reopen your shell or add ~/.cargo/bin to PATH)."
fi

# Build the rustbpe Tokenizer (if not already built)
if ! python -c "import rustbpe" >/dev/null 2>&1; then
  uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# -----------------------------------------------------------------------------
# 4) Download dataset (pretraining) and train tokenizer

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

# -----------------------------------------------------------------------------
# 5) Model params (tiny run)

DEPTH=4
MAX_SEQ_LEN=512
NUM_ITERATIONS=20

MODEL_TAG="d${DEPTH}"
MODEL_STEP="${NUM_ITERATIONS}"
WANDB_RUN_NAME="${WANDB_RUN:-dummy}"

# -----------------------------------------------------------------------------
# 6) Train / eval pipeline with GPU metrics per phase

run_phase "01_pretrain_base_train" \
  torchrun --standalone -m scripts.base_train \
    --depth="$DEPTH" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size=1 \
    --eval-tokens=512 \
    --core-metric-every=-1 \
    --total-batch-size=512 \
    --num-iterations="$NUM_ITERATIONS"

run_phase "02_pretrain_base_loss" \
  torchrun --standalone -m scripts.base_loss \
    --device-batch-size=1 \
    --split-tokens=512 \
    --model-tag="$MODEL_TAG" \
    --model-step="$MODEL_STEP"

run_phase "03_pretrain_base_eval" \
  torchrun --standalone -m scripts.base_eval \
    --max-per-task=16 \
    --model-tag="$MODEL_TAG" \
    --step="$MODEL_STEP"

# -----------------------------------------------------------------------------
# 7) Download identity conversations (for SFT personality)

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# -----------------------------------------------------------------------------
# 8) Mid training + eval

run_phase "04_mid_train" \
  torchrun --standalone -m scripts.mid_train -- \
    --device-batch-size=1 \
    --max-seq-len="$MAX_SEQ_LEN" \
    --model-tag="$MODEL_TAG" \
    --model-step="$MODEL_STEP" \
    --run="$WANDB_RUN_NAME"

run_phase "05_mid_eval" \
  torchrun --standalone -m scripts.chat_eval -- \
    -i mid \
    --model-tag="$MODEL_TAG"

# -----------------------------------------------------------------------------
# 9) SFT training + eval

run_phase "06_sft_train" \
  torchrun --standalone -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN_NAME"

run_phase "07_sft_eval" \
  torchrun --standalone -m scripts.chat_eval -- \
    -i sft \
    --model-tag="$MODEL_TAG"

# -----------------------------------------------------------------------------
# 10) Report generation (training + HTML GPU metrics dashboard)

python -m nanochat.report generate

# Best-effort HTML GPU metrics report; do not fail the training run if it errors.
python -m scripts.metrics_report --metrics-dir "$METRICS_DIR" --out "$METRICS_DIR/report.html" || \
  echo "Warning: metrics HTML report generation failed."

echo ""
echo "Done."
echo "GPU metrics CSVs saved in: $METRICS_DIR"
echo "Example: ls -lh $METRICS_DIR"
echo "HTML GPU dashboard (if generated): $METRICS_DIR/report.html"