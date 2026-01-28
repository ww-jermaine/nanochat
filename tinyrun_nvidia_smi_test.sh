#!/usr/bin/env bash
set -euo pipefail

# Mirrors `tinyrun.sh`, but adds per-phase GPU metrics logging via `nvidia-smi`.
# Output: CSV + metadata files under $NANOCHAT_BASE_DIR/metrics

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

SMI_QUERY="timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu,memory.used,memory.total"
SMI_FORMAT="csv"
SMI_INTERVAL_SECONDS="${SMI_INTERVAL_SECONDS:-1}"

GPU_LOG_PID=""

start_gpu_log() {
  local phase="$1"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local out="$METRICS_DIR/gpu_${phase}_${ts}.csv"

  echo "=== [metrics] START phase=$phase -> $out"
  stdbuf -oL nvidia-smi --query-gpu="$SMI_QUERY" --format="$SMI_FORMAT" -l "$SMI_INTERVAL_SECONDS" \
    > "$out" 2> "${out}.err" &
  GPU_LOG_PID=$!

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

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
else
  echo "Warning: $HOME/.cargo/env not found; ensure Cargo is on your PATH (e.g., reopen your shell or add ~/.cargo/bin to PATH)."
fi

if ! python -c "import rustbpe" >/dev/null 2>&1; then
  uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# -----------------------------------------------------------------------------
# 4) Download dataset (pretraining) and train tokenizer
#
# For a faster end-to-end GPU sanity test, you can set FAST_TEST=1 in the
# environment to drastically cut dataset size and tokenizer work.

FAST_TEST="${FAST_TEST:-0}"

if [ "$FAST_TEST" -eq 1 ]; then
  run_phase "download_dataset_and_tokenizer" bash -lc '
    set -euo pipefail
    # tiny subset for quick smoke test
    python -m nanochat.dataset -n 2
    python -m scripts.tok_train --max-chars=50000000
    python -m scripts.tok_eval
  '
else
  run_phase "download_dataset_and_tokenizer" bash -lc '
    set -euo pipefail
    python -m nanochat.dataset -n 8
    python -m nanochat.dataset -n 240 &
    DATASET_DOWNLOAD_PID=$!
    python -m scripts.tok_train --max-chars=2000000000
    python -m scripts.tok_eval
    echo "Waiting for dataset download to complete..."
    wait "$DATASET_DOWNLOAD_PID"
  '
fi

# -----------------------------------------------------------------------------
# 5) Tiny run parameters (defaults match tinyrun.sh)

if [ "$FAST_TEST" -eq 1 ]; then
  DEPTH="${DEPTH:-2}"
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
  NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
else
  DEPTH="${DEPTH:-4}"
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
  NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
fi

MODEL_TAG="d${DEPTH}"
MODEL_STEP="${NUM_ITERATIONS}"
WANDB_RUN_NAME="${WANDB_RUN:-dummy}"

# -----------------------------------------------------------------------------
# 6) Base train / eval

run_phase "base_train" torchrun --standalone -m scripts.base_train \
  --depth="$DEPTH" \
  --max-seq-len="$MAX_SEQ_LEN" \
  --device-batch-size=1 \
  --eval-tokens=512 \
  --core-metric-every=-1 \
  --total-batch-size=512 \
  --num-iterations="$NUM_ITERATIONS"

run_phase "base_loss" torchrun --standalone -m scripts.base_loss \
  --device-batch-size=1 \
  --split-tokens=512 \
  --model-tag="$MODEL_TAG" \
  --model-step="$MODEL_STEP"

run_phase "base_eval" torchrun --standalone -m scripts.base_eval \
  --max-per-task=16 \
  --model-tag="$MODEL_TAG" \
  --step="$MODEL_STEP"

FAST_SKIPS_LATE="${FAST_SKIPS_LATE:-$FAST_TEST}"

# -----------------------------------------------------------------------------
# 7) Identity conversations + mid train / eval

if [ "$FAST_SKIPS_LATE" -eq 1 ]; then
  echo "FAST_TEST=1: Skipping mid + SFT phases; only base training/eval will run."
else
  run_phase "download_identity_conversations" curl -L \
    -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

  run_phase "mid_train" torchrun --standalone -m scripts.mid_train -- \
    --device-batch-size=1 \
    --max-seq-len="$MAX_SEQ_LEN" \
    --model-tag="$MODEL_TAG" \
    --model-step="$MODEL_STEP" \
    --run="$WANDB_RUN_NAME"

  run_phase "chat_eval_mid" torchrun --standalone -m scripts.chat_eval -- \
    -i mid \
    --model-tag="$MODEL_TAG"

  # -----------------------------------------------------------------------------
  # 8) SFT + eval

  run_phase "chat_sft" torchrun --standalone -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN_NAME"

  run_phase "chat_eval_sft" torchrun --standalone -m scripts.chat_eval -- \
    -i sft \
    --model-tag="$MODEL_TAG"
fi

# -----------------------------------------------------------------------------
# 9) Report

run_phase "report_generate" python -m nanochat.report generate

# Best-effort HTML GPU metrics report; do not fail the training run if it errors.
python -m scripts.metrics_report --metrics-dir "$NANOCHAT_BASE_DIR/metrics" --out "$NANOCHAT_BASE_DIR/metrics/report.html" || \
  echo "Warning: metrics HTML report generation failed."

# Notes:
# - Chat CLI: `python -m scripts.chat_cli -p "Why is the sky blue?"`
# - Web UI:   `python -m scripts.chat_web`

