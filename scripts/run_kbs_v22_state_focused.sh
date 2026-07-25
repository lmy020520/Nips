#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_v22_state_focused}"
DEFAULT_DATA_ROOT="data/hotpotqa_distractor_v22_state_focused"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BUILD_DATA="${BUILD_DATA:-1}"
FORCE_DATA="${FORCE_DATA:-0}"
TRAIN="${TRAIN:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
MAX_QIDS="${MAX_QIDS:-0}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-10}"
DEEP_REPEAT="${DEEP_REPEAT:-2}"
SEED="${SEED:-20260725}"
CONFIG="${CONFIG:-configs/train_ranker_deberta_v22_state_focused.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ranker/deberta_v3_large_v22_state_focused}"
LOG_DIR="${LOG_DIR:-outputs/logs/kbs_v22_state_focused}"

if [[ "$BUILD_DATA" == "1" ]]; then
  build_args=(
    --output-root "$DATA_ROOT"
    --candidate-top-k "$CANDIDATE_TOP_K"
    --deep-repeat "$DEEP_REPEAT"
    --seed "$SEED"
    --max-qids "$MAX_QIDS"
  )
  if [[ "$FORCE_DATA" == "1" ]]; then
    build_args+=(--force)
  elif [[ -f "$DATA_ROOT/manifest.json" ]]; then
    echo "[SKIP] data already exists: $DATA_ROOT (set FORCE_DATA=1 to rebuild)"
    build_args=()
  fi
  if (( ${#build_args[@]} )); then
    python3 scripts/build_kbs_v22_state_focused_data.py "${build_args[@]}"
  fi
fi

python3 scripts/check_kbs_v22_state_focused.py \
  --data-root "$DATA_ROOT" \
  --require-paths

if [[ "$TRAIN" != "1" ]]; then
  echo "[OK] v22 state-focused data and readiness checks completed"
  exit 0
fi

if [[ "$DATA_ROOT" != "$DEFAULT_DATA_ROOT" ]]; then
  echo "[ERROR] training config expects DATA_ROOT=$DEFAULT_DATA_ROOT; custom roots are data-only" >&2
  exit 1
fi

if [[ -f "$OUTPUT_DIR/best_model.pt" && "$FORCE_TRAIN" != "1" ]]; then
  echo "[SKIP] checkpoint exists: $OUTPUT_DIR/best_model.pt"
  exit 0
fi

mkdir -p "$LOG_DIR"
log="$LOG_DIR/train.log"
echo "[START] v22 state-focused training gpu=$CUDA_DEVICE config=$CONFIG log=$log"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 src/train/train_ranker.py --config "$CONFIG" \
  2>&1 | tee "$log"
echo "[DONE] checkpoint=$OUTPUT_DIR/best_model.pt"
