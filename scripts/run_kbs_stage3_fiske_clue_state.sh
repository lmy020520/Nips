#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_v26_fiske_clue_state}"
CONFIG="${CONFIG:-configs/train_ranker_deberta_v26_fiske_clue_state.yaml}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-configs/train_ranker_deberta_v22_state_focused.yaml}"
READINESS_OUTPUT="${READINESS_OUTPUT:-outputs/analysis/kbs_stage3_fiske_clue_state_readiness.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ranker/deberta_v3_large_v26_fiske_clue_state}"
LOG_DIR="${LOG_DIR:-outputs/logs/kbs_stage3_fiske_clue_state}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_QIDS="${MAX_QIDS:-0}"
BUILD_DATA="${BUILD_DATA:-1}"
FORCE_DATA="${FORCE_DATA:-0}"
CHECK_ONLY="${CHECK_ONLY:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

if [[ "$BUILD_DATA" == "1" ]]; then
  if [[ -f "$DATA_ROOT/manifest.json" && "$FORCE_DATA" != "1" ]]; then
    echo "[SKIP] clue-state data exists: $DATA_ROOT (set FORCE_DATA=1 to rebuild)"
  else
    build_args=(
      --output-root "$DATA_ROOT"
      --max-qids "$MAX_QIDS"
      --seed 42
    )
    if [[ "$FORCE_DATA" == "1" ]]; then
      build_args+=(--force)
    fi
    python3 scripts/build_kbs_v26_fiske_clue_state_data.py "${build_args[@]}"
  fi
fi

python3 scripts/check_kbs_v26_fiske_clue_state.py \
  --data-root "$DATA_ROOT" \
  --config "$CONFIG" \
  --reference-config "$REFERENCE_CONFIG" \
  --output "$READINESS_OUTPUT" \
  --require-paths

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "[OK] Stage 3.3 readiness passed; training was not started"
  exit 0
fi

if [[ "$MAX_QIDS" != "0" ]]; then
  echo "[ERROR] training is forbidden for MAX_QIDS smoke data" >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/best_model.pt" && "$FORCE_TRAIN" != "1" ]]; then
  echo "[SKIP] checkpoint exists: $OUTPUT_DIR/best_model.pt"
  exit 0
fi

mkdir -p "$LOG_DIR"
log="$LOG_DIR/train.log"
echo "[START] Stage 3.3 FiSKE-inspired clue-state training gpu=$CUDA_DEVICE config=$CONFIG"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 src/train/train_ranker.py --config "$CONFIG" \
  2>&1 | tee "$log"
echo "[DONE] checkpoint=$OUTPUT_DIR/best_model.pt"
