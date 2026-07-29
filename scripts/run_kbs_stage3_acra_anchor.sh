#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/train_ranker_deberta_v23_acra_anchor.yaml}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-configs/train_ranker_deberta_v22_state_focused.yaml}"
READINESS_OUTPUT="${READINESS_OUTPUT:-outputs/analysis/kbs_stage3_acra_anchor_readiness.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ranker/deberta_v3_large_v23_acra_anchor}"
LOG_DIR="${LOG_DIR:-outputs/logs/kbs_stage3_acra_anchor}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CHECK_ONLY="${CHECK_ONLY:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

python3 scripts/check_kbs_stage3_anchor_readiness.py \
  --config "$CONFIG" \
  --reference-config "$REFERENCE_CONFIG" \
  --output "$READINESS_OUTPUT" \
  --require-paths

if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "[OK] Stage 3.1 readiness passed; training was not started"
  exit 0
fi

if [[ -f "$OUTPUT_DIR/best_model.pt" && "$FORCE_TRAIN" != "1" ]]; then
  echo "[SKIP] checkpoint exists: $OUTPUT_DIR/best_model.pt"
  exit 0
fi

mkdir -p "$LOG_DIR"
log="$LOG_DIR/train.log"
echo "[START] Stage 3.1 ACRA-style anchor training gpu=$CUDA_DEVICE config=$CONFIG"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 src/train/train_ranker.py --config "$CONFIG" \
  2>&1 | tee "$log"
echo "[DONE] checkpoint=$OUTPUT_DIR/best_model.pt"
