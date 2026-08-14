#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOURCE_ROOT="${SOURCE_ROOT:-data/hotpotqa_distractor_v27_counterfactual_dual}"
DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_v29_coverage_greedy}"
CONFIG="${CONFIG:-configs/train_ranker_deberta_v29_coverage_greedy.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ranker/deberta_v3_large_v29_coverage_greedy}"
READINESS_OUTPUT="${READINESS_OUTPUT:-outputs/analysis/kbs_stage6_coverage_student/readiness.json}"
LOG_DIR="${LOG_DIR:-outputs/logs/kbs_stage6_coverage_student}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BUILD_DATA="${BUILD_DATA:-1}"
CHECK_ONLY="${CHECK_ONLY:-1}"
TRAIN="${TRAIN:-0}"

if [[ "$BUILD_DATA" == "1" ]]; then
  if [[ -f "$DATA_ROOT/manifest.json" ]]; then
    echo "[SKIP] data exists: $DATA_ROOT"
  else
    python3 scripts/build_kbs_stage6_coverage_student_data.py \
      --source-root "$SOURCE_ROOT" \
      --output-root "$DATA_ROOT"
  fi
fi

mkdir -p "$(dirname "$READINESS_OUTPUT")"
python3 scripts/check_kbs_stage6_coverage_student.py \
  --source-root "$SOURCE_ROOT" \
  --data-root "$DATA_ROOT" \
  --config "$CONFIG" \
  --require-paths \
  --output "$READINESS_OUTPUT"

if [[ "$CHECK_ONLY" == "1" || "$TRAIN" != "1" ]]; then
  echo "[OK] Stage 6.1 Coverage Student readiness passed; training was not started"
  exit 0
fi

if [[ -f "$OUTPUT_DIR/best_model.pt" ]]; then
  echo "[SKIP] checkpoint exists: $OUTPUT_DIR/best_model.pt"
  exit 0
fi

mkdir -p "$LOG_DIR"
echo "[START] Stage 6.1 Coverage Student training gpu=$CUDA_DEVICE"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 src/train/train_ranker.py --config "$CONFIG" \
  2>&1 | tee "$LOG_DIR/train.log"
echo "[DONE] checkpoint=$OUTPUT_DIR/best_model.pt"
