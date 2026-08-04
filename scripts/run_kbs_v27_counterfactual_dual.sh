#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOURCE_ROOT="${SOURCE_ROOT:-data/hotpotqa_distractor_v22_state_focused}"
FULL_DATA_ROOT="data/hotpotqa_distractor_v27_counterfactual_dual"
MAX_QIDS="${MAX_QIDS:-20}"
if [[ "$MAX_QIDS" == "0" ]]; then
  DATA_ROOT="${DATA_ROOT:-$FULL_DATA_ROOT}"
else
  DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_v27_counterfactual_dual_smoke${MAX_QIDS}}"
fi
CONFIG="${CONFIG:-configs/train_ranker_deberta_v27_counterfactual_dual.yaml}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BUILD_DATA="${BUILD_DATA:-1}"
FORCE_DATA="${FORCE_DATA:-0}"
CHECK_ONLY="${CHECK_ONLY:-1}"
TRAIN="${TRAIN:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ranker/deberta_v3_large_v27_counterfactual_dual}"
LOG_DIR="${LOG_DIR:-outputs/logs/kbs_v27_counterfactual_dual}"

if [[ "$BUILD_DATA" == "1" ]]; then
  if [[ -f "$DATA_ROOT/manifest.json" && "$FORCE_DATA" != "1" ]]; then
    echo "[SKIP] data exists: $DATA_ROOT (set FORCE_DATA=1 to rebuild)"
  else
    build_args=(
      --source-root "$SOURCE_ROOT"
      --output-root "$DATA_ROOT"
      --later-repeat 2
      --max-qids "$MAX_QIDS"
      --seed 20260805
    )
    if [[ "$FORCE_DATA" == "1" ]]; then
      build_args+=(--force)
    fi
    python3 scripts/build_kbs_v27_counterfactual_data.py "${build_args[@]}"
  fi
fi

python3 scripts/check_kbs_v27_counterfactual_dual.py \
  --data-root "$DATA_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --config "$CONFIG" \
  --require-paths

if [[ "$CHECK_ONLY" == "1" || "$TRAIN" != "1" ]]; then
  echo "[OK] v27 data/readiness passed; training was not started"
  exit 0
fi

if [[ "$MAX_QIDS" != "0" || "$DATA_ROOT" != "$FULL_DATA_ROOT" ]]; then
  echo "[ERROR] training is allowed only with MAX_QIDS=0 and DATA_ROOT=$FULL_DATA_ROOT" >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/best_model.pt" && "$FORCE_TRAIN" != "1" ]]; then
  echo "[SKIP] checkpoint exists: $OUTPUT_DIR/best_model.pt"
  exit 0
fi

mkdir -p "$LOG_DIR"
log="$LOG_DIR/train.log"
echo "[START] v27 counterfactual dual training gpu=$CUDA_DEVICE config=$CONFIG log=$log"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 src/train/train_ranker.py --config "$CONFIG" \
  2>&1 | tee "$log"
echo "[DONE] checkpoint=$OUTPUT_DIR/best_model.pt"
