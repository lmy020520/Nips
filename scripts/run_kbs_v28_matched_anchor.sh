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
DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_v28_matched_anchor}"
BUILD_DATA="${BUILD_DATA:-1}"
CHECK_ONLY="${CHECK_ONLY:-1}"
TRAIN="${TRAIN:-0}"
RUN="${RUN:-seed42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
READINESS_OUTPUT="${READINESS_OUTPUT:-outputs/analysis/kbs_v28_matched_anchor/readiness.json}"

case "$RUN" in
  seed42)
    CONFIG="configs/train_ranker_deberta_v28_matched_anchor.yaml"
    OUTPUT_DIR="outputs/ranker/deberta_v3_large_v28_matched_anchor"
    ;;
  seed43)
    CONFIG="configs/train_ranker_deberta_v28_matched_anchor_seed43.yaml"
    OUTPUT_DIR="outputs/ranker/deberta_v3_large_v28_matched_anchor_seed43"
    ;;
  seed44)
    CONFIG="configs/train_ranker_deberta_v28_matched_anchor_seed44.yaml"
    OUTPUT_DIR="outputs/ranker/deberta_v3_large_v28_matched_anchor_seed44"
    ;;
  *)
    echo "[ERROR] unsupported RUN=$RUN; expected seed42, seed43, or seed44" >&2
    exit 2
    ;;
esac

if [[ "$BUILD_DATA" == "1" ]]; then
  if [[ -f "$DATA_ROOT/manifest.json" ]]; then
    echo "[SKIP] matched Anchor data exists: $DATA_ROOT"
  else
    python3 scripts/build_kbs_v28_matched_anchor_data.py \
      --source-root "$SOURCE_ROOT" \
      --output-root "$DATA_ROOT"
  fi
fi

mkdir -p "$(dirname "$READINESS_OUTPUT")"
python3 scripts/check_kbs_v28_matched_anchor.py \
  --data-root "$DATA_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --require-paths | tee "$READINESS_OUTPUT"

if [[ "$CHECK_ONLY" == "1" || "$TRAIN" != "1" ]]; then
  echo "[OK] Stage 5 matched Anchor readiness passed; training was not started"
  exit 0
fi

if [[ -f "$OUTPUT_DIR/best_model.pt" ]]; then
  echo "[ERROR] refusing to overwrite existing checkpoint: $OUTPUT_DIR/best_model.pt" >&2
  exit 1
fi

LOG_DIR="outputs/logs/kbs_v28_matched_anchor/$RUN"
mkdir -p "$LOG_DIR"
echo "[START] matched Anchor training run=$RUN gpu=$CUDA_DEVICE config=$CONFIG"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 src/train/train_ranker.py --config "$CONFIG" \
  2>&1 | tee "$LOG_DIR/train.log"
echo "[DONE] checkpoint=$OUTPUT_DIR/best_model.pt"
