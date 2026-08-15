#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/kbs_stage6_compression_funnel}"
READINESS_OUTPUT="$OUTPUT_DIR/readiness.json"
SUMMARY_OUTPUT="$OUTPUT_DIR/summary.json"
RECORDS_OUTPUT="$OUTPUT_DIR/records.jsonl"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_QIDS="${MAX_QIDS:-0}"
RUN="${RUN:-0}"

mkdir -p "$OUTPUT_DIR"

python3 scripts/check_kbs_stage6_compression_funnel.py \
  --output "$READINESS_OUTPUT"

if [[ "$RUN" != "1" ]]; then
  echo "[OK] Stage 6.2 readiness passed; GPU inference was not started"
  exit 0
fi

if [[ -f "$SUMMARY_OUTPUT" ]]; then
  echo "[SKIP] completed summary exists: $SUMMARY_OUTPUT"
  exit 0
fi

echo "[START] Stage 6.2 compression funnel gpu=$CUDA_DEVICE max_qids=$MAX_QIDS"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/analyze_kbs_stage6_compression_funnel.py \
  --samples data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl \
  --memory data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl \
  --checkpoint outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt \
  --model-dir models/deberta-v3-large \
  --dense-model models/bge-large-en-v1.5 \
  --device cuda \
  --front-pool-k 30 \
  --candidate-top-k 10 \
  --select-top-k 5 \
  --state-update-top-k 1 \
  --policy-blend-weight 0.5 \
  --local-expansion-window 1 \
  --mmr-lambda 0.7 \
  --mmr-same-doc-similarity 0.35 \
  --online-state-max-raw 8 \
  --online-state-max-chars 260 \
  --max-qids "$MAX_QIDS" \
  --output "$SUMMARY_OUTPUT" \
  --records-output "$RECORDS_OUTPUT"

echo "[DONE] $SUMMARY_OUTPUT"
