#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_QIDS="${MAX_QIDS:-0}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
DATA_ONLY="${DATA_ONLY:-0}"

SAMPLES="${SAMPLES:-data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl}"
MEMORY="${MEMORY:-data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl}"
QUERIES="${QUERIES:-data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl}"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/kbs_state_phase1_v21}"

args=(
  --samples "$SAMPLES"
  --memory "$MEMORY"
  --queries "$QUERIES"
  --model-dir "$MODEL_DIR"
  --checkpoint "$CHECKPOINT"
  --dense-model "$DENSE_MODEL"
  --device cuda
  --batch-size 16
  --dense-batch-size 64
  --max-length 320
  --max-qids "$MAX_QIDS"
  --front-pool-k 30
  --front-fusion rrf
  --hybrid-alpha 0.5
  --dense-query-mode state
  --candidate-top-k 10
  --local-expansion-window 1
  --mmr-lambda 0.7
  --mmr-same-doc-similarity 0.35
  --n-bootstrap "$N_BOOTSTRAP"
  --seed 20260722
  --output-dir "$OUTPUT_DIR"
)

if [[ "$DATA_ONLY" == "1" ]]; then
  args+=(--data-only)
  python3 scripts/analyze_kbs_state_mechanism.py "${args[@]}"
else
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    python3 scripts/analyze_kbs_state_mechanism.py "${args[@]}"
fi
