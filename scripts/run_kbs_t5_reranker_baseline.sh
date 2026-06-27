#!/usr/bin/env bash
set -euo pipefail

# monoT5/RankT5 baseline for KBS HotpotQA cand50 evaluation.
#
# Expected local model examples:
#   models/monot5-base-msmarco
#   models/rankt5-base
#
# Override with:
#   RERANKER_MODEL=models/your-t5-reranker bash scripts/run_kbs_t5_reranker_baseline.sh

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_eval_3000_cand50}"
SAMPLES="${SAMPLES:-$DATA_ROOT/samples/test.jsonl}"
MEMORY="${MEMORY:-$DATA_ROOT/unit_registry/raw_units_test.jsonl}"
QUERIES="${QUERIES:-$DATA_ROOT/queries/test.jsonl}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v17_candidate_contribution_lr5e7/best_model.pt}"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
RERANKER_MODEL="${RERANKER_MODEL:-models/monot5-base-msmarco}"
CUDA_DEVICE="${CUDA_DEVICE:-5}"
MAX_QIDS="${MAX_QIDS:-3000}"
SELECT_TOP_K="${SELECT_TOP_K:-5}"
RERANKER_BATCH_SIZE="${RERANKER_BATCH_SIZE:-8}"
RERANKER_MAX_LENGTH="${RERANKER_MAX_LENGTH:-512}"
OUTPUT="${OUTPUT:-outputs/rag/monot5_reranker_baseline_eval3000.json}"
ANSWER_CACHE_DIR="${ANSWER_CACHE_DIR:-outputs/rag/cache_monot5_reranker_baseline_eval3000}"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[ERROR] DEEPSEEK_API_KEY is not set. Run: read -s DEEPSEEK_API_KEY; export DEEPSEEK_API_KEY" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
  --samples "$SAMPLES" \
  --memory "$MEMORY" \
  --queries "$QUERIES" \
  --checkpoint "$CHECKPOINT" \
  --model-dir "$MODEL_DIR" \
  --state-mode policy \
  --policy-context-source online_state \
  --selector t5_reranker \
  --reranker-model "$RERANKER_MODEL" \
  --reranker-batch-size "$RERANKER_BATCH_SIZE" \
  --reranker-max-length "$RERANKER_MAX_LENGTH" \
  --select-top-k "$SELECT_TOP_K" \
  --answer-mode json \
  --generate-answers \
  --refresh-answer-cache \
  --max-qids "$MAX_QIDS" \
  --answer-cache-dir "$ANSWER_CACHE_DIR" \
  --output "$OUTPUT"
