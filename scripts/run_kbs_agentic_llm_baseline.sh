#!/usr/bin/env bash
set -euo pipefail

# Iterative / agentic RAG baseline.
# At each evidence-acquisition step, an LLM sees Question + current K_t +
# candidate evidence and selects evidence indices directly. No trained student
# policy is used for selection.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_eval_3000_cand50}"
SAMPLES="${SAMPLES:-$DATA_ROOT/samples/test.jsonl}"
MEMORY="${MEMORY:-$DATA_ROOT/unit_registry/raw_units_test.jsonl}"
QUERIES="${QUERIES:-$DATA_ROOT/queries/test.jsonl}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v17_candidate_contribution_lr5e7/best_model.pt}"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"

CUDA_DEVICE="${CUDA_DEVICE:-7}"
MAX_QIDS="${MAX_QIDS:-3000}"
SELECT_TOP_K="${SELECT_TOP_K:-5}"
AGENTIC_MAX_CANDIDATES="${AGENTIC_MAX_CANDIDATES:-50}"
GENERATE_ANSWERS="${GENERATE_ANSWERS:-1}"
REFRESH_CACHE="${REFRESH_CACHE:-1}"
OUTPUT="${OUTPUT:-outputs/rag/agentic_llm_eval3000.json}"
ANSWER_CACHE_DIR="${ANSWER_CACHE_DIR:-outputs/rag/cache_agentic_llm_eval3000_answers}"
AGENTIC_CACHE_DIR="${AGENTIC_CACHE_DIR:-outputs/rag/cache_agentic_llm_eval3000_selection}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
  echo
  export DEEPSEEK_API_KEY
fi

cmd=(
  python3 scripts/run_hotpotqa_policy_rag.py
  --samples "$SAMPLES"
  --memory "$MEMORY"
  --queries "$QUERIES"
  --checkpoint "$CHECKPOINT"
  --model-dir "$MODEL_DIR"
  --state-mode policy
  --policy-context-source online_state
  --selector agentic_llm
  --select-top-k "$SELECT_TOP_K"
  --agentic-max-candidates "$AGENTIC_MAX_CANDIDATES"
  --agentic-cache-dir "$AGENTIC_CACHE_DIR"
  --answer-mode json
  --max-qids "$MAX_QIDS"
  --answer-cache-dir "$ANSWER_CACHE_DIR"
  --output "$OUTPUT"
)

if [[ "$GENERATE_ANSWERS" == "1" ]]; then
  cmd+=(--generate-answers)
fi

if [[ "$REFRESH_CACHE" == "1" ]]; then
  cmd+=(--refresh-answer-cache)
fi

echo "[INFO] running agentic LLM baseline"
echo "[INFO] max_qids=$MAX_QIDS"
echo "[INFO] select_top_k=$SELECT_TOP_K"
echo "[INFO] agentic_max_candidates=$AGENTIC_MAX_CANDIDATES"
echo "[INFO] output=$OUTPUT"
echo "[INFO] answer_cache=$ANSWER_CACHE_DIR"
echo "[INFO] agentic_cache=$AGENTIC_CACHE_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"
