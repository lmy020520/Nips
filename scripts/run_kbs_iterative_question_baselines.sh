#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN="${RUN:-hotpot_iterative_hybrid}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
REFRESH_ANSWER_CACHE="${REFRESH_ANSWER_CACHE:-1}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt}"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"

case "$RUN" in
  hotpot_iterative_hybrid|hotpot_question_only)
    data_root="data/hotpotqa_distractor_eval_3000_cand50"
    max_qids="${MAX_QIDS:-3000}"
    ;;
  2wiki_iterative_hybrid|2wiki_question_only)
    data_root="data/2wiki_multihopqa_eval_1000_cand50"
    max_qids="${MAX_QIDS:-1000}"
    ;;
  *)
    echo "[ERROR] unknown RUN=$RUN" >&2
    echo "Supported: hotpot_iterative_hybrid 2wiki_iterative_hybrid hotpot_question_only 2wiki_question_only" >&2
    exit 1
    ;;
esac

queries="$data_root/queries/test.jsonl"
output="${OUTPUT:-outputs/rag/missing_baselines/${RUN}.json}"
cache="${ANSWER_CACHE_DIR:-outputs/rag/cache_missing_baselines/${RUN}}"
mkdir -p "$(dirname "$output")" "$cache"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
  echo
  export DEEPSEEK_API_KEY
fi

refresh=()
if [[ "$REFRESH_ANSWER_CACHE" == "1" ]]; then
  refresh=(--refresh-answer-cache)
fi

if [[ "$RUN" == *_question_only ]]; then
  python3 scripts/run_question_only_llm.py \
    --queries "$queries" \
    --max-qids "$max_qids" \
    --answer-cache-dir "$cache" \
    "${refresh[@]}" \
    --llm-max-retries 8 \
    --llm-retry-sleep 2.0 \
    --seed 20260608 \
    --output "$output"
  exit 0
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
  --samples "$data_root/samples/test.jsonl" \
  --memory "$data_root/unit_registry/raw_units_test.jsonl" \
  --queries "$queries" \
  --checkpoint "$CHECKPOINT" \
  --model-dir "$MODEL_DIR" \
  --state-mode policy \
  --policy-context-source online_state \
  --selector iterative_hybrid \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state \
  --hybrid-alpha 0.5 \
  --select-top-k 5 \
  --answer-mode json \
  --generate-answers \
  "${refresh[@]}" \
  --max-qids "$max_qids" \
  --ks 1,2,3,5 \
  --save-online-states \
  --answer-cache-dir "$cache" \
  --llm-max-retries 8 \
  --llm-retry-sleep 2.0 \
  --seed 20260608 \
  --output "$output"
