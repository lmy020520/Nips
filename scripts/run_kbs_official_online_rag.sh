#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_eval_3000_cand50}"
SAMPLES="${SAMPLES:-$DATA_ROOT/samples/test.jsonl}"
MEMORY="${MEMORY:-$DATA_ROOT/unit_registry/raw_units_test.jsonl}"
QUERIES="${QUERIES:-$DATA_ROOT/queries/test.jsonl}"

MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v17_candidate_contribution_lr5e7/best_model.pt}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"

CUDA_DEVICE="${CUDA_DEVICE:-5}"
MAX_QIDS="${MAX_QIDS:-0}"
MAX_POLICY_STEPS="${MAX_POLICY_STEPS:-0}"
GENERATE_ANSWERS="${GENERATE_ANSWERS:-1}"
SAVE_ONLINE_STATES="${SAVE_ONLINE_STATES:-1}"
REFRESH_ANSWER_CACHE="${REFRESH_ANSWER_CACHE:-1}"
REQUIRE_DEEPSEEK_API_KEY="${REQUIRE_DEEPSEEK_API_KEY:-1}"

FRONT_POOL_K="${FRONT_POOL_K:-30}"
FRONT_FUSION="${FRONT_FUSION:-rrf}"
LOCAL_EXPANSION_WINDOW="${LOCAL_EXPANSION_WINDOW:-1}"
MMR_LAMBDA="${MMR_LAMBDA:-0.7}"
MMR_SAME_DOC_SIMILARITY="${MMR_SAME_DOC_SIMILARITY:-0.35}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-10}"
SELECT_TOP_K="${SELECT_TOP_K:-5}"
POLICY_BLEND_WEIGHT="${POLICY_BLEND_WEIGHT:-0.35}"

ANSWER_MODE="${ANSWER_MODE:-json}"
ANSWER_CACHE_DIR="${ANSWER_CACHE_DIR:-outputs/rag/cache_kbs_official_online_state_v1}"
OUTPUT="${OUTPUT:-outputs/rag/kbs_official_online_state_v1.json}"

required_files=("$SAMPLES" "$MEMORY" "$QUERIES" "$MODEL_DIR" "$CHECKPOINT" "$DENSE_MODEL")
for path in "${required_files[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done

cmd=(
  python3 scripts/run_hotpotqa_policy_rag.py
  --samples "$SAMPLES"
  --memory "$MEMORY"
  --queries "$QUERIES"
  --checkpoint "$CHECKPOINT"
  --model-dir "$MODEL_DIR"
  --state-mode policy
  --policy-context-source online_state
  --selector hybrid_policy
  --dense-model "$DENSE_MODEL"
  --dense-query-mode state
  --front-pool-k "$FRONT_POOL_K"
  --front-fusion "$FRONT_FUSION"
  --local-expansion-window "$LOCAL_EXPANSION_WINDOW"
  --mmr-lambda "$MMR_LAMBDA"
  --mmr-same-doc-similarity "$MMR_SAME_DOC_SIMILARITY"
  --candidate-top-k "$CANDIDATE_TOP_K"
  --select-top-k "$SELECT_TOP_K"
  --policy-score-mode front_policy_blend
  --policy-blend-weight "$POLICY_BLEND_WEIGHT"
  --answer-mode "$ANSWER_MODE"
  --answer-cache-dir "$ANSWER_CACHE_DIR"
  --ks 1,2,3,5
  --output "$OUTPUT"
)

if [[ "$MAX_QIDS" != "0" ]]; then
  cmd+=(--max-qids "$MAX_QIDS")
fi

if [[ "$MAX_POLICY_STEPS" != "0" ]]; then
  cmd+=(--max-policy-steps "$MAX_POLICY_STEPS")
fi

if [[ "$SAVE_ONLINE_STATES" == "1" ]]; then
  cmd+=(--save-online-states)
fi

if [[ "$GENERATE_ANSWERS" == "1" ]]; then
  if [[ "$REQUIRE_DEEPSEEK_API_KEY" == "1" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
    echo
    export DEEPSEEK_API_KEY
  fi
  cmd+=(--generate-answers)
  if [[ "$REFRESH_ANSWER_CACHE" == "1" ]]; then
    cmd+=(--refresh-answer-cache)
  fi
fi

echo "[INFO] KBS official online RAG route"
echo "[INFO] data_root=$DATA_ROOT"
echo "[INFO] checkpoint=$CHECKPOINT"
echo "[INFO] output=$OUTPUT"
echo "[INFO] cache=$ANSWER_CACHE_DIR"
echo "[INFO] refresh_answer_cache=$REFRESH_ANSWER_CACHE"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"
