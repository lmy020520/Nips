#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

RUN="${RUN:-hotpot_full_compact}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
REFRESH_ANSWER_CACHE="${REFRESH_ANSWER_CACHE:-1}"
GENERATE_ANSWERS="${GENERATE_ANSWERS:-1}"

FULL_CHECKPOINT="outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt"
RANKING_ONLY_CHECKPOINT="outputs/ranker/deberta_v3_large_v21_unified_ranking_only/best_model.pt"
NO_DEFICIT_CHECKPOINT="outputs/ranker/deberta_v3_large_v21_unified_no_deficit/best_model.pt"
NO_CONTRIBUTION_CHECKPOINT="outputs/ranker/deberta_v3_large_v21_unified_no_contribution/best_model.pt"

case "$RUN" in
  hotpot_full_compact)
    data_root="data/hotpotqa_distractor_eval_3000_cand50"
    checkpoint="$FULL_CHECKPOINT"
    max_qids="${MAX_QIDS:-3000}"
    front_pool_k=30
    candidate_top_k=10
    ;;
  hotpot_full_recall)
    data_root="data/hotpotqa_distractor_eval_3000_cand50"
    checkpoint="$FULL_CHECKPOINT"
    max_qids="${MAX_QIDS:-3000}"
    front_pool_k=50
    candidate_top_k=50
    ;;
  2wiki_full_compact)
    data_root="data/2wiki_multihopqa_eval_1000_cand50"
    checkpoint="$FULL_CHECKPOINT"
    max_qids="${MAX_QIDS:-1000}"
    front_pool_k=30
    candidate_top_k=10
    ;;
  2wiki_full_recall)
    data_root="data/2wiki_multihopqa_eval_1000_cand50"
    checkpoint="$FULL_CHECKPOINT"
    max_qids="${MAX_QIDS:-1000}"
    front_pool_k=30
    candidate_top_k=50
    ;;
  hotpot_ranking_only_compact)
    data_root="data/hotpotqa_distractor_eval_3000_cand50"
    checkpoint="$RANKING_ONLY_CHECKPOINT"
    max_qids="${MAX_QIDS:-3000}"
    front_pool_k=30
    candidate_top_k=10
    ;;
  hotpot_no_deficit_compact)
    data_root="data/hotpotqa_distractor_eval_3000_cand50"
    checkpoint="$NO_DEFICIT_CHECKPOINT"
    max_qids="${MAX_QIDS:-3000}"
    front_pool_k=30
    candidate_top_k=10
    ;;
  hotpot_no_contribution_compact)
    data_root="data/hotpotqa_distractor_eval_3000_cand50"
    checkpoint="$NO_CONTRIBUTION_CHECKPOINT"
    max_qids="${MAX_QIDS:-3000}"
    front_pool_k=30
    candidate_top_k=10
    ;;
  *)
    echo "[ERROR] unknown RUN=$RUN" >&2
    echo "Supported: hotpot_full_compact hotpot_full_recall 2wiki_full_compact 2wiki_full_recall" >&2
    echo "           hotpot_ranking_only_compact hotpot_no_deficit_compact hotpot_no_contribution_compact" >&2
    exit 1
    ;;
esac

samples="$data_root/samples/test.jsonl"
memory="$data_root/unit_registry/raw_units_test.jsonl"
queries="$data_root/queries/test.jsonl"
model_dir="${MODEL_DIR:-models/deberta-v3-large}"
dense_model="${DENSE_MODEL:-models/bge-large-en-v1.5}"
output="${OUTPUT:-outputs/rag/kbs_v21_unified/${RUN}.json}"
cache_dir="${ANSWER_CACHE_DIR:-outputs/rag/cache_kbs_v21_unified/${RUN}}"

for path in "$samples" "$memory" "$queries" "$model_dir" "$dense_model" "$checkpoint"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$output")" "$cache_dir"

cmd=(
  python3 scripts/run_hotpotqa_policy_rag.py
  --samples "$samples"
  --memory "$memory"
  --queries "$queries"
  --checkpoint "$checkpoint"
  --model-dir "$model_dir"
  --state-mode policy
  --policy-context-source online_state
  --selector hybrid_policy
  --dense-model "$dense_model"
  --dense-query-mode state
  --hybrid-alpha 0.5
  --front-pool-k "$front_pool_k"
  --front-fusion rrf
  --local-expansion-window 1
  --mmr-lambda 0.7
  --mmr-same-doc-similarity 0.35
  --candidate-top-k "$candidate_top_k"
  --select-top-k 5
  --policy-score-mode front_policy_blend
  --policy-blend-weight 0.35
  --answer-mode json
  --answer-cache-dir "$cache_dir"
  --max-qids "$max_qids"
  --ks 1,2,3,5
  --save-online-states
  --llm-max-retries 8
  --llm-retry-sleep 2.0
  --seed 20260608
  --output "$output"
)

if [[ "$GENERATE_ANSWERS" == "1" ]]; then
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
    echo
    export DEEPSEEK_API_KEY
  fi
  cmd+=(--generate-answers)
  if [[ "$REFRESH_ANSWER_CACHE" == "1" ]]; then
    cmd+=(--refresh-answer-cache)
  fi
fi

echo "[INFO] run=$RUN gpu=$CUDA_DEVICE checkpoint=$checkpoint qids=$max_qids"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"
