#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

ABLATION="${ABLATION:-official}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_eval_3000_cand50}"
SAMPLES="${SAMPLES:-$DATA_ROOT/samples/test.jsonl}"
MEMORY="${MEMORY:-$DATA_ROOT/unit_registry/raw_units_test.jsonl}"
QUERIES="${QUERIES:-$DATA_ROOT/queries/test.jsonl}"

MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v17_candidate_contribution_lr5e7/best_model.pt}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
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
DEFICIT_ROLE_WEIGHT="${DEFICIT_ROLE_WEIGHT:-0.5}"
DEFICIT_CONTRIBUTION_WEIGHT="${DEFICIT_CONTRIBUTION_WEIGHT:-0.5}"

ANSWER_MODE="${ANSWER_MODE:-json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rag/ablations_full3000}"
CACHE_ROOT="${CACHE_ROOT:-outputs/rag/cache_ablations_full3000}"

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"

usage() {
  cat <<'EOF'
Usage:
  ABLATION=<name> bash scripts/run_kbs_ablation_experiment.sh

Supported ABLATION values:
  official                       Full KBS official system.
  no_online_state                Use legacy selected-evidence state instead of online K_t.
  query_only_policy              Remove knowledge state; score candidates with Question only.
  no_front_policy_blend          Use policy ranking only after hybrid front-end compression.
  no_policy                      Use hybrid front-end only; remove student policy selection.
  no_local_expansion             Disable local sentence expansion in hybrid front-end.
  no_dense                       Use BM25 only; remove dense retriever.
  no_bm25                        Use dense only; remove BM25 retriever.
  no_deficit_contribution_score  Disable explicit deficit/contribution scoring.
  deficit_role_score             Optional variant: use deficit-role compatibility score.
  deficit_contribution_score     Optional variant: use deficit-contribution compatibility score.
EOF
}

if [[ "$ABLATION" == "-h" || "$ABLATION" == "--help" ]]; then
  usage
  exit 0
fi

required_files=("$SAMPLES" "$MEMORY" "$QUERIES")
for path in "$MODEL_DIR" "$CHECKPOINT"; do
  required_files+=("$path")
done

selector="hybrid_policy"
policy_context_source="online_state"
dense_query_mode="state"
policy_score_mode="front_policy_blend"
local_expansion_window="$LOCAL_EXPANSION_WINDOW"
dense_model_arg="$DENSE_MODEL"
ablation_note=""

case "$ABLATION" in
  official)
    ;;
  no_online_state)
    policy_context_source="legacy"
    ;;
  query_only_policy)
    policy_context_source="query_only"
    ;;
  no_front_policy_blend)
    policy_score_mode="rank"
    ;;
  no_policy)
    selector="hybrid"
    ;;
  no_local_expansion)
    local_expansion_window="0"
    ;;
  no_dense)
    selector="bm25"
    dense_model_arg=""
    ;;
  no_bm25)
    selector="dense"
    ;;
  no_deficit_contribution_score)
    policy_score_mode="front_policy_blend"
    ablation_note="Official v1 does not explicitly use deficit/contribution score at inference; this run documents that setting."
    ;;
  deficit_role_score)
    policy_score_mode="deficit_role"
    ;;
  deficit_contribution_score)
    policy_score_mode="deficit_contribution"
    ;;
  *)
    echo "[ERROR] unknown ABLATION=$ABLATION" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "$selector" == "dense" || "$selector" == "hybrid" || "$selector" == "hybrid_policy" ]]; then
  required_files+=("$DENSE_MODEL")
fi

for path in "${required_files[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done

output="${OUTPUT:-$OUTPUT_DIR/${ABLATION}.json}"
answer_cache_dir="${ANSWER_CACHE_DIR:-$CACHE_ROOT/$ABLATION}"

cmd=(
  python3 scripts/run_hotpotqa_policy_rag.py
  --samples "$SAMPLES"
  --memory "$MEMORY"
  --queries "$QUERIES"
  --checkpoint "$CHECKPOINT"
  --model-dir "$MODEL_DIR"
  --state-mode policy
  --policy-context-source "$policy_context_source"
  --selector "$selector"
  --dense-query-mode "$dense_query_mode"
  --hybrid-alpha 0.5
  --front-pool-k "$FRONT_POOL_K"
  --front-fusion "$FRONT_FUSION"
  --local-expansion-window "$local_expansion_window"
  --mmr-lambda "$MMR_LAMBDA"
  --mmr-same-doc-similarity "$MMR_SAME_DOC_SIMILARITY"
  --candidate-top-k "$CANDIDATE_TOP_K"
  --select-top-k "$SELECT_TOP_K"
  --policy-score-mode "$policy_score_mode"
  --policy-blend-weight "$POLICY_BLEND_WEIGHT"
  --deficit-role-weight "$DEFICIT_ROLE_WEIGHT"
  --deficit-contribution-weight "$DEFICIT_CONTRIBUTION_WEIGHT"
  --answer-mode "$ANSWER_MODE"
  --answer-cache-dir "$answer_cache_dir"
  --ks 1,2,3,5
  --output "$output"
)

if [[ -n "$dense_model_arg" ]]; then
  cmd+=(--dense-model "$dense_model_arg")
fi

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

echo "[INFO] KBS ablation experiment"
echo "[INFO] ablation=$ABLATION"
echo "[INFO] selector=$selector"
echo "[INFO] policy_context_source=$policy_context_source"
echo "[INFO] policy_score_mode=$policy_score_mode"
echo "[INFO] local_expansion_window=$local_expansion_window"
echo "[INFO] max_qids=$MAX_QIDS (0 means all qids)"
echo "[INFO] output=$output"
echo "[INFO] cache=$answer_cache_dir"
echo "[INFO] refresh_answer_cache=$REFRESH_ANSWER_CACHE"
if [[ -n "$ablation_note" ]]; then
  echo "[INFO] note=$ablation_note"
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"
