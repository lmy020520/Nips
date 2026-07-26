#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PLAN_FILE="md/kbs_three_review_execution_plan.md"
if [[ ! -f "$PLAN_FILE" ]]; then
  echo "[ERROR] missing experiment governance plan: $PLAN_FILE" >&2
  exit 1
fi

RUN="${RUN:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SMOKE="${SMOKE:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

DATA_ROOT="data/hotpotqa_distractor_eval_3000_cand50"
CHECKPOINT="outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"
FROZEN_ALPHA="0.35"
FROZEN_DEEPSEEK_MODEL="deepseek-v4-flash"
FROZEN_DEEPSEEK_THINKING_MODE="disabled"

usage() {
  cat <<'EOF'
Usage:
  RUN=<name> CUDA_DEVICE=<gpu> bash scripts/run_kbs_v22_stage2_hotpot.sh

Core Stage 2.2 runs:
  full_compact
  full_recall
  query_only_compact
  query_only_recall

Optional diagnostic inference runs:
  previous_only_compact
  previous_only_recall

Set SMOKE=1 for a 20-qid preflight with separate smoke outputs.
EOF
}

if [[ -z "$RUN" || "$RUN" == "-h" || "$RUN" == "--help" ]]; then
  usage
  [[ -n "$RUN" ]] && exit 0
  exit 2
fi

policy_context_source="online_state"
front_pool_k=30
candidate_top_k=10
case "$RUN" in
  full_compact)
    ;;
  full_recall)
    front_pool_k=50
    candidate_top_k=50
    ;;
  query_only_compact)
    policy_context_source="query_only"
    ;;
  query_only_recall)
    policy_context_source="query_only"
    front_pool_k=50
    candidate_top_k=50
    ;;
  previous_only_compact)
    policy_context_source="previous_evidence_only"
    ;;
  previous_only_recall)
    policy_context_source="previous_evidence_only"
    front_pool_k=50
    candidate_top_k=50
    ;;
  *)
    echo "[ERROR] unknown RUN=$RUN" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "$SMOKE" == "1" ]]; then
  max_qids=20
  run_tag="${RUN}_smoke20"
else
  max_qids=3000
  run_tag="$RUN"
fi
if [[ -n "$RUN_SUFFIX" ]]; then
  if [[ ! "$RUN_SUFFIX" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "[ERROR] RUN_SUFFIX may contain only letters, numbers, dot, underscore, and hyphen" >&2
    exit 2
  fi
  run_tag="${run_tag}_${RUN_SUFFIX}"
fi

samples="$DATA_ROOT/samples/test.jsonl"
memory="$DATA_ROOT/unit_registry/raw_units_test.jsonl"
queries="$DATA_ROOT/queries/test.jsonl"
output="outputs/rag/kbs_v22_stage2_hotpot/${run_tag}.json"
cache_dir="outputs/rag/cache_kbs_v22_stage2_hotpot/${run_tag}"

for path in "$samples" "$memory" "$queries" "$CHECKPOINT" "$MODEL_DIR" "$DENSE_MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ -t 0 ]]; then
    read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
    echo
    export DEEPSEEK_API_KEY
  else
    echo "[ERROR] DEEPSEEK_API_KEY must be exported before a non-interactive run" >&2
    exit 1
  fi
fi
if [[ -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
  echo "[ERROR] DEEPSEEK_API_KEY is blank" >&2
  exit 1
fi
export DEEPSEEK_API_KEY
export DEEPSEEK_MODEL="$FROZEN_DEEPSEEK_MODEL"
export DEEPSEEK_THINKING_MODE="$FROZEN_DEEPSEEK_THINKING_MODE"

if [[ "$ALLOW_OVERWRITE" != "1" ]]; then
  if [[ -e "$output" ]]; then
    echo "[ERROR] refusing to overwrite existing report: $output" >&2
    exit 1
  fi
  if [[ -d "$cache_dir" && -n "$(find "$cache_dir" -mindepth 1 -print -quit)" ]]; then
    echo "[ERROR] refusing to reuse non-empty answer cache: $cache_dir" >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "$output")" "$cache_dir"

echo "[INFO] checking DeepSeek connectivity and authentication before retrieval"
python3 scripts/check_deepseek_api.py

echo "[PLAN] Stage 2.2 HotpotQA v22 end-to-end evaluation"
echo "[INFO] run=$RUN smoke=$SMOKE gpu=$CUDA_DEVICE qids=$max_qids"
echo "[INFO] checkpoint=$CHECKPOINT"
echo "[INFO] context_source=$policy_context_source"
echo "[INFO] front_pool_k=$front_pool_k candidate_top_k=$candidate_top_k select_top_k=5"
echo "[INFO] frozen_alpha=$FROZEN_ALPHA"
echo "[INFO] answer_model=$DEEPSEEK_MODEL thinking_mode=$DEEPSEEK_THINKING_MODE"
echo "[INFO] fresh_answer_calls=required output=$output cache=$cache_dir"
if [[ "$RUN" == previous_only_* ]]; then
  echo "[INFO] previous-only is an inference diagnostic, not a separately trained anchor baseline"
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
  --samples "$samples" \
  --memory "$memory" \
  --queries "$queries" \
  --checkpoint "$CHECKPOINT" \
  --model-dir "$MODEL_DIR" \
  --state-mode policy \
  --policy-context-source "$policy_context_source" \
  --selector hybrid_policy \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state \
  --hybrid-alpha 0.5 \
  --front-pool-k "$front_pool_k" \
  --front-fusion rrf \
  --local-expansion-window 1 \
  --mmr-lambda 0.7 \
  --mmr-same-doc-similarity 0.35 \
  --candidate-top-k "$candidate_top_k" \
  --select-top-k 5 \
  --policy-score-mode front_policy_blend \
  --policy-blend-weight "$FROZEN_ALPHA" \
  --answer-mode json \
  --generate-answers \
  --refresh-answer-cache \
  --answer-cache-dir "$cache_dir" \
  --max-qids "$max_qids" \
  --ks 1,2,3,5 \
  --save-online-states \
  --profile-runtime \
  --profile-warmup-qids 20 \
  --llm-max-retries 8 \
  --llm-retry-sleep 2.0 \
  --seed 20260608 \
  --output "$output"

python3 scripts/check_kbs_v22_stage2_report.py \
  --report "$output" \
  --run "$RUN" \
  --expected-qids "$max_qids"
