#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_alpha_val_1000_cand50}"
SPLIT="${SPLIT:-val}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt}"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/alpha_sensitivity_val}"
ALPHAS="${ALPHAS:-0,0.2,0.35,0.5,0.8,1.0}"
GPU_LIST="${GPU_LIST:-0}"
MAX_QIDS="${MAX_QIDS:-1000}"
GENERATE_ANSWERS="${GENERATE_ANSWERS:-0}"
REFRESH_ANSWER_CACHE="${REFRESH_ANSWER_CACHE:-0}"
POLICY_CONTEXT_SOURCE="${POLICY_CONTEXT_SOURCE:-online_state}"
STATE_UPDATE_TOP_K="${STATE_UPDATE_TOP_K:-0}"

samples="$DATA_ROOT/samples/${SPLIT}.jsonl"
memory="$DATA_ROOT/unit_registry/raw_units_${SPLIT}.jsonl"
queries="$DATA_ROOT/queries/${SPLIT}.jsonl"
for path in "$samples" "$memory" "$queries" "$CHECKPOINT" "$MODEL_DIR" "$DENSE_MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/cache"
IFS=',' read -r -a alpha_values <<< "$ALPHAS"
IFS=',' read -r -a gpu_values <<< "$GPU_LIST"
if [[ "${#gpu_values[@]}" -eq 0 ]]; then
  echo "[ERROR] GPU_LIST is empty" >&2
  exit 1
fi

if [[ "$GENERATE_ANSWERS" == "1" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
  echo
  export DEEPSEEK_API_KEY
fi

run_alpha() {
  local alpha="$1"
  local gpu="$2"
  local tag
  tag="$(python3 -c 'import sys; print(f"{float(sys.argv[1]):.2f}".replace(".", "p"))' "$alpha")"
  local output="$OUTPUT_DIR/alpha_${tag}.json"
  local cache="$OUTPUT_DIR/cache/alpha_${tag}"
  local cmd=(
    python3 scripts/run_hotpotqa_policy_rag.py
    --samples "$samples"
    --memory "$memory"
    --queries "$queries"
    --checkpoint "$CHECKPOINT"
    --model-dir "$MODEL_DIR"
    --state-mode policy
    --policy-context-source "$POLICY_CONTEXT_SOURCE"
    --selector hybrid_policy
    --dense-model "$DENSE_MODEL"
    --dense-query-mode state
    --hybrid-alpha 0.5
    --front-pool-k 30
    --front-fusion rrf
    --local-expansion-window 1
    --mmr-lambda 0.7
    --mmr-same-doc-similarity 0.35
    --candidate-top-k 10
    --select-top-k 5
    --state-update-top-k "$STATE_UPDATE_TOP_K"
    --policy-score-mode front_policy_blend
    --policy-blend-weight "$alpha"
    --answer-mode json
    --max-qids "$MAX_QIDS"
    --ks 1,2,3,5
    --seed 20260608
    --output "$output"
  )
  if [[ "$GENERATE_ANSWERS" == "1" ]]; then
    cmd+=(--generate-answers --answer-cache-dir "$cache")
    if [[ "$REFRESH_ANSWER_CACHE" == "1" ]]; then
      cmd+=(--refresh-answer-cache)
    fi
  fi
  echo "[INFO] alpha=$alpha gpu=$gpu output=$output"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
}

# One alpha per GPU in each batch; a single GPU therefore runs sequentially.
for ((start=0; start<${#alpha_values[@]}; start+=${#gpu_values[@]})); do
  pids=()
  for ((offset=0; offset<${#gpu_values[@]} && start+offset<${#alpha_values[@]}; offset++)); do
    run_alpha "${alpha_values[start+offset]}" "${gpu_values[offset]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

python3 scripts/summarize_kbs_alpha_sensitivity.py \
  --input-dir "$OUTPUT_DIR" \
  --alphas "$ALPHAS" \
  --expected-qids "$MAX_QIDS" \
  --output "$OUTPUT_DIR/summary.json" \
  --tsv-output "$OUTPUT_DIR/summary.tsv"
