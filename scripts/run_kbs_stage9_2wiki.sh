#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

ACTION="${ACTION:-readiness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_stage9_2wiki}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4}"
CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"

for plan in \
  md/kbs_three_review_execution_plan.md \
  md/kbs_review_75_85_execution_plan.md; do
  if [[ ! -f "$plan" ]]; then
    echo "[ERROR] missing experiment plan: $plan" >&2
    exit 1
  fi
done
mkdir -p "$OUTPUT_ROOT"

readiness_is_valid() {
  python3 - "$OUTPUT_ROOT/readiness.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing readiness report: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "OK" or report.get("failures"):
    raise SystemExit(f"readiness is not a clean OK: {path}")
PY
}

run_selection_smoke() {
  if [[ "${KBS_STAGE9_2WIKI_SELECTION_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.4 selection smoke is locked pending readiness review" >&2
    exit 1
  fi
  readiness_is_valid
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"
  if [[ "${#gpu_ids[@]}" -lt 5 ]]; then
    echo "[ERROR] GPU_LIST needs five GPUs, for example 0,1,2,3,4" >&2
    exit 1
  fi

  local output_dir="$OUTPUT_ROOT/selection_smoke20"
  mkdir -p "$output_dir"
  local method
  for method in compact_seed42 balanced_knee_seed42 recall_seed42 hybrid bge_reranker; do
    if [[ -e "$output_dir/$method.json" ]]; then
      echo "[ERROR] refusing to overwrite smoke report: $output_dir/$method.json" >&2
      exit 1
    fi
  done

  run_method() {
    local method="$1"
    local device="$2"
    local selector candidate_top_k front_pool_k state_update_top_k
    selector="hybrid_policy"
    candidate_top_k=10
    front_pool_k=30
    state_update_top_k=1
    case "$method" in
      compact_seed42)
        ;;
      balanced_knee_seed42)
        candidate_top_k=15
        ;;
      recall_seed42)
        candidate_top_k=50
        front_pool_k=50
        ;;
      hybrid)
        selector="hybrid"
        candidate_top_k=8
        state_update_top_k=5
        ;;
      bge_reranker)
        selector="generic_reranker"
        candidate_top_k=8
        state_update_top_k=5
        ;;
      *)
        echo "[ERROR] unknown Stage 9.4 smoke method: $method" >&2
        return 1
        ;;
    esac

    local -a command=(
      python3 scripts/run_hotpotqa_policy_rag.py
      --samples data/2wiki_multihopqa_eval_1000_cand50/samples/test.jsonl
      --memory data/2wiki_multihopqa_eval_1000_cand50/unit_registry/raw_units_test.jsonl
      --queries data/2wiki_multihopqa_eval_1000_cand50/queries/test.jsonl
      --checkpoint "$CHECKPOINT"
      --state-mode policy
      --policy-context-source online_state
      --selector "$selector"
      --hybrid-alpha 0.5
      --front-pool-k "$front_pool_k"
      --candidate-top-k "$candidate_top_k"
      --select-top-k 5
      --state-update-top-k "$state_update_top_k"
      --answer-mode json
      --max-qids 20
      --ks 1,2,3,5
      --seed 20260608
      --device cuda
      --output "$output_dir/$method.json"
    )
    if [[ "$selector" == "hybrid_policy" ]]; then
      command+=(
        --dense-model models/bge-large-en-v1.5
        --dense-query-mode state
        --front-fusion rrf
        --local-expansion-window 1
        --mmr-lambda 0.7
        --mmr-same-doc-similarity 0.35
        --policy-score-mode front_policy_blend
        --policy-blend-weight 0.5
        --save-online-states
      )
    elif [[ "$selector" == "hybrid" ]]; then
      command+=(--dense-model models/bge-large-en-v1.5 --dense-query-mode state)
    else
      command+=(
        --dense-query-mode question
        --reranker-model models/bge-reranker-large
      )
    fi
    CUDA_VISIBLE_DEVICES="$device" "${command[@]}" >"$output_dir/$method.log" 2>&1
  }

  echo "[START] Stage 9.4 five-method 20-qid selection smoke; no API"
  run_method compact_seed42 "${gpu_ids[0]}" &
  local compact_pid=$!
  run_method balanced_knee_seed42 "${gpu_ids[1]}" &
  local balanced_pid=$!
  run_method recall_seed42 "${gpu_ids[2]}" &
  local recall_pid=$!
  run_method hybrid "${gpu_ids[3]}" &
  local hybrid_pid=$!
  run_method bge_reranker "${gpu_ids[4]}" &
  local bge_pid=$!

  local failed=0 pid
  for pid in "$compact_pid" "$balanced_pid" "$recall_pid" "$hybrid_pid" "$bge_pid"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "[ERROR] at least one Stage 9.4 smoke failed; inspect $output_dir/*.log" >&2
    exit 1
  fi

  python3 scripts/check_kbs_stage9_2wiki_selection.py \
    --report "compact_seed42=$output_dir/compact_seed42.json" \
    --report "balanced_knee_seed42=$output_dir/balanced_knee_seed42.json" \
    --report "recall_seed42=$output_dir/recall_seed42.json" \
    --report "hybrid=$output_dir/hybrid.json" \
    --report "bge_reranker=$output_dir/bge_reranker.json" \
    --expected-qids 20 \
    --smoke \
    --output "$output_dir/summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_4_SELECTION_SMOKE_OK"
  echo "No training or answer API call was started."
}

case "$ACTION" in
  readiness)
    python3 scripts/check_kbs_stage9_2wiki_readiness.py \
      --output-root "$OUTPUT_ROOT" \
      --output "$OUTPUT_ROOT/readiness.json"
    echo "FINISHED_OK"
    echo "status=STAGE9_4_READINESS_OK"
    echo "No training, GPU inference, or API call was started."
    ;;
  selection_smoke)
    run_selection_smoke
    ;;
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness, selection_smoke" >&2
    exit 2
    ;;
esac
