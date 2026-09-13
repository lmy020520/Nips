#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

ACTION="${ACTION:-readiness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_stage9_state_rollout}"
CORRECT_REPORT="${CORRECT_REPORT:-outputs/rag/kbs_v27_final_hotpot/full_compact.json}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
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

run_selection() {
  local condition="$1"
  local device="$2"
  local output="$3"
  local -a command=(
    python3 scripts/run_hotpotqa_policy_rag.py
    --samples data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl
    --memory data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl
    --queries data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl
    --checkpoint "$CHECKPOINT"
    --model-dir models/deberta-v3-large
    --state-mode policy
    --policy-context-source "$condition"
    --selector hybrid_policy
    --dense-model models/bge-large-en-v1.5
    --dense-query-mode state
    --hybrid-alpha 0.5
    --front-pool-k 30
    --front-fusion rrf
    --local-expansion-window 1
    --mmr-lambda 0.7
    --mmr-same-doc-similarity 0.35
    --candidate-top-k 10
    --select-top-k 5
    --state-update-top-k 1
    --policy-score-mode front_policy_blend
    --policy-blend-weight 0.5
    --answer-mode json
    --max-qids 20
    --ks 1,2,3,5
    --save-online-states
    --seed 20260608
    --device cuda
    --output "$output"
  )
  if [[ "$condition" == "other_question_state" ]]; then
    command+=(--external-policy-state-report "$OUTPUT_ROOT/selection_smoke20/online_state.json")
  fi
  CUDA_VISIBLE_DEVICES="$device" "${command[@]}" >"${output%.json}.log" 2>&1
}

run_selection_smoke() {
  if [[ "${KBS_STAGE9_STATE_SELECTION_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.5 selection smoke is locked pending readiness review" >&2
    exit 1
  fi
  readiness_is_valid
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"
  if [[ "${#gpu_ids[@]}" -lt 4 ]]; then
    echo "[ERROR] GPU_LIST needs four GPUs, for example 0,1,2,3" >&2
    exit 1
  fi
  local output_dir="$OUTPUT_ROOT/selection_smoke20"
  mkdir -p "$output_dir"
  local condition
  for condition in online_state query_only frozen_initial_state other_question_state previous_evidence_only; do
    if [[ -e "$output_dir/$condition.json" ]]; then
      echo "[ERROR] refusing to overwrite smoke report: $output_dir/$condition.json" >&2
      exit 1
    fi
  done

  echo "[START] correct online-state smoke; no answer API"
  run_selection online_state "${gpu_ids[0]}" "$output_dir/online_state.json"

  echo "[START] four matched state interventions; no answer API"
  run_selection query_only "${gpu_ids[0]}" "$output_dir/query_only.json" &
  local query_pid=$!
  run_selection frozen_initial_state "${gpu_ids[1]}" "$output_dir/frozen_initial_state.json" &
  local frozen_pid=$!
  run_selection other_question_state "${gpu_ids[2]}" "$output_dir/other_question_state.json" &
  local other_pid=$!
  run_selection previous_evidence_only "${gpu_ids[3]}" "$output_dir/previous_evidence_only.json" &
  local previous_pid=$!
  local failed=0 pid
  for pid in "$query_pid" "$frozen_pid" "$other_pid" "$previous_pid"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "[ERROR] at least one state intervention failed; inspect $output_dir/*.log" >&2
    exit 1
  fi

  python3 scripts/check_kbs_stage9_state_rollout_selection.py \
    --report "online_state=$output_dir/online_state.json" \
    --report "query_only=$output_dir/query_only.json" \
    --report "frozen_initial_state=$output_dir/frozen_initial_state.json" \
    --report "other_question_state=$output_dir/other_question_state.json" \
    --report "previous_evidence_only=$output_dir/previous_evidence_only.json" \
    --expected-qids 20 \
    --output "$output_dir/summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_5_SELECTION_SMOKE_OK"
  echo "No training or answer API call was started."
}

case "$ACTION" in
  readiness)
    python3 scripts/check_kbs_stage9_state_rollout_readiness.py \
      --correct-report "$CORRECT_REPORT" \
      --output "$OUTPUT_ROOT/readiness.json"
    echo "FINISHED_OK"
    echo "status=STAGE9_5_READINESS_OK"
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
