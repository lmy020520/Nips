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
  local max_qids="$4"
  local external_report="${5:-}"
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
    --max-qids "$max_qids"
    --ks 1,2,3,5
    --save-online-states
    --seed 20260608
    --device cuda
    --output "$output"
  )
  if [[ "$condition" == "other_question_state" ]]; then
    command+=(--external-policy-state-report "$external_report")
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
  run_selection online_state "${gpu_ids[0]}" "$output_dir/online_state.json" 20

  echo "[START] four matched state interventions; no answer API"
  run_selection query_only "${gpu_ids[0]}" "$output_dir/query_only.json" 20 &
  local query_pid=$!
  run_selection frozen_initial_state "${gpu_ids[1]}" "$output_dir/frozen_initial_state.json" 20 &
  local frozen_pid=$!
  run_selection other_question_state "${gpu_ids[2]}" "$output_dir/other_question_state.json" 20 "$output_dir/online_state.json" &
  local other_pid=$!
  run_selection previous_evidence_only "${gpu_ids[3]}" "$output_dir/previous_evidence_only.json" 20 &
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
    --smoke \
    --n-bootstrap 200 \
    --output "$output_dir/summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_5_SELECTION_SMOKE_OK"
  echo "No training or answer API call was started."
}

run_selection_full() {
  if [[ "${KBS_STAGE9_STATE_SELECTION_FULL_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.5 full selection is locked pending smoke review" >&2
    exit 1
  fi
  readiness_is_valid
  local smoke_summary="$OUTPUT_ROOT/selection_smoke20/summary.json"
  python3 - "$smoke_summary" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing smoke summary: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "SMOKE_OK" or report.get("failures"):
    raise SystemExit(f"selection smoke is not a clean SMOKE_OK: {path}")
PY
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"
  if [[ "${#gpu_ids[@]}" -lt 4 ]]; then
    echo "[ERROR] GPU_LIST needs four GPUs, for example 0,1,2,3" >&2
    exit 1
  fi
  local output_dir="$OUTPUT_ROOT/selection3000"
  mkdir -p "$output_dir"
  echo "$$" > "$output_dir/run.pid"
  local condition
  for condition in online_state query_only frozen_initial_state other_question_state previous_evidence_only; do
    if [[ -e "$output_dir/$condition.json" ]]; then
      echo "[ERROR] refusing to overwrite full report: $output_dir/$condition.json" >&2
      exit 1
    fi
  done

  echo "[START] correct online-state 3,000-qid rollout; no answer API"
  run_selection online_state "${gpu_ids[0]}" "$output_dir/online_state.json" 3000

  echo "[START] four matched 3,000-qid state interventions; no answer API"
  run_selection query_only "${gpu_ids[0]}" "$output_dir/query_only.json" 3000 &
  local query_pid=$!
  run_selection frozen_initial_state "${gpu_ids[1]}" "$output_dir/frozen_initial_state.json" 3000 &
  local frozen_pid=$!
  run_selection other_question_state "${gpu_ids[2]}" "$output_dir/other_question_state.json" 3000 "$output_dir/online_state.json" &
  local other_pid=$!
  run_selection previous_evidence_only "${gpu_ids[3]}" "$output_dir/previous_evidence_only.json" 3000 &
  local previous_pid=$!
  local failed=0 pid
  for pid in "$query_pid" "$frozen_pid" "$other_pid" "$previous_pid"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "[ERROR] at least one full state intervention failed; inspect $output_dir/*.log" >&2
    exit 1
  fi

  python3 scripts/check_kbs_stage9_state_rollout_selection.py \
    --report "online_state=$output_dir/online_state.json" \
    --report "query_only=$output_dir/query_only.json" \
    --report "frozen_initial_state=$output_dir/frozen_initial_state.json" \
    --report "other_question_state=$output_dir/other_question_state.json" \
    --report "previous_evidence_only=$output_dir/previous_evidence_only.json" \
    --expected-qids 3000 \
    --n-bootstrap 10000 \
    --output "$output_dir/summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_5_SELECTION_FULL_OK"
  echo "No training or answer API call was started."
}

selection_status() {
  local output_dir="$OUTPUT_ROOT/selection3000"
  local summary="$output_dir/summary.json"
  if [[ -s "$summary" ]] && python3 - "$summary" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "OK" and not report.get("failures") else 1)
PY
  then
    echo "selection_full: FINISHED_OK"
    echo "summary: $summary"
    return
  fi
  local pid=""
  [[ -s "$output_dir/run.pid" ]] && pid="$(cat "$output_dir/run.pid")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    if [[ -s "$output_dir/online_state.json" ]]; then
      echo "selection_full: RUNNING_INTERVENTIONS pid=$pid"
    else
      echo "selection_full: RUNNING_ONLINE_STATE pid=$pid"
    fi
    find "$output_dir" -maxdepth 1 -name '*.json' -size +0c -printf '%f\n' | sort
  else
    echo "selection_full: FAILED_OR_INCOMPLETE"
    echo "inspect: $output_dir/*.log"
  fi
  echo "Status check completed; no training, GPU inference, or API call was started."
}

prepare_answer_caches() {
  local selection_summary="$OUTPUT_ROOT/selection3000/summary.json"
  python3 - "$selection_summary" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing full selection summary: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "OK" or report.get("failures"):
    raise SystemExit(f"full selection is not a clean OK: {path}")
PY
  python3 scripts/prepare_kbs_stage9_state_rollout_answer_caches.py \
    --selection-root "$OUTPUT_ROOT/selection3000" \
    --cache-root outputs/rag/cache_kbs_stage9_state_rollout \
    --output "$OUTPUT_ROOT/answer_cache_readiness.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_5_ANSWER_CACHE_READINESS_OK"
  echo "No training, GPU inference, or API call was started."
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
  selection_full)
    run_selection_full
    ;;
  selection_status)
    selection_status
    ;;
  prepare_answer_caches)
    prepare_answer_caches
    ;;
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness, selection_smoke, selection_full, selection_status, prepare_answer_caches" >&2
    exit 2
    ;;
esac
