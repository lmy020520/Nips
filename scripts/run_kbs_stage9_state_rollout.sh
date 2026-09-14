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
CUDA_DEVICE="${CUDA_DEVICE:-0}"
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

run_answer_smoke() {
  if [[ "${KBS_STAGE9_STATE_ANSWER_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.5 answer smoke is locked pending cache review" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
    echo "[ERROR] export a non-empty DEEPSEEK_API_KEY" >&2
    exit 1
  fi
  export DEEPSEEK_MODEL=deepseek-v4-flash
  export DEEPSEEK_THINKING_MODE=disabled

  local readiness="$OUTPUT_ROOT/answer_cache_readiness.json"
  local selection="$OUTPUT_ROOT/selection3000/other_question_state.json"
  local online_report="$OUTPUT_ROOT/selection3000/online_state.json"
  local cache_dir="outputs/rag/cache_kbs_stage9_state_rollout/other_question_state"
  local report="outputs/rag/kbs_stage9_state_rollout/other_question_state_smoke20.json"
  local audit="$OUTPUT_ROOT/other_question_state_answer_smoke20.json"
  python3 - "$readiness" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing cache readiness: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("mode") != "state_rollout_exact_context_answer_cache_preparation"
    or report.get("failures")
):
    raise SystemExit(f"cache readiness is not a clean registered OK: {path}")
PY
  for path in "$selection" "$online_report" "$cache_dir"; do
    if [[ ! -e "$path" ]]; then
      echo "[ERROR] missing answer-smoke prerequisite: $path" >&2
      exit 1
    fi
  done
  if [[ -e "$report" || -e "$audit" ]]; then
    if [[ "${KBS_STAGE9_STATE_RETRY_FAILED_SMOKE:-0}" != "1" ]]; then
      echo "[ERROR] answer-smoke output already exists; refusing overwrite" >&2
      echo "Set KBS_STAGE9_STATE_RETRY_FAILED_SMOKE=1 only for the registered boundary-pairing retry." >&2
      exit 1
    fi
    python3 - "$report" "$audit" "$cache_dir" <<'PY'
import json
import re
import sys
from pathlib import Path

report_path, audit_path, cache_dir = map(Path, sys.argv[1:])
if not report_path.is_file() or not audit_path.is_file():
    raise SystemExit("registered retry requires both the failed report and audit")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if (
    audit.get("status") != "FAIL"
    or audit.get("method") != "other_question_state"
    or audit.get("qids") != 20
    or audit.get("failures") != ["selection context mismatches=1"]
    or (audit.get("cache") or {}).get("invalid") != 0
):
    raise SystemExit("existing smoke is not the registered one-context boundary failure")

answer = json.loads(report_path.read_text(encoding="utf-8"))
selection_path = Path(str(audit["selection_report"]))
selection = json.loads(selection_path.read_text(encoding="utf-8"))
answer_rows = answer.get("results") or []
selection_rows = (selection.get("results") or [])[:20]
if len(answer_rows) != 20 or len(selection_rows) != 20:
    raise SystemExit("registered retry requires two complete 20-qid prefixes")

mismatched = []
for current, frozen in zip(answer_rows, selection_rows):
    if current.get("qid") != frozen.get("qid"):
        raise SystemExit("qid order differs; refusing automatic retry cleanup")
    if any(
        current.get(key) != frozen.get(key)
        for key in ("question", "gold_answer", "selected_unit_ids")
    ):
        mismatched.append(str(current["qid"]))
if len(mismatched) != 1:
    raise SystemExit(f"expected exactly one mismatched qid, found {mismatched}")

safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", mismatched[0])
bad_cache = cache_dir / f"{safe_qid}.json"
if not bad_cache.is_file():
    raise SystemExit(f"missing mismatched cache file: {bad_cache}")
bad_cache.unlink()
report_path.unlink()
audit_path.unlink()
print(f"[RETRY] removed only mismatched qid cache: {mismatched[0]}")
PY
  fi
  mkdir -p "$cache_dir" "$(dirname "$report")"

  echo "[INFO] checking frozen DeepSeek endpoint before bounded state-rollout smoke"
  python3 scripts/check_deepseek_api.py
  echo "[START] Stage 9.5 other-question-state answer smoke; qids=20"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
    --samples data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl \
    --memory data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl \
    --queries data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl \
    --checkpoint "$CHECKPOINT" \
    --model-dir models/deberta-v3-large \
    --state-mode policy \
    --policy-context-source other_question_state \
    --external-policy-state-report "$online_report" \
    --selector hybrid_policy \
    --dense-model models/bge-large-en-v1.5 \
    --dense-query-mode state \
    --hybrid-alpha 0.5 \
    --front-pool-k 30 \
    --front-fusion rrf \
    --local-expansion-window 1 \
    --mmr-lambda 0.7 \
    --mmr-same-doc-similarity 0.35 \
    --candidate-top-k 10 \
    --select-top-k 5 \
    --state-update-top-k 1 \
    --policy-score-mode front_policy_blend \
    --policy-blend-weight 0.5 \
    --answer-mode json \
    --generate-answers \
    --answer-cache-dir "$cache_dir" \
    --save-online-states \
    --max-qids 20 \
    --ks 1,2,3,5 \
    --llm-max-retries 8 \
    --llm-retry-sleep 2.0 \
    --seed 20260608 \
    --device cuda \
    --output "$report"

  python3 scripts/check_kbs_stage9_state_rollout_answer_report.py \
    --method other_question_state \
    --report "$report" \
    --selection-report "$selection" \
    --online-state-report "$online_report" \
    --cache-dir "$cache_dir" \
    --expected-qids 20 \
    --smoke \
    --output "$audit"
  echo "FINISHED_OK"
  echo "status=STAGE9_5_OTHER_QUESTION_ANSWER_SMOKE_OK"
}

answer_smoke_is_valid() {
  python3 - "$OUTPUT_ROOT/other_question_state_answer_smoke20.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing answer smoke audit: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "SMOKE_OK"
    or report.get("method") != "other_question_state"
    or report.get("qids") != 20
    or report.get("failures")
):
    raise SystemExit(f"answer smoke is not a clean registered SMOKE_OK: {path}")
PY
}

run_full_answer_condition() {
  local condition="$1"
  if [[ "${KBS_STAGE9_STATE_FULL_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.5 full answers are locked pending smoke review" >&2
    exit 1
  fi
  answer_smoke_is_valid

  local selection="$OUTPUT_ROOT/selection3000/$condition.json"
  local online_report="$OUTPUT_ROOT/selection3000/online_state.json"
  local cache_dir="outputs/rag/cache_kbs_stage9_state_rollout/$condition"
  local report="outputs/rag/kbs_stage9_state_rollout/${condition}_full3000.json"
  local audit="$OUTPUT_ROOT/${condition}_answer_full3000.json"
  if [[ "$condition" == "online_state" ]]; then
    report="$CORRECT_REPORT"
  fi
  mkdir -p "$cache_dir" "$(dirname "$report")"

  if [[ -s "$report" ]]; then
    if [[ ! -s "$audit" ]]; then
      python3 scripts/check_kbs_stage9_state_rollout_answer_report.py \
        --method "$condition" \
        --report "$report" \
        --selection-report "$selection" \
        --online-state-report "$online_report" \
        --cache-dir "$cache_dir" \
        --expected-qids 3000 \
        --output "$audit"
    fi
    python3 - "$audit" "$condition" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
condition = sys.argv[2]
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("method") != condition
    or report.get("qids") != 3000
    or report.get("failures")
):
    raise SystemExit(f"existing answer audit is not a clean matching OK: {path}")
PY
    echo "[SKIP] completed answer report: $condition"
    return
  fi
  if [[ -e "$audit" ]]; then
    echo "[ERROR] audit exists without a completed answer report: $audit" >&2
    exit 1
  fi

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
    --generate-answers
    --answer-cache-dir "$cache_dir"
    --save-online-states
    --max-qids 3000
    --ks 1,2,3,5
    --llm-max-retries 8
    --llm-retry-sleep 2.0
    --seed 20260608
    --device cuda
    --output "$report"
  )
  if [[ "$condition" == "other_question_state" ]]; then
    command+=(--external-policy-state-report "$online_report")
  fi

  echo "[START] Stage 9.5 full answers condition=$condition qids=3000"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${command[@]}"
  python3 scripts/check_kbs_stage9_state_rollout_answer_report.py \
    --method "$condition" \
    --report "$report" \
    --selection-report "$selection" \
    --online-state-report "$online_report" \
    --cache-dir "$cache_dir" \
    --expected-qids 3000 \
    --output "$audit"
  echo "FINISHED_OK condition=$condition"
}

run_full_answer_chain() {
  if [[ "${KBS_STAGE9_STATE_FULL_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.5 full answers are locked pending smoke review" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
    echo "[ERROR] export a non-empty DEEPSEEK_API_KEY" >&2
    exit 1
  fi
  export DEEPSEEK_MODEL=deepseek-v4-flash
  export DEEPSEEK_THINKING_MODE=disabled
  answer_smoke_is_valid
  echo "[INFO] checking frozen DeepSeek endpoint before full state-condition answer chain"
  python3 scripts/check_deepseek_api.py

  local propagation_root="$OUTPUT_ROOT/cache_propagation"
  mkdir -p "$propagation_root"
  run_full_answer_condition online_state
  local condition
  for condition in other_question_state query_only frozen_initial_state previous_evidence_only; do
    python3 scripts/prepare_kbs_stage9_state_rollout_answer_caches.py \
      --selection-root "$OUTPUT_ROOT/selection3000" \
      --cache-root outputs/rag/cache_kbs_stage9_state_rollout \
      --output "$propagation_root/before_${condition}.json"
    run_full_answer_condition "$condition"
  done
  python3 scripts/prepare_kbs_stage9_state_rollout_answer_caches.py \
    --selection-root "$OUTPUT_ROOT/selection3000" \
    --cache-root outputs/rag/cache_kbs_stage9_state_rollout \
    --output "$propagation_root/after_all.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_5_ALL_STATE_ANSWERS_OK"
}

start_full_answer_chain() {
  if [[ "${KBS_STAGE9_STATE_FULL_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.5 full answers are locked pending smoke review" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
    echo "[ERROR] export a non-empty DEEPSEEK_API_KEY" >&2
    exit 1
  fi
  answer_smoke_is_valid
  local log_dir="outputs/logs/kbs_stage9_state_rollout"
  local pid_file="$log_dir/answer_chain.pid"
  mkdir -p "$log_dir"
  if [[ -s "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(cat "$pid_file")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "[ERROR] answer chain is already running pid=$existing_pid" >&2
      exit 1
    fi
  fi
  nohup env \
    KBS_STAGE9_STATE_FULL_ANSWERS_AUTHORIZED=1 \
    ACTION=answer_full_chain \
    CUDA_DEVICE="$CUDA_DEVICE" \
    bash scripts/run_kbs_stage9_state_rollout.sh \
    >"$log_dir/answer_chain_launcher.log" 2>&1 < /dev/null &
  echo "$!" >"$pid_file"
  echo "answer_chain: STARTED pid=$! gpu=$CUDA_DEVICE"
  echo "Check with: ACTION=answer_status bash scripts/run_kbs_stage9_state_rollout.sh"
}

show_answer_status() {
  local log_dir="outputs/logs/kbs_stage9_state_rollout"
  local condition audit report
  for condition in online_state other_question_state query_only frozen_initial_state previous_evidence_only; do
    audit="$OUTPUT_ROOT/${condition}_answer_full3000.json"
    report="outputs/rag/kbs_stage9_state_rollout/${condition}_full3000.json"
    if [[ "$condition" == "online_state" ]]; then
      report="$CORRECT_REPORT"
    fi
    if [[ -s "$audit" ]] && python3 - "$audit" "$condition" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("method") != sys.argv[2]
    or report.get("qids") != 3000
    or report.get("failures")
):
    raise SystemExit(1)
PY
    then
      echo "$condition: FINISHED_OK"
    elif [[ -s "$report" ]]; then
      echo "$condition: REPORT_WRITTEN_AUDIT_PENDING"
    else
      echo "$condition: PENDING_OR_RUNNING"
    fi
  done
  if [[ -s "$log_dir/answer_chain.pid" ]]; then
    local pid
    pid="$(cat "$log_dir/answer_chain.pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "answer_chain: RUNNING pid=$pid"
    elif grep -aFq "status=STAGE9_5_ALL_STATE_ANSWERS_OK" \
      "$log_dir/answer_chain_launcher.log" 2>/dev/null; then
      echo "answer_chain: FINISHED_OK"
    else
      echo "answer_chain: FAILED_OR_INCOMPLETE"
      echo "  inspect: $log_dir/answer_chain_launcher.log"
    fi
  else
    echo "answer_chain: NOT_STARTED"
  fi
  echo "Status check completed; no GPU inference or API call was started."
}

finalize_answers() {
  local output_dir="$OUTPUT_ROOT/downstream3000"
  local standard_summary="$output_dir/standard_metrics.json"
  local standard_records="$output_dir/standard_metric_records.jsonl"
  local bootstrap="$output_dir/online_state_paired_bootstrap.json"
  local final_summary="$output_dir/final_summary.json"
  local selection_summary="$OUTPUT_ROOT/selection3000/summary.json"
  local cache_summary="$OUTPUT_ROOT/cache_propagation/after_all.json"
  local online_report="$CORRECT_REPORT"
  local gold_report="outputs/rag/full3000_gold_oracle.json"
  local condition path audit report

  for path in "$selection_summary" "$cache_summary" "$online_report" "$gold_report"; do
    if [[ ! -s "$path" ]]; then
      echo "[ERROR] missing finalization prerequisite: $path" >&2
      exit 1
    fi
  done
  for condition in online_state other_question_state query_only frozen_initial_state previous_evidence_only; do
    audit="$OUTPUT_ROOT/${condition}_answer_full3000.json"
    if [[ ! -s "$audit" ]]; then
      echo "[ERROR] missing answer audit: $audit" >&2
      exit 1
    fi
    python3 - "$audit" "$condition" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("method") != sys.argv[2]
    or report.get("qids") != 3000
    or report.get("failures")
):
    raise SystemExit(f"answer audit is not a clean matching OK: {sys.argv[1]}")
PY
  done
  if [[ -s "$final_summary" ]] && python3 - "$final_summary" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "OK" and not report.get("failures") else 1)
PY
  then
    echo "FINISHED_OK"
    echo "status=STAGE9_5_DOWNSTREAM_OK"
    echo "summary=$final_summary"
    echo "Existing clean summary was retained; no GPU inference or API call was started."
    return
  fi
  mkdir -p "$output_dir"

  python3 scripts/evaluate_kbs_standard_metrics.py \
    --report "Online-State=$online_report" \
    --report "Query-Only=outputs/rag/kbs_stage9_state_rollout/query_only_full3000.json" \
    --report "Frozen-Initial=outputs/rag/kbs_stage9_state_rollout/frozen_initial_state_full3000.json" \
    --report "Other-Question=outputs/rag/kbs_stage9_state_rollout/other_question_state_full3000.json" \
    --report "Previous-Evidence-Only=outputs/rag/kbs_stage9_state_rollout/previous_evidence_only_full3000.json" \
    --report "Gold-Oracle=$gold_report" \
    --gold-oracle-name Gold-Oracle \
    --expected-qids 3000 \
    --closure-unit-budgets 10 \
    --output "$standard_summary" \
    --records-output "$standard_records"

  local metrics="answer_em,answer_f1,supporting_fact_f1,supporting_fact_em,joint_f1,joint_em,full_support_coverage,closure_success_at_10"
  python3 scripts/bootstrap_kbs_stage4_metrics.py \
    --records "$standard_records" \
    --primary Online-State \
    --baseline Query-Only \
    --baseline Frozen-Initial \
    --baseline Other-Question \
    --baseline Previous-Evidence-Only \
    --metrics "$metrics" \
    --n-bootstrap 10000 \
    --seed 20260914 \
    --output "$bootstrap"

  python3 scripts/summarize_kbs_stage9_state_rollout_downstream.py \
    --standard-summary "$standard_summary" \
    --selection-summary "$selection_summary" \
    --bootstrap "$bootstrap" \
    --cache-summary "$cache_summary" \
    --answer-audit "online_state=$OUTPUT_ROOT/online_state_answer_full3000.json" \
    --answer-audit "query_only=$OUTPUT_ROOT/query_only_answer_full3000.json" \
    --answer-audit "frozen_initial_state=$OUTPUT_ROOT/frozen_initial_state_answer_full3000.json" \
    --answer-audit "other_question_state=$OUTPUT_ROOT/other_question_state_answer_full3000.json" \
    --answer-audit "previous_evidence_only=$OUTPUT_ROOT/previous_evidence_only_answer_full3000.json" \
    --output "$final_summary"
  echo "FINISHED_OK"
  echo "status=STAGE9_5_DOWNSTREAM_OK"
  echo "summary=$final_summary"
  echo "No training, GPU inference, or API call was started by finalization."
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
  answer_smoke)
    run_answer_smoke
    ;;
  answer_full_start)
    start_full_answer_chain
    ;;
  answer_full_chain)
    run_full_answer_chain
    ;;
  answer_status)
    show_answer_status
    ;;
  finalize_answers)
    finalize_answers
    ;;
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness, selection_smoke, selection_full, selection_status, prepare_answer_caches, answer_smoke, answer_full_start, answer_full_chain, answer_status, finalize_answers" >&2
    exit 2
    ;;
esac
