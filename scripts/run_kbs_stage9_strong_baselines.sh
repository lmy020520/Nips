#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ACTION="${ACTION:-readiness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_stage9_strong_baselines}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
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

run_selection_smoke() {
  if [[ "${KBS_STAGE9_BASELINE_SELECTION_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.3 selection smoke is locked pending readiness review" >&2
    exit 1
  fi
  local readiness="$OUTPUT_ROOT/readiness.json"
  python3 - "$readiness" <<'PY'
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

  local data_root="data/hotpotqa_distractor_eval_3000_cand50"
  local smoke_root="$OUTPUT_ROOT/selection_smoke20"
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"
  if [[ "${#gpu_ids[@]}" -lt 4 ]]; then
    echo "[ERROR] GPU_LIST needs four GPUs, for example 0,1,2,3" >&2
    exit 1
  fi
  mkdir -p "$smoke_root"

  local method
  for method in bm25 dense hybrid iterative_hybrid bge_reranker; do
    if [[ -e "$smoke_root/$method.json" ]]; then
      echo "[ERROR] smoke report already exists; refusing overwrite: $smoke_root/$method.json" >&2
      exit 1
    fi
  done

  run_method() {
    local method="$1"
    local device="$2"
    local selector dense_model dense_query_mode reranker_model
    case "$method" in
      bm25)
        selector="bm25"; dense_model=""; dense_query_mode="question"; reranker_model=""
        ;;
      dense)
        selector="dense"; dense_model="models/bge-large-en-v1.5"; dense_query_mode="state"; reranker_model=""
        ;;
      hybrid)
        selector="hybrid"; dense_model="models/bge-large-en-v1.5"; dense_query_mode="state"; reranker_model=""
        ;;
      iterative_hybrid)
        selector="iterative_hybrid"; dense_model="models/bge-large-en-v1.5"; dense_query_mode="state"; reranker_model=""
        ;;
      bge_reranker)
        selector="generic_reranker"; dense_model=""; dense_query_mode="question"; reranker_model="models/bge-reranker-large"
        ;;
      *)
        echo "[ERROR] unknown baseline: $method" >&2
        return 1
        ;;
    esac

    local -a command=(
      python3 scripts/run_hotpotqa_policy_rag.py
      --samples "$data_root/samples/test.jsonl"
      --memory "$data_root/unit_registry/raw_units_test.jsonl"
      --queries "$data_root/queries/test.jsonl"
      --checkpoint "$CHECKPOINT"
      --state-mode policy
      --policy-context-source online_state
      --selector "$selector"
      --dense-query-mode "$dense_query_mode"
      --hybrid-alpha 0.5
      --candidate-top-k 8
      --select-top-k 5
      --state-update-top-k 5
      --max-qids 20
      --answer-mode json
      --output "$smoke_root/$method.json"
      --device "$device"
    )
    if [[ -n "$dense_model" ]]; then
      command+=(--dense-model "$dense_model")
    fi
    if [[ -n "$reranker_model" ]]; then
      command+=(--reranker-model "$reranker_model")
    fi
    "${command[@]}" >"$smoke_root/$method.log" 2>&1
  }

  echo "[START] Stage 9.3 five-baseline 20-qid selection smoke; no API"
  run_method bm25 cpu &
  local bm25_pid=$!
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" run_method dense cuda &
  local dense_pid=$!
  CUDA_VISIBLE_DEVICES="${gpu_ids[1]}" run_method hybrid cuda &
  local hybrid_pid=$!
  CUDA_VISIBLE_DEVICES="${gpu_ids[2]}" run_method iterative_hybrid cuda &
  local iterative_pid=$!
  CUDA_VISIBLE_DEVICES="${gpu_ids[3]}" run_method bge_reranker cuda &
  local reranker_pid=$!

  local failed=0 pid
  for pid in "$bm25_pid" "$dense_pid" "$hybrid_pid" "$iterative_pid" "$reranker_pid"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "[ERROR] at least one baseline smoke failed; inspect $smoke_root/*.log" >&2
    exit 1
  fi

  python3 scripts/check_kbs_stage9_strong_baselines_selection.py \
    --report "bm25=$smoke_root/bm25.json" \
    --report "dense=$smoke_root/dense.json" \
    --report "hybrid=$smoke_root/hybrid.json" \
    --report "iterative_hybrid=$smoke_root/iterative_hybrid.json" \
    --report "bge_reranker=$smoke_root/bge_reranker.json" \
    --checkpoint "$CHECKPOINT" \
    --expected-qids 20 \
    --smoke \
    --output "$smoke_root/summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_3_SELECTION_SMOKE_OK"
  echo "No training or answer API call was started."
}

selection_smoke_is_valid() {
  python3 - "$OUTPUT_ROOT/selection_smoke20/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing selection smoke summary: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "SMOKE_OK" or report.get("failures"):
    raise SystemExit(f"selection smoke is not a clean SMOKE_OK: {path}")
PY
}

run_full_selection_worker() {
  if [[ "${KBS_STAGE9_BASELINE_FULL_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.3 full selection is locked pending smoke review" >&2
    exit 1
  fi
  selection_smoke_is_valid

  local method="${BASELINE_METHOD:-}"
  local selector dense_model dense_query_mode reranker_model device
  case "$method" in
    bm25)
      selector="bm25"; dense_model=""; dense_query_mode="question"; reranker_model=""; device="cpu"
      ;;
    dense)
      selector="dense"; dense_model="models/bge-large-en-v1.5"; dense_query_mode="state"; reranker_model=""; device="cuda"
      ;;
    hybrid)
      selector="hybrid"; dense_model="models/bge-large-en-v1.5"; dense_query_mode="state"; reranker_model=""; device="cuda"
      ;;
    iterative_hybrid)
      selector="iterative_hybrid"; dense_model="models/bge-large-en-v1.5"; dense_query_mode="state"; reranker_model=""; device="cuda"
      ;;
    bge_reranker)
      selector="generic_reranker"; dense_model=""; dense_query_mode="question"; reranker_model="models/bge-reranker-large"; device="cuda"
      ;;
    *)
      echo "[ERROR] BASELINE_METHOD must be bm25, dense, hybrid, iterative_hybrid, or bge_reranker" >&2
      exit 1
      ;;
  esac

  local data_root="data/hotpotqa_distractor_eval_3000_cand50"
  local output_dir="$OUTPUT_ROOT/selection3000"
  local output="$output_dir/$method.json"
  mkdir -p "$output_dir"
  if [[ -e "$output" ]]; then
    echo "[ERROR] refusing to overwrite full selection report: $output" >&2
    exit 1
  fi

  local -a command=(
    python3 scripts/run_hotpotqa_policy_rag.py
    --samples "$data_root/samples/test.jsonl"
    --memory "$data_root/unit_registry/raw_units_test.jsonl"
    --queries "$data_root/queries/test.jsonl"
    --checkpoint "$CHECKPOINT"
    --state-mode policy
    --policy-context-source online_state
    --selector "$selector"
    --dense-query-mode "$dense_query_mode"
    --hybrid-alpha 0.5
    --candidate-top-k 8
    --select-top-k 5
    --state-update-top-k 5
    --max-qids 3000
    --answer-mode json
    --profile-runtime
    --profile-warmup-qids 20
    --output "$output"
    --device "$device"
  )
  if [[ -n "$dense_model" ]]; then
    command+=(--dense-model "$dense_model")
  fi
  if [[ -n "$reranker_model" ]]; then
    command+=(--reranker-model "$reranker_model")
  fi

  echo "[START] Stage 9.3 method=$method qids=3000 device=$device"
  if [[ "$device" == "cuda" ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${command[@]}"
  else
    "${command[@]}"
  fi
  echo "FINISHED_OK"
  echo "status=STAGE9_3_${method}_SELECTION3000_OK"
}

start_full_selection() {
  if [[ "${KBS_STAGE9_BASELINE_FULL_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.3 full selection is locked pending smoke review" >&2
    exit 1
  fi
  selection_smoke_is_valid
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"
  if [[ "${#gpu_ids[@]}" -lt 4 ]]; then
    echo "[ERROR] GPU_LIST needs four GPUs, for example 0,1,2,3" >&2
    exit 1
  fi

  local output_dir="$OUTPUT_ROOT/selection3000"
  local log_dir="outputs/logs/kbs_stage9_strong_baselines"
  mkdir -p "$output_dir" "$log_dir"
  local method
  for method in bm25 dense hybrid iterative_hybrid bge_reranker; do
    if [[ -e "$output_dir/$method.json" || -e "$log_dir/$method.pid" ]]; then
      echo "[ERROR] existing output or pid file for $method; inspect before restarting" >&2
      exit 1
    fi
  done

  launch_worker() {
    local method="$1"
    local gpu="$2"
    nohup env \
      KBS_STAGE9_BASELINE_FULL_SELECTION_AUTHORIZED=1 \
      ACTION=selection_full_worker \
      BASELINE_METHOD="$method" \
      CUDA_DEVICE="$gpu" \
      bash scripts/run_kbs_stage9_strong_baselines.sh \
      >"$log_dir/${method}_launcher.log" 2>&1 < /dev/null &
    echo "$!" >"$log_dir/$method.pid"
    echo "$method: STARTED pid=$! device=$gpu"
  }

  launch_worker bm25 cpu
  launch_worker dense "${gpu_ids[0]}"
  launch_worker hybrid "${gpu_ids[1]}"
  launch_worker iterative_hybrid "${gpu_ids[2]}"
  launch_worker bge_reranker "${gpu_ids[3]}"
  echo "All five selection-only workers were started; no answer API call is enabled."
  echo "Check with: ACTION=selection_status bash scripts/run_kbs_stage9_strong_baselines.sh"
}

show_selection_status() {
  local output_dir="$OUTPUT_ROOT/selection3000"
  local log_dir="outputs/logs/kbs_stage9_strong_baselines"
  local method pid
  for method in bm25 dense hybrid iterative_hybrid bge_reranker; do
    if [[ -s "$output_dir/$method.json" ]]; then
      if python3 - "$output_dir/$method.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
summary = report.get("summary") or {}
results = report.get("results") or []
if summary.get("qids") != 3000 or summary.get("answer_judged") != 0:
    raise SystemExit(1)
if len(results) != 3000 or summary.get("skipped") != 0:
    raise SystemExit(1)
PY
      then
        echo "$method: FINISHED_OK"
      else
        echo "$method: INVALID_REPORT"
      fi
      continue
    fi
    if [[ -s "$log_dir/$method.pid" ]]; then
      pid="$(cat "$log_dir/$method.pid")"
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "$method: RUNNING pid=$pid"
        continue
      fi
    fi
    echo "$method: FAILED_OR_INCOMPLETE"
    echo "  inspect: $log_dir/${method}_launcher.log"
  done
  echo "Status check completed; no GPU inference or API call was started."
}

finalize_full_selection() {
  local full_root="$OUTPUT_ROOT/selection3000"
  python3 scripts/check_kbs_stage9_strong_baselines_selection.py \
    --report "bm25=$full_root/bm25.json" \
    --report "dense=$full_root/dense.json" \
    --report "hybrid=$full_root/hybrid.json" \
    --report "iterative_hybrid=$full_root/iterative_hybrid.json" \
    --report "bge_reranker=$full_root/bge_reranker.json" \
    --checkpoint "$CHECKPOINT" \
    --expected-qids 3000 \
    --output "$full_root/summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_3_SELECTION3000_FINALIZED"
  echo "No GPU inference or answer API call was started by finalization."
}

case "$ACTION" in
  readiness)
    python3 scripts/check_kbs_stage9_strong_baselines_readiness.py \
      --output-root "$OUTPUT_ROOT" \
      --output "$OUTPUT_ROOT/readiness.json"
    echo "FINISHED_OK"
    echo "status=STAGE9_3_READINESS_OK"
    echo "No training, GPU inference, or API call was started."
    ;;
  selection_smoke)
    run_selection_smoke
    ;;
  selection_full_start)
    start_full_selection
    ;;
  selection_full_worker)
    run_full_selection_worker
    ;;
  selection_status)
    show_selection_status
    ;;
  selection_finalize)
    finalize_full_selection
    ;;
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness, selection_smoke, selection_full_start, selection_full_worker, selection_status, selection_finalize" >&2
    exit 2
    ;;
esac
