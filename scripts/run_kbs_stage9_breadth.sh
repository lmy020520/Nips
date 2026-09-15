#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

ACTION="${ACTION:-readiness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_stage9_breadth}"
MUSIQUE_PATH="${MUSIQUE_PATH:-}"
TARGET_QIDS="${TARGET_QIDS:-1000}"
SEED="${SEED:-20260914}"
DATA_ROOT="${DATA_ROOT:-data/musique_ans_eval_1000_paragraph20}"
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

ready_musique_source() {
  python3 - "$OUTPUT_ROOT/readiness.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing readiness report: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "READY_MUSIQUE" or report.get("readiness_failures"):
    raise SystemExit(f"readiness is not a clean READY_MUSIQUE: {path}")
source = ((report.get("schema_audit") or {}).get("path") or "").strip()
if not source or not Path(source).is_file():
    raise SystemExit(f"audited MuSiQue source is missing: {source!r}")
print(source)
PY
}

audit_adapter() {
  local source_path="$1"
  python3 scripts/check_kbs_stage9_musique_adapter.py \
    --source-json "$source_path" \
    --data-root "$DATA_ROOT" \
    --size "$TARGET_QIDS" \
    --seed "$SEED" \
    --output "$OUTPUT_ROOT/adapter_readiness.json"
}

adapter_is_valid() {
  python3 - "$OUTPUT_ROOT/adapter_readiness.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing adapter audit: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "OK" or report.get("failure_count") != 0:
    raise SystemExit(f"adapter audit is not a clean OK: {path}")
PY
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

run_full_selection_method() {
  local method="$1"
  local device="$2"
  local output_dir="$OUTPUT_ROOT/selection1000"
  local output="$output_dir/$method.json"
  local selector candidate_top_k state_update_top_k
  selector="hybrid_policy"
  candidate_top_k=10
  state_update_top_k=1

  case "$method" in
    compact_seed42)
      ;;
    balanced_seed42)
      candidate_top_k=15
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
      echo "[ERROR] unknown Stage 9.6 full-selection method: $method" >&2
      return 1
      ;;
  esac
  if [[ -e "$output" ]]; then
    echo "[ERROR] refusing to overwrite full selection report: $output" >&2
    return 1
  fi

  local -a command=(
    python3 scripts/run_hotpotqa_policy_rag.py
    --samples "$DATA_ROOT/samples/test.jsonl"
    --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl"
    --queries "$DATA_ROOT/queries/test.jsonl"
    --checkpoint "$CHECKPOINT"
    --state-mode policy
    --policy-context-source online_state
    --selector "$selector"
    --hybrid-alpha 0.5
    --front-pool-k 30
    --candidate-top-k "$candidate_top_k"
    --select-top-k 5
    --state-update-top-k "$state_update_top_k"
    --answer-mode json
    --max-qids 1000
    --ks 1,2,3,5
    --seed 20260608
    --profile-runtime
    --profile-warmup-qids 20
    --device cuda
    --output "$output"
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

  echo "[START] Stage 9.6 method=$method qids=1000 gpu=$device"
  CUDA_VISIBLE_DEVICES="$device" "${command[@]}"
  echo "FINISHED_OK method=$method"
}

run_full_selection_worker() {
  if [[ "${KBS_STAGE9_MUSIQUE_FULL_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] MuSiQue full selection is locked pending smoke review" >&2
    exit 1
  fi
  selection_smoke_is_valid
  if [[ -z "${METHOD:-}" ]]; then
    echo "[ERROR] METHOD is required" >&2
    exit 1
  fi
  run_full_selection_method "$METHOD" "$CUDA_DEVICE"
}

start_full_selection() {
  if [[ "${KBS_STAGE9_MUSIQUE_FULL_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] MuSiQue full selection is locked pending smoke review" >&2
    exit 1
  fi
  adapter_is_valid
  selection_smoke_is_valid
  for path in \
    "$CHECKPOINT" \
    models/deberta-v3-large \
    models/bge-large-en-v1.5 \
    models/bge-reranker-large; do
    if [[ ! -e "$path" ]]; then
      echo "[ERROR] missing full-selection prerequisite: $path" >&2
      exit 1
    fi
  done

  local -a gpu_ids methods
  IFS=',' read -r -a gpu_ids <<< "${GPU_LIST:-0,1,2,3}"
  methods=(compact_seed42 balanced_seed42 hybrid bge_reranker)
  if [[ "${#gpu_ids[@]}" -lt "${#methods[@]}" ]]; then
    echo "[ERROR] GPU_LIST needs four GPUs, for example 0,1,2,3" >&2
    exit 1
  fi

  local output_dir="$OUTPUT_ROOT/selection1000"
  local log_dir="outputs/logs/kbs_stage9_breadth"
  mkdir -p "$output_dir" "$log_dir"
  local index method
  for method in "${methods[@]}"; do
    if [[ -e "$output_dir/$method.json" || -e "$log_dir/${method}.pid" ]]; then
      echo "[ERROR] existing output or pid for $method; inspect before restarting" >&2
      exit 1
    fi
  done

  for index in "${!methods[@]}"; do
    method="${methods[$index]}"
    nohup env \
      KBS_STAGE9_MUSIQUE_FULL_SELECTION_AUTHORIZED=1 \
      ACTION=selection_full_worker \
      METHOD="$method" \
      CUDA_DEVICE="${gpu_ids[$index]}" \
      bash scripts/run_kbs_stage9_breadth.sh \
      >"$log_dir/${method}_launcher.log" 2>&1 < /dev/null &
    echo "$!" >"$log_dir/${method}.pid"
    echo "$method: STARTED pid=$! gpu=${gpu_ids[$index]}"
  done
  echo "All MuSiQue selection workers started; no answer API call is enabled."
  echo "Check with: ACTION=selection_status bash scripts/run_kbs_stage9_breadth.sh"
}

show_selection_status() {
  local output_dir="$OUTPUT_ROOT/selection1000"
  local log_dir="outputs/logs/kbs_stage9_breadth"
  local method pid
  for method in compact_seed42 balanced_seed42 hybrid bge_reranker; do
    if [[ -s "$output_dir/$method.json" ]] && python3 - "$output_dir/$method.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
summary = report.get("summary") or {}
results = report.get("results") or []
if (
    summary.get("qids") != 1000
    or summary.get("answer_judged") != 0
    or summary.get("skipped") != 0
    or len(results) != 1000
):
    raise SystemExit(1)
PY
    then
      echo "$method: FINISHED_OK"
    elif [[ -s "$log_dir/${method}.pid" ]]; then
      pid="$(cat "$log_dir/${method}.pid")"
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "$method: RUNNING pid=$pid"
      else
        echo "$method: FAILED_OR_INCOMPLETE"
        echo "  inspect: $log_dir/${method}_launcher.log"
      fi
    else
      echo "$method: NOT_STARTED_OR_INCOMPLETE"
    fi
  done
  echo "Status check completed; no GPU inference or API call was started."
}

finalize_full_selection() {
  local output_dir="$OUTPUT_ROOT/selection1000"
  python3 scripts/check_kbs_stage9_musique_full_selection.py \
    --report "compact_seed42=$output_dir/compact_seed42.json" \
    --report "balanced_seed42=$output_dir/balanced_seed42.json" \
    --report "hybrid=$output_dir/hybrid.json" \
    --report "bge_reranker=$output_dir/bge_reranker.json" \
    --data-root "$DATA_ROOT" \
    --expected-qids 1000 \
    --adapter-audit "$OUTPUT_ROOT/adapter_readiness.json" \
    --output "$output_dir/summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_6_MUSIQUE_SELECTION1000_FINALIZED"
  echo "No GPU inference or answer API call was started by finalization."
}

run_selection_bootstrap() {
  local output_dir="$OUTPUT_ROOT/selection1000"
  local selection_summary="$output_dir/summary.json"
  local bootstrap_report="$output_dir/paired_bootstrap.json"
  python3 - "$selection_summary" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing full selection summary: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("mode") != "musique_zero_shot_selection"
    or report.get("qids") != 1000
    or report.get("failures")
):
    raise SystemExit(f"full selection summary is not a clean registered OK: {path}")
PY
  if [[ -e "$bootstrap_report" ]]; then
    echo "[ERROR] refusing to overwrite bootstrap report: $bootstrap_report" >&2
    exit 1
  fi
  python3 scripts/bootstrap_kbs_stage9_musique_selection.py \
    --report "compact_seed42=$output_dir/compact_seed42.json" \
    --report "balanced_seed42=$output_dir/balanced_seed42.json" \
    --report "hybrid=$output_dir/hybrid.json" \
    --report "bge_reranker=$output_dir/bge_reranker.json" \
    --selection-summary "$selection_summary" \
    --expected-qids 1000 \
    --n-bootstrap 10000 \
    --seed 20260914 \
    --output "$bootstrap_report"
  echo "FINISHED_OK"
  echo "status=STAGE9_6_MUSIQUE_SELECTION_BOOTSTRAP_OK"
  echo "report=$bootstrap_report"
  echo "No GPU inference or answer API call was started."
}

prepare_answer_caches() {
  local output_dir="$OUTPUT_ROOT/selection1000"
  local bootstrap_report="$output_dir/paired_bootstrap.json"
  python3 - "$bootstrap_report" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing selection bootstrap report: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("mode") != "musique_zero_shot_selection_bootstrap"
    or report.get("qids") != 1000
    or report.get("failures")
):
    raise SystemExit(f"selection bootstrap is not a clean registered OK: {path}")
PY
  python3 scripts/prepare_kbs_stage9_musique_answer_caches.py \
    --selection-root "$output_dir" \
    --queries "$DATA_ROOT/queries/test.jsonl" \
    --cache-root outputs/rag/cache_kbs_stage9_musique \
    --output "$OUTPUT_ROOT/answer_cache_readiness.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_6_MUSIQUE_ANSWER_CACHE_READY"
  echo "report=$OUTPUT_ROOT/answer_cache_readiness.json"
  echo "No GPU inference or answer API call was started."
}

run_answer_smoke() {
  if [[ "${KBS_STAGE9_MUSIQUE_ANSWER_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] MuSiQue answer smoke is locked pending cache review" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
    echo "[ERROR] export a non-empty DEEPSEEK_API_KEY" >&2
    exit 1
  fi
  export DEEPSEEK_MODEL=deepseek-v4-flash
  export DEEPSEEK_THINKING_MODE=disabled

  local readiness="$OUTPUT_ROOT/answer_cache_readiness.json"
  local selection="$OUTPUT_ROOT/selection1000/compact_seed42.json"
  local cache_dir="outputs/rag/cache_kbs_stage9_musique/compact_seed42"
  local report="outputs/rag/kbs_stage9_musique/compact_seed42_smoke20.json"
  local audit="$OUTPUT_ROOT/compact_seed42_answer_smoke20.json"
  local path
  for path in "$readiness" "$selection" "$DATA_ROOT/queries/test.jsonl" "$cache_dir"; do
    if [[ ! -e "$path" ]]; then
      echo "[ERROR] missing answer-smoke prerequisite: $path" >&2
      exit 1
    fi
  done
  python3 - "$readiness" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("mode") != "musique_exact_context_answer_cache_preparation"
    or (report.get("query_audit") or {}).get("qids_with_aliases") != 287
    or report.get("unique_fresh_contexts_with_cross_method_deduplication") != 3758
    or report.get("failures")
):
    raise SystemExit(f"cache readiness is not the reviewed registered OK: {path}")
PY
  if [[ -e "$report" || -e "$audit" ]]; then
    echo "[ERROR] answer-smoke output already exists; refusing overwrite" >&2
    exit 1
  fi
  mkdir -p "$cache_dir" "$(dirname "$report")"

  echo "[INFO] checking frozen DeepSeek endpoint before bounded MuSiQue smoke"
  python3 scripts/check_deepseek_api.py
  echo "[START] Stage 9.6 Compact answer smoke; qids=20"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
    --samples "$DATA_ROOT/samples/test.jsonl" \
    --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl" \
    --queries "$DATA_ROOT/queries/test.jsonl" \
    --checkpoint "$CHECKPOINT" \
    --state-mode policy \
    --policy-context-source online_state \
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

  python3 scripts/check_kbs_stage9_musique_answer_report.py \
    --method compact_seed42 \
    --report "$report" \
    --selection-report "$selection" \
    --queries "$DATA_ROOT/queries/test.jsonl" \
    --cache-dir "$cache_dir" \
    --checkpoint "$CHECKPOINT" \
    --expected-qids 20 \
    --smoke \
    --output "$audit"
  echo "FINISHED_OK"
  echo "status=STAGE9_6_MUSIQUE_COMPACT_ANSWER_SMOKE_OK"
}

case "$ACTION" in
  readiness)
    command=(
      python3 scripts/check_kbs_stage9_breadth_readiness.py
      --target-qids "$TARGET_QIDS"
      --seed "$SEED"
      --output "$OUTPUT_ROOT/readiness.json"
    )
    if [[ -n "$MUSIQUE_PATH" ]]; then
      command+=(--musique-path "$MUSIQUE_PATH")
    fi
    "${command[@]}"
    echo "AUDIT_COMPLETE"
    echo "report=$OUTPUT_ROOT/readiness.json"
    echo "No download, training, GPU inference, or API call was started."
    ;;
  build_adapter)
    if [[ "${KBS_STAGE9_MUSIQUE_ADAPTER_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] MuSiQue adapter build is locked pending readiness review" >&2
      exit 1
    fi
    source_path="$(ready_musique_source)"
    python3 scripts/prepare_musique_policy_rag_eval.py \
      --source-json "$source_path" \
      --output-root "$DATA_ROOT" \
      --size "$TARGET_QIDS" \
      --seed "$SEED"
    audit_adapter "$source_path"
    echo "FINISHED_OK"
    echo "status=STAGE9_6_MUSIQUE_ADAPTER_OK"
    echo "report=$OUTPUT_ROOT/adapter_readiness.json"
    echo "No training, GPU inference, or API call was started."
    ;;
  audit_adapter)
    source_path="$(ready_musique_source)"
    audit_adapter "$source_path"
    echo "AUDIT_COMPLETE"
    echo "report=$OUTPUT_ROOT/adapter_readiness.json"
    echo "No training, GPU inference, or API call was started."
    ;;
  selection_smoke)
    if [[ "${KBS_STAGE9_MUSIQUE_SELECTION_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] MuSiQue selection smoke is locked pending adapter review" >&2
      exit 1
    fi
    adapter_is_valid
    smoke_dir="$OUTPUT_ROOT/selection_smoke20"
    report="$smoke_dir/compact_seed42.json"
    summary="$smoke_dir/summary.json"
    if [[ -e "$report" || -e "$summary" ]]; then
      echo "[ERROR] refusing to overwrite existing MuSiQue smoke outputs: $smoke_dir" >&2
      exit 1
    fi
    mkdir -p "$smoke_dir"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
      --samples "$DATA_ROOT/samples/test.jsonl" \
      --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl" \
      --queries "$DATA_ROOT/queries/test.jsonl" \
      --checkpoint "$CHECKPOINT" \
      --state-mode policy \
      --policy-context-source online_state \
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
      --max-qids 20 \
      --ks 1,2,3,5 \
      --seed 20260608 \
      --save-online-states \
      --device cuda \
      --output "$report"
    python3 scripts/check_kbs_stage9_musique_selection.py \
      --report "$report" \
      --data-root "$DATA_ROOT" \
      --expected-qids 20 \
      --adapter-audit "$OUTPUT_ROOT/adapter_readiness.json" \
      --output "$summary"
    echo "FINISHED_OK"
    echo "status=STAGE9_6_MUSIQUE_SELECTION_SMOKE_OK"
    echo "report=$summary"
    echo "No answer API call was started."
    ;;
  audit_selection_smoke)
    adapter_is_valid
    smoke_dir="$OUTPUT_ROOT/selection_smoke20"
    report="$smoke_dir/compact_seed42.json"
    summary="$smoke_dir/summary.json"
    if [[ ! -s "$report" ]]; then
      echo "[ERROR] missing existing MuSiQue smoke report: $report" >&2
      exit 1
    fi
    python3 scripts/check_kbs_stage9_musique_selection.py \
      --report "$report" \
      --data-root "$DATA_ROOT" \
      --expected-qids 20 \
      --adapter-audit "$OUTPUT_ROOT/adapter_readiness.json" \
      --output "$summary"
    echo "AUDIT_COMPLETE"
    echo "status=STAGE9_6_MUSIQUE_SELECTION_SMOKE_AUDITED"
    echo "report=$summary"
    echo "The existing GPU report was reused; no inference or API call was started."
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
  selection_bootstrap)
    run_selection_bootstrap
    ;;
  prepare_answer_caches)
    prepare_answer_caches
    ;;
  answer_smoke)
    run_answer_smoke
    ;;
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness, build_adapter, audit_adapter, selection_smoke, audit_selection_smoke, selection_full_start, selection_full_worker, selection_status, selection_finalize, selection_bootstrap, prepare_answer_caches, answer_smoke" >&2
    exit 2
    ;;
esac
