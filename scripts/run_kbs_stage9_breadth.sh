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

answer_smoke_is_valid() {
  python3 - "$OUTPUT_ROOT/compact_seed42_answer_smoke20.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing answer smoke audit: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "SMOKE_OK"
    or report.get("mode") != "musique_answer_smoke"
    or report.get("method") != "compact_seed42"
    or report.get("qids") != 20
    or report.get("failures")
):
    raise SystemExit(f"answer smoke is not a clean registered SMOKE_OK: {path}")
PY
}

run_full_answer_method() {
  local method="$1"
  if [[ "${KBS_STAGE9_MUSIQUE_FULL_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] MuSiQue full answers are locked pending smoke review" >&2
    exit 1
  fi
  answer_smoke_is_valid

  local selector dense_model dense_query_mode reranker_model
  local candidate_top_k state_update_top_k
  selector="hybrid_policy"
  dense_model="models/bge-large-en-v1.5"
  dense_query_mode="state"
  reranker_model=""
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
      dense_model=""
      dense_query_mode="question"
      reranker_model="models/bge-reranker-large"
      candidate_top_k=8
      state_update_top_k=5
      ;;
    *)
      echo "[ERROR] unknown Stage 9.6 answer method: $method" >&2
      exit 1
      ;;
  esac

  local selection="$OUTPUT_ROOT/selection1000/$method.json"
  local cache_dir="outputs/rag/cache_kbs_stage9_musique/$method"
  local report="outputs/rag/kbs_stage9_musique/${method}_full1000.json"
  local audit="$OUTPUT_ROOT/${method}_answer_full1000.json"
  mkdir -p "$cache_dir" "$(dirname "$report")"

  if [[ -s "$report" ]]; then
    if [[ ! -s "$audit" ]]; then
      python3 scripts/check_kbs_stage9_musique_answer_report.py \
        --method "$method" \
        --report "$report" \
        --selection-report "$selection" \
        --queries "$DATA_ROOT/queries/test.jsonl" \
        --cache-dir "$cache_dir" \
        --checkpoint "$CHECKPOINT" \
        --expected-qids 1000 \
        --output "$audit"
    fi
    python3 - "$audit" "$method" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
method = sys.argv[2]
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("method") != method
    or report.get("qids") != 1000
    or report.get("failures")
):
    raise SystemExit(f"existing answer audit is not a clean matching OK: {path}")
PY
    echo "[SKIP] completed answer report: $method"
    return
  fi
  if [[ -e "$audit" ]]; then
    echo "[ERROR] audit exists without a completed answer report: $audit" >&2
    exit 1
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
    --dense-query-mode "$dense_query_mode"
    --hybrid-alpha 0.5
    --front-pool-k 30
    --candidate-top-k "$candidate_top_k"
    --select-top-k 5
    --state-update-top-k "$state_update_top_k"
    --answer-mode json
    --generate-answers
    --answer-cache-dir "$cache_dir"
    --max-qids 1000
    --ks 1,2,3,5
    --llm-max-retries 8
    --llm-retry-sleep 2.0
    --seed 20260608
    --device cuda
    --output "$report"
  )
  if [[ -n "$dense_model" ]]; then
    command+=(--dense-model "$dense_model")
  fi
  if [[ -n "$reranker_model" ]]; then
    command+=(--reranker-model "$reranker_model")
  fi
  if [[ "$selector" == "hybrid_policy" ]]; then
    command+=(
      --front-fusion rrf
      --local-expansion-window 1
      --mmr-lambda 0.7
      --mmr-same-doc-similarity 0.35
      --policy-score-mode front_policy_blend
      --policy-blend-weight 0.5
      --save-online-states
    )
  fi

  echo "[START] Stage 9.6 full answers method=$method qids=1000"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${command[@]}"
  python3 scripts/check_kbs_stage9_musique_answer_report.py \
    --method "$method" \
    --report "$report" \
    --selection-report "$selection" \
    --queries "$DATA_ROOT/queries/test.jsonl" \
    --cache-dir "$cache_dir" \
    --checkpoint "$CHECKPOINT" \
    --expected-qids 1000 \
    --output "$audit"
  echo "FINISHED_OK method=$method"
}

run_full_answer_chain() {
  if [[ "${KBS_STAGE9_MUSIQUE_FULL_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] MuSiQue full answers are locked pending smoke review" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
    echo "[ERROR] export a non-empty DEEPSEEK_API_KEY" >&2
    exit 1
  fi
  export DEEPSEEK_MODEL=deepseek-v4-flash
  export DEEPSEEK_THINKING_MODE=disabled
  answer_smoke_is_valid
  echo "[INFO] checking frozen DeepSeek endpoint before full MuSiQue answer chain"
  python3 scripts/check_deepseek_api.py

  local propagation_root="$OUTPUT_ROOT/cache_propagation"
  mkdir -p "$propagation_root"
  local method
  for method in compact_seed42 balanced_seed42 hybrid bge_reranker; do
    python3 scripts/prepare_kbs_stage9_musique_answer_caches.py \
      --selection-root "$OUTPUT_ROOT/selection1000" \
      --queries "$DATA_ROOT/queries/test.jsonl" \
      --cache-root outputs/rag/cache_kbs_stage9_musique \
      --output "$propagation_root/before_${method}.json"
    run_full_answer_method "$method"
  done
  python3 scripts/prepare_kbs_stage9_musique_answer_caches.py \
    --selection-root "$OUTPUT_ROOT/selection1000" \
    --queries "$DATA_ROOT/queries/test.jsonl" \
    --cache-root outputs/rag/cache_kbs_stage9_musique \
    --output "$propagation_root/after_all.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_6_ALL_MUSIQUE_ANSWERS_OK"
}

start_full_answer_chain() {
  if [[ "${KBS_STAGE9_MUSIQUE_FULL_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] MuSiQue full answers are locked pending smoke review" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
    echo "[ERROR] export a non-empty DEEPSEEK_API_KEY" >&2
    exit 1
  fi
  answer_smoke_is_valid
  local log_dir="outputs/logs/kbs_stage9_breadth"
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
    KBS_STAGE9_MUSIQUE_FULL_ANSWERS_AUTHORIZED=1 \
    ACTION=answer_full_chain \
    CUDA_DEVICE="$CUDA_DEVICE" \
    bash scripts/run_kbs_stage9_breadth.sh \
    >"$log_dir/answer_chain_launcher.log" 2>&1 < /dev/null &
  echo "$!" >"$pid_file"
  echo "answer_chain: STARTED pid=$! gpu=$CUDA_DEVICE"
  echo "Check with: ACTION=answer_status bash scripts/run_kbs_stage9_breadth.sh"
}

show_answer_status() {
  local log_dir="outputs/logs/kbs_stage9_breadth"
  local method audit cache_count
  for method in compact_seed42 balanced_seed42 hybrid bge_reranker; do
    audit="$OUTPUT_ROOT/${method}_answer_full1000.json"
    if [[ -d "outputs/rag/cache_kbs_stage9_musique/$method" ]]; then
      cache_count="$(find "outputs/rag/cache_kbs_stage9_musique/$method" \
        -maxdepth 1 -type f -name '*.json' | wc -l)"
    else
      cache_count=0
    fi
    if [[ -s "$audit" ]] && python3 - "$audit" "$method" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("method") != sys.argv[2]
    or report.get("qids") != 1000
    or report.get("failures")
):
    raise SystemExit(1)
PY
    then
      echo "$method: FINISHED_OK cache_files=$cache_count/1000 remaining=0"
    elif [[ -s "outputs/rag/kbs_stage9_musique/${method}_full1000.json" ]]; then
      echo "$method: REPORT_WRITTEN_AUDIT_PENDING cache_files=$cache_count/1000"
    else
      echo "$method: PENDING_OR_RUNNING cache_files=$cache_count/1000 remaining=$((1000-cache_count))"
    fi
  done
  if [[ -s "$log_dir/answer_chain.pid" ]]; then
    local pid
    pid="$(cat "$log_dir/answer_chain.pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "answer_chain: RUNNING pid=$pid"
    elif grep -aFq "status=STAGE9_6_ALL_MUSIQUE_ANSWERS_OK" \
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

repair_hybrid_single_api_error() {
  if [[ "${KBS_STAGE9_MUSIQUE_HYBRID_REPAIR_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Hybrid repair is locked pending failure review" >&2
    exit 1
  fi
  python3 - "$OUTPUT_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
report_path = Path("outputs/rag/kbs_stage9_musique/hybrid_full1000.json")
audit_path = output_root / "hybrid_answer_full1000.json"
log_path = Path("outputs/logs/kbs_stage9_breadth/answer_chain_launcher.log")
pid_path = Path("outputs/logs/kbs_stage9_breadth/answer_chain.pid")

for path in (report_path, audit_path):
    if not path.is_file():
        raise SystemExit(f"missing Hybrid repair prerequisite: {path}")

audit = json.loads(audit_path.read_text(encoding="utf-8"))
expected_failures = {
    "summary answer_errors=1 != 0",
    "empty answers=1, error answers=1",
}
if (
    audit.get("status") != "FAIL"
    or audit.get("method") != "hybrid"
    or audit.get("qids") != 1000
    or set(audit.get("failures") or []) != expected_failures
):
    raise SystemExit("Hybrid failure differs from the reviewed single-API-error case")

report = json.loads(report_path.read_text(encoding="utf-8"))
rows = report.get("results") or []
bad_rows = [
    row for row in rows
    if (
        not str(row.get("answer") or "").strip()
        or not str(row.get("raw_answer") or "").strip()
        or str(row.get("raw_answer") or "").startswith("ERROR:")
    )
]
if len(rows) != 1000 or len(bad_rows) != 1:
    raise SystemExit(
        f"expected 1000 Hybrid rows and one failed answer; "
        f"found rows={len(rows)}, failures={len(bad_rows)}"
    )

qid = str(bad_rows[0].get("qid") or "")
safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
cache_path = Path("outputs/rag/cache_kbs_stage9_musique/hybrid") / f"{safe_qid}.json"
if not qid or not cache_path.is_file():
    raise SystemExit(f"failed Hybrid cache is missing for qid={qid!r}: {cache_path}")
cache = json.loads(cache_path.read_text(encoding="utf-8"))
if (
    str(cache.get("qid") or "") != qid
    or (
        str(cache.get("answer") or "").strip()
        and str(cache.get("raw_answer") or "").strip()
        and not str(cache.get("raw_answer") or "").startswith("ERROR:")
    )
):
    raise SystemExit(f"cache is not the reviewed failed answer for qid={qid}")

archive = output_root / "failed_attempts" / f"hybrid_single_api_error_{safe_qid}"
if archive.exists():
    raise SystemExit(f"repair archive already exists: {archive}")
archive.mkdir(parents=True)
artifacts = {
    report_path: archive / "hybrid_full1000_failed.json",
    audit_path: archive / "hybrid_answer_full1000_failed_audit.json",
    cache_path: archive / f"{safe_qid}_failed_cache.json",
}
if log_path.is_file():
    artifacts[log_path] = archive / "answer_chain_failed.log"
if pid_path.is_file():
    artifacts[pid_path] = archive / "answer_chain_failed.pid"
for source, destination in artifacts.items():
    source.replace(destination)

manifest = {
    "status": "READY_TO_RETRY_ONE_ANSWER",
    "stage": 9,
    "step": "9.6",
    "method": "hybrid",
    "failed_qid": qid,
    "preserved_valid_cache_files": 999,
    "quarantined_cache_files": 1,
    "quarantined_artifacts": [str(path) for path in artifacts.values()],
    "api_calls": 0,
    "next_gate": "Resume the guarded sequential answer chain.",
}
(archive / "repair_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f"archive: {archive}")
PY
  echo "FINISHED_OK"
  echo "status=STAGE9_6_HYBRID_SINGLE_API_ERROR_QUARANTINED"
  echo "No GPU inference or API call was started."
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
  answer_full_start)
    start_full_answer_chain
    ;;
  answer_full_chain)
    run_full_answer_chain
    ;;
  answer_status)
    show_answer_status
    ;;
  repair_hybrid_single_api_error)
    repair_hybrid_single_api_error
    ;;
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness, build_adapter, audit_adapter, selection_smoke, audit_selection_smoke, selection_full_start, selection_full_worker, selection_status, selection_finalize, selection_bootstrap, prepare_answer_caches, answer_smoke, answer_full_start, answer_full_chain, answer_status, repair_hybrid_single_api_error" >&2
    exit 2
    ;;
esac
