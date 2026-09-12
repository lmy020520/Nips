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
  local selector checkpoint candidate_top_k front_pool_k state_update_top_k
  selector="hybrid_policy"
  checkpoint="$CHECKPOINT"
  candidate_top_k=10
  front_pool_k=30
  state_update_top_k=1

  case "$method" in
    compact_seed42|balanced_knee_seed42|recall_seed42)
      ;;
    compact_seed43|balanced_knee_seed43|recall_seed43)
      checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt"
      ;;
    compact_seed44|balanced_knee_seed44|recall_seed44)
      checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt"
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
      echo "[ERROR] unknown Stage 9.4 full-selection method: $method" >&2
      return 1
      ;;
  esac
  if [[ "$method" == balanced_knee_* ]]; then
    candidate_top_k=15
  elif [[ "$method" == recall_* ]]; then
    candidate_top_k=50
    front_pool_k=50
  fi
  if [[ -e "$output" ]]; then
    echo "[ERROR] refusing to overwrite full selection report: $output" >&2
    return 1
  fi

  local -a command=(
    python3 scripts/run_hotpotqa_policy_rag.py
    --samples data/2wiki_multihopqa_eval_1000_cand50/samples/test.jsonl
    --memory data/2wiki_multihopqa_eval_1000_cand50/unit_registry/raw_units_test.jsonl
    --queries data/2wiki_multihopqa_eval_1000_cand50/queries/test.jsonl
    --checkpoint "$checkpoint"
    --state-mode policy
    --policy-context-source online_state
    --selector "$selector"
    --hybrid-alpha 0.5
    --front-pool-k "$front_pool_k"
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

  echo "[START] Stage 9.4 method=$method qids=1000 gpu=$device"
  CUDA_VISIBLE_DEVICES="$device" "${command[@]}"
  echo "FINISHED_OK method=$method"
}

run_full_selection_worker() {
  if [[ "${KBS_STAGE9_2WIKI_FULL_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.4 full selection is locked pending smoke review" >&2
    exit 1
  fi
  selection_smoke_is_valid
  local chain="${METHOD_CHAIN:-}"
  if [[ -z "$chain" ]]; then
    echo "[ERROR] METHOD_CHAIN is required" >&2
    exit 1
  fi
  local -a methods
  IFS=',' read -r -a methods <<< "$chain"
  local method
  for method in "${methods[@]}"; do
    run_full_selection_method "$method" "$CUDA_DEVICE"
  done
  echo "FINISHED_OK chain=$chain"
}

start_full_selection() {
  if [[ "${KBS_STAGE9_2WIKI_FULL_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.4 full selection is locked pending smoke review" >&2
    exit 1
  fi
  selection_smoke_is_valid
  local -a gpu_ids
  IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"
  if [[ "${#gpu_ids[@]}" -lt 8 ]]; then
    echo "[ERROR] GPU_LIST needs eight GPUs, for example 0,1,2,3,4,5,6,7" >&2
    exit 1
  fi
  local output_dir="$OUTPUT_ROOT/selection1000"
  local log_dir="outputs/logs/kbs_stage9_2wiki"
  mkdir -p "$output_dir" "$log_dir"
  local method worker
  for method in \
    compact_seed42 balanced_knee_seed42 recall_seed42 \
    compact_seed43 balanced_knee_seed43 recall_seed43 \
    compact_seed44 balanced_knee_seed44 recall_seed44 \
    hybrid bge_reranker; do
    if [[ -e "$output_dir/$method.json" ]]; then
      echo "[ERROR] existing output for $method; inspect before restarting" >&2
      exit 1
    fi
  done
  for worker in 0 1 2 3 4 5 6 7; do
    if [[ -e "$log_dir/selection_worker${worker}.pid" ]]; then
      echo "[ERROR] existing pid file for worker$worker; inspect before restarting" >&2
      exit 1
    fi
  done

  local -a chains=(
    "compact_seed42,compact_seed43"
    "balanced_knee_seed42,compact_seed44"
    "recall_seed42"
    "balanced_knee_seed43"
    "recall_seed43"
    "balanced_knee_seed44"
    "recall_seed44"
    "hybrid,bge_reranker"
  )
  for worker in 0 1 2 3 4 5 6 7; do
    nohup env \
      KBS_STAGE9_2WIKI_FULL_SELECTION_AUTHORIZED=1 \
      ACTION=selection_full_worker \
      METHOD_CHAIN="${chains[$worker]}" \
      CUDA_DEVICE="${gpu_ids[$worker]}" \
      bash scripts/run_kbs_stage9_2wiki.sh \
      >"$log_dir/selection_worker${worker}_launcher.log" 2>&1 < /dev/null &
    echo "$!" >"$log_dir/selection_worker${worker}.pid"
    echo "worker$worker: STARTED pid=$! gpu=${gpu_ids[$worker]} methods=${chains[$worker]}"
  done
  echo "All Stage 9.4 selection workers started; no answer API call is enabled."
  echo "Check with: ACTION=selection_status bash scripts/run_kbs_stage9_2wiki.sh"
}

show_selection_status() {
  local output_dir="$OUTPUT_ROOT/selection1000"
  local log_dir="outputs/logs/kbs_stage9_2wiki"
  local method worker pid
  for method in \
    compact_seed42 balanced_knee_seed42 recall_seed42 \
    compact_seed43 balanced_knee_seed43 recall_seed43 \
    compact_seed44 balanced_knee_seed44 recall_seed44 \
    hybrid bge_reranker; do
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
    else
      echo "$method: PENDING_OR_INVALID"
    fi
  done
  for worker in 0 1 2 3 4 5 6 7; do
    if [[ -s "$log_dir/selection_worker${worker}.pid" ]]; then
      pid="$(cat "$log_dir/selection_worker${worker}.pid")"
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "worker$worker: RUNNING pid=$pid"
      elif grep -aFq "FINISHED_OK chain=" "$log_dir/selection_worker${worker}_launcher.log" 2>/dev/null; then
        echo "worker$worker: FINISHED_OK"
      else
        echo "worker$worker: FAILED_OR_INCOMPLETE"
        echo "  inspect: $log_dir/selection_worker${worker}_launcher.log"
      fi
    else
      echo "worker$worker: NOT_STARTED"
    fi
  done
  echo "Status check completed; no GPU inference or API call was started."
}

finalize_full_selection() {
  local output_dir="$OUTPUT_ROOT/selection1000"
  python3 scripts/check_kbs_stage9_2wiki_selection.py \
    --report "compact_seed42=$output_dir/compact_seed42.json" \
    --report "balanced_knee_seed42=$output_dir/balanced_knee_seed42.json" \
    --report "recall_seed42=$output_dir/recall_seed42.json" \
    --report "compact_seed43=$output_dir/compact_seed43.json" \
    --report "balanced_knee_seed43=$output_dir/balanced_knee_seed43.json" \
    --report "recall_seed43=$output_dir/recall_seed43.json" \
    --report "compact_seed44=$output_dir/compact_seed44.json" \
    --report "balanced_knee_seed44=$output_dir/balanced_knee_seed44.json" \
    --report "recall_seed44=$output_dir/recall_seed44.json" \
    --report "hybrid=$output_dir/hybrid.json" \
    --report "bge_reranker=$output_dir/bge_reranker.json" \
    --expected-qids 1000 \
    --output "$output_dir/summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_4_SELECTION1000_FINALIZED"
  echo "No GPU inference or answer API call was started by finalization."
}

prepare_answer_caches() {
  if [[ "${KBS_STAGE9_2WIKI_CACHE_PREP_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.4 answer-cache preparation is locked pending selection review" >&2
    exit 1
  fi
  python3 scripts/prepare_kbs_stage9_2wiki_answer_caches.py \
    --selection-root "$OUTPUT_ROOT/selection1000" \
    --cache-root outputs/rag/cache_kbs_stage9_2wiki \
    --output "$OUTPUT_ROOT/answer_cache_readiness.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_4_ANSWER_CACHE_READINESS_OK"
  echo "No training, GPU inference, or answer API call was started."
}

run_answer_smoke() {
  if [[ "${KBS_STAGE9_2WIKI_ANSWER_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.4 answer smoke is locked pending cache review" >&2
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
  local cache_dir="outputs/rag/cache_kbs_stage9_2wiki/compact_seed42"
  local report="outputs/rag/kbs_stage9_2wiki/compact_seed42_smoke20.json"
  local audit="$OUTPUT_ROOT/compact_seed42_answer_smoke20.json"
  local path
  for path in "$readiness" "$selection" "$cache_dir"; do
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
    or report.get("mode")
    != "final_policy_2wiki_exact_context_answer_cache_preparation"
    or report.get("failures")
):
    raise SystemExit(f"cache readiness is not a clean registered OK: {path}")
PY
  if [[ -e "$report" || -e "$audit" ]]; then
    echo "[ERROR] answer-smoke output already exists; refusing overwrite" >&2
    exit 1
  fi
  mkdir -p "$cache_dir" "$(dirname "$report")"

  echo "[INFO] checking frozen DeepSeek endpoint before bounded 2Wiki Compact smoke"
  python3 scripts/check_deepseek_api.py
  echo "[START] Stage 9.4 Compact answer smoke; qids=20"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
    --samples data/2wiki_multihopqa_eval_1000_cand50/samples/test.jsonl \
    --memory data/2wiki_multihopqa_eval_1000_cand50/unit_registry/raw_units_test.jsonl \
    --queries data/2wiki_multihopqa_eval_1000_cand50/queries/test.jsonl \
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

  python3 scripts/check_kbs_stage9_2wiki_answer_report.py \
    --method compact_seed42 \
    --report "$report" \
    --selection-report "$selection" \
    --cache-dir "$cache_dir" \
    --checkpoint "$CHECKPOINT" \
    --expected-qids 20 \
    --smoke \
    --output "$audit"
  echo "FINISHED_OK"
  echo "status=STAGE9_4_COMPACT_ANSWER_SMOKE_OK"
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
    or report.get("mode") != "final_policy_2wiki_answer_smoke"
    or report.get("method") != "compact_seed42"
    or report.get("failures")
):
    raise SystemExit(f"answer smoke is not a clean registered SMOKE_OK: {path}")
PY
}

run_full_answer_method() {
  local method="$1"
  if [[ "${KBS_STAGE9_2WIKI_FULL_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.4 full answers are locked pending smoke review" >&2
    exit 1
  fi
  answer_smoke_is_valid

  local selector dense_model dense_query_mode reranker_model
  local candidate_top_k front_pool_k state_update_top_k
  selector="hybrid_policy"
  dense_model="models/bge-large-en-v1.5"
  dense_query_mode="state"
  reranker_model=""
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
      dense_model=""
      dense_query_mode="question"
      reranker_model="models/bge-reranker-large"
      candidate_top_k=8
      state_update_top_k=5
      ;;
    *)
      echo "[ERROR] unknown Stage 9.4 answer method: $method" >&2
      exit 1
      ;;
  esac

  local selection="$OUTPUT_ROOT/selection1000/$method.json"
  local cache_dir="outputs/rag/cache_kbs_stage9_2wiki/$method"
  local report="outputs/rag/kbs_stage9_2wiki/${method}_full1000.json"
  local audit="$OUTPUT_ROOT/${method}_answer_full1000.json"
  mkdir -p "$cache_dir" "$(dirname "$report")"

  if [[ -s "$report" ]]; then
    if [[ ! -s "$audit" ]]; then
      python3 scripts/check_kbs_stage9_2wiki_answer_report.py \
        --method "$method" \
        --report "$report" \
        --selection-report "$selection" \
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
    --samples data/2wiki_multihopqa_eval_1000_cand50/samples/test.jsonl
    --memory data/2wiki_multihopqa_eval_1000_cand50/unit_registry/raw_units_test.jsonl
    --queries data/2wiki_multihopqa_eval_1000_cand50/queries/test.jsonl
    --checkpoint "$CHECKPOINT"
    --state-mode policy
    --policy-context-source online_state
    --selector "$selector"
    --dense-query-mode "$dense_query_mode"
    --hybrid-alpha 0.5
    --front-pool-k "$front_pool_k"
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

  echo "[START] Stage 9.4 full answers method=$method qids=1000"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${command[@]}"
  python3 scripts/check_kbs_stage9_2wiki_answer_report.py \
    --method "$method" \
    --report "$report" \
    --selection-report "$selection" \
    --cache-dir "$cache_dir" \
    --checkpoint "$CHECKPOINT" \
    --expected-qids 1000 \
    --output "$audit"
  echo "FINISHED_OK method=$method"
}

run_full_answer_chain() {
  if [[ "${KBS_STAGE9_2WIKI_FULL_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.4 full answers are locked pending smoke review" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
    echo "[ERROR] export a non-empty DEEPSEEK_API_KEY" >&2
    exit 1
  fi
  export DEEPSEEK_MODEL=deepseek-v4-flash
  export DEEPSEEK_THINKING_MODE=disabled
  answer_smoke_is_valid
  echo "[INFO] checking frozen DeepSeek endpoint before full 2Wiki answer chain"
  python3 scripts/check_deepseek_api.py

  local propagation_root="$OUTPUT_ROOT/cache_propagation"
  mkdir -p "$propagation_root"
  local method
  for method in \
    compact_seed42 balanced_knee_seed42 recall_seed42 hybrid bge_reranker; do
    python3 scripts/prepare_kbs_stage9_2wiki_answer_caches.py \
      --selection-root "$OUTPUT_ROOT/selection1000" \
      --cache-root outputs/rag/cache_kbs_stage9_2wiki \
      --output "$propagation_root/before_${method}.json"
    run_full_answer_method "$method"
  done
  python3 scripts/prepare_kbs_stage9_2wiki_answer_caches.py \
    --selection-root "$OUTPUT_ROOT/selection1000" \
    --cache-root outputs/rag/cache_kbs_stage9_2wiki \
    --output "$propagation_root/after_all.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_4_ALL_ANSWERS_OK"
}

start_full_answer_chain() {
  if [[ "${KBS_STAGE9_2WIKI_FULL_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.4 full answers are locked pending smoke review" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
    echo "[ERROR] export a non-empty DEEPSEEK_API_KEY" >&2
    exit 1
  fi
  answer_smoke_is_valid
  local log_dir="outputs/logs/kbs_stage9_2wiki"
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
    KBS_STAGE9_2WIKI_FULL_ANSWERS_AUTHORIZED=1 \
    ACTION=answer_full_chain \
    CUDA_DEVICE="$CUDA_DEVICE" \
    bash scripts/run_kbs_stage9_2wiki.sh \
    >"$log_dir/answer_chain_launcher.log" 2>&1 < /dev/null &
  echo "$!" >"$pid_file"
  echo "answer_chain: STARTED pid=$! gpu=$CUDA_DEVICE"
  echo "Check with: ACTION=answer_status bash scripts/run_kbs_stage9_2wiki.sh"
}

show_answer_status() {
  local log_dir="outputs/logs/kbs_stage9_2wiki"
  local method audit
  for method in \
    compact_seed42 balanced_knee_seed42 recall_seed42 hybrid bge_reranker; do
    audit="$OUTPUT_ROOT/${method}_answer_full1000.json"
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
      echo "$method: FINISHED_OK"
    elif [[ -s "outputs/rag/kbs_stage9_2wiki/${method}_full1000.json" ]]; then
      echo "$method: REPORT_WRITTEN_AUDIT_PENDING"
    else
      echo "$method: PENDING_OR_RUNNING"
    fi
  done
  if [[ -s "$log_dir/answer_chain.pid" ]]; then
    local pid
    pid="$(cat "$log_dir/answer_chain.pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "answer_chain: RUNNING pid=$pid"
    elif grep -aFq "status=STAGE9_4_ALL_ANSWERS_OK" \
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
  local output_dir="$OUTPUT_ROOT/downstream1000"
  local standard_summary="$output_dir/standard_metrics.json"
  local standard_records="$output_dir/standard_metric_records.jsonl"
  local selection_summary="$OUTPUT_ROOT/selection1000/summary.json"
  local cache_summary="$OUTPUT_ROOT/cache_propagation/after_all.json"
  local gold_report="outputs/rag/2wiki_generalization_1000/gold_oracle.json"
  local method path audit report

  for path in "$selection_summary" "$cache_summary" "$gold_report"; do
    if [[ ! -s "$path" ]]; then
      echo "[ERROR] missing finalization prerequisite: $path" >&2
      exit 1
    fi
  done
  for method in \
    compact_seed42 balanced_knee_seed42 recall_seed42 hybrid bge_reranker; do
    audit="$OUTPUT_ROOT/${method}_answer_full1000.json"
    report="outputs/rag/kbs_stage9_2wiki/${method}_full1000.json"
    for path in "$audit" "$report"; do
      if [[ ! -s "$path" ]]; then
        echo "[ERROR] missing $method finalization prerequisite: $path" >&2
        exit 1
      fi
    done
    python3 - "$audit" "$method" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "OK"
    or report.get("method") != sys.argv[2]
    or report.get("qids") != 1000
    or report.get("failures")
):
    raise SystemExit(f"answer audit is not a clean matching OK: {path}")
PY
  done

  if [[ -s "$output_dir/final_summary.json" ]] && \
    python3 - "$output_dir/final_summary.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "OK" and not report.get("failures") else 1)
PY
  then
    echo "FINISHED_OK"
    echo "status=STAGE9_4_DOWNSTREAM_OK"
    echo "summary=$output_dir/final_summary.json"
    echo "Existing clean summary was retained; no GPU inference or API call was started."
    return
  fi
  mkdir -p "$output_dir"

  python3 scripts/evaluate_kbs_standard_metrics.py \
    --report "KSG-EA-Compact=outputs/rag/kbs_stage9_2wiki/compact_seed42_full1000.json" \
    --report "KSG-EA-Balanced=outputs/rag/kbs_stage9_2wiki/balanced_knee_seed42_full1000.json" \
    --report "KSG-EA-Recall=outputs/rag/kbs_stage9_2wiki/recall_seed42_full1000.json" \
    --report "Hybrid-RAG=outputs/rag/kbs_stage9_2wiki/hybrid_full1000.json" \
    --report "BGE-Reranker-RAG=outputs/rag/kbs_stage9_2wiki/bge_reranker_full1000.json" \
    --report "Gold-Oracle=$gold_report" \
    --gold-oracle-name Gold-Oracle \
    --expected-qids 1000 \
    --closure-unit-budgets 10,15,50 \
    --output "$standard_summary" \
    --records-output "$standard_records"

  local metrics_common="answer_em,answer_f1,supporting_fact_f1,supporting_fact_em,joint_f1,joint_em,full_support_coverage"
  local -a baseline_args=(--baseline Hybrid-RAG --baseline BGE-Reranker-RAG)
  python3 scripts/bootstrap_kbs_stage4_metrics.py \
    --records "$standard_records" \
    --primary KSG-EA-Compact \
    "${baseline_args[@]}" \
    --metrics "$metrics_common,closure_success_at_10" \
    --n-bootstrap 10000 \
    --seed 20260918 \
    --output "$output_dir/compact_paired_bootstrap.json"
  python3 scripts/bootstrap_kbs_stage4_metrics.py \
    --records "$standard_records" \
    --primary KSG-EA-Balanced \
    "${baseline_args[@]}" \
    --metrics "$metrics_common,closure_success_at_15" \
    --n-bootstrap 10000 \
    --seed 20260919 \
    --output "$output_dir/balanced_paired_bootstrap.json"
  python3 scripts/bootstrap_kbs_stage4_metrics.py \
    --records "$standard_records" \
    --primary KSG-EA-Recall \
    "${baseline_args[@]}" \
    --metrics "$metrics_common,closure_success_at_50" \
    --n-bootstrap 10000 \
    --seed 20260920 \
    --output "$output_dir/recall_paired_bootstrap.json"

  python3 scripts/summarize_kbs_stage9_2wiki.py \
    --standard-summary "$standard_summary" \
    --selection-summary "$selection_summary" \
    --compact-bootstrap "$output_dir/compact_paired_bootstrap.json" \
    --balanced-bootstrap "$output_dir/balanced_paired_bootstrap.json" \
    --recall-bootstrap "$output_dir/recall_paired_bootstrap.json" \
    --cache-summary "$cache_summary" \
    --answer-audit "compact_seed42=$OUTPUT_ROOT/compact_seed42_answer_full1000.json" \
    --answer-audit "balanced_knee_seed42=$OUTPUT_ROOT/balanced_knee_seed42_answer_full1000.json" \
    --answer-audit "recall_seed42=$OUTPUT_ROOT/recall_seed42_answer_full1000.json" \
    --answer-audit "hybrid=$OUTPUT_ROOT/hybrid_answer_full1000.json" \
    --answer-audit "bge_reranker=$OUTPUT_ROOT/bge_reranker_answer_full1000.json" \
    --output "$output_dir/final_summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_4_DOWNSTREAM_OK"
  echo "summary=$output_dir/final_summary.json"
  echo "No training, GPU inference, or API call was started by finalization."
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
    echo "Allowed: readiness, selection_smoke, selection_full_start, selection_full_worker, selection_status, selection_finalize, prepare_answer_caches, answer_smoke, answer_full_start, answer_full_chain, answer_status, finalize_answers" >&2
    exit 2
    ;;
esac
