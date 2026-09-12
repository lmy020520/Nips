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
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness, selection_smoke" >&2
    exit 2
    ;;
esac
