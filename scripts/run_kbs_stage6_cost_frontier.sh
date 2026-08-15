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
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/kbs_stage6_cost_frontier}"
CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
DATA_ROOT="data/hotpotqa_distractor_eval_3000_cand50"

mkdir -p "$OUTPUT_DIR"

run_readiness() {
  python3 scripts/check_kbs_stage6_cost_frontier.py \
    --output "$OUTPUT_DIR/readiness.json"
}

run_selection() {
  local budget="$1"
  local max_qids="$2"
  local output="$3"
  local front_pool=30
  if [[ "$budget" == "50" ]]; then
    front_pool=50
  fi
  if [[ -f "$output" ]]; then
    echo "[SKIP] report exists: $output"
    return
  fi
  echo "[START] Stage 6.3 selection-only cand$budget qids=$max_qids gpu=$CUDA_DEVICE"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
    --samples "$DATA_ROOT/samples/test.jsonl" \
    --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl" \
    --queries "$DATA_ROOT/queries/test.jsonl" \
    --checkpoint "$CHECKPOINT" \
    --model-dir models/deberta-v3-large \
    --state-mode policy \
    --policy-context-source online_state \
    --selector hybrid_policy \
    --dense-model models/bge-large-en-v1.5 \
    --dense-query-mode state \
    --hybrid-alpha 0.5 \
    --front-pool-k "$front_pool" \
    --front-fusion rrf \
    --local-expansion-window 1 \
    --mmr-lambda 0.7 \
    --mmr-same-doc-similarity 0.35 \
    --candidate-top-k "$budget" \
    --select-top-k 5 \
    --state-update-top-k 1 \
    --policy-score-mode front_policy_blend \
    --policy-blend-weight 0.5 \
    --max-qids "$max_qids" \
    --ks 1,2,3,5 \
    --save-online-states \
    --profile-runtime \
    --profile-warmup-qids 5 \
    --seed 20260608 \
    --output "$output"
  echo "[DONE] $output"
}

case "$ACTION" in
  readiness)
    run_readiness
    echo "[OK] Stage 6.3 readiness finished; GPU/API work was not started"
    ;;
  smoke20)
    run_readiness
    run_selection 15 20 "$OUTPUT_DIR/cand15_smoke20.json"
    run_selection 20 20 "$OUTPUT_DIR/cand20_smoke20.json"
    echo "[DONE] Stage 6.3 cand15/cand20 selection-only smoke"
    ;;
  selection3000)
    if [[ "${KBS_STAGE6_COST_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] full selection-only run is locked by the execution plan" >&2
      exit 1
    fi
    run_readiness
    run_selection 15 3000 "$OUTPUT_DIR/cand15_selection3000.json"
    run_selection 20 3000 "$OUTPUT_DIR/cand20_selection3000.json"
    echo "[DONE] Stage 6.3 cand15/cand20 full selection-only reports"
    ;;
  *)
    echo "[ERROR] ACTION must be readiness, smoke20, or selection3000" >&2
    exit 2
    ;;
esac
