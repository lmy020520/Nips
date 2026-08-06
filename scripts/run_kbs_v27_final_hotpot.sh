#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

PLAN_FILE="md/kbs_three_review_execution_plan.md"
CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
MODE="${MODE:-smoke20}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

for path in "$PLAN_FILE" "$CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done

case "$MODE" in
  smoke20)
    smoke=1
    ;;
  final3000)
    if [[ "${KBS_V27_FINAL_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] final3000 remains locked until the smoke report is reviewed" >&2
      echo "[INFO] set KBS_V27_FINAL_AUTHORIZED=1 only after the plan authorizes it" >&2
      exit 1
    fi
    smoke=0
    ;;
  *)
    echo "[ERROR] MODE must be smoke20 or final3000" >&2
    exit 2
    ;;
esac

echo "[PLAN] Stage 3X v27 final HotpotQA protocol"
echo "[INFO] mode=$MODE checkpoint=$CHECKPOINT seed=42"
echo "[INFO] alpha=0.5 candidate_top_k=10 select_top_k=5 state_update_top_k=1"
echo "[INFO] answers=DeepSeek V4-Flash non-thinking with a fresh cache"

RUN=full_compact \
SMOKE="$smoke" \
RUN_SUFFIX="$RUN_SUFFIX" \
CUDA_DEVICE="$CUDA_DEVICE" \
KBS_CHECKPOINT="$CHECKPOINT" \
KBS_EXPECTED_CHECKPOINT_SUFFIX="deberta_v3_large_v27_counterfactual_dual/best_model.pt" \
KBS_OUTPUT_NAMESPACE="kbs_v27_final_hotpot" \
KBS_POLICY_BLEND_WEIGHT=0.5 \
KBS_STATE_UPDATE_TOP_K=1 \
bash scripts/run_kbs_v22_stage2_hotpot.sh

echo "[DONE] v27 final protocol mode=$MODE"
