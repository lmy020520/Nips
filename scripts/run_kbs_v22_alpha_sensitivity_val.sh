#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

PLAN_FILE="md/kbs_three_review_execution_plan.md"
if [[ ! -f "$PLAN_FILE" ]]; then
  echo "[ERROR] missing experiment governance plan: $PLAN_FILE" >&2
  exit 1
fi

echo "[PLAN] Stage 2.1: v22 alpha sensitivity on the question-disjoint validation set"
echo "[PLAN] no answer API; select alpha by Step@5 with the pre-registered near-tie rule"

export DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_alpha_val_1000_cand50}"
export SPLIT="${SPLIT:-val}"
export CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/kbs_v22_alpha_sensitivity_val}"
export ALPHAS="${ALPHAS:-0,0.2,0.35,0.5,0.8,1.0}"
export MAX_QIDS="${MAX_QIDS:-1000}"
export GENERATE_ANSWERS=0
export REFRESH_ANSWER_CACHE=0

exec bash "$SCRIPT_DIR/run_kbs_alpha_sensitivity_val.sh"
