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

CHECKPOINT="outputs/ranker/deberta_v3_large_v23_acra_anchor/best_model.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] missing trained Stage 3.1 anchor checkpoint: $CHECKPOINT" >&2
  exit 1
fi

export RUN="previous_only_compact"
export KBS_CHECKPOINT="$CHECKPOINT"
export KBS_EXPECTED_CHECKPOINT_SUFFIX="deberta_v3_large_v23_acra_anchor/best_model.pt"
export KBS_OUTPUT_NAMESPACE="kbs_stage3_acra_anchor_hotpot"
export KBS_POLICY_BLEND_WEIGHT="0.5"
export KBS_SEPARATELY_TRAINED_ANCHOR="1"

echo "[PLAN] Stage 3.1 separately trained ACRA-style anchor evaluation"
echo "[INFO] input=(question, last predicted evidence, candidate)"
echo "[INFO] matched Compact budget: front=30 candidate=10 selected=5 alpha=0.5"
echo "[INFO] SMOKE=${SMOKE:-0}; fresh V4-Flash non-thinking answers required"

exec bash scripts/run_kbs_v22_stage2_hotpot.sh
