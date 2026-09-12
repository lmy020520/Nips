#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

ACTION="${ACTION:-readiness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_stage9_strong_baselines}"

for plan in \
  md/kbs_three_review_execution_plan.md \
  md/kbs_review_75_85_execution_plan.md; do
  if [[ ! -f "$plan" ]]; then
    echo "[ERROR] missing experiment plan: $plan" >&2
    exit 1
  fi
done
mkdir -p "$OUTPUT_ROOT"

case "$ACTION" in
  readiness)
    python3 scripts/check_kbs_stage9_strong_baselines_readiness.py \
      --output-root "$OUTPUT_ROOT" \
      --output "$OUTPUT_ROOT/readiness.json"
    echo "FINISHED_OK"
    echo "status=STAGE9_3_READINESS_OK"
    echo "No training, GPU inference, or API call was started."
    ;;
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness" >&2
    exit 2
    ;;
esac
