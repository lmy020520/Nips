#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

ACTION="${ACTION:-readiness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_stage9_breadth}"
MUSIQUE_PATH="${MUSIQUE_PATH:-}"
TARGET_QIDS="${TARGET_QIDS:-1000}"
SEED="${SEED:-20260914}"

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
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness" >&2
    exit 2
    ;;
esac
