#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PLAN="md/kbs_three_review_execution_plan.md"
REPORT="${REPORT:-outputs/rag/agentic_llm_eval3000.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/kbs_stage6_agentic_cost}"

if [[ "${KBS_STAGE6_AGENTIC_COST_AUTHORIZED:-0}" != "1" ]]; then
  echo "[ERROR] Stage 6.4 offline cost accounting is locked by the execution plan" >&2
  exit 1
fi
for path in "$PLAN" "$REPORT"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] missing required file: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
python3 scripts/summarize_kbs_stage6_agentic_cost.py \
  --report "$REPORT" \
  --output "$OUTPUT_DIR/summary.json" \
  --tsv-output "$OUTPUT_DIR/agentic_cost.tsv"

echo "[DONE] Stage 6.4 stored agentic-cost audit passed; no API was called"
