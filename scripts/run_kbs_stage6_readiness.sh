#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/kbs_stage6_readiness}"
mkdir -p "$OUTPUT_DIR"

echo "[PLAN] Stage 6 offline artifact inventory"
echo "[INFO] api_calls=0 training_runs=0 gpu_required=0"
python3 scripts/audit_kbs_stage6_artifacts.py \
  --output "$OUTPUT_DIR/artifact_inventory.json"

echo "[DONE] Stage 6 readiness inventory: $OUTPUT_DIR/artifact_inventory.json"
