#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/kbs_stage6_teacher_objectives}"
OUTPUT="$OUTPUT_DIR/objective_audit.json"

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Stage 6.1 same-state Teacher objective audit"
echo "[INFO] no API, GPU inference, training, or trajectory rebuild"

python3 scripts/analyze_kbs_stage6_teacher_objectives.py \
  --output "$OUTPUT"

echo "[DONE] $OUTPUT"
