#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PLAN_FILE="md/kbs_three_review_execution_plan.md"
DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_alpha_val_1000_cand50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_v25_validation_gate}"
GPU_LIST="${GPU_LIST:-0}"
MAX_QIDS="${MAX_QIDS:-1000}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
FORCE="${FORCE:-0}"

V25_CHECKPOINT="outputs/ranker/deberta_v3_large_v25_rollout_aligned/best_model.pt"
ANCHOR_CHECKPOINT="outputs/ranker/deberta_v3_large_v23_acra_anchor/best_model.pt"
DIRECT_CHECKPOINT="outputs/ranker/deberta_v3_large_v24_ecdr_direct_indirect/best_model.pt"

for path in \
  "$PLAN_FILE" \
  "$V25_CHECKPOINT" \
  "$ANCHOR_CHECKPOINT" \
  "$DIRECT_CHECKPOINT" \
  "$DATA_ROOT/samples/val.jsonl" \
  "$DATA_ROOT/unit_registry/raw_units_val.jsonl" \
  "$DATA_ROOT/queries/val.jsonl" \
  models/deberta-v3-large \
  models/bge-large-en-v1.5; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done
if [[ "$MAX_QIDS" != "1000" ]]; then
  echo "[ERROR] the pre-registered validation gate requires MAX_QIDS=1000" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
echo "[PLAN] Stage 3R v25 validation-only gate"
echo "[INFO] qids=1000 alpha=0.5 state_update_top_k=1 answers=disabled"

run_method() {
  local name="$1"
  local checkpoint="$2"
  local context="$3"
  local output_dir="$OUTPUT_ROOT/$name"
  local report="$output_dir/alpha_0p50.json"
  if [[ -s "$report" && "$FORCE" != "1" ]]; then
    echo "[ERROR] report already exists: $report (set FORCE=1 only to replace a failed run)" >&2
    exit 1
  fi
  DATA_ROOT="$DATA_ROOT" \
  CHECKPOINT="$checkpoint" \
  OUTPUT_DIR="$output_dir" \
  ALPHAS="0.5" \
  GPU_LIST="$GPU_LIST" \
  MAX_QIDS=1000 \
  GENERATE_ANSWERS=0 \
  POLICY_CONTEXT_SOURCE="$context" \
  STATE_UPDATE_TOP_K=1 \
  bash scripts/run_kbs_alpha_sensitivity_val.sh
}

run_method "v25_full" "$V25_CHECKPOINT" "online_state"
run_method "v23_anchor" "$ANCHOR_CHECKPOINT" "previous_evidence_only"
run_method "v24_direct_indirect" "$DIRECT_CHECKPOINT" "direct_evidence_only"

python3 scripts/analyze_kbs_v25_validation_gate.py \
  --v25-report "$OUTPUT_ROOT/v25_full/alpha_0p50.json" \
  --anchor-report "$OUTPUT_ROOT/v23_anchor/alpha_0p50.json" \
  --direct-report "$OUTPUT_ROOT/v24_direct_indirect/alpha_0p50.json" \
  --expected-qids 1000 \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed 20260731 \
  --output "$OUTPUT_ROOT/validation_gate.json"

echo "[DONE] validation gate=$OUTPUT_ROOT/validation_gate.json"
