#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PLAN_FILE="md/kbs_three_review_execution_plan.md"
if [[ ! -f "$PLAN_FILE" ]]; then
  echo "[ERROR] missing experiment governance plan: $PLAN_FILE" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_alpha_val_1000_cand50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_stage3_rollout_aligned_state_val}"
GPU_LIST="${GPU_LIST:-0}"
MAX_QIDS="${MAX_QIDS:-1000}"
FULL_CHECKPOINT="outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt"
ANCHOR_CHECKPOINT="outputs/ranker/deberta_v3_large_v23_acra_anchor/best_model.pt"

for path in \
  "$FULL_CHECKPOINT" \
  "$ANCHOR_CHECKPOINT" \
  "$DATA_ROOT/samples/val.jsonl" \
  "$DATA_ROOT/unit_registry/raw_units_val.jsonl" \
  "$DATA_ROOT/queries/val.jsonl"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done

echo "[PLAN] Stage 3.1 validation-only rollout-aligned state repair"
echo "[INFO] final evidence budget remains top-5; only top-1 updates online K_t"
echo "[INFO] no answer API; alpha remains frozen at 0.5"

DATA_ROOT="$DATA_ROOT" \
CHECKPOINT="$FULL_CHECKPOINT" \
OUTPUT_DIR="$OUTPUT_ROOT/full_state_update1" \
ALPHAS="0.5" \
GPU_LIST="$GPU_LIST" \
MAX_QIDS="$MAX_QIDS" \
GENERATE_ANSWERS=0 \
POLICY_CONTEXT_SOURCE="online_state" \
STATE_UPDATE_TOP_K=1 \
bash scripts/run_kbs_alpha_sensitivity_val.sh

DATA_ROOT="$DATA_ROOT" \
CHECKPOINT="$ANCHOR_CHECKPOINT" \
OUTPUT_DIR="$OUTPUT_ROOT/trained_anchor" \
ALPHAS="0.5" \
GPU_LIST="$GPU_LIST" \
MAX_QIDS="$MAX_QIDS" \
GENERATE_ANSWERS=0 \
POLICY_CONTEXT_SOURCE="previous_evidence_only" \
STATE_UPDATE_TOP_K=1 \
bash scripts/run_kbs_alpha_sensitivity_val.sh

python3 scripts/analyze_kbs_stage3_anchor_vs_full.py \
  --full-report "$OUTPUT_ROOT/full_state_update1/alpha_0p50.json" \
  --anchor-report "$OUTPUT_ROOT/trained_anchor/alpha_0p50.json" \
  --queries "$DATA_ROOT/queries/val.jsonl" \
  --n-bootstrap 10000 \
  --seed 20260730 \
  --output "$OUTPUT_ROOT/diagnostic.json"

echo "[DONE] validation diagnostic=$OUTPUT_ROOT/diagnostic.json"
