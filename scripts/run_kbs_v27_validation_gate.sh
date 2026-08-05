#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_alpha_val_1000_cand50}"
SMOKE="${SMOKE:-1}"
if [[ "$SMOKE" == "1" ]]; then
  EXPECTED_QIDS=20
  N_BOOTSTRAP="${N_BOOTSTRAP:-200}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_v27_validation_gate_smoke20}"
else
  EXPECTED_QIDS=1000
  N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_v27_validation_gate}"
fi
MAX_QIDS="${MAX_QIDS:-$EXPECTED_QIDS}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
FORCE="${FORCE:-0}"

V27_CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
V22_CHECKPOINT="outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt"
ANCHOR_CHECKPOINT="outputs/ranker/deberta_v3_large_v23_acra_anchor/best_model.pt"
DIRECT_CHECKPOINT="outputs/ranker/deberta_v3_large_v24_ecdr_direct_indirect/best_model.pt"

for path in \
  "$V27_CHECKPOINT" \
  "$V22_CHECKPOINT" \
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
if [[ "$MAX_QIDS" != "$EXPECTED_QIDS" ]]; then
  echo "[ERROR] expected MAX_QIDS=$EXPECTED_QIDS for SMOKE=$SMOKE" >&2
  exit 1
fi

IFS=',' read -r -a gpus <<< "$GPU_LIST"
if [[ "${#gpus[@]}" -lt 4 ]]; then
  echo "[ERROR] GPU_LIST must contain four GPUs, for example 0,1,2,3" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
echo "[PLAN] Stage 3X v27 validation gate runtime (answers disabled)"
echo "[INFO] smoke=$SMOKE qids=$MAX_QIDS alpha=0.5 state_update_top_k=1"

run_method() {
  local name="$1"
  local checkpoint="$2"
  local context="$3"
  local gpu="$4"
  local output_dir="$OUTPUT_ROOT/$name"
  local report="$output_dir/alpha_0p50.json"
  if [[ -s "$report" && "$FORCE" != "1" ]]; then
    echo "[ERROR] report already exists: $report" >&2
    return 1
  fi
  DATA_ROOT="$DATA_ROOT" \
  CHECKPOINT="$checkpoint" \
  OUTPUT_DIR="$output_dir" \
  ALPHAS="0.5" \
  GPU_LIST="$gpu" \
  MAX_QIDS="$MAX_QIDS" \
  GENERATE_ANSWERS=0 \
  POLICY_CONTEXT_SOURCE="$context" \
  STATE_UPDATE_TOP_K=1 \
  SAVE_ONLINE_STATES=0 \
  bash scripts/run_kbs_alpha_sensitivity_val.sh
}

run_method "v27_dual" "$V27_CHECKPOINT" "online_state" "${gpus[0]}" &
pid_v27=$!
run_method "v22_full" "$V22_CHECKPOINT" "online_state" "${gpus[1]}" &
pid_v22=$!
run_method "v23_anchor" "$ANCHOR_CHECKPOINT" "previous_evidence_only" "${gpus[2]}" &
pid_anchor=$!
run_method "v24_direct_indirect" "$DIRECT_CHECKPOINT" "direct_evidence_only" "${gpus[3]}" &
pid_direct=$!

status=0
for pid in "$pid_v27" "$pid_v22" "$pid_anchor" "$pid_direct"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" != "0" ]]; then
  echo "[ERROR] at least one validation method failed" >&2
  exit 1
fi

smoke_arg=()
if [[ "$SMOKE" == "1" ]]; then
  smoke_arg+=(--smoke)
fi
python3 scripts/analyze_kbs_v27_validation_gate.py \
  --v27-report "$OUTPUT_ROOT/v27_dual/alpha_0p50.json" \
  --full-report "$OUTPUT_ROOT/v22_full/alpha_0p50.json" \
  --anchor-report "$OUTPUT_ROOT/v23_anchor/alpha_0p50.json" \
  --direct-report "$OUTPUT_ROOT/v24_direct_indirect/alpha_0p50.json" \
  --expected-qids "$MAX_QIDS" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed 20260805 \
  --output "$OUTPUT_ROOT/validation_gate.json" \
  "${smoke_arg[@]}"

echo "[DONE] validation gate=$OUTPUT_ROOT/validation_gate.json"
