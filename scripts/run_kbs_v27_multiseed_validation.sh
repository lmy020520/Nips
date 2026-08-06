#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_alpha_val_1000_cand50}"
BASE_GATE_ROOT="${BASE_GATE_ROOT:-outputs/analysis/kbs_v27_validation_gate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_v27_multiseed_validation}"
GPU_LIST="${GPU_LIST:-0,1}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
FORCE="${FORCE:-0}"

SEED43_CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt"
SEED44_CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt"
FULL_REPORT="$BASE_GATE_ROOT/v22_full/alpha_0p50.json"
ANCHOR_REPORT="$BASE_GATE_ROOT/v23_anchor/alpha_0p50.json"
DIRECT_REPORT="$BASE_GATE_ROOT/v24_direct_indirect/alpha_0p50.json"
SEED42_GATE="$BASE_GATE_ROOT/validation_gate.json"

for path in \
  "$SEED43_CHECKPOINT" "$SEED44_CHECKPOINT" \
  "$FULL_REPORT" "$ANCHOR_REPORT" "$DIRECT_REPORT" "$SEED42_GATE" \
  "$DATA_ROOT/samples/val.jsonl" \
  "$DATA_ROOT/unit_registry/raw_units_val.jsonl" \
  "$DATA_ROOT/queries/val.jsonl" \
  models/deberta-v3-large models/bge-large-en-v1.5; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done

IFS=',' read -r -a gpus <<< "$GPU_LIST"
if [[ "${#gpus[@]}" -lt 2 ]]; then
  echo "[ERROR] GPU_LIST must contain two GPUs, for example 0,1" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
echo "[PLAN] v27 seed-43/44 validation robustness; answers disabled"

run_seed() {
  local seed="$1"
  local checkpoint="$2"
  local gpu="$3"
  local output_dir="$OUTPUT_ROOT/seed${seed}"
  local report="$output_dir/alpha_0p50.json"
  if [[ -s "$report" && "$FORCE" != "1" ]]; then
    echo "[ERROR] report already exists: $report" >&2
    return 1
  fi
  DATA_ROOT="$DATA_ROOT" \
  CHECKPOINT="$checkpoint" \
  OUTPUT_DIR="$output_dir" \
  ALPHAS="0.5" GPU_LIST="$gpu" MAX_QIDS=1000 \
  GENERATE_ANSWERS=0 POLICY_CONTEXT_SOURCE=online_state \
  STATE_UPDATE_TOP_K=1 SAVE_ONLINE_STATES=0 \
  bash scripts/run_kbs_alpha_sensitivity_val.sh
}

run_seed 43 "$SEED43_CHECKPOINT" "${gpus[0]}" &
pid43=$!
run_seed 44 "$SEED44_CHECKPOINT" "${gpus[1]}" &
pid44=$!

status=0
for pid in "$pid43" "$pid44"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" != "0" ]]; then
  echo "[ERROR] at least one seed validation run failed" >&2
  exit 1
fi

for seed in 43 44; do
  checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed${seed}/best_model.pt"
  python3 scripts/analyze_kbs_v27_validation_gate.py \
    --v27-report "$OUTPUT_ROOT/seed${seed}/alpha_0p50.json" \
    --full-report "$FULL_REPORT" \
    --anchor-report "$ANCHOR_REPORT" \
    --direct-report "$DIRECT_REPORT" \
    --expected-v27-checkpoint "$checkpoint" \
    --expected-qids 1000 \
    --n-bootstrap "$N_BOOTSTRAP" \
    --seed "$((20260805 + seed))" \
    --output "$OUTPUT_ROOT/seed${seed}_validation_gate.json"
done

python3 scripts/summarize_kbs_v27_multiseed.py \
  --gate "42=$SEED42_GATE" \
  --gate "43=$OUTPUT_ROOT/seed43_validation_gate.json" \
  --gate "44=$OUTPUT_ROOT/seed44_validation_gate.json" \
  --output "$OUTPUT_ROOT/summary.json"

echo "[DONE] multiseed summary=$OUTPUT_ROOT/summary.json"
