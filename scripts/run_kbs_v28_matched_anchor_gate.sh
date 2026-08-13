#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

MODE="${MODE:-smoke20}"
SEED="${SEED:-42}"
GPU_LIST="${GPU_LIST:-0,1}"
DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_alpha_val_1000_cand50}"
FORCE="${FORCE:-0}"

if [[ "$SEED" == "42" ]]; then
  FULL_CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
  ANCHOR_CHECKPOINT="outputs/ranker/deberta_v3_large_v28_matched_anchor/best_model.pt"
elif [[ "$SEED" == "43" || "$SEED" == "44" ]]; then
  FULL_CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed${SEED}/best_model.pt"
  ANCHOR_CHECKPOINT="outputs/ranker/deberta_v3_large_v28_matched_anchor_seed${SEED}/best_model.pt"
else
  echo "[ERROR] unsupported SEED=$SEED" >&2
  exit 2
fi

case "$MODE" in
  smoke20)
    EXPECTED_QIDS=20
    N_BOOTSTRAP="${N_BOOTSTRAP:-200}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_v28_matched_anchor_gate/seed${SEED}_smoke20}"
    SMOKE_ARG=(--smoke)
    ;;
  validation1000)
    if [[ "${ALLOW_FULL_VALIDATION:-0}" != "1" ]]; then
      echo "[ERROR] validation1000 is locked; set ALLOW_FULL_VALIDATION=1 only after smoke review" >&2
      exit 1
    fi
    EXPECTED_QIDS=1000
    N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_v28_matched_anchor_gate/seed${SEED}_validation1000}"
    SMOKE_ARG=()
    ;;
  *)
    echo "[ERROR] unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

for path in \
  "$FULL_CHECKPOINT" "$ANCHOR_CHECKPOINT" \
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
if [[ -e "$OUTPUT_ROOT/validation_gate.json" && "$FORCE" != "1" ]]; then
  echo "[ERROR] output exists: $OUTPUT_ROOT/validation_gate.json" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
echo "[PLAN] Stage 5 v27 Full versus v28 matched Anchor; answers disabled"
echo "[INFO] mode=$MODE seed=$SEED qids=$EXPECTED_QIDS alpha=0.5 state_update_top_k=1"

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
  CHECKPOINT="$checkpoint" OUTPUT_DIR="$output_dir" \
  ALPHAS="0.5" GPU_LIST="$gpu" MAX_QIDS="$EXPECTED_QIDS" \
  GENERATE_ANSWERS=0 POLICY_CONTEXT_SOURCE="$context" \
  STATE_UPDATE_TOP_K=1 SAVE_ONLINE_STATES=0 \
  bash scripts/run_kbs_alpha_sensitivity_val.sh
}

run_method v27_full "$FULL_CHECKPOINT" online_state "${gpus[0]}" &
pid_full=$!
run_method v28_anchor "$ANCHOR_CHECKPOINT" previous_evidence_only "${gpus[1]}" &
pid_anchor=$!

status=0
for pid in "$pid_full" "$pid_anchor"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" != "0" ]]; then
  echo "[ERROR] at least one validation method failed" >&2
  exit 1
fi

python3 scripts/analyze_kbs_v28_matched_anchor_gate.py \
  --full-report "$OUTPUT_ROOT/v27_full/alpha_0p50.json" \
  --anchor-report "$OUTPUT_ROOT/v28_anchor/alpha_0p50.json" \
  --full-checkpoint "$FULL_CHECKPOINT" \
  --anchor-checkpoint "$ANCHOR_CHECKPOINT" \
  --expected-qids "$EXPECTED_QIDS" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$((20260813 + SEED))" \
  --output "$OUTPUT_ROOT/validation_gate.json" \
  "${SMOKE_ARG[@]}"

echo "[DONE] matched Anchor gate=$OUTPUT_ROOT/validation_gate.json"
