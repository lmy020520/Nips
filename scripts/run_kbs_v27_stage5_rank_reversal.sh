#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

RUN="${RUN:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SMOKE="${SMOKE:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_v27_stage5_rank_reversal}"

python3 scripts/check_kbs_v27_stage5_rank_reversal.py \
  --check-only \
  --output "$OUTPUT_ROOT/readiness.json"
if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "[OK] Stage 5 rank-reversal readiness passed; no diagnostic was started"
  exit 0
fi

case "$RUN" in
  seed42)
    seed=42
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
    ;;
  seed43)
    seed=43
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt"
    ;;
  seed44)
    seed=44
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt"
    ;;
  *)
    echo "[ERROR] RUN must be seed42, seed43, or seed44" >&2
    exit 2
    ;;
esac

if [[ "$SMOKE" == "1" ]]; then
  max_qids=20
  n_bootstrap=200
  run_name="${RUN}_smoke20"
  expected_states=0
else
  max_qids=3000
  n_bootstrap=10000
  run_name="$RUN"
  expected_states=7296
fi

output_dir="$OUTPUT_ROOT/$run_name"
summary="$output_dir/summary.json"
if [[ -e "$summary" ]]; then
  echo "[ERROR] refusing to overwrite existing report: $summary" >&2
  echo "[ERROR] use a different OUTPUT_ROOT for an intentional rerun" >&2
  exit 1
fi
mkdir -p "$output_dir"

echo "[PLAN] Stage 5 fixed-pool rank reversal"
echo "[INFO] run=$RUN checkpoint=$checkpoint gpu=$CUDA_DEVICE smoke=$SMOKE"
echo "[INFO] qids=$max_qids bootstrap=$n_bootstrap api_calls=0"

CUDA_DEVICE="$CUDA_DEVICE" \
MAX_QIDS="$max_qids" \
N_BOOTSTRAP="$n_bootstrap" \
CHECKPOINT="$checkpoint" \
OUTPUT_DIR="$output_dir" \
bash scripts/run_kbs_state_phase1.sh

python3 scripts/check_kbs_v27_stage5_rank_reversal.py \
  --report "$summary" \
  --expected-seed "$seed" \
  --expected-qids "$max_qids" \
  --expected-states "$expected_states" \
  --output "$output_dir/report_audit.json"

echo "[DONE] Stage 5 rank reversal run=$RUN smoke=$SMOKE"
