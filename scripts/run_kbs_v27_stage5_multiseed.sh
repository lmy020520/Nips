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
CHECK_ONLY="${CHECK_ONLY:-0}"
READINESS_OUTPUT="${READINESS_OUTPUT:-outputs/analysis/kbs_v27_stage5/readiness.json}"

python3 scripts/check_kbs_v27_stage5_readiness.py --output "$READINESS_OUTPUT"
if [[ "$CHECK_ONLY" == "1" ]]; then
  echo "[OK] Stage 5 readiness passed; no API run was started"
  exit 0
fi

if [[ -z "$RUN" ]]; then
  cat >&2 <<'EOF'
[ERROR] RUN is required. Currently authorized runs:
  compact_seed43
  compact_seed44
  recall_seed42
  recall_seed43
  recall_seed44

Each operating point still requires its explicit authorization environment
variable. The execution plan is the source of truth for authorization.
EOF
  exit 2
fi

case "$RUN" in
  compact_seed43)
    seed=43
    operating_point=compact
    runtime_run=full_compact
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt"
    ;;
  compact_seed44)
    seed=44
    operating_point=compact
    runtime_run=full_compact
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt"
    ;;
  recall_seed42)
    seed=42
    operating_point=recall
    runtime_run=full_recall
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
    ;;
  recall_seed43)
    seed=43
    operating_point=recall
    runtime_run=full_recall
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt"
    ;;
  recall_seed44)
    seed=44
    operating_point=recall
    runtime_run=full_recall
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt"
    ;;
  *)
    echo "[ERROR] unknown RUN=$RUN" >&2
    exit 2
    ;;
esac

if [[ "$operating_point" == "compact" ]]; then
  if [[ "${KBS_STAGE5_COMPACT_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Compact Stage-5 API run is locked" >&2
    exit 1
  fi
else
  if [[ "${KBS_STAGE5_RECALL_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Recall Stage-5 API run is locked" >&2
    exit 1
  fi
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
  echo "[ERROR] export a non-empty DEEPSEEK_API_KEY before launching Stage 5" >&2
  exit 1
fi

checkpoint_suffix="$(basename "$(dirname "$checkpoint")")/best_model.pt"
namespace="kbs_v27_stage5_multiseed/seed${seed}"

echo "[PLAN] Stage 5 v27 multiseed end-to-end evaluation"
echo "[INFO] run=$RUN seed=$seed operating_point=$operating_point gpu=$CUDA_DEVICE"
echo "[INFO] alpha=0.5 select_top_k=5 state_update_top_k=1 fresh_cache=required"

RUN="$runtime_run" \
SMOKE=0 \
CUDA_DEVICE="$CUDA_DEVICE" \
KBS_MAX_QIDS=3000 \
KBS_CHECKPOINT="$checkpoint" \
KBS_EXPECTED_CHECKPOINT_SUFFIX="$checkpoint_suffix" \
KBS_OUTPUT_NAMESPACE="$namespace" \
KBS_POLICY_BLEND_WEIGHT=0.5 \
KBS_STATE_UPDATE_TOP_K=1 \
bash scripts/run_kbs_v22_stage2_hotpot.sh

echo "[DONE] Stage 5 run=$RUN"
