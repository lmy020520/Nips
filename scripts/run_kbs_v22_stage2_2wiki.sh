#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

RUN="${RUN:-}"
case "$RUN" in
  full_compact|full_recall|query_only_compact)
    ;;
  *)
    echo "[ERROR] RUN must be full_compact, full_recall, or query_only_compact" >&2
    exit 2
    ;;
esac
export RUN

export KBS_DATA_ROOT="data/2wiki_multihopqa_eval_1000_cand50"
export KBS_MAX_QIDS="1000"
export KBS_CHECKPOINT="outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt"
export KBS_OUTPUT_NAMESPACE="kbs_v22_stage2_2wiki"
export KBS_EXPECTED_CHECKPOINT_SUFFIX="deberta_v3_large_v22_state_focused/best_model.pt"
export KBS_POLICY_BLEND_WEIGHT="0.5"

echo "[PLAN] Stage 2.4 zero-shot 2Wiki transfer"
echo "[INFO] checkpoint=v22_state_focused alpha=0.5 run=$RUN"
echo "[INFO] no 2Wiki fine-tuning; fresh V4-Flash non-thinking answers required"

exec bash scripts/run_kbs_v22_stage2_hotpot.sh
