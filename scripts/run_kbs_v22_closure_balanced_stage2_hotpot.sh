#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

RUN="${RUN:-}"
case "$RUN" in
  full_compact|query_only_compact)
    ;;
  *)
    echo "[ERROR] RUN must be full_compact or query_only_compact" >&2
    exit 2
    ;;
esac
export RUN

export KBS_CHECKPOINT="outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt"
export KBS_OUTPUT_NAMESPACE="kbs_v22_stage2_closure_balanced"
export KBS_EXPECTED_CHECKPOINT_SUFFIX="deberta_v3_large_v22_state_focused/best_model.pt"
export KBS_POLICY_BLEND_WEIGHT="0.5"

echo "[PLAN] Stage 2 validation-only closure-balance repair"
echo "[INFO] alpha=0.5 selected by validation harmonic mean of Alignment@5 and full-unit coverage"
echo "[INFO] run=$RUN fresh_cache=required; no test alpha sweep is permitted"

exec bash scripts/run_kbs_v22_stage2_hotpot.sh
