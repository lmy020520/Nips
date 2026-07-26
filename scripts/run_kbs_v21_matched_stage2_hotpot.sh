#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

RUN="${RUN:-}"
case "$RUN" in
  full_compact|full_recall)
    ;;
  *)
    echo "[ERROR] RUN must be full_compact or full_recall" >&2
    exit 2
    ;;
esac
export RUN

export KBS_CHECKPOINT="outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt"
export KBS_OUTPUT_NAMESPACE="kbs_v21_matched_stage2_hotpot"
export KBS_EXPECTED_CHECKPOINT_SUFFIX="deberta_v3_large_v21_unified_full/best_model.pt"

echo "[PLAN] Matched v21-v22 Stage 2 audit"
echo "[INFO] variant=v21_matched run=$RUN"
echo "[INFO] generator=deepseek-v4-flash thinking=disabled fresh_cache=required"

exec bash scripts/run_kbs_v22_stage2_hotpot.sh
