#!/usr/bin/env bash
set -euo pipefail

# Main experiment for the official KBS online-state RAG system.
# This wrapper intentionally refreshes answer cache by default so each main run
# calls DeepSeek again instead of silently reusing old generations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_eval_3000_cand50}"
export CUDA_DEVICE="${CUDA_DEVICE:-5}"
export MAX_QIDS="${MAX_QIDS:-0}"
export GENERATE_ANSWERS="${GENERATE_ANSWERS:-1}"
export SAVE_ONLINE_STATES="${SAVE_ONLINE_STATES:-1}"
export REFRESH_ANSWER_CACHE="${REFRESH_ANSWER_CACHE:-1}"
export REQUIRE_DEEPSEEK_API_KEY="${REQUIRE_DEEPSEEK_API_KEY:-1}"

export OUTPUT="${OUTPUT:-outputs/rag/main_kbs_official_online_state_v1_full.json}"
export ANSWER_CACHE_DIR="${ANSWER_CACHE_DIR:-outputs/rag/cache_main_kbs_official_online_state_v1_full}"

echo "[INFO] Running main KBS official experiment"
echo "[INFO] MAX_QIDS=$MAX_QIDS (0 means all qids)"
echo "[INFO] OUTPUT=$OUTPUT"

bash scripts/run_kbs_official_online_rag.sh

python3 scripts/check_kbs_pipeline_readiness.py \
  --manifest configs/kbs_official_online_rag_v1_manifest.json \
  --rag-report "$OUTPUT" \
  --output outputs/diagnostics/main_kbs_official_online_state_v1_readiness.json
