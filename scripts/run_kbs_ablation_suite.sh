#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_QIDS="${MAX_QIDS:-0}"
GENERATE_ANSWERS="${GENERATE_ANSWERS:-1}"
REFRESH_ANSWER_CACHE="${REFRESH_ANSWER_CACHE:-1}"
REQUIRE_DEEPSEEK_API_KEY="${REQUIRE_DEEPSEEK_API_KEY:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/rag/ablations_full3000}"
CACHE_ROOT="${CACHE_ROOT:-outputs/rag/cache_ablations_full3000}"

ABLATIONS="${ABLATIONS:-official no_online_state query_only_policy no_front_policy_blend no_policy no_local_expansion no_dense no_bm25 no_deficit_contribution_score}"

if [[ "$GENERATE_ANSWERS" == "1" && "$REQUIRE_DEEPSEEK_API_KEY" == "1" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
  echo
  export DEEPSEEK_API_KEY
fi

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"

echo "[INFO] KBS ablation suite"
echo "[INFO] CUDA_DEVICE=$CUDA_DEVICE"
echo "[INFO] MAX_QIDS=$MAX_QIDS (0 means all qids)"
echo "[INFO] GENERATE_ANSWERS=$GENERATE_ANSWERS"
echo "[INFO] REFRESH_ANSWER_CACHE=$REFRESH_ANSWER_CACHE"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] CACHE_ROOT=$CACHE_ROOT"
echo "[INFO] ABLATIONS=$ABLATIONS"

for ablation in $ABLATIONS; do
  echo
  echo "===== Running ablation: $ablation ====="
  ABLATION="$ablation" \
  CUDA_DEVICE="$CUDA_DEVICE" \
  MAX_QIDS="$MAX_QIDS" \
  GENERATE_ANSWERS="$GENERATE_ANSWERS" \
  REFRESH_ANSWER_CACHE="$REFRESH_ANSWER_CACHE" \
  REQUIRE_DEEPSEEK_API_KEY=0 \
  OUTPUT_DIR="$OUTPUT_DIR" \
  CACHE_ROOT="$CACHE_ROOT" \
  bash scripts/run_kbs_ablation_experiment.sh
done

echo
echo "[INFO] Ablation suite complete."
