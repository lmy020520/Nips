#!/usr/bin/env bash
set -euo pipefail

# Strict w/o contribution-supervision ablation:
# train the official KBS student with contribution_aux_weight=0.00, then
# evaluate the resulting checkpoint on the same official online-state RAG route.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CUDA_DEVICE="${CUDA_DEVICE:-5}"
TRAIN_MODEL="${TRAIN_MODEL:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
MAX_QIDS="${MAX_QIDS:-3000}"
GENERATE_ANSWERS="${GENERATE_ANSWERS:-1}"
REFRESH_ANSWER_CACHE="${REFRESH_ANSWER_CACHE:-1}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_ranker_deberta_v20_no_contribution_supervision.yaml}"
FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-outputs/ranker/deberta_v3_large_v20_no_contribution_supervision/best_model.pt}"

OUTPUT="${OUTPUT:-outputs/rag/no_contribution_supervision_eval3000.json}"
ANSWER_CACHE_DIR="${ANSWER_CACHE_DIR:-outputs/rag/cache_no_contribution_supervision_eval3000}"

if [[ "$TRAIN_MODEL" == "1" ]]; then
  echo "[INFO] training no-contribution student: $TRAIN_CONFIG"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  python3 src/train/train_ranker.py --config "$TRAIN_CONFIG"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ ! -e "$FINAL_CHECKPOINT" ]]; then
    echo "[ERROR] missing final checkpoint: $FINAL_CHECKPOINT" >&2
    exit 1
  fi
  if [[ "$GENERATE_ANSWERS" == "1" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
    echo
    export DEEPSEEK_API_KEY
  fi

  echo "[INFO] evaluating no-contribution checkpoint: $FINAL_CHECKPOINT"
  CHECKPOINT="$FINAL_CHECKPOINT" \
  CUDA_DEVICE="$CUDA_DEVICE" \
  MAX_QIDS="$MAX_QIDS" \
  GENERATE_ANSWERS="$GENERATE_ANSWERS" \
  REFRESH_ANSWER_CACHE="$REFRESH_ANSWER_CACHE" \
  OUTPUT="$OUTPUT" \
  ANSWER_CACHE_DIR="$ANSWER_CACHE_DIR" \
  bash scripts/run_kbs_official_online_rag.sh
fi
