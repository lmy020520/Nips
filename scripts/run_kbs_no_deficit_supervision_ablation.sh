#!/usr/bin/env bash
set -euo pipefail

# Strict w/o deficit-supervision ablation:
# 1) train a v16-style student without deficit_aux_weight
# 2) train a v17-style candidate-role finetune without deficit_aux_weight
# 3) evaluate the final checkpoint on the official KBS RAG route

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CUDA_DEVICE="${CUDA_DEVICE:-5}"
TRAIN_PRETRAIN="${TRAIN_PRETRAIN:-1}"
TRAIN_FINAL="${TRAIN_FINAL:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
MAX_QIDS="${MAX_QIDS:-3000}"
GENERATE_ANSWERS="${GENERATE_ANSWERS:-1}"
REFRESH_ANSWER_CACHE="${REFRESH_ANSWER_CACHE:-1}"

PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/train_ranker_deberta_v19_no_deficit_pretrain.yaml}"
FINAL_CONFIG="${FINAL_CONFIG:-configs/train_ranker_deberta_v19_no_deficit_supervision.yaml}"
FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-outputs/ranker/deberta_v3_large_v19_no_deficit_supervision/best_model.pt}"

OUTPUT="${OUTPUT:-outputs/rag/no_deficit_supervision_eval3000.json}"
ANSWER_CACHE_DIR="${ANSWER_CACHE_DIR:-outputs/rag/cache_no_deficit_supervision_eval3000}"

if [[ "$TRAIN_PRETRAIN" == "1" ]]; then
  echo "[INFO] training no-deficit pretrain: $PRETRAIN_CONFIG"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  python3 src/train/train_ranker.py --config "$PRETRAIN_CONFIG"
fi

if [[ "$TRAIN_FINAL" == "1" ]]; then
  echo "[INFO] training no-deficit final student: $FINAL_CONFIG"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  python3 src/train/train_ranker.py --config "$FINAL_CONFIG"
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

  echo "[INFO] evaluating no-deficit final checkpoint: $FINAL_CHECKPOINT"
  CHECKPOINT="$FINAL_CHECKPOINT" \
  CUDA_DEVICE="$CUDA_DEVICE" \
  MAX_QIDS="$MAX_QIDS" \
  GENERATE_ANSWERS="$GENERATE_ANSWERS" \
  REFRESH_ANSWER_CACHE="$REFRESH_ANSWER_CACHE" \
  OUTPUT="$OUTPUT" \
  ANSWER_CACHE_DIR="$ANSWER_CACHE_DIR" \
  bash scripts/run_kbs_official_online_rag.sh
fi
