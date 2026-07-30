#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOURCE_ROOT="${SOURCE_ROOT:-data/hotpotqa_distractor_v22_state_focused}"
TEACHER_ROOT="${TEACHER_ROOT:-data/hotpotqa_distractor_v7_10k_llm_prestep}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt}"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"
CONFIG="${CONFIG:-configs/train_ranker_deberta_v25_rollout_aligned.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ranker/deberta_v3_large_v25_rollout_aligned}"
LOG_DIR="${LOG_DIR:-outputs/logs/kbs_v25_rollout_aligned}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_QIDS="${MAX_QIDS:-0}"
if [[ "$MAX_QIDS" == "0" ]]; then
  WORKSPACE="${WORKSPACE:-outputs/analysis/kbs_v25_rollout_workspace}"
  DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_v25_rollout_aligned}"
else
  WORKSPACE="${WORKSPACE:-outputs/analysis/kbs_v25_rollout_workspace_smoke${MAX_QIDS}}"
  DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_v25_rollout_aligned_smoke${MAX_QIDS}}"
fi
BUILD_CANONICAL="${BUILD_CANONICAL:-1}"
COLLECT_ROLLOUTS="${COLLECT_ROLLOUTS:-1}"
BUILD_DATA="${BUILD_DATA:-1}"
FORCE="${FORCE:-0}"
TRAIN="${TRAIN:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

force_arg=()
if [[ "$FORCE" == "1" ]]; then
  force_arg=(--force)
fi

if [[ "$BUILD_CANONICAL" == "1" ]]; then
  if [[ -f "$WORKSPACE/canonical_manifest.json" && "$FORCE" != "1" ]]; then
    echo "[SKIP] canonical data exists: $WORKSPACE/canonical_manifest.json"
  else
    python3 scripts/build_kbs_v25_rollout_aligned_data.py canonicalize \
      --source-root "$SOURCE_ROOT" \
      --workspace "$WORKSPACE" \
      --max-qids "$MAX_QIDS" \
      "${force_arg[@]}"
  fi
fi

if [[ "$COLLECT_ROLLOUTS" == "1" ]]; then
  mkdir -p "$WORKSPACE/rollouts"
  for split in train val test; do
    rollout="$WORKSPACE/rollouts/$split.json"
    if [[ -s "$rollout" && "$FORCE" != "1" ]]; then
      echo "[SKIP] rollout exists: $rollout"
      continue
    fi
    echo "[START] frozen-v22 rollout split=$split gpu=$CUDA_DEVICE"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
      --samples "$WORKSPACE/canonical/$split.jsonl" \
      --memory "$TEACHER_ROOT/unit_registry/raw_units_$split.jsonl" \
      --checkpoint "$SOURCE_CHECKPOINT" \
      --model-dir "$MODEL_DIR" \
      --device cuda \
      --batch-size 16 \
      --max-length 320 \
      --state-mode policy \
      --policy-context-source online_state \
      --selector hybrid_policy \
      --dense-model "$DENSE_MODEL" \
      --dense-query-mode state \
      --hybrid-alpha 0.5 \
      --front-pool-k 30 \
      --front-fusion rrf \
      --local-expansion-window 1 \
      --mmr-lambda 0.7 \
      --mmr-same-doc-similarity 0.35 \
      --candidate-top-k 10 \
      --select-top-k 5 \
      --state-update-top-k 1 \
      --policy-score-mode front_policy_blend \
      --policy-blend-weight 0.5 \
      --save-online-states \
      --online-state-max-raw 8 \
      --online-state-max-chars 260 \
      --stop-control none \
      --seed 20260608 \
      --output "$rollout"
    echo "[DONE] frozen-v22 rollout split=$split output=$rollout"
  done
fi

if [[ "$BUILD_DATA" == "1" ]]; then
  if [[ -f "$DATA_ROOT/manifest.json" && "$FORCE" != "1" ]]; then
    echo "[SKIP] v25 data exists: $DATA_ROOT/manifest.json"
  else
    python3 scripts/build_kbs_v25_rollout_aligned_data.py build \
      --workspace "$WORKSPACE" \
      --teacher-root "$TEACHER_ROOT" \
      --output-root "$DATA_ROOT" \
      --source-checkpoint "$SOURCE_CHECKPOINT" \
      "${force_arg[@]}"
  fi
fi

python3 scripts/check_kbs_v25_rollout_aligned.py \
  --data-root "$DATA_ROOT" \
  --workspace "$WORKSPACE" \
  --teacher-root "$TEACHER_ROOT" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --model-dir "$MODEL_DIR" \
  --dense-model "$DENSE_MODEL" \
  --require-paths

if [[ "$TRAIN" != "1" ]]; then
  echo "[OK] Stage 3R v25 rollout-aligned data readiness passed; training was not started"
  exit 0
fi

if [[ "$MAX_QIDS" != "0" ]]; then
  echo "[ERROR] training is forbidden when MAX_QIDS is nonzero" >&2
  exit 1
fi
if [[ "$DATA_ROOT" != "data/hotpotqa_distractor_v25_rollout_aligned" ]]; then
  echo "[ERROR] training config expects the frozen full v25 DATA_ROOT" >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/best_model.pt" && "$FORCE_TRAIN" != "1" ]]; then
  echo "[SKIP] checkpoint exists: $OUTPUT_DIR/best_model.pt"
  exit 0
fi

mkdir -p "$LOG_DIR"
echo "[START] v25 rollout-aligned training gpu=$CUDA_DEVICE config=$CONFIG"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 src/train/train_ranker.py --config "$CONFIG" \
  2>&1 | tee "$LOG_DIR/train.log"
echo "[DONE] checkpoint=$OUTPUT_DIR/best_model.pt"
