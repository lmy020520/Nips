#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MANIFEST="${MANIFEST:-configs/kbs_teacher_student_build_v1_manifest.json}"
SOURCE_ROOT="${SOURCE_ROOT:-data/hotpotqa_distractor_v7_10k_llm_prestep}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/kbs_teacher_student_v1_hybrid_frontend}"
SPLITS="${SPLITS:-train,val,test}"

DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"
CUDA_DEVICE="${CUDA_DEVICE:-5}"
FRONT_POOL_K="${FRONT_POOL_K:-30}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-10}"
LOCAL_EXPANSION_WINDOW="${LOCAL_EXPANSION_WINDOW:-1}"
MMR_LAMBDA="${MMR_LAMBDA:-0.7}"
MMR_SAME_DOC_SIMILARITY="${MMR_SAME_DOC_SIMILARITY:-0.35}"
HARD_NEGATIVE_SOURCE="${HARD_NEGATIVE_SOURCE:-mixed}"
HARD_NEGATIVE_COUNT="${HARD_NEGATIVE_COUNT:-4}"
NATURAL_ONLY="${NATURAL_ONLY:-0}"
CORRUPT_STATE_VARIANTS="${CORRUPT_STATE_VARIANTS:-0}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_ranker_deberta_kbs_official_student_v1.yaml}"

RUN_BUILD="${RUN_BUILD:-1}"
RUN_SCHEMA_VALIDATE="${RUN_SCHEMA_VALIDATE:-1}"
RUN_FRONTEND_VALIDATE="${RUN_FRONTEND_VALIDATE:-0}"
RUN_CONTRIBUTION_SCOPE_CHECK="${RUN_CONTRIBUTION_SCOPE_CHECK:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
FORCE="${FORCE:-1}"

required_paths=("$MANIFEST" "$SOURCE_ROOT" "$DENSE_MODEL")
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done

echo "[INFO] KBS teacher-student build route"
echo "[INFO] manifest=$MANIFEST"
echo "[INFO] source_root=$SOURCE_ROOT"
echo "[INFO] output_root=$OUTPUT_ROOT"
echo "[INFO] dense_model=$DENSE_MODEL"
echo "[INFO] train_config=$TRAIN_CONFIG"

if [[ "$RUN_BUILD" == "1" ]]; then
  build_cmd=(
    python3 scripts/rebuild_hotpotqa_frontend_dataset.py
    --source-root "$SOURCE_ROOT"
    --output-root "$OUTPUT_ROOT"
    --splits "$SPLITS"
    --dense-model "$DENSE_MODEL"
    --device cuda
    --front-pool-k "$FRONT_POOL_K"
    --candidate-top-k "$CANDIDATE_TOP_K"
    --local-expansion-window "$LOCAL_EXPANSION_WINDOW"
    --mmr-lambda "$MMR_LAMBDA"
    --mmr-same-doc-similarity "$MMR_SAME_DOC_SIMILARITY"
    --hard-negative-source "$HARD_NEGATIVE_SOURCE"
    --hard-negative-count "$HARD_NEGATIVE_COUNT"
    --corrupt-state-variants "$CORRUPT_STATE_VARIANTS"
  )
  if [[ "$NATURAL_ONLY" == "1" ]]; then
    build_cmd+=(--natural-only)
  fi
  if [[ "$FORCE" == "1" ]]; then
    build_cmd+=(--force)
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${build_cmd[@]}"
fi

if [[ "$RUN_SCHEMA_VALIDATE" == "1" ]]; then
  python3 scripts/validate_kbs_sample_schema.py \
    --data-root "$OUTPUT_ROOT" \
    --splits "$SPLITS" \
    --output outputs/diagnostics/kbs_teacher_student_v1_schema.json
fi

if [[ "$RUN_FRONTEND_VALIDATE" == "1" ]]; then
  python3 scripts/validate_hotpotqa_frontend_dataset.py \
    --data-root "$OUTPUT_ROOT" \
    --expected-candidates "$CANDIDATE_TOP_K" \
    --splits "$SPLITS"
fi

if [[ "$RUN_CONTRIBUTION_SCOPE_CHECK" == "1" ]]; then
  python3 scripts/check_kbs_contribution_scope.py \
    --data-root "$OUTPUT_ROOT" \
    --splits "$SPLITS" \
    --output outputs/diagnostics/kbs_teacher_student_v1_contribution_scope.json
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  if [[ ! -f "$TRAIN_CONFIG" ]]; then
    echo "[ERROR] missing train config: $TRAIN_CONFIG" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  python3 src/train/train_ranker.py --config "$TRAIN_CONFIG"
fi
