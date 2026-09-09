#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ACTION="${ACTION:-readiness}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
READINESS_DIR="${READINESS_DIR:-outputs/analysis/kbs_stage9_teacher_objective}"
LOG_DIR="${LOG_DIR:-outputs/logs/kbs_stage9_teacher_objective}"

for plan in \
  md/kbs_three_review_execution_plan.md \
  md/kbs_review_75_85_execution_plan.md; do
  if [[ ! -f "$plan" ]]; then
    echo "[ERROR] missing experiment plan: $plan" >&2
    exit 1
  fi
done

mkdir -p "$READINESS_DIR" "$LOG_DIR"

run_readiness() {
  local mode="$1"
  python3 scripts/check_kbs_stage9_teacher_objective_readiness.py \
    --mode "$mode" \
    --output "$READINESS_DIR/${mode}_readiness.json"
}

case "$ACTION" in
  readiness)
    run_readiness pretrain
    echo "FINISHED_OK"
    echo "status=PRETRAIN_READINESS_OK"
    echo "No training or API call was started."
    ;;
  train_seed43|train_seed44)
    if [[ "${KBS_STAGE9_COVERAGE_TRAIN_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] Coverage training is locked pending readiness review" >&2
      exit 1
    fi
    run_readiness pretrain
    seed="${ACTION#train_seed}"
    config="configs/train_ranker_deberta_v29_coverage_greedy_seed${seed}.yaml"
    output_dir="outputs/ranker/deberta_v3_large_v29_coverage_greedy_seed${seed}"
    if [[ -e "$output_dir/best_model.pt" ]]; then
      echo "[ERROR] refusing to overwrite existing checkpoint: $output_dir/best_model.pt" >&2
      exit 1
    fi
    echo "[START] Stage 9.1 Coverage-teacher seed=$seed gpu=$CUDA_DEVICE"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
      python3 src/train/train_ranker.py --config "$config" \
      2>&1 | tee "$LOG_DIR/coverage_seed${seed}.log"
    for artifact in best_model.pt best_val_metrics.json test_metrics.json train_history.json; do
      if [[ ! -s "$output_dir/$artifact" ]]; then
        echo "[ERROR] missing completed training artifact: $output_dir/$artifact" >&2
        exit 1
      fi
    done
    echo "FINISHED_OK"
    echo "status=COVERAGE_SEED_${seed}_TRAINING_OK"
    ;;
  check_training)
    run_readiness posttrain
    echo "FINISHED_OK"
    echo "status=POSTTRAIN_READINESS_OK"
    echo "No API call was started."
    ;;
  *)
    echo "[ERROR] ACTION must be readiness, train_seed43, train_seed44, or check_training" >&2
    exit 2
    ;;
esac
