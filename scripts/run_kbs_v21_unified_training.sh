#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUNS_TEXT="${RUNS:-full ranking_only no_deficit no_contribution}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
PARALLEL="${PARALLEL:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
LOG_DIR="${LOG_DIR:-outputs/logs/kbs_v21_unified_training}"

declare -A CONFIGS=(
  [full]="configs/train_ranker_deberta_v21_unified_full.yaml"
  [ranking_only]="configs/train_ranker_deberta_v21_unified_ranking_only.yaml"
  [no_deficit]="configs/train_ranker_deberta_v21_unified_no_deficit.yaml"
  [no_contribution]="configs/train_ranker_deberta_v21_unified_no_contribution.yaml"
)

declare -A OUTPUT_DIRS=(
  [full]="outputs/ranker/deberta_v3_large_v21_unified_full"
  [ranking_only]="outputs/ranker/deberta_v3_large_v21_unified_ranking_only"
  [no_deficit]="outputs/ranker/deberta_v3_large_v21_unified_no_deficit"
  [no_contribution]="outputs/ranker/deberta_v3_large_v21_unified_no_contribution"
)

python3 scripts/check_kbs_unified_student_experiment.py --require-paths
mkdir -p "$LOG_DIR"

read -r -a run_names <<< "$RUNS_TEXT"
IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"
if (( ${#gpu_ids[@]} < ${#run_names[@]} )); then
  echo "[ERROR] GPU_LIST has ${#gpu_ids[@]} devices for ${#run_names[@]} runs" >&2
  exit 1
fi

pids=()
cleanup() {
  if (( ${#pids[@]} )); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

run_one() {
  local name="$1"
  local gpu="$2"
  local config="${CONFIGS[$name]:-}"
  local output_dir="${OUTPUT_DIRS[$name]:-}"
  local log="$LOG_DIR/${name}.log"

  if [[ -z "$config" || -z "$output_dir" ]]; then
    echo "[ERROR] unknown run: $name" >&2
    return 1
  fi
  if [[ -f "$output_dir/best_model.pt" && "$FORCE_TRAIN" != "1" ]]; then
    echo "[SKIP] $name already has $output_dir/best_model.pt (set FORCE_TRAIN=1 to retrain)"
    return 0
  fi

  echo "[START] run=$name gpu=$gpu config=$config log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" python3 src/train/train_ranker.py --config "$config" \
    >"$log" 2>&1
  echo "[DONE] run=$name checkpoint=$output_dir/best_model.pt"
}

for index in "${!run_names[@]}"; do
  name="${run_names[$index]}"
  gpu="${gpu_ids[$index]}"
  if [[ "$PARALLEL" == "1" ]]; then
    run_one "$name" "$gpu" &
    pids+=("$!")
  else
    run_one "$name" "$gpu"
  fi
done

if [[ "$PARALLEL" == "1" ]]; then
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" == "1" ]]; then
    echo "[ERROR] at least one training run failed; inspect $LOG_DIR" >&2
    exit 1
  fi
fi

echo "[OK] requested unified training runs completed"
