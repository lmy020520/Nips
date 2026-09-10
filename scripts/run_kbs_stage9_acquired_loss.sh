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
OUTPUT_ROOT="outputs/analysis/kbs_stage9_acquired_loss"
LOG_ROOT="outputs/logs/kbs_stage9_acquired_loss"

for plan in \
  md/kbs_three_review_execution_plan.md \
  md/kbs_review_75_85_execution_plan.md; do
  if [[ ! -f "$plan" ]]; then
    echo "[ERROR] missing experiment plan: $plan" >&2
    exit 1
  fi
done
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

run_readiness() {
  local mode="$1"
  python3 scripts/check_kbs_stage9_acquired_loss_readiness.py \
    --mode "$mode" \
    --output "$OUTPUT_ROOT/${mode}_readiness.json"
}

variant_output_dir() {
  local variant="$1"
  local seed="$2"
  echo "outputs/ranker/deberta_v3_large_v30_${variant}_seed${seed}"
}

show_status() {
  local variant seed output_dir pid_file artifact complete
  for variant in ranking_only ce_margin ce_acquired; do
    for seed in 42 43 44; do
      output_dir="$(variant_output_dir "$variant" "$seed")"
      pid_file="$LOG_ROOT/${variant}_seed${seed}.pid"
      if [[ -s "$pid_file" ]]; then
        local pid
        pid="$(cat "$pid_file")"
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
          echo "${variant}_seed${seed}: RUNNING pid=$pid"
          continue
        fi
      fi
      complete=1
      for artifact in best_model.pt best_val_metrics.json test_metrics.json train_history.json; do
        if [[ ! -s "$output_dir/$artifact" ]]; then
          complete=0
        fi
      done
      if [[ "$complete" == "1" ]]; then
        echo "${variant}_seed${seed}: FINISHED_OK"
      else
        echo "${variant}_seed${seed}: NOT_STARTED_OR_INCOMPLETE"
        echo "  inspect: $LOG_ROOT/${variant}_seed${seed}_launcher.log"
      fi
    done
  done
  echo "Status check completed; no training, GPU inference, or API call was started."
}

run_training() {
  local variant="$1"
  local seed="$2"
  if [[ "${KBS_STAGE9_ACQUIRED_TRAIN_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.2 training is locked pending readiness review" >&2
    exit 1
  fi
  local readiness="$OUTPUT_ROOT/pretrain_readiness.json"
  local config="$OUTPUT_ROOT/configs/${variant}_seed${seed}.yaml"
  local output_dir
  output_dir="$(variant_output_dir "$variant" "$seed")"
  if [[ ! -s "$readiness" || ! -s "$config" ]]; then
    echo "[ERROR] missing audited readiness/config: $readiness or $config" >&2
    exit 1
  fi
  python3 - "$readiness" "$config" "$variant" "$seed" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

readiness_path, config_path = map(Path, sys.argv[1:3])
variant, seed = sys.argv[3:5]
report = json.loads(readiness_path.read_text(encoding="utf-8"))
if report.get("status") != "OK" or report.get("failures"):
    raise SystemExit(f"pre-training readiness is not a clean OK: {readiness_path}")
entry = report["config_audit"]["resolved_configs"][variant][seed]
digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
if digest != entry.get("sha256"):
    raise SystemExit(f"resolved config hash changed after readiness: {config_path}")
PY
  if [[ -e "$output_dir/best_model.pt" ]]; then
    echo "[ERROR] refusing to overwrite checkpoint: $output_dir/best_model.pt" >&2
    exit 1
  fi
  echo "[START] Stage 9.2 variant=$variant seed=$seed gpu=$CUDA_DEVICE"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    python3 src/train/train_ranker.py --config "$config"
  local artifact
  for artifact in best_model.pt best_val_metrics.json test_metrics.json train_history.json; do
    if [[ ! -s "$output_dir/$artifact" ]]; then
      echo "[ERROR] missing completed artifact: $output_dir/$artifact" >&2
      exit 1
    fi
  done
  echo "FINISHED_OK"
  echo "status=STAGE9_2_${variant}_SEED${seed}_TRAINING_OK"
}

case "$ACTION" in
  readiness)
    run_readiness pretrain
    echo "FINISHED_OK"
    echo "status=STAGE9_2_PRETRAIN_READINESS_OK"
    echo "No training, GPU inference, or API call was started."
    ;;
  check_training)
    run_readiness posttrain
    echo "FINISHED_OK"
    echo "status=STAGE9_2_POSTTRAIN_READINESS_OK"
    echo "No training, GPU inference, or API call was started."
    ;;
  status)
    show_status
    ;;
  train_ranking_only_seed42|train_ranking_only_seed43|train_ranking_only_seed44|\
  train_ce_margin_seed42|train_ce_margin_seed43|train_ce_margin_seed44|\
  train_ce_acquired_seed42|train_ce_acquired_seed43|train_ce_acquired_seed44)
    remainder="${ACTION#train_}"
    seed="${remainder##*_seed}"
    variant="${remainder%_seed*}"
    run_training "$variant" "$seed"
    ;;
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness, status, check_training, train_{ranking_only,ce_margin,ce_acquired}_seed{42,43,44}" >&2
    exit 2
    ;;
esac
