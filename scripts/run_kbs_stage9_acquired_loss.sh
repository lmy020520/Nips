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
GPU_LIST="${GPU_LIST:-0,1,2,3}"
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

checkpoint_for() {
  local variant="$1"
  local seed="$2"
  if [[ "$variant" == "full" ]]; then
    local suffix=""
    if [[ "$seed" != "42" ]]; then
      suffix="_seed${seed}"
    fi
    echo "outputs/ranker/deberta_v3_large_v27_counterfactual_dual${suffix}/best_model.pt"
  else
    echo "outputs/ranker/deberta_v3_large_v30_${variant}_seed${seed}/best_model.pt"
  fi
}

run_selection_smoke() {
  if [[ "${KBS_STAGE9_ACQUIRED_SELECTION_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.2 selection smoke is locked pending training-metric review" >&2
    exit 1
  fi
  run_readiness posttrain
  python3 scripts/summarize_kbs_stage9_acquired_loss_training.py \
    --output "$OUTPUT_ROOT/training_metrics_summary.json"

  local data_root="data/hotpotqa_distractor_eval_3000_cand50"
  local smoke_root="$OUTPUT_ROOT/selection_smoke20"
  local -a variants=(full ranking_only ce_margin ce_acquired)
  local -a gpus
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  if [[ "${#gpus[@]}" -lt 4 ]]; then
    echo "[ERROR] GPU_LIST must contain four GPUs, for example 0,1,2,3" >&2
    exit 1
  fi

  local variant
  for variant in "${variants[@]}"; do
    local report="$smoke_root/${variant}_seed42/alpha_0p50.json"
    if [[ -e "$report" ]]; then
      echo "[ERROR] smoke report already exists; refusing silent reuse: $report" >&2
      exit 1
    fi
  done

  run_smoke_method() {
    local variant="$1"
    local gpu="$2"
    local output_dir="$smoke_root/${variant}_seed42"
    mkdir -p "$output_dir"
    DATA_ROOT="$data_root" \
    SPLIT=test \
    CHECKPOINT="$(checkpoint_for "$variant" 42)" \
    OUTPUT_DIR="$output_dir" \
    ALPHAS="0.5" \
    GPU_LIST="$gpu" \
    MAX_QIDS=20 \
    GENERATE_ANSWERS=0 \
    POLICY_CONTEXT_SOURCE=online_state \
    STATE_UPDATE_TOP_K=1 \
    SAVE_ONLINE_STATES=0 \
    bash scripts/run_kbs_alpha_sensitivity_val.sh
  }

  mkdir -p "$smoke_root"
  echo "[START] Stage 9.2 four-variant selection smoke; answers disabled"
  local -a pids=()
  local index=0
  for variant in "${variants[@]}"; do
    run_smoke_method "$variant" "${gpus[$index]}" &
    pids+=("$!")
    index=$((index + 1))
  done
  local run_status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      run_status=1
    fi
  done
  if [[ "$run_status" != "0" ]]; then
    echo "[ERROR] at least one Stage 9.2 selection smoke run failed" >&2
    exit 1
  fi

  python3 scripts/analyze_kbs_stage9_acquired_loss_selection.py \
    --report "full=$smoke_root/full_seed42/alpha_0p50.json" \
    --report "ranking_only=$smoke_root/ranking_only_seed42/alpha_0p50.json" \
    --report "ce_margin=$smoke_root/ce_margin_seed42/alpha_0p50.json" \
    --report "ce_acquired=$smoke_root/ce_acquired_seed42/alpha_0p50.json" \
    --checkpoint "full=$(checkpoint_for full 42)" \
    --checkpoint "ranking_only=$(checkpoint_for ranking_only 42)" \
    --checkpoint "ce_margin=$(checkpoint_for ce_margin 42)" \
    --checkpoint "ce_acquired=$(checkpoint_for ce_acquired 42)" \
    --reference full \
    --expected-qids 20 \
    --n-bootstrap 200 \
    --seed 20260911 \
    --smoke \
    --output "$smoke_root/summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_2_SELECTION_SMOKE_OK"
  echo "No answer API call was made."
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
  summarize_training)
    run_readiness posttrain
    python3 scripts/summarize_kbs_stage9_acquired_loss_training.py \
      --output "$OUTPUT_ROOT/training_metrics_summary.json"
    echo "FINISHED_OK"
    echo "status=STAGE9_2_TRAINING_METRICS_OK"
    echo "No training, GPU inference, or API call was started."
    ;;
  selection_smoke)
    run_selection_smoke
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
    echo "Allowed: readiness, status, check_training, summarize_training, selection_smoke, train_{ranking_only,ce_margin,ce_acquired}_seed{42,43,44}" >&2
    exit 2
    ;;
esac
