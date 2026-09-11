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

selection_report_is_valid() {
  local report="$1"
  local checkpoint="$2"
  local expected_qids="$3"
  python3 - "$report" "$checkpoint" "$expected_qids" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
checkpoint = sys.argv[2]
expected_qids = int(sys.argv[3])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
summary = report.get("summary")
results = report.get("results")
expected = {
    "checkpoint": checkpoint,
    "policy_context_source": "online_state",
    "selector": "hybrid_policy",
    "front_pool_k": 30,
    "front_fusion": "rrf",
    "local_expansion_window": 1,
    "candidate_top_k": 10,
    "select_top_k": 5,
    "state_update_top_k": 1,
    "policy_score_mode": "front_policy_blend",
    "policy_blend_weight": 0.5,
    "generate_answers": False,
    "answer_judged": 0,
    "answer_errors": 0,
    "qids": expected_qids,
}
if not isinstance(summary, dict) or not isinstance(results, list):
    raise SystemExit(1)
if any(summary.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
qids = [str(row.get("qid") or "") for row in results]
raise SystemExit(0 if len(qids) == expected_qids and len(set(qids)) == expected_qids else 1)
PY
}

full_selection_reference() {
  local seed="$1"
  echo "outputs/analysis/kbs_stage9_teacher_objective/selection3000/closure_seed${seed}/alpha_0p50.json"
}

run_full_selection_method() {
  local variant="$1"
  local seed="$2"
  local gpu="$3"
  local output_dir="$OUTPUT_ROOT/selection3000/${variant}_seed${seed}"
  local report="$output_dir/alpha_0p50.json"
  local checkpoint
  checkpoint="$(checkpoint_for "$variant" "$seed")"
  if [[ -e "$report" ]]; then
    if selection_report_is_valid "$report" "$checkpoint" 3000; then
      echo "[SKIP] clean existing selection report: $report"
      return
    fi
    echo "[ERROR] invalid existing report; refusing overwrite: $report" >&2
    return 1
  fi
  mkdir -p "$output_dir"
  DATA_ROOT=data/hotpotqa_distractor_eval_3000_cand50 \
  SPLIT=test \
  CHECKPOINT="$checkpoint" \
  OUTPUT_DIR="$output_dir" \
  ALPHAS="0.5" \
  GPU_LIST="$gpu" \
  MAX_QIDS=3000 \
  GENERATE_ANSWERS=0 \
  POLICY_CONTEXT_SOURCE=online_state \
  STATE_UPDATE_TOP_K=1 \
  SAVE_ONLINE_STATES=0 \
  bash scripts/run_kbs_alpha_sensitivity_val.sh
  selection_report_is_valid "$report" "$checkpoint" 3000
}

run_full_selection_wave() {
  local -a variants=(ranking_only ce_margin ce_acquired)
  local -a seeds=("$@")
  local -a gpus
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  local required=$(( ${#variants[@]} * ${#seeds[@]} ))
  if [[ "${#gpus[@]}" -lt "$required" ]]; then
    echo "[ERROR] wave requires $required GPUs; GPU_LIST has ${#gpus[@]}" >&2
    return 1
  fi
  local -a pids=()
  local index=0
  local seed variant
  for seed in "${seeds[@]}"; do
    for variant in "${variants[@]}"; do
      run_full_selection_method "$variant" "$seed" "${gpus[$index]}" &
      pids+=("$!")
      index=$((index + 1))
    done
  done
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

run_full_selection() {
  if [[ "${KBS_STAGE9_ACQUIRED_FULL_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.2 full selection is locked pending smoke review" >&2
    exit 1
  fi
  run_readiness posttrain
  local smoke="$OUTPUT_ROOT/selection_smoke20/summary.json"
  python3 - "$smoke" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if (
    report.get("status") != "SMOKE_OK"
    or report.get("mode") != "acquired_loss_selection_smoke"
    or int(report.get("qids") or 0) != 20
    or report.get("failures")
):
    raise SystemExit(f"selection smoke is not a clean SMOKE_OK: {path}")
PY

  local seed
  for seed in 42 43 44; do
    local reference
    reference="$(full_selection_reference "$seed")"
    if ! selection_report_is_valid "$reference" "$(checkpoint_for full "$seed")" 3000; then
      echo "[ERROR] missing or invalid Full reference selection: $reference" >&2
      exit 1
    fi
  done

  echo "[START] Stage 9.2 selection wave 1: seeds 42 and 43; answers disabled"
  run_full_selection_wave 42 43
  echo "[START] Stage 9.2 selection wave 2: seed 44; answers disabled"
  run_full_selection_wave 44

  local root="$OUTPUT_ROOT/selection3000"
  for seed in 42 43 44; do
    python3 scripts/analyze_kbs_stage9_acquired_loss_selection.py \
      --report "full=$(full_selection_reference "$seed")" \
      --report "ranking_only=$root/ranking_only_seed${seed}/alpha_0p50.json" \
      --report "ce_margin=$root/ce_margin_seed${seed}/alpha_0p50.json" \
      --report "ce_acquired=$root/ce_acquired_seed${seed}/alpha_0p50.json" \
      --checkpoint "full=$(checkpoint_for full "$seed")" \
      --checkpoint "ranking_only=$(checkpoint_for ranking_only "$seed")" \
      --checkpoint "ce_margin=$(checkpoint_for ce_margin "$seed")" \
      --checkpoint "ce_acquired=$(checkpoint_for ce_acquired "$seed")" \
      --reference full \
      --expected-qids 3000 \
      --n-bootstrap 10000 \
      --seed "$((20260911 + seed))" \
      --output "$root/seed${seed}_paired.json"
  done
  python3 scripts/summarize_kbs_stage9_acquired_loss_selection.py \
    --summary "42=$root/seed42_paired.json" \
    --summary "43=$root/seed43_paired.json" \
    --summary "44=$root/seed44_paired.json" \
    --expected-qids 3000 \
    --output "$root/multiseed_summary.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_2_FULL_SELECTION_OK"
  echo "summary=$root/multiseed_summary.json"
  echo "No answer API call was made."
}

start_full_selection() {
  if [[ "${KBS_STAGE9_ACQUIRED_FULL_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.2 full selection is locked pending smoke review" >&2
    exit 1
  fi
  local pid_file="$LOG_ROOT/selection_full.pid"
  local log_file="$LOG_ROOT/selection_full_launcher.log"
  if [[ -s "$pid_file" ]]; then
    local existing_pid
    existing_pid="$(cat "$pid_file")"
    if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "[ERROR] full selection is already running pid=$existing_pid" >&2
      exit 1
    fi
  fi
  nohup env \
    KBS_STAGE9_ACQUIRED_FULL_SELECTION_AUTHORIZED=1 \
    ACTION=selection_full_worker \
    GPU_LIST="$GPU_LIST" \
    bash "$SCRIPT_DIR/run_kbs_stage9_acquired_loss.sh" \
    > "$log_file" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$pid_file"
  echo "RUNNING pid=$pid"
  echo "log=$log_file"
  echo "Check with: ACTION=selection_status bash scripts/run_kbs_stage9_acquired_loss.sh"
}

full_selection_status() {
  local summary="$OUTPUT_ROOT/selection3000/multiseed_summary.json"
  local pid_file="$LOG_ROOT/selection_full.pid"
  local completed=0
  local variant seed report
  for seed in 42 43 44; do
    for variant in ranking_only ce_margin ce_acquired; do
      report="$OUTPUT_ROOT/selection3000/${variant}_seed${seed}/alpha_0p50.json"
      if selection_report_is_valid "$report" "$(checkpoint_for "$variant" "$seed")" 3000; then
        completed=$((completed + 1))
      fi
    done
  done
  if [[ -s "$summary" ]] && python3 - "$summary" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if report.get("status") == "OK" and not report.get("failures") else 1)
PY
  then
    echo "FINISHED_OK reports=$completed/9"
    echo "summary=$summary"
    return
  fi
  if [[ -s "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "RUNNING pid=$pid reports=$completed/9"
      return
    fi
  fi
  echo "FAILED_OR_INCOMPLETE reports=$completed/9"
  echo "inspect=$LOG_ROOT/selection_full_launcher.log"
  return 1
}

prepare_primary_answer_caches() {
  if [[ "${KBS_STAGE9_ACQUIRED_CACHE_PREP_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] Stage 9.2 cache preparation is locked pending selection review" >&2
    exit 1
  fi
  python3 scripts/prepare_kbs_stage9_acquired_loss_answer_caches.py \
    --selection-root "$OUTPUT_ROOT/selection3000" \
    --cache-root outputs/rag/cache_kbs_stage9_acquired_loss \
    --output "$OUTPUT_ROOT/answer_cache_readiness.json"
  echo "FINISHED_OK"
  echo "status=STAGE9_2_PRIMARY_ANSWER_CACHE_READINESS_OK"
  echo "No training, GPU inference, or API call was started."
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
  selection_full_start)
    start_full_selection
    ;;
  selection_full_worker)
    run_full_selection
    ;;
  selection_status)
    full_selection_status
    ;;
  prepare_primary_answer_caches)
    prepare_primary_answer_caches
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
    echo "Allowed: readiness, status, check_training, summarize_training, selection_smoke, selection_full_start, selection_full_worker, selection_status, prepare_primary_answer_caches, train_{ranking_only,ce_margin,ce_acquired}_seed{42,43,44}" >&2
    exit 2
    ;;
esac
