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
GPU_LIST="${GPU_LIST:-2,3}"
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

training_status() {
  local seed="$1"
  local pid_file="$LOG_DIR/seed${seed}.pid"
  local output_dir="outputs/ranker/deberta_v3_large_v29_coverage_greedy_seed${seed}"
  local artifact
  local complete=1

  for artifact in best_model.pt best_val_metrics.json test_metrics.json train_history.json; do
    if [[ ! -s "$output_dir/$artifact" ]]; then
      complete=0
    fi
  done

  if [[ -s "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "seed${seed}: RUNNING pid=$pid"
      return
    fi
  fi

  if [[ "$complete" == "1" ]]; then
    echo "seed${seed}: FINISHED_OK"
  else
    echo "seed${seed}: FAILED_OR_INCOMPLETE"
    echo "  inspect: $LOG_DIR/seed${seed}_launcher.log"
  fi
}

run_selection_smoke() {
  if [[ "${KBS_STAGE9_SELECTION_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] selection smoke is locked pending post-training review" >&2
    exit 1
  fi
  run_readiness posttrain

  local data_root="data/hotpotqa_distractor_eval_3000_cand50"
  local output_root="$READINESS_DIR/selection_smoke20"
  local closure_checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
  local coverage_checkpoint="outputs/ranker/deberta_v3_large_v29_coverage_greedy/best_model.pt"
  local closure_dir="$output_root/closure_seed42"
  local coverage_dir="$output_root/coverage_seed42"
  local closure_report="$closure_dir/alpha_0p50.json"
  local coverage_report="$coverage_dir/alpha_0p50.json"

  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  if [[ "${#gpus[@]}" -lt 2 ]]; then
    echo "[ERROR] GPU_LIST must contain two GPUs, for example 2,3" >&2
    exit 1
  fi
  if [[ -e "$closure_report" || -e "$coverage_report" ]]; then
    echo "[ERROR] smoke report already exists; refusing silent reuse" >&2
    exit 1
  fi
  mkdir -p "$closure_dir" "$coverage_dir"

  run_smoke_method() {
    local checkpoint="$1"
    local output_dir="$2"
    local gpu="$3"
    DATA_ROOT="$data_root" \
    SPLIT=test \
    CHECKPOINT="$checkpoint" \
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

  echo "[START] Stage 9.1 paired selection smoke; answers disabled"
  run_smoke_method "$closure_checkpoint" "$closure_dir" "${gpus[0]}" &
  local closure_pid=$!
  run_smoke_method "$coverage_checkpoint" "$coverage_dir" "${gpus[1]}" &
  local coverage_pid=$!
  local status=0
  if ! wait "$closure_pid"; then
    status=1
  fi
  if ! wait "$coverage_pid"; then
    status=1
  fi
  if [[ "$status" != "0" ]]; then
    echo "[ERROR] at least one selection smoke run failed" >&2
    exit 1
  fi

  python3 scripts/analyze_kbs_stage9_teacher_objective_selection.py \
    --closure-report "$closure_report" \
    --coverage-report "$coverage_report" \
    --closure-checkpoint "$closure_checkpoint" \
    --coverage-checkpoint "$coverage_checkpoint" \
    --expected-qids 20 \
    --n-bootstrap 200 \
    --seed 20260910 \
    --smoke \
    --output "$output_root/summary.json"
  echo "FINISHED_OK"
  echo "status=SELECTION_SMOKE_OK"
}

run_full_selection() {
  if [[ "${KBS_STAGE9_FULL_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] full selection is locked pending smoke review" >&2
    exit 1
  fi
  run_readiness posttrain

  local data_root="data/hotpotqa_distractor_eval_3000_cand50"
  local output_root="$READINESS_DIR/selection3000"
  local -a seeds=(42 43 44)
  local -a methods=(closure coverage)
  local -a gpus
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  if [[ "${#gpus[@]}" -lt 6 ]]; then
    echo "[ERROR] GPU_LIST must contain six GPUs, for example 2,3,4,5,6,7" >&2
    exit 1
  fi

  checkpoint_for() {
    local method="$1"
    local seed="$2"
    local suffix=""
    if [[ "$seed" != "42" ]]; then
      suffix="_seed${seed}"
    fi
    if [[ "$method" == "closure" ]]; then
      echo "outputs/ranker/deberta_v3_large_v27_counterfactual_dual${suffix}/best_model.pt"
    else
      echo "outputs/ranker/deberta_v3_large_v29_coverage_greedy${suffix}/best_model.pt"
    fi
  }

  local method
  local seed
  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      local existing_report="$output_root/${method}_seed${seed}/alpha_0p50.json"
      if [[ -e "$existing_report" ]]; then
        echo "[ERROR] report already exists; refusing silent reuse: $existing_report" >&2
        exit 1
      fi
    done
  done

  run_full_method() {
    local method="$1"
    local seed="$2"
    local gpu="$3"
    local checkpoint
    checkpoint="$(checkpoint_for "$method" "$seed")"
    local output_dir="$output_root/${method}_seed${seed}"
    DATA_ROOT="$data_root" \
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
  }

  mkdir -p "$output_root"
  echo "[START] Stage 9.1 six-run paired selection evaluation; answers disabled"
  local -a pids=()
  local index=0
  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      run_full_method "$method" "$seed" "${gpus[$index]}" &
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
  if [[ "$status" != "0" ]]; then
    echo "[ERROR] at least one full selection run failed" >&2
    exit 1
  fi

  for seed in "${seeds[@]}"; do
    python3 scripts/analyze_kbs_stage9_teacher_objective_selection.py \
      --closure-report "$output_root/closure_seed${seed}/alpha_0p50.json" \
      --coverage-report "$output_root/coverage_seed${seed}/alpha_0p50.json" \
      --closure-checkpoint "$(checkpoint_for closure "$seed")" \
      --coverage-checkpoint "$(checkpoint_for coverage "$seed")" \
      --expected-qids 3000 \
      --n-bootstrap 10000 \
      --seed "$((20260910 + seed))" \
      --output "$output_root/seed${seed}_paired.json"
  done

  python3 scripts/summarize_kbs_stage9_teacher_objective_multiseed.py \
    --summary "42=$output_root/seed42_paired.json" \
    --summary "43=$output_root/seed43_paired.json" \
    --summary "44=$output_root/seed44_paired.json" \
    --expected-qids 3000 \
    --output "$output_root/multiseed_summary.json"
  echo "FINISHED_OK"
  echo "status=FULL_SELECTION_OK"
}

prepare_answer_caches() {
  if [[ "${KBS_STAGE9_CACHE_PREP_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] answer-cache preparation is locked pending selection review" >&2
    exit 1
  fi
  python3 scripts/prepare_kbs_stage9_teacher_objective_answer_caches.py \
    --selection-root "$READINESS_DIR/selection3000" \
    --cache-root outputs/rag/cache_kbs_stage9_teacher_objective \
    --output "$READINESS_DIR/answer_cache_readiness.json"
  echo "FINISHED_OK"
  echo "status=ANSWER_CACHE_READINESS_OK"
  echo "No API call was started."
}

run_answer_smoke() {
  if [[ "${KBS_STAGE9_ANSWER_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
    echo "[ERROR] answer smoke is locked pending cache-readiness review" >&2
    exit 1
  fi
  if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
    echo "[ERROR] export a non-empty DEEPSEEK_API_KEY" >&2
    exit 1
  fi
  export DEEPSEEK_MODEL=deepseek-v4-flash
  export DEEPSEEK_THINKING_MODE=disabled

  local readiness="$READINESS_DIR/answer_cache_readiness.json"
  local selection="$READINESS_DIR/selection3000/coverage_seed42/alpha_0p50.json"
  local checkpoint="outputs/ranker/deberta_v3_large_v29_coverage_greedy/best_model.pt"
  local cache_dir="outputs/rag/cache_kbs_stage9_teacher_objective/coverage_seed42"
  local output="outputs/rag/kbs_stage9_teacher_objective/coverage_seed42_smoke20.json"
  local audit="$READINESS_DIR/coverage_seed42_answer_smoke20.json"
  for path in "$readiness" "$selection" "$checkpoint" "$cache_dir"; do
    if [[ ! -e "$path" ]]; then
      echo "[ERROR] missing answer-smoke prerequisite: $path" >&2
      exit 1
    fi
  done
  python3 - "$readiness" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "OK" or report.get("failures"):
    raise SystemExit(f"answer cache readiness is not a clean OK: {path}")
PY
  if [[ -e "$output" || -e "$audit" ]]; then
    echo "[ERROR] answer-smoke output already exists; refusing overwrite" >&2
    exit 1
  fi

  echo "[INFO] checking frozen DeepSeek endpoint before the bounded smoke"
  python3 scripts/check_deepseek_api.py
  echo "[START] Stage 9.1 Coverage seed-42 answer smoke; max_qids=20"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
    --samples data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl \
    --memory data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl \
    --queries data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl \
    --checkpoint "$checkpoint" \
    --model-dir models/deberta-v3-large \
    --state-mode policy \
    --policy-context-source online_state \
    --selector hybrid_policy \
    --dense-model models/bge-large-en-v1.5 \
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
    --answer-mode json \
    --generate-answers \
    --answer-cache-dir "$cache_dir" \
    --max-qids 20 \
    --ks 1,2,3,5 \
    --save-online-states \
    --profile-runtime \
    --profile-warmup-qids 20 \
    --llm-max-retries 8 \
    --llm-retry-sleep 2.0 \
    --seed 20260608 \
    --output "$output"

  python3 scripts/check_kbs_stage9_teacher_objective_answer_report.py \
    --seed 42 \
    --report "$output" \
    --selection-report "$selection" \
    --cache-dir "$cache_dir" \
    --expected-qids 20 \
    --smoke \
    --output "$audit"
  echo "FINISHED_OK"
  echo "status=ANSWER_SMOKE_OK"
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
  status)
    training_status 43
    training_status 44
    echo "Status check completed; no training or API call was started."
    ;;
  smoke_selection)
    run_selection_smoke
    ;;
  full_selection)
    run_full_selection
    ;;
  prepare_answer_caches)
    prepare_answer_caches
    ;;
  answer_smoke)
    run_answer_smoke
    ;;
  *)
    echo "[ERROR] ACTION must be readiness, train_seed43, train_seed44, check_training, status, smoke_selection, full_selection, prepare_answer_caches, or answer_smoke" >&2
    exit 2
    ;;
esac
