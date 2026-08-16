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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/kbs_stage6_cost_frontier}"
CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
DATA_ROOT="data/hotpotqa_distractor_eval_3000_cand50"
CACHE_ROOT="${CACHE_ROOT:-outputs/rag/cache_kbs_stage6_cost_frontier}"
ANSWER_OUTPUT_ROOT="${ANSWER_OUTPUT_ROOT:-outputs/rag/kbs_stage6_cost_frontier}"
ANSWER_AUDIT_ROOT="${ANSWER_AUDIT_ROOT:-outputs/analysis/kbs_stage6_cost_frontier_answers}"
CACHE_PREP_AUDIT="${CACHE_PREP_AUDIT:-outputs/analysis/kbs_stage6_cost_frontier_cacheprep/cache_reuse_readiness.json}"
PROFILE_ROOT="${PROFILE_ROOT:-outputs/analysis/kbs_stage6_cost_frontier_profile500}"
FINAL_ROOT="${FINAL_ROOT:-outputs/analysis/kbs_stage6_cost_frontier_final}"

mkdir -p "$OUTPUT_DIR"

run_readiness() {
  python3 scripts/check_kbs_stage6_cost_frontier.py \
    --output "$OUTPUT_DIR/readiness.json"
}

run_selection() {
  local budget="$1"
  local max_qids="$2"
  local output="$3"
  local warmup_qids="$4"
  local front_pool=30
  if [[ "$budget" == "50" ]]; then
    front_pool=50
  fi
  if [[ -f "$output" ]]; then
    echo "[SKIP] report exists: $output"
    return
  fi
  echo "[START] Stage 6.3 selection-only cand$budget qids=$max_qids gpu=$CUDA_DEVICE"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
    --samples "$DATA_ROOT/samples/test.jsonl" \
    --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl" \
    --queries "$DATA_ROOT/queries/test.jsonl" \
    --checkpoint "$CHECKPOINT" \
    --model-dir models/deberta-v3-large \
    --state-mode policy \
    --policy-context-source online_state \
    --selector hybrid_policy \
    --dense-model models/bge-large-en-v1.5 \
    --dense-query-mode state \
    --hybrid-alpha 0.5 \
    --front-pool-k "$front_pool" \
    --front-fusion rrf \
    --local-expansion-window 1 \
    --mmr-lambda 0.7 \
    --mmr-same-doc-similarity 0.35 \
    --candidate-top-k "$budget" \
    --select-top-k 5 \
    --state-update-top-k 1 \
    --policy-score-mode front_policy_blend \
    --policy-blend-weight 0.5 \
    --max-qids "$max_qids" \
    --ks 1,2,3,5 \
    --save-online-states \
    --profile-runtime \
    --profile-warmup-qids "$warmup_qids" \
    --seed 20260608 \
    --output "$output"
  echo "[DONE] $output"
}

run_answers() {
  local budget="$1"
  local selection_report="outputs/analysis/kbs_stage6_cost_frontier_selection3000/cand${budget}_selection3000.json"
  local report="$ANSWER_OUTPUT_ROOT/cand${budget}_answers3000.json"
  local cache_dir="$CACHE_ROOT/cand${budget}"
  local audit="$ANSWER_AUDIT_ROOT/cand${budget}_audit.json"
  if [[ ! -f "$report" ]]; then
    echo "[START] Stage 6.3 cand$budget answers; exact caches are reused, missing qids call the API"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
      --samples "$DATA_ROOT/samples/test.jsonl" \
      --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl" \
      --queries "$DATA_ROOT/queries/test.jsonl" \
      --checkpoint "$CHECKPOINT" \
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
      --candidate-top-k "$budget" \
      --select-top-k 5 \
      --state-update-top-k 1 \
      --policy-score-mode front_policy_blend \
      --policy-blend-weight 0.5 \
      --answer-mode json \
      --generate-answers \
      --answer-cache-dir "$cache_dir" \
      --max-qids 3000 \
      --ks 1,2,3,5 \
      --save-online-states \
      --profile-runtime \
      --profile-warmup-qids 20 \
      --llm-max-retries 8 \
      --llm-retry-sleep 2.0 \
      --seed 20260608 \
      --output "$report"
  else
    echo "[SKIP] answer report exists; validating it: $report"
  fi
  python3 scripts/check_kbs_stage6_cost_answer_report.py \
    --budget "$budget" \
    --report "$report" \
    --selection-report "$selection_report" \
    --cache-dir "$cache_dir" \
    --output "$audit"
  echo "[PASS] Stage 6.3 cand$budget answer report"
}

case "$ACTION" in
  readiness)
    run_readiness
    echo "[OK] Stage 6.3 readiness finished; GPU/API work was not started"
    ;;
  smoke20)
    run_readiness
    run_selection 15 20 "$OUTPUT_DIR/cand15_smoke20.json" 5
    run_selection 20 20 "$OUTPUT_DIR/cand20_smoke20.json" 5
    echo "[DONE] Stage 6.3 cand15/cand20 selection-only smoke"
    ;;
  selection3000)
    if [[ "${KBS_STAGE6_COST_SELECTION_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] full selection-only run is locked by the execution plan" >&2
      exit 1
    fi
    run_readiness
    run_selection 15 3000 "$OUTPUT_DIR/cand15_selection3000.json" 20
    run_selection 20 3000 "$OUTPUT_DIR/cand20_selection3000.json" 20
    echo "[DONE] Stage 6.3 cand15/cand20 full selection-only reports"
    ;;
  profile500)
    if [[ "${KBS_STAGE6_COST_PROFILE_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] unified four-budget runtime profile is locked by the execution plan" >&2
      exit 1
    fi
    run_readiness
    run_selection 10 500 "$OUTPUT_DIR/cand10_profile500.json" 20
    run_selection 15 500 "$OUTPUT_DIR/cand15_profile500.json" 20
    run_selection 20 500 "$OUTPUT_DIR/cand20_profile500.json" 20
    run_selection 50 500 "$OUTPUT_DIR/cand50_profile500.json" 20
    echo "[DONE] Stage 6.3 unified four-budget runtime profile"
    ;;
  cache-prep)
    if [[ "${KBS_STAGE6_CACHE_PREP_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] exact-context answer-cache preparation is locked by the execution plan" >&2
      exit 1
    fi
    python3 scripts/prepare_kbs_stage6_cost_answer_caches.py \
      --output "$OUTPUT_DIR/cache_reuse_readiness.json"
    echo "[DONE] Stage 6.3 exact-context answer-cache preparation; no API was called"
    ;;
  answers3000)
    if [[ "${KBS_STAGE6_ANSWERS_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] Stage 6.3 answer generation is locked by the execution plan" >&2
      exit 1
    fi
    if [[ -z "${DEEPSEEK_API_KEY:-}" || -z "${DEEPSEEK_API_KEY//[[:space:]]/}" ]]; then
      echo "[ERROR] export a non-empty DEEPSEEK_API_KEY before launching" >&2
      exit 1
    fi
    export DEEPSEEK_MODEL="deepseek-v4-flash"
    export DEEPSEEK_THINKING_MODE="disabled"
    python3 - "$CACHE_PREP_AUDIT" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
obj = json.loads(path.read_text(encoding="utf-8"))
if obj.get("status") != "OK" or obj.get("total_new_api_answers_required") != 3615:
    raise SystemExit(f"cache-preparation audit is not accepted: {path}")
for budget, reused in (("15", 1259), ("20", 1126)):
    if (obj.get("budgets") or {}).get(budget, {}).get("reused_exact_context") != reused:
        raise SystemExit(f"cache-preparation count mismatch for cand{budget}")
print(f"[OK] accepted cache-preparation audit: {path}")
PY
    mkdir -p "$ANSWER_OUTPUT_ROOT" "$ANSWER_AUDIT_ROOT"
    echo "[INFO] one API preflight, then 3,615 expected fresh answer calls across both budgets"
    python3 scripts/check_deepseek_api.py
    run_answers 15
    run_answers 20
    echo "[DONE] Stage 6.3 cand15/cand20 answer reports passed all audits"
    ;;
  finalize)
    if [[ "${KBS_STAGE6_FINALIZE_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] Stage 6.3 final offline aggregation is locked by the execution plan" >&2
      exit 1
    fi
    mkdir -p "$FINAL_ROOT"
    python3 scripts/evaluate_kbs_standard_metrics.py \
      --report cand10=outputs/rag/kbs_v27_final_hotpot/full_compact.json \
      --report cand15="$ANSWER_OUTPUT_ROOT/cand15_answers3000.json" \
      --report cand20="$ANSWER_OUTPUT_ROOT/cand20_answers3000.json" \
      --report cand50=outputs/rag/kbs_v27_stage5_multiseed/seed42/full_recall.json \
      --report Gold-Oracle=outputs/rag/full3000_gold_oracle.json \
      --gold-oracle-name Gold-Oracle \
      --expected-qids 3000 \
      --closure-unit-budgets 5,10,15,20,50 \
      --output "$FINAL_ROOT/standard_metrics.json" \
      --records-output "$FINAL_ROOT/standard_metric_records.jsonl"
    python3 scripts/summarize_kbs_stage6_cost_frontier.py \
      --report 10=outputs/rag/kbs_v27_final_hotpot/full_compact.json \
      --report 15="$ANSWER_OUTPUT_ROOT/cand15_answers3000.json" \
      --report 20="$ANSWER_OUTPUT_ROOT/cand20_answers3000.json" \
      --report 50=outputs/rag/kbs_v27_stage5_multiseed/seed42/full_recall.json \
      --profile-report 10="$PROFILE_ROOT/cand10_profile500.json" \
      --profile-report 15="$PROFILE_ROOT/cand15_profile500.json" \
      --profile-report 20="$PROFILE_ROOT/cand20_profile500.json" \
      --profile-report 50="$PROFILE_ROOT/cand50_profile500.json" \
      --standard-summary "$FINAL_ROOT/standard_metrics.json" \
      --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl" \
      --output "$FINAL_ROOT/summary.json" \
      --tsv-output "$FINAL_ROOT/frontier.tsv"
    echo "[DONE] Stage 6.3 final offline cost-closure frontier passed"
    ;;
  *)
    echo "[ERROR] ACTION must be readiness, smoke20, selection3000, profile500, cache-prep, answers3000, or finalize" >&2
    exit 2
    ;;
esac
