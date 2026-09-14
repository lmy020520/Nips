#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

ACTION="${ACTION:-readiness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_stage9_breadth}"
MUSIQUE_PATH="${MUSIQUE_PATH:-}"
TARGET_QIDS="${TARGET_QIDS:-1000}"
SEED="${SEED:-20260914}"
DATA_ROOT="${DATA_ROOT:-data/musique_ans_eval_1000_paragraph20}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CHECKPOINT="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"

for plan in \
  md/kbs_three_review_execution_plan.md \
  md/kbs_review_75_85_execution_plan.md; do
  if [[ ! -f "$plan" ]]; then
    echo "[ERROR] missing experiment plan: $plan" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"

ready_musique_source() {
  python3 - "$OUTPUT_ROOT/readiness.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing readiness report: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "READY_MUSIQUE" or report.get("readiness_failures"):
    raise SystemExit(f"readiness is not a clean READY_MUSIQUE: {path}")
source = ((report.get("schema_audit") or {}).get("path") or "").strip()
if not source or not Path(source).is_file():
    raise SystemExit(f"audited MuSiQue source is missing: {source!r}")
print(source)
PY
}

audit_adapter() {
  local source_path="$1"
  python3 scripts/check_kbs_stage9_musique_adapter.py \
    --source-json "$source_path" \
    --data-root "$DATA_ROOT" \
    --size "$TARGET_QIDS" \
    --seed "$SEED" \
    --output "$OUTPUT_ROOT/adapter_readiness.json"
}

adapter_is_valid() {
  python3 - "$OUTPUT_ROOT/adapter_readiness.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing adapter audit: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "OK" or report.get("failure_count") != 0:
    raise SystemExit(f"adapter audit is not a clean OK: {path}")
PY
}

case "$ACTION" in
  readiness)
    command=(
      python3 scripts/check_kbs_stage9_breadth_readiness.py
      --target-qids "$TARGET_QIDS"
      --seed "$SEED"
      --output "$OUTPUT_ROOT/readiness.json"
    )
    if [[ -n "$MUSIQUE_PATH" ]]; then
      command+=(--musique-path "$MUSIQUE_PATH")
    fi
    "${command[@]}"
    echo "AUDIT_COMPLETE"
    echo "report=$OUTPUT_ROOT/readiness.json"
    echo "No download, training, GPU inference, or API call was started."
    ;;
  build_adapter)
    if [[ "${KBS_STAGE9_MUSIQUE_ADAPTER_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] MuSiQue adapter build is locked pending readiness review" >&2
      exit 1
    fi
    source_path="$(ready_musique_source)"
    python3 scripts/prepare_musique_policy_rag_eval.py \
      --source-json "$source_path" \
      --output-root "$DATA_ROOT" \
      --size "$TARGET_QIDS" \
      --seed "$SEED"
    audit_adapter "$source_path"
    echo "FINISHED_OK"
    echo "status=STAGE9_6_MUSIQUE_ADAPTER_OK"
    echo "report=$OUTPUT_ROOT/adapter_readiness.json"
    echo "No training, GPU inference, or API call was started."
    ;;
  audit_adapter)
    source_path="$(ready_musique_source)"
    audit_adapter "$source_path"
    echo "AUDIT_COMPLETE"
    echo "report=$OUTPUT_ROOT/adapter_readiness.json"
    echo "No training, GPU inference, or API call was started."
    ;;
  selection_smoke)
    if [[ "${KBS_STAGE9_MUSIQUE_SELECTION_SMOKE_AUTHORIZED:-0}" != "1" ]]; then
      echo "[ERROR] MuSiQue selection smoke is locked pending adapter review" >&2
      exit 1
    fi
    adapter_is_valid
    smoke_dir="$OUTPUT_ROOT/selection_smoke20"
    report="$smoke_dir/compact_seed42.json"
    summary="$smoke_dir/summary.json"
    if [[ -e "$report" || -e "$summary" ]]; then
      echo "[ERROR] refusing to overwrite existing MuSiQue smoke outputs: $smoke_dir" >&2
      exit 1
    fi
    mkdir -p "$smoke_dir"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
      --samples "$DATA_ROOT/samples/test.jsonl" \
      --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl" \
      --queries "$DATA_ROOT/queries/test.jsonl" \
      --checkpoint "$CHECKPOINT" \
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
      --max-qids 20 \
      --ks 1,2,3,5 \
      --seed 20260608 \
      --save-online-states \
      --device cuda \
      --output "$report"
    python3 scripts/check_kbs_stage9_musique_selection.py \
      --report "$report" \
      --data-root "$DATA_ROOT" \
      --expected-qids 20 \
      --adapter-audit "$OUTPUT_ROOT/adapter_readiness.json" \
      --output "$summary"
    echo "FINISHED_OK"
    echo "status=STAGE9_6_MUSIQUE_SELECTION_SMOKE_OK"
    echo "report=$summary"
    echo "No answer API call was started."
    ;;
  *)
    echo "[ERROR] unsupported ACTION=$ACTION" >&2
    echo "Allowed: readiness, build_adapter, audit_adapter, selection_smoke" >&2
    exit 2
    ;;
esac
