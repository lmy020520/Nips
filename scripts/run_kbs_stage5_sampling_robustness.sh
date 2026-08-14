#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_validation_full_cand50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/kbs_stage5_sampling_robustness}"
BUILD_DATA="${BUILD_DATA:-0}"
CHECK_ONLY="${CHECK_ONLY:-1}"
RUN="${RUN:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"

if [[ "$BUILD_DATA" == "1" ]]; then
  if [[ -f "$DATA_ROOT/manifest.json" ]]; then
    echo "[SKIP] full validation data exists: $DATA_ROOT"
  else
    python3 scripts/prepare_hotpotqa_policy_rag_eval.py \
      --output-root "$DATA_ROOT" \
      --source-split validation \
      --output-split val \
      --size 0 \
      --max-candidates 50 \
      --seed 20260814
  fi
fi

mkdir -p "$OUTPUT_ROOT"
if [[ "$CHECK_ONLY" == "1" ]]; then
  python3 scripts/check_kbs_stage5_sampling_data.py \
    --data-root "$DATA_ROOT" \
    --output "$OUTPUT_ROOT/readiness.json"
  echo "[OK] Stage 5 sampling-data readiness passed; no model diagnostic was started"
  exit 0
fi

python3 - "$OUTPUT_ROOT/readiness.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"[ERROR] missing reviewed readiness report: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("status") != "OK" or report.get("failures"):
    raise SystemExit(f"[ERROR] readiness report is not a clean OK: {path}")
if int(report.get("queries", 0)) != 6903:
    raise SystemExit(f"[ERROR] readiness qids changed: {report.get('queries')}")
print(f"[OK] reusing reviewed readiness report: {path}")
PY

case "$RUN" in
  seed42)
    seed=42
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
    ;;
  seed43)
    seed=43
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt"
    ;;
  seed44)
    seed=44
    checkpoint="outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt"
    ;;
  *)
    echo "[ERROR] RUN must be seed42, seed43, or seed44 when CHECK_ONLY=0" >&2
    exit 2
    ;;
esac

expected_qids="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["size"])' "$DATA_ROOT/manifest.json")"
output_dir="$OUTPUT_ROOT/$RUN"
if [[ -e "$output_dir/summary.json" ]]; then
  echo "[ERROR] refusing to overwrite existing report: $output_dir/summary.json" >&2
  exit 1
fi
mkdir -p "$output_dir"

echo "[PLAN] Stage 5 full-validation fixed-pool sampling robustness"
echo "[INFO] run=$RUN qids=$expected_qids gpu=$CUDA_DEVICE api_calls=0"
SAMPLES="$DATA_ROOT/samples/val.jsonl" \
MEMORY="$DATA_ROOT/unit_registry/raw_units_val.jsonl" \
QUERIES="$DATA_ROOT/queries/val.jsonl" \
CHECKPOINT="$checkpoint" \
CUDA_DEVICE="$CUDA_DEVICE" \
MAX_QIDS=0 \
N_BOOTSTRAP="$N_BOOTSTRAP" \
OUTPUT_DIR="$output_dir" \
bash scripts/run_kbs_state_phase1.sh

python3 scripts/check_kbs_v27_stage5_rank_reversal.py \
  --report "$output_dir/summary.json" \
  --expected-seed "$seed" \
  --expected-qids "$expected_qids" \
  --expected-states 0 \
  --output "$output_dir/report_audit.json"

echo "[DONE] Stage 5 sampling robustness run=$RUN"
