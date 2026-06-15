#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_eval_3000_cand50}"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/frozen_deberta_v3_large_v7_main_val08252/best_model.pt}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"
CUDA_DEVICE="${CUDA_DEVICE:-5}"
MAX_QIDS="${MAX_QIDS:-0}"
SELECT_TOP_K="${SELECT_TOP_K:-5}"
HYBRID_ALPHA="${HYBRID_ALPHA:-0.5}"
LOCAL_EXPANSION_WINDOW="${LOCAL_EXPANSION_WINDOW:-1}"
ANSWER_MODE="${ANSWER_MODE:-json}"
LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-10}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
  echo
  export DEEPSEEK_API_KEY
fi

python3 scripts/validate_hotpotqa_policy_rag_eval.py --data-root "$DATA_ROOT"

for CAND_K in 10 15 20; do
  SUFFIX="hybrid_policy_cand${CAND_K}_top${SELECT_TOP_K}_local${LOCAL_EXPANSION_WINDOW}"
  EXTRA_ARGS=()
  if [[ "$MAX_QIDS" != "0" ]]; then
    SUFFIX="${SUFFIX}_q${MAX_QIDS}"
    EXTRA_ARGS+=(--max-qids "$MAX_QIDS")
  fi

  echo "===== Running ${SUFFIX} ====="
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
    --samples "$DATA_ROOT/samples/test.jsonl" \
    --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl" \
    --queries "$DATA_ROOT/queries/test.jsonl" \
    --checkpoint "$CHECKPOINT" \
    --model-dir "$MODEL_DIR" \
    --state-mode policy \
    --selector hybrid_policy \
    --dense-model "$DENSE_MODEL" \
    --dense-query-mode question \
    --hybrid-alpha "$HYBRID_ALPHA" \
    --candidate-top-k "$CAND_K" \
    --local-expansion-window "$LOCAL_EXPANSION_WINDOW" \
    --select-top-k "$SELECT_TOP_K" \
    --answer-mode "$ANSWER_MODE" \
    --ks 1,2,3,5 \
    --generate-answers \
    "${EXTRA_ARGS[@]}" \
    --llm-max-retries "$LLM_MAX_RETRIES" \
    --answer-cache-dir "outputs/rag/cache_eval3000_cand50_${SUFFIX}" \
    --output "outputs/rag/eval3000_cand50_${SUFFIX}.json"
done

python3 - <<'PY'
import json
from pathlib import Path

rows = []
for path in sorted(Path("outputs/rag").glob("eval3000_cand50_hybrid_policy_cand*_top*_local*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data.get("summary", data)
    rows.append(
        {
            "file": str(path),
            "qids": s.get("qids"),
            "candidate_top_k": s.get("candidate_top_k"),
            "local_expansion_window": s.get("local_expansion_window"),
            "answer_em": s.get("answer_em"),
            "answer_f1": s.get("answer_f1"),
            "answer_contains": s.get("answer_contains"),
            "full_gold_doc_coverage": s.get("full_gold_doc_coverage"),
            "full_gold_unit_coverage": s.get("full_gold_unit_coverage"),
            "step_selected_contains_gold": s.get("step_selected_contains_gold"),
            "step_acc@1": s.get("step_acc@1"),
            "step_acc@5": s.get("step_acc@5"),
        }
    )

print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
