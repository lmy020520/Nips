#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/2wiki_multihopqa_eval_1000_cand50}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rag/2wiki_generalization_1000}"
CACHE_ROOT="${CACHE_ROOT:-outputs/rag/cache_2wiki_generalization_1000}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v17_candidate_contribution_lr5e7/best_model.pt}"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"
BGE_RERANKER_MODEL="${BGE_RERANKER_MODEL:-models/bge-reranker-large}"
CUDA_DEVICE="${CUDA_DEVICE:-5}"
MAX_QIDS="${MAX_QIDS:-1000}"
SELECT_TOP_K="${SELECT_TOP_K:-5}"
FRONT_POOL_K="${FRONT_POOL_K:-30}"
COMPACT_CANDIDATE_TOP_K="${COMPACT_CANDIDATE_TOP_K:-10}"
RECALL_CANDIDATE_TOP_K="${RECALL_CANDIDATE_TOP_K:-50}"
LOCAL_EXPANSION_WINDOW="${LOCAL_EXPANSION_WINDOW:-1}"
MMR_LAMBDA="${MMR_LAMBDA:-0.7}"
MMR_SAME_DOC_SIMILARITY="${MMR_SAME_DOC_SIMILARITY:-0.35}"
POLICY_BLEND_WEIGHT="${POLICY_BLEND_WEIGHT:-0.35}"
REFRESH_ANSWER_CACHE="${REFRESH_ANSWER_CACHE:-1}"

mkdir -p "$OUTPUT_DIR" "$CACHE_ROOT"
export OUTPUT_DIR

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  read -r -s -p "DeepSeek API Key: " DEEPSEEK_API_KEY
  echo
  export DEEPSEEK_API_KEY
fi

refresh_flag=()
if [[ "$REFRESH_ANSWER_CACHE" == "1" ]]; then
  refresh_flag=(--refresh-answer-cache)
fi

common_args=(
  --samples "$DATA_ROOT/samples/test.jsonl"
  --memory "$DATA_ROOT/unit_registry/raw_units_test.jsonl"
  --queries "$DATA_ROOT/queries/test.jsonl"
  --checkpoint "$CHECKPOINT"
  --model-dir "$MODEL_DIR"
  --state-mode policy
  --policy-context-source online_state
  --select-top-k "$SELECT_TOP_K"
  --answer-mode json
  --generate-answers
  --max-qids "$MAX_QIDS"
)

run_method() {
  local name="$1"
  shift
  echo "===== Running 2Wiki generalization: ${name} ====="
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
    "${common_args[@]}" \
    "${refresh_flag[@]}" \
    --answer-cache-dir "$CACHE_ROOT/$name" \
    --output "$OUTPUT_DIR/$name.json" \
    "$@"
}

run_method bm25_rag \
  --selector bm25

run_method dense_rag \
  --selector dense \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state

run_method hybrid_rag \
  --selector hybrid \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state \
  --hybrid-alpha 0.5

run_method bge_reranker_rag \
  --selector generic_reranker \
  --reranker-model "$BGE_RERANKER_MODEL" \
  --reranker-batch-size 8

run_method ksg_ea_compact \
  --selector hybrid_policy \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state \
  --hybrid-alpha 0.5 \
  --front-pool-k "$FRONT_POOL_K" \
  --front-fusion rrf \
  --local-expansion-window "$LOCAL_EXPANSION_WINDOW" \
  --mmr-lambda "$MMR_LAMBDA" \
  --mmr-same-doc-similarity "$MMR_SAME_DOC_SIMILARITY" \
  --candidate-top-k "$COMPACT_CANDIDATE_TOP_K" \
  --policy-score-mode front_policy_blend \
  --policy-blend-weight "$POLICY_BLEND_WEIGHT"

run_method ksg_ea_recall \
  --selector hybrid_policy \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state \
  --hybrid-alpha 0.5 \
  --front-pool-k "$FRONT_POOL_K" \
  --front-fusion rrf \
  --local-expansion-window "$LOCAL_EXPANSION_WINDOW" \
  --mmr-lambda "$MMR_LAMBDA" \
  --mmr-same-doc-similarity "$MMR_SAME_DOC_SIMILARITY" \
  --candidate-top-k "$RECALL_CANDIDATE_TOP_K" \
  --policy-score-mode front_policy_blend \
  --policy-blend-weight "$POLICY_BLEND_WEIGHT"

run_method gold_oracle \
  --selector gold_oracle

python3 - <<'PY'
import json
from pathlib import Path

out = Path(__import__("os").environ.get("OUTPUT_DIR", "outputs/rag/2wiki_generalization_1000"))
rows = []
for path in sorted(out.glob("*.json")):
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary", obj)
    rows.append(
        {
            "method": path.stem,
            "qids": summary.get("qids"),
            "answer_em": summary.get("answer_em"),
            "answer_f1": summary.get("answer_f1"),
            "answer_contains": summary.get("answer_contains"),
            "full_gold_doc_coverage": summary.get("full_gold_doc_coverage"),
            "full_gold_unit_coverage": summary.get("full_gold_unit_coverage"),
            "step_selected_contains_gold": summary.get("step_selected_contains_gold"),
            "step_acc@1": summary.get("step_acc@1"),
            "step_acc@5": summary.get("step_acc@5"),
            "avg_answer_tokens": summary.get("avg_answer_tokens"),
            "avg_answer_latency": summary.get("avg_answer_latency"),
        }
    )

summary_path = out / "summary.json"
summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"summary: {summary_path}")
PY
