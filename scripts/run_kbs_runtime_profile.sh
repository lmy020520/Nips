#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_QIDS="${MAX_QIDS:-500}"
WARMUP_QIDS="${WARMUP_QIDS:-20}"
DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_eval_3000_cand50}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/runtime_profiles_hotpot}"
CHECKPOINT="${CHECKPOINT:-outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt}"
MODEL_DIR="${MODEL_DIR:-models/deberta-v3-large}"
DENSE_MODEL="${DENSE_MODEL:-models/bge-large-en-v1.5}"
BGE_MODEL="${BGE_MODEL:-models/bge-reranker-large}"
MONOT5_MODEL="${MONOT5_MODEL:-models/monot5-base-msmarco}"
METHODS="${METHODS:-bm25 dense hybrid bge monot5 ksg_compact ksg_recall iterative_hybrid}"

samples="$DATA_ROOT/samples/test.jsonl"
memory="$DATA_ROOT/unit_registry/raw_units_test.jsonl"
queries="$DATA_ROOT/queries/test.jsonl"
for path in "$samples" "$memory" "$queries" "$CHECKPOINT" "$MODEL_DIR" "$DENSE_MODEL" "$BGE_MODEL" "$MONOT5_MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] missing required path: $path" >&2
    exit 1
  fi
done
mkdir -p "$OUTPUT_DIR"

common=(
  --samples "$samples"
  --memory "$memory"
  --queries "$queries"
  --checkpoint "$CHECKPOINT"
  --model-dir "$MODEL_DIR"
  --state-mode policy
  --policy-context-source online_state
  --select-top-k 5
  --max-qids "$MAX_QIDS"
  --ks 1,2,3,5
  --profile-runtime
  --profile-warmup-qids "$WARMUP_QIDS"
  --seed 20260608
)

run_profile() {
  local name="$1"
  shift
  echo "===== Profiling $name on GPU $CUDA_DEVICE ====="
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 scripts/run_hotpotqa_policy_rag.py \
    "${common[@]}" \
    --output "$OUTPUT_DIR/${name}.json" \
    "$@"
}

should_run() {
  [[ " $METHODS " == *" $1 "* ]]
}

# Run sequentially on one otherwise-idle GPU. Do not parallelize this block.
if should_run bm25; then run_profile bm25 \
  --selector bm25
fi

if should_run dense; then run_profile dense \
  --selector dense \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state
fi

if should_run hybrid; then run_profile hybrid \
  --selector hybrid \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state \
  --hybrid-alpha 0.5
fi

if should_run bge; then run_profile bge \
  --selector generic_reranker \
  --reranker-model "$BGE_MODEL" \
  --reranker-batch-size 8
fi

if should_run monot5; then run_profile monot5 \
  --selector t5_reranker \
  --reranker-model "$MONOT5_MODEL" \
  --reranker-batch-size 8 \
  --reranker-max-length 512
fi

if should_run ksg_compact; then run_profile ksg_compact \
  --selector hybrid_policy \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state \
  --hybrid-alpha 0.5 \
  --front-pool-k 30 \
  --front-fusion rrf \
  --local-expansion-window 1 \
  --mmr-lambda 0.7 \
  --mmr-same-doc-similarity 0.35 \
  --candidate-top-k 10 \
  --policy-score-mode front_policy_blend \
  --policy-blend-weight 0.35
fi

if should_run ksg_recall; then run_profile ksg_recall \
  --selector hybrid_policy \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state \
  --hybrid-alpha 0.5 \
  --front-pool-k 50 \
  --front-fusion rrf \
  --local-expansion-window 1 \
  --mmr-lambda 0.7 \
  --mmr-same-doc-similarity 0.35 \
  --candidate-top-k 50 \
  --policy-score-mode front_policy_blend \
  --policy-blend-weight 0.35
fi

if should_run iterative_hybrid; then run_profile iterative_hybrid \
  --selector iterative_hybrid \
  --dense-model "$DENSE_MODEL" \
  --dense-query-mode state \
  --hybrid-alpha 0.5
fi

python3 scripts/summarize_kbs_runtime_profiles.py \
  --profile BM25="$OUTPUT_DIR/bm25.json" \
  --profile Dense="$OUTPUT_DIR/dense.json" \
  --profile Hybrid="$OUTPUT_DIR/hybrid.json" \
  --profile BGE="$OUTPUT_DIR/bge.json" \
  --profile MonoT5="$OUTPUT_DIR/monot5.json" \
  --profile KSG-EA-Compact="$OUTPUT_DIR/ksg_compact.json" \
  --profile KSG-EA-Recall="$OUTPUT_DIR/ksg_recall.json" \
  --profile Iterative-Hybrid="$OUTPUT_DIR/iterative_hybrid.json" \
  --answer-report BM25=outputs/rag/full3000_bm25.json \
  --answer-report Dense=outputs/rag/full3000_dense.json \
  --answer-report Hybrid=outputs/rag/full3000_hybrid.json \
  --answer-report BGE=outputs/rag/bge_reranker_large_eval3000.json \
  --answer-report MonoT5=outputs/rag/monot5_reranker_eval3000.json \
  --answer-report KSG-EA-Compact=outputs/rag/kbs_v21_unified/hotpot_full_compact.json \
  --answer-report KSG-EA-Recall=outputs/rag/kbs_v21_unified/hotpot_full_recall.json \
  --answer-report Iterative-Hybrid=outputs/rag/missing_baselines/hotpot_iterative_hybrid.json \
  --answer-report Question-Only=outputs/rag/missing_baselines/hotpot_question_only.json \
  --output "$OUTPUT_DIR/summary.json" \
  --tsv-output "$OUTPUT_DIR/summary.tsv"
