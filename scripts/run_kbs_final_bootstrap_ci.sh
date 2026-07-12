#!/usr/bin/env bash
set -euo pipefail

N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEED="${SEED:-20260628}"
METRICS="answer_em,answer_f1,step_at_5,full_unit_coverage"

required_reports=(
  outputs/rag/kbs_v21_unified/hotpot_full_compact.json
  outputs/rag/kbs_v21_unified/hotpot_full_recall.json
  outputs/rag/full3000_hybrid.json
  outputs/rag/bge_reranker_large_eval3000.json
  outputs/rag/kbs_v21_unified/2wiki_full_compact.json
  outputs/rag/kbs_v21_unified/2wiki_full_recall.json
  outputs/rag/2wiki_generalization_1000/hybrid_rag.json
  outputs/rag/2wiki_generalization_1000/bge_reranker_rag.json
)

for report in "${required_reports[@]}"; do
  if [[ ! -f "$report" ]]; then
    echo "Missing per-qid report: $report" >&2
    exit 1
  fi
done

run_ci() {
  local dataset="$1"
  local baseline="$2"
  local expected_qids="$3"
  shift 3
  local stem="outputs/analysis/${dataset}_v21_paired_ci_vs_${baseline,,}"

  python3 scripts/bootstrap_kbs_ci.py \
    "$@" \
    --baseline "$baseline" \
    --metrics "$METRICS" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --seed "$SEED" \
    --require-identical-qids \
    --require-summary-match \
    --expected-qids "$expected_qids" \
    --output "${stem}.json" \
    --tsv-output "${stem}.tsv"
}

hotpot_ksg_reports=(
  --report KSG-EA-Compact=outputs/rag/kbs_v21_unified/hotpot_full_compact.json
  --report KSG-EA-Recall=outputs/rag/kbs_v21_unified/hotpot_full_recall.json
)

wiki_ksg_reports=(
  --report KSG-EA-Compact=outputs/rag/kbs_v21_unified/2wiki_full_compact.json
  --report KSG-EA-Recall=outputs/rag/kbs_v21_unified/2wiki_full_recall.json
)

run_ci hotpot Hybrid 3000 \
  "${hotpot_ksg_reports[@]}" \
  --report Hybrid=outputs/rag/full3000_hybrid.json
run_ci hotpot BGE 3000 \
  "${hotpot_ksg_reports[@]}" \
  --report BGE=outputs/rag/bge_reranker_large_eval3000.json
run_ci 2wiki Hybrid 1000 \
  "${wiki_ksg_reports[@]}" \
  --report Hybrid=outputs/rag/2wiki_generalization_1000/hybrid_rag.json
run_ci 2wiki BGE 1000 \
  "${wiki_ksg_reports[@]}" \
  --report BGE=outputs/rag/2wiki_generalization_1000/bge_reranker_rag.json

echo "Bootstrap CI reports written under outputs/analysis/."
