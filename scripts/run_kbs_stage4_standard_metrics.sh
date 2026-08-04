#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/kbs_stage4_standard_metrics}"
COMPACT_REPORT="${COMPACT_REPORT:-outputs/rag/kbs_v22_stage2_closure_balanced/full_compact.json}"
RECALL_REPORT="${RECALL_REPORT:-outputs/rag/kbs_v22_stage2_hotpot/full_recall.json}"
ANCHOR_REPORT="${ANCHOR_REPORT:-outputs/rag/kbs_stage3_acra_anchor_hotpot/previous_only_compact.json}"
DIRECT_REPORT="${DIRECT_REPORT:-outputs/rag/kbs_stage3_ecdr_direct_indirect_hotpot/direct_only_compact.json}"
GOLD_REPORT="${GOLD_REPORT:-outputs/rag/full3000_gold_oracle.json}"
EXPECTED_QIDS="${EXPECTED_QIDS:-3000}"

mkdir -p "$OUTPUT_DIR"

python3 scripts/check_kbs_stage4_metric_readiness.py \
  --require-paths \
  --expected-qids "$EXPECTED_QIDS" \
  --report "KSG-EA-Compact=$COMPACT_REPORT" \
  --report "KSG-EA-Recall=$RECALL_REPORT" \
  --report "Anchor=$ANCHOR_REPORT" \
  --report "Direct-Indirect=$DIRECT_REPORT" \
  --report "Gold-Oracle=$GOLD_REPORT" \
  --output "$OUTPUT_DIR/readiness.json"

python3 scripts/evaluate_kbs_standard_metrics.py \
  --report "KSG-EA-Compact=$COMPACT_REPORT" \
  --report "KSG-EA-Recall=$RECALL_REPORT" \
  --report "Anchor=$ANCHOR_REPORT" \
  --report "Direct-Indirect=$DIRECT_REPORT" \
  --report "Gold-Oracle=$GOLD_REPORT" \
  --gold-oracle-name "Gold-Oracle" \
  --expected-qids "$EXPECTED_QIDS" \
  --closure-unit-budgets 5,10,15,20,50 \
  --output "$OUTPUT_DIR/summary.json" \
  --records-output "$OUTPUT_DIR/records.jsonl"

echo "FINISHED_OK"
echo "summary=$OUTPUT_DIR/summary.json"
echo "records=$OUTPUT_DIR/records.jsonl"
