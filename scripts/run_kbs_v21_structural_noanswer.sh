#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

GPU_LIST="${GPU_LIST:-3,4,5,6,7}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rag/v21_structural_noanswer}"
CHECKPOINT="outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt"
ABLATIONS="${ABLATIONS:-official no_online_state query_only_policy no_front_policy_blend no_policy no_local_expansion no_dense no_bm25}"
mkdir -p "$OUTPUT_DIR"

IFS=',' read -r -a gpus <<< "$GPU_LIST"
read -r -a variants <<< "$ABLATIONS"
for ((start=0; start<${#variants[@]}; start+=${#gpus[@]})); do
  pids=()
  for ((offset=0; offset<${#gpus[@]} && start+offset<${#variants[@]}; offset++)); do
    variant="${variants[start+offset]}"
    echo "[INFO] v21 no-answer ablation=$variant gpu=${gpus[offset]}"
    TRANSFORMERS_OFFLINE=1 \
    ABLATION="$variant" \
    CHECKPOINT="$CHECKPOINT" \
    CUDA_DEVICE="${gpus[offset]}" \
    MAX_QIDS=3000 \
    GENERATE_ANSWERS=0 \
    REQUIRE_DEEPSEEK_API_KEY=0 \
    SAVE_ONLINE_STATES=0 \
    OUTPUT_DIR="$OUTPUT_DIR" \
    bash scripts/run_kbs_ablation_experiment.sh &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

python3 - "$OUTPUT_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
keys = ["qids", "step_acc@1", "step_acc@5", "full_gold_doc_coverage", "full_gold_unit_coverage"]
rows = []
for path in sorted(root.glob("*.json")):
    summary = json.loads(path.read_text(encoding="utf-8")).get("summary", {})
    rows.append({"variant": path.stem, **{key: summary.get(key) for key in keys}})
(root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
