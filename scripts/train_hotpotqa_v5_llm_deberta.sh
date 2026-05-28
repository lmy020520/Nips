#!/usr/bin/env bash
set -euo pipefail

# One-command training entry for the HotpotQA v5 LLM-role dataset.
#
# Usage:
#   bash scripts/train_hotpotqa_v5_llm_deberta.sh
#
# Optional overrides:
#   CONFIG=configs/train_ranker_deberta_v5_llm_quick.yaml bash scripts/train_hotpotqa_v5_llm_deberta.sh
#   SKIP_VALIDATE=1 bash scripts/train_hotpotqa_v5_llm_deberta.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_hotpotqa_v5_llm_deberta.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/train_ranker_deberta_v5_llm.yaml}"
DATA_ROOT="${DATA_ROOT:-data/hotpotqa_distractor_v5_llm_dataset}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

echo "===== HotpotQA v5 LLM DeBERTa Training ====="
echo "project_root: ${PROJECT_ROOT}"
echo "config:       ${CONFIG}"
echo "data_root:    ${DATA_ROOT}"
echo

required_files=(
  "${CONFIG}"
  "${DATA_ROOT}/samples/train.jsonl"
  "${DATA_ROOT}/samples/val.jsonl"
  "${DATA_ROOT}/samples/test.jsonl"
  "${DATA_ROOT}/unit_registry/raw_units_train.jsonl"
  "${DATA_ROOT}/unit_registry/raw_units_val.jsonl"
  "${DATA_ROOT}/unit_registry/raw_units_test.jsonl"
)

missing=0
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing required file: ${path}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "[ERROR] required training files are incomplete; aborting." >&2
  exit 1
fi

echo "===== data size ====="
du -sh "${DATA_ROOT}/samples" "${DATA_ROOT}/unit_registry"
echo

echo "===== python / torch / cuda ====="
python - <<'PY'
import sys
print("python:", sys.version.replace("\n", " "))
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device_count:", torch.cuda.device_count())
        print("cuda_device_0:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("cuda_memory_gb:", round(props.total_memory / 1024**3, 2))
except Exception as exc:
    print("torch_check_error:", repr(exc))
PY
echo

if [[ "${SKIP_VALIDATE}" != "1" ]]; then
  echo "===== dataset validation ====="
  python scripts/validate_hotpotqa_dataset_release.py \
    --data_root "${DATA_ROOT}" \
    --strict
  echo
else
  echo "===== dataset validation skipped ====="
  echo
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "===== start training ====="
python src/train/train_ranker.py --config "${CONFIG}"
