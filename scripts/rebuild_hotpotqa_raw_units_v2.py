import os
import json
from pathlib import Path
from typing import Iterable, List, Dict


SPLITS = ["train", "val", "test"]

DEFAULT_INPUT_BASE = "data/hotpotqa_distractor/processed"
DEFAULT_OUTPUT_BASE = os.environ.get("HOTPOTQA_UNIT_REGISTRY_BASE", os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v2") + "/unit_registry")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSONL 解析失败: file={path}, line={line_idx}, error={e}"
                ) from e


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_unit_id(qid: str, doc_id: str, sent_id: int) -> str:
    return f"{qid}::{doc_id}::{sent_id}"


def build_parent_chunk_id(qid: str, doc_id: str) -> str:
    return f"{qid}::{doc_id}"


def build_raw_units_from_sample(sample: dict) -> List[dict]:
    required_fields = ["qid", "context"]
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"processed 样本缺少字段: {field}")

    qid = str(sample["qid"])
    context = sample["context"]

    if not isinstance(context, list):
        raise ValueError(f"context 必须是 list: qid={qid}")

    raw_units = []
    seen_unit_ids = set()

    for block_idx, block in enumerate(context):
        if not isinstance(block, dict):
            raise ValueError(f"context[{block_idx}] 必须是 dict: qid={qid}")

        if "title" not in block or "sentences" not in block:
            raise ValueError(f"context[{block_idx}] 缺少 title 或 sentences: qid={qid}")

        doc_id = str(block["title"])
        sentences = block["sentences"]

        if not isinstance(sentences, list):
            raise ValueError(f"sentences 必须是 list: qid={qid}, title={doc_id}")

        parent_chunk_id = build_parent_chunk_id(qid, doc_id)

        for sent_id, text in enumerate(sentences):
            if not isinstance(text, str):
                raise ValueError(
                    f"sentence 必须是 str: qid={qid}, title={doc_id}, sent_id={sent_id}"
                )

            text = text.strip()
            if not text:
                continue

            unit_id = build_unit_id(qid, doc_id, sent_id)
            if unit_id in seen_unit_ids:
                raise ValueError(f"发现重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen_unit_ids.add(unit_id)

            raw_units.append(
                {
                    "unit_id": unit_id,
                    "text": text,
                    "doc_id": doc_id,
                    "parent_chunk_id": parent_chunk_id,
                    "span_start": None,
                    "span_end": None,
                    "provenance": "raw",
                    "candidate_granularity": "sentence",
                }
            )

    if not raw_units:
        raise ValueError(f"该样本没有可写入的 raw units: qid={qid}")

    return raw_units


def convert_split(input_path: Path, output_path: Path) -> int:
    seen_global_unit_ids = set()

    def generator():
        for row_idx, sample in enumerate(read_jsonl(input_path), start=1):
            try:
                raw_units = build_raw_units_from_sample(sample)
            except Exception as e:
                raise ValueError(
                    f"构建 RawUnit 失败: file={input_path}, row={row_idx}, qid={sample.get('qid', 'UNKNOWN')}, error={e}"
                ) from e

            for unit in raw_units:
                unit_id = unit["unit_id"]
                if unit_id in seen_global_unit_ids:
                    raise ValueError(f"split 内出现重复 unit_id: file={input_path}, unit_id={unit_id}")
                seen_global_unit_ids.add(unit_id)
                yield unit

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent

    input_base = project_root / DEFAULT_INPUT_BASE
    output_base = project_root / DEFAULT_OUTPUT_BASE
    output_base.mkdir(parents=True, exist_ok=True)

    output_name_map = {
        "train": "raw_units_train.jsonl",
        "val": "raw_units_val.jsonl",
        "test": "raw_units_test.jsonl",
    }

    stats: Dict[str, int] = {}

    for split in SPLITS:
        input_path = input_base / f"{split}.jsonl"
        output_path = output_base / output_name_map[split]

        if not input_path.exists():
            raise FileNotFoundError(f"找不到 processed 文件: {input_path}")

        stats[split] = convert_split(input_path, output_path)

    print("RawUnit v2 重建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {output_base / output_name_map[split]}")


if __name__ == "__main__":
    main()