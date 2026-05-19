import os
import json
from pathlib import Path
from typing import Iterable, Dict


SPLITS = ["train", "val", "test"]

DEFAULT_INPUT_BASE = "data/hotpotqa_distractor/processed"
DEFAULT_OUTPUT_BASE = os.environ.get("HOTPOTQA_QUERY_BASE", os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v2") + "/queries")


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


def build_query_record(sample: dict, split: str) -> dict:
    required_fields = ["qid", "question", "answer", "type", "level"]
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"processed 样本缺少字段: {field}")

    qid = str(sample["qid"])
    question = str(sample["question"]).strip()
    answer = str(sample["answer"]).strip()
    q_type = str(sample["type"]).strip()
    level = str(sample["level"]).strip()

    if not qid:
        raise ValueError("qid 为空")
    if not question:
        raise ValueError(f"question 为空: qid={qid}")
    if not answer:
        raise ValueError(f"answer 为空: qid={qid}")

    return {
        "qid": qid,
        "question": question,
        "answer": answer,
        "metadata": {
            "dataset": "hotpotqa_distractor",
            "split": split,
            "type": q_type,
            "level": level,
        },
    }


def convert_split(input_path: Path, output_path: Path, split: str) -> int:
    seen_qids = set()

    def generator():
        for row_idx, sample in enumerate(read_jsonl(input_path), start=1):
            try:
                record = build_query_record(sample, split=split)
            except Exception as e:
                raise ValueError(
                    f"构建 QueryRecord 失败: file={input_path}, row={row_idx}, qid={sample.get('qid', 'UNKNOWN')}, error={e}"
                ) from e

            qid = record["qid"]
            if qid in seen_qids:
                raise ValueError(f"split 内出现重复 qid: file={input_path}, qid={qid}")
            seen_qids.add(qid)

            yield record

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent

    input_base = project_root / DEFAULT_INPUT_BASE
    output_base = project_root / DEFAULT_OUTPUT_BASE
    output_base.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {}

    for split in SPLITS:
        input_path = input_base / f"{split}.jsonl"
        output_path = output_base / f"{split}.jsonl"

        if not input_path.exists():
            raise FileNotFoundError(f"找不到 processed 文件: {input_path}")

        stats[split] = convert_split(input_path, output_path, split=split)

    print("QueryRecord v2 重建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {output_base / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()