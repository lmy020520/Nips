import json
from pathlib import Path
from typing import Dict, Iterable


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


def build_index_record(memory_record: dict) -> dict:
    required_fields = ["qid", "unit_id", "title", "sent_id", "text"]
    for field in required_fields:
        if field not in memory_record:
            raise ValueError(f"memory record 缺少必要字段: {field}")

    qid = str(memory_record["qid"])
    unit_id = str(memory_record["unit_id"])
    title = str(memory_record["title"])
    sent_id = int(memory_record["sent_id"])
    text = str(memory_record["text"]).strip()

    if not text:
        raise ValueError(f"空 text 不允许进入 index: unit_id={unit_id}")

    retrieval_text = f"{title} [SEP] {text}"

    return {
        "qid": qid,
        "unit_id": unit_id,
        "retrieval_text": retrieval_text,
        "title": title,
        "sent_id": sent_id,
        "text": text,
    }


def convert_split(memory_path: Path, index_path: Path) -> Dict[str, int]:
    total_in = 0
    total_out = 0
    seen_unit_ids = set()

    def record_generator():
        nonlocal total_in, total_out
        for row_idx, memory_record in enumerate(read_jsonl(memory_path), start=1):
            total_in += 1
            try:
                index_record = build_index_record(memory_record)
            except Exception as e:
                raise ValueError(
                    f"构建 index record 失败: file={memory_path}, row={row_idx}, error={e}"
                ) from e

            unit_id = index_record["unit_id"]
            if unit_id in seen_unit_ids:
                raise ValueError(
                    f"发现重复 unit_id: file={memory_path}, row={row_idx}, unit_id={unit_id}"
                )
            seen_unit_ids.add(unit_id)

            total_out += 1
            yield index_record

    written = write_jsonl(record_generator(), index_path)

    if written != total_out:
        raise RuntimeError(
            f"写入条数异常: file={index_path}, written={written}, expected={total_out}"
        )

    return {
        "memory_records": total_in,
        "index_records": total_out,
        "unique_unit_ids": len(seen_unit_ids),
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    memory_dir = base_dir / "memory"
    index_dir = base_dir / "index"

    input_files = {
        "train": memory_dir / "train.jsonl",
        "val": memory_dir / "val.jsonl",
        "test": memory_dir / "test.jsonl",
    }
    output_files = {
        "train": index_dir / "train.jsonl",
        "val": index_dir / "val.jsonl",
        "test": index_dir / "test.jsonl",
    }

    for split, path in input_files.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到 memory 文件: split={split}, path={path}")

    index_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = convert_split(input_files[split], output_files[split])
        all_stats[split] = stats

    print("HotpotQA index 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"memory_records={stats['memory_records']}, "
            f"index_records={stats['index_records']}, "
            f"unique_unit_ids={stats['unique_unit_ids']}, "
            f"output={output_files[split]}"
        )


if __name__ == "__main__":
    main()