import json
from pathlib import Path
from collections import OrderedDict
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
                raise ValueError(f"JSONL 解析失败: file={path}, line={line_idx}, error={e}") from e


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_targets_from_memory(memory_path: Path, targets_path: Path) -> Dict[str, int]:
    """
    从 memory/*.jsonl 中提取 is_supporting=true 的句子，
    按 qid 聚合成：
    {
      "qid": "...",
      "raw_targets": [
        {
          "unit_id": "...",
          "title": "...",
          "sent_id": 0,
          "text": "...",
          "raw_role": "sup"
        }
      ]
    }
    """
    grouped = OrderedDict()
    total_memory_records = 0
    total_supporting_records = 0

    for row_idx, record in enumerate(read_jsonl(memory_path), start=1):
        total_memory_records += 1

        required_fields = ["qid", "unit_id", "title", "sent_id", "text", "is_supporting"]
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"memory 记录缺少字段: file={memory_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        unit_id = str(record["unit_id"])
        title = str(record["title"])
        sent_id = int(record["sent_id"])
        text = str(record["text"])
        is_supporting = bool(record["is_supporting"])

        if qid not in grouped:
            grouped[qid] = {
                "qid": qid,
                "raw_targets": [],
                "_seen_unit_ids": set(),
            }

        if is_supporting:
            total_supporting_records += 1

            if unit_id in grouped[qid]["_seen_unit_ids"]:
                raise ValueError(
                    f"发现重复 supporting unit_id: file={memory_path}, qid={qid}, unit_id={unit_id}"
                )

            grouped[qid]["_seen_unit_ids"].add(unit_id)
            grouped[qid]["raw_targets"].append(
                {
                    "unit_id": unit_id,
                    "title": title,
                    "sent_id": sent_id,
                    "text": text,
                    "raw_role": "sup",
                }
            )

    # 只输出 raw_targets 非空的问题
    output_records = []
    for qid, item in grouped.items():
        raw_targets = item["raw_targets"]
        if not raw_targets:
            raise ValueError(f"qid={qid} 没有 supporting 句子，无法构建 raw_targets")
        output_records.append(
            {
                "qid": qid,
                "raw_targets": raw_targets,
            }
        )

    written = write_jsonl(output_records, targets_path)

    if written != len(output_records):
        raise RuntimeError(
            f"写入条数异常: file={targets_path}, written={written}, expected={len(output_records)}"
        )

    total_target_units = sum(len(x["raw_targets"]) for x in output_records)
    if total_target_units != total_supporting_records:
        raise RuntimeError(
            f"supporting 数量不一致: file={targets_path}, "
            f"from_memory={total_supporting_records}, in_targets={total_target_units}"
        )

    return {
        "questions": len(output_records),
        "memory_records": total_memory_records,
        "supporting_records": total_supporting_records,
        "target_units": total_target_units,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    memory_dir = base_dir / "memory"
    targets_dir = base_dir / "targets"

    input_files = {
        "train": memory_dir / "train.jsonl",
        "val": memory_dir / "val.jsonl",
        "test": memory_dir / "test.jsonl",
    }
    output_files = {
        "train": targets_dir / "train.jsonl",
        "val": targets_dir / "val.jsonl",
        "test": targets_dir / "test.jsonl",
    }

    for split, path in input_files.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到 memory 文件: split={split}, path={path}")

    targets_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = build_targets_from_memory(input_files[split], output_files[split])
        all_stats[split] = stats

    print("HotpotQA raw targets 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"questions={stats['questions']}, "
            f"memory_records={stats['memory_records']}, "
            f"supporting_records={stats['supporting_records']}, "
            f"target_units={stats['target_units']}, "
            f"output={output_files[split]}"
        )


if __name__ == "__main__":
    main()