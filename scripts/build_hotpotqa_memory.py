import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


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


def normalize_supporting_set(sample: dict) -> Set[Tuple[str, int]]:
    supporting_facts = sample.get("supporting_facts", [])
    if not isinstance(supporting_facts, list):
        raise ValueError("supporting_facts 必须是 list")

    support_set: Set[Tuple[str, int]] = set()
    for i, item in enumerate(supporting_facts):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"supporting_facts[{i}] 格式错误，应为 [title, sent_id]")
        title, sent_id = item
        support_set.add((str(title), int(sent_id)))
    return support_set


def build_memory_records_from_sample(sample: dict) -> List[dict]:
    required_fields = [
        "qid",
        "question",
        "answer",
        "type",
        "level",
        "supporting_facts",
        "context",
    ]
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"样本缺少必要字段: {field}")

    qid = str(sample["qid"])
    context = sample["context"]
    if not isinstance(context, list):
        raise ValueError("context 必须是 list")

    support_set = normalize_supporting_set(sample)

    records: List[dict] = []
    seen_unit_ids: Set[str] = set()
    written_supporting_set: Set[Tuple[str, int]] = set()

    for block_idx, block in enumerate(context):
        if not isinstance(block, dict):
            raise ValueError(f"context[{block_idx}] 必须是 dict")

        if "title" not in block or "sentences" not in block:
            raise ValueError(f"context[{block_idx}] 缺少 title 或 sentences")

        title = str(block["title"])
        sentences = block["sentences"]

        if not isinstance(sentences, list):
            raise ValueError(f"context[{block_idx}].sentences 必须是 list")

        for sent_id, text in enumerate(sentences):
            if not isinstance(text, str):
                raise ValueError(
                    f"context[{block_idx}].sentences[{sent_id}] 必须是 str，实际为 {type(text)}"
                )

            text = text.strip()
            if not text:
                # 保持 sent_id 原编号，不写入空句子，也不重排
                continue

            unit_id = f"{qid}::{title}::{sent_id}"
            if unit_id in seen_unit_ids:
                raise ValueError(f"发现重复 unit_id: {unit_id}")
            seen_unit_ids.add(unit_id)

            is_supporting = (title, sent_id) in support_set
            if is_supporting:
                written_supporting_set.add((title, sent_id))

            records.append(
                {
                    "qid": qid,
                    "unit_id": unit_id,
                    "title": title,
                    "sent_id": sent_id,
                    "text": text,
                    "is_supporting": is_supporting,
                }
            )

    # sanity check: supporting_facts 必须都能在 memory 中找到
    if written_supporting_set != support_set:
        missing = sorted(list(support_set - written_supporting_set))
        extra = sorted(list(written_supporting_set - support_set))
        raise ValueError(
            f"supporting 覆盖校验失败: qid={qid}, missing={missing}, extra={extra}"
        )

    return records


def convert_split(processed_path: Path, memory_path: Path) -> Dict[str, int]:
    total_questions = 0
    total_units = 0
    total_supporting_units = 0

    def record_generator():
        nonlocal total_questions, total_units, total_supporting_units
        for sample_idx, sample in enumerate(read_jsonl(processed_path), start=1):
            try:
                records = build_memory_records_from_sample(sample)
            except Exception as e:
                raise ValueError(
                    f"处理样本失败: file={processed_path}, sample_idx={sample_idx}, qid={sample.get('qid', 'UNKNOWN')}, error={e}"
                ) from e

            total_questions += 1
            total_units += len(records)
            total_supporting_units += sum(1 for r in records if r["is_supporting"])

            for record in records:
                yield record

    written_count = write_jsonl(record_generator(), memory_path)

    if written_count != total_units:
        raise RuntimeError(
            f"写入条数异常: file={memory_path}, written_count={written_count}, total_units={total_units}"
        )

    return {
        "questions": total_questions,
        "units": total_units,
        "supporting_units": total_supporting_units,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    processed_dir = base_dir / "processed"
    memory_dir = base_dir / "memory"

    input_files = {
        "train": processed_dir / "train.jsonl",
        "val": processed_dir / "val.jsonl",
        "test": processed_dir / "test.jsonl",
    }
    output_files = {
        "train": memory_dir / "train.jsonl",
        "val": memory_dir / "val.jsonl",
        "test": memory_dir / "test.jsonl",
    }

    for split, path in input_files.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到 processed 文件: split={split}, path={path}")

    memory_dir.mkdir(parents=True, exist_ok=True)

    all_stats: Dict[str, Dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        stats = convert_split(input_files[split], output_files[split])
        all_stats[split] = stats

    print("HotpotQA memory 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"questions={stats['questions']}, "
            f"units={stats['units']}, "
            f"supporting_units={stats['supporting_units']}, "
            f"output={output_files[split]}"
        )


if __name__ == "__main__":
    main()