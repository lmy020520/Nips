import json
from pathlib import Path
from typing import Dict, Iterable, List


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


def normalize_seed_units(seed_units: List[dict], qid: str) -> List[dict]:
    if not isinstance(seed_units, list):
        raise ValueError(f"seed_units 必须是 list: qid={qid}")

    normalized = []
    seen_unit_ids = set()
    seen_ranks = set()

    for i, item in enumerate(seed_units):
        required_fields = ["unit_id", "rank", "title", "sent_id", "text"]
        for field in required_fields:
            if field not in item:
                raise ValueError(
                    f"seed_unit 缺少字段: qid={qid}, idx={i}, field={field}"
                )

        unit_id = str(item["unit_id"])
        rank = int(item["rank"])
        title = str(item["title"])
        sent_id = int(item["sent_id"])
        text = str(item["text"]).strip()

        if not text:
            raise ValueError(f"seed_unit 的 text 为空: qid={qid}, unit_id={unit_id}")
        if unit_id in seen_unit_ids:
            raise ValueError(f"重复 unit_id: qid={qid}, unit_id={unit_id}")
        if rank in seen_ranks:
            raise ValueError(f"重复 rank: qid={qid}, rank={rank}")

        seen_unit_ids.add(unit_id)
        seen_ranks.add(rank)

        normalized.append(
            {
                "unit_id": unit_id,
                "rank": rank,
                "title": title,
                "sent_id": sent_id,
                "text": text,
            }
        )

    normalized.sort(key=lambda x: x["rank"])

    expected_ranks = list(range(1, len(normalized) + 1))
    actual_ranks = [x["rank"] for x in normalized]
    if actual_ranks != expected_ranks:
        raise ValueError(
            f"rank 不连续: qid={qid}, actual={actual_ranks}, expected={expected_ranks}"
        )

    return normalized


def build_init_state_record(retrieval_record: dict) -> dict:
    required_fields = ["qid", "seed_units"]
    for field in required_fields:
        if field not in retrieval_record:
            raise ValueError(f"retrieval record 缺少字段: {field}")

    qid = str(retrieval_record["qid"])
    seed_units = normalize_seed_units(retrieval_record["seed_units"], qid=qid)

    h0 = [x["unit_id"] for x in seed_units]
    s0 = {"unit_ids": list(h0)}
    k0 = "\n".join(
        f"{x['title']} [{x['sent_id']}] {x['text']}"
        for x in seed_units
    )

    return {
        "qid": qid,
        "H0": h0,
        "S0": s0,
        "K0": k0,
    }


def convert_split(retrieval_path: Path, init_state_path: Path) -> Dict[str, int]:
    total_in = 0
    total_out = 0
    total_units = 0
    seen_qids = set()

    def record_generator():
        nonlocal total_in, total_out, total_units

        for row_idx, retrieval_record in enumerate(read_jsonl(retrieval_path), start=1):
            total_in += 1

            if "qid" not in retrieval_record:
                raise ValueError(f"retrieval 记录缺少 qid: file={retrieval_path}, row={row_idx}")

            qid = str(retrieval_record["qid"])
            if qid in seen_qids:
                raise ValueError(f"重复 qid: file={retrieval_path}, qid={qid}")
            seen_qids.add(qid)

            try:
                init_record = build_init_state_record(retrieval_record)
            except Exception as e:
                raise ValueError(
                    f"构建 init_state 失败: file={retrieval_path}, row={row_idx}, qid={qid}, error={e}"
                ) from e

            total_units += len(init_record["H0"])
            total_out += 1
            yield init_record

    written = write_jsonl(record_generator(), init_state_path)

    if written != total_out:
        raise RuntimeError(
            f"写入条数异常: file={init_state_path}, written={written}, expected={total_out}"
        )

    return {
        "retrieval_records": total_in,
        "init_records": total_out,
        "total_units": total_units,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    retrieval_dir = base_dir / "retrieval"
    init_state_dir = base_dir / "init_state"

    input_files = {
        "train": retrieval_dir / "train.jsonl",
        "val": retrieval_dir / "val.jsonl",
        "test": retrieval_dir / "test.jsonl",
    }
    output_files = {
        "train": init_state_dir / "train.jsonl",
        "val": init_state_dir / "val.jsonl",
        "test": init_state_dir / "test.jsonl",
    }

    for split, path in input_files.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到 retrieval 文件: split={split}, path={path}")

    init_state_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = convert_split(input_files[split], output_files[split])
        all_stats[split] = stats

    print("HotpotQA init_state 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"retrieval_records={stats['retrieval_records']}, "
            f"init_records={stats['init_records']}, "
            f"total_units={stats['total_units']}, "
            f"output={output_files[split]}"
        )


if __name__ == "__main__":
    main()