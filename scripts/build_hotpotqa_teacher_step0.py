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


def load_targets(targets_path: Path) -> Dict[str, dict]:
    targets = {}
    for row_idx, record in enumerate(read_jsonl(targets_path), start=1):
        required_fields = ["qid", "raw_targets"]
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"targets 记录缺少字段: file={targets_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in targets:
            raise ValueError(f"targets 中发现重复 qid: file={targets_path}, qid={qid}")

        raw_targets = record["raw_targets"]
        if not isinstance(raw_targets, list):
            raise ValueError(f"raw_targets 必须是 list: qid={qid}")

        target_unit_ids = []
        seen = set()
        for i, item in enumerate(raw_targets):
            if not isinstance(item, dict):
                raise ValueError(f"raw_targets[{i}] 必须是 dict: qid={qid}")
            if "unit_id" not in item:
                raise ValueError(f"raw_targets[{i}] 缺少 unit_id: qid={qid}")

            unit_id = str(item["unit_id"])
            if unit_id in seen:
                raise ValueError(f"raw_targets 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen.add(unit_id)
            target_unit_ids.append(unit_id)

        targets[qid] = {
            "qid": qid,
            "target_unit_ids": target_unit_ids,
            "target_unit_id_set": set(target_unit_ids),
        }

    return targets


def load_rollout_step0(rollout_path: Path) -> List[dict]:
    records = []
    seen_qids = set()

    for row_idx, record in enumerate(read_jsonl(rollout_path), start=1):
        required_fields = ["qid", "t", "H_t", "R_t"]
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"rollout 记录缺少字段: file={rollout_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in seen_qids:
            raise ValueError(f"rollout 中发现重复 qid: file={rollout_path}, qid={qid}")
        seen_qids.add(qid)

        t = int(record["t"])
        if t != 0:
            raise ValueError(f"当前脚本只处理 step0，但发现 t={t}: qid={qid}")

        h_t = record["H_t"]
        if not isinstance(h_t, list):
            raise ValueError(f"H_t 必须是 list: qid={qid}")

        normalized_h_t = []
        seen_h = set()
        for i, unit_id in enumerate(h_t):
            unit_id = str(unit_id)
            if unit_id in seen_h:
                raise ValueError(f"H_t 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen_h.add(unit_id)
            normalized_h_t.append(unit_id)

        r_t = record["R_t"]
        if not isinstance(r_t, list):
            raise ValueError(f"R_t 必须是 list: qid={qid}")

        normalized_r_t = []
        seen_r = set()
        seen_ranks = set()
        for i, item in enumerate(r_t):
            required_r_fields = ["unit_id", "rank"]
            for field in required_r_fields:
                if field not in item:
                    raise ValueError(f"R_t[{i}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item["unit_id"])
            rank = int(item["rank"])

            if unit_id in seen_r:
                raise ValueError(f"R_t 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            if rank in seen_ranks:
                raise ValueError(f"R_t 中重复 rank: qid={qid}, rank={rank}")

            seen_r.add(unit_id)
            seen_ranks.add(rank)

            normalized_r_t.append(
                {
                    "unit_id": unit_id,
                    "rank": rank,
                }
            )

        normalized_r_t.sort(key=lambda x: x["rank"])
        expected_ranks = list(range(1, len(normalized_r_t) + 1))
        actual_ranks = [x["rank"] for x in normalized_r_t]
        if actual_ranks != expected_ranks:
            raise ValueError(
                f"R_t 的 rank 不连续: qid={qid}, actual={actual_ranks}, expected={expected_ranks}"
            )

        records.append(
            {
                "qid": qid,
                "t": t,
                "H_t": normalized_h_t,
                "R_t": normalized_r_t,
            }
        )

    return records


def build_teacher_record(rollout_record: dict, target_record: dict):
    qid = rollout_record["qid"]
    h_t = rollout_record["H_t"]
    r_t = rollout_record["R_t"]

    target_set = target_record["target_unit_id_set"]
    covered_set = set(h_t)
    remaining_target_set = target_set - covered_set

    candidate_labels = []
    positive_unit_id = None

    for item in r_t:
        unit_id = item["unit_id"]
        label = 1 if unit_id in remaining_target_set else 0
        candidate_labels.append(
            {
                "unit_id": unit_id,
                "label": label,
            }
        )
        if label == 1 and positive_unit_id is None:
            positive_unit_id = unit_id

    if positive_unit_id is None:
        return None

    positive_count = sum(x["label"] for x in candidate_labels)
    if positive_count <= 0:
        raise RuntimeError(f"内部错误：positive_count 应该 > 0，但 qid={qid}")

    return {
        "qid": qid,
        "t": 0,
        "positive_unit_id": positive_unit_id,
        "candidate_labels": candidate_labels,
    }


def build_teacher_split(rollout_path: Path, targets_path: Path, teacher_path: Path) -> Dict[str, int]:
    rollout_records = load_rollout_step0(rollout_path)
    targets = load_targets(targets_path)

    total_rollout = 0
    total_written = 0
    total_skipped = 0
    total_positive_labels = 0

    def record_generator():
        nonlocal total_rollout, total_written, total_skipped, total_positive_labels

        for record in rollout_records:
            total_rollout += 1
            qid = record["qid"]

            if qid not in targets:
                raise ValueError(f"targets 中找不到 qid: {qid}")

            teacher_record = build_teacher_record(record, targets[qid])

            if teacher_record is None:
                total_skipped += 1
                continue

            pos_count = sum(x["label"] for x in teacher_record["candidate_labels"])
            total_positive_labels += pos_count
            total_written += 1
            yield teacher_record

    written = write_jsonl(record_generator(), teacher_path)
    if written != total_written:
        raise RuntimeError(
            f"写入条数异常: file={teacher_path}, written={written}, expected={total_written}"
        )

    return {
        "rollout_records": total_rollout,
        "teacher_records": total_written,
        "skipped_no_positive": total_skipped,
        "positive_labels": total_positive_labels,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    rollout_dir = base_dir / "rollout"
    targets_dir = base_dir / "targets"
    teacher_dir = base_dir / "teacher"

    rollout_files = {
        "train": rollout_dir / "step0_train.jsonl",
        "val": rollout_dir / "step0_val.jsonl",
        "test": rollout_dir / "step0_test.jsonl",
    }
    targets_files = {
        "train": targets_dir / "train.jsonl",
        "val": targets_dir / "val.jsonl",
        "test": targets_dir / "test.jsonl",
    }
    teacher_files = {
        "train": teacher_dir / "step0_train.jsonl",
        "val": teacher_dir / "step0_val.jsonl",
        "test": teacher_dir / "step0_test.jsonl",
    }

    for split in ["train", "val", "test"]:
        if not rollout_files[split].exists():
            raise FileNotFoundError(f"找不到 rollout 文件: {rollout_files[split]}")
        if not targets_files[split].exists():
            raise FileNotFoundError(f"找不到 targets 文件: {targets_files[split]}")

    teacher_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = build_teacher_split(
            rollout_path=rollout_files[split],
            targets_path=targets_files[split],
            teacher_path=teacher_files[split],
        )
        all_stats[split] = stats

    print("HotpotQA teacher step0 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"rollout_records={stats['rollout_records']}, "
            f"teacher_records={stats['teacher_records']}, "
            f"skipped_no_positive={stats['skipped_no_positive']}, "
            f"positive_labels={stats['positive_labels']}, "
            f"output={teacher_files[split]}"
        )


if __name__ == "__main__":
    main()