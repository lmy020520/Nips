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


def load_init_state(init_state_path: Path) -> Dict[str, dict]:
    states = {}
    for row_idx, record in enumerate(read_jsonl(init_state_path), start=1):
        required_fields = ["qid", "H0", "S0", "K0"]
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"init_state 记录缺少字段: file={init_state_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in states:
            raise ValueError(f"init_state 中发现重复 qid: file={init_state_path}, qid={qid}")

        h0 = record["H0"]
        if not isinstance(h0, list):
            raise ValueError(f"H0 必须是 list: qid={qid}")

        normalized_h0 = []
        seen = set()
        for unit_id in h0:
            unit_id = str(unit_id)
            if unit_id in seen:
                raise ValueError(f"H0 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen.add(unit_id)
            normalized_h0.append(unit_id)

        states[qid] = {
            "qid": qid,
            "H0": normalized_h0,
            "S0": record["S0"],
            "K0": str(record["K0"]),
        }

    return states


def load_teacher_step0(teacher_path: Path) -> Dict[str, dict]:
    teacher = {}
    for row_idx, record in enumerate(read_jsonl(teacher_path), start=1):
        required_fields = ["qid", "t", "positive_unit_id", "candidate_labels"]
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"teacher 记录缺少字段: file={teacher_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in teacher:
            raise ValueError(f"teacher 中发现重复 qid: file={teacher_path}, qid={qid}")

        t = int(record["t"])
        if t != 0:
            raise ValueError(f"当前脚本只接 step0 teacher，但发现 t={t}: qid={qid}")

        positive_unit_id = str(record["positive_unit_id"]).strip()
        if not positive_unit_id:
            raise ValueError(f"positive_unit_id 为空: qid={qid}")

        candidate_labels = record["candidate_labels"]
        if not isinstance(candidate_labels, list):
            raise ValueError(f"candidate_labels 必须是 list: qid={qid}")

        positive_count = 0
        positive_in_candidates = False
        seen_candidate_ids = set()

        for i, item in enumerate(candidate_labels):
            if not isinstance(item, dict):
                raise ValueError(f"candidate_labels[{i}] 必须是 dict: qid={qid}")
            if "unit_id" not in item or "label" not in item:
                raise ValueError(f"candidate_labels[{i}] 缺少 unit_id 或 label: qid={qid}")

            unit_id = str(item["unit_id"])
            label = int(item["label"])

            if unit_id in seen_candidate_ids:
                raise ValueError(f"candidate_labels 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen_candidate_ids.add(unit_id)

            if label not in (0, 1):
                raise ValueError(f"label 只能是 0/1: qid={qid}, unit_id={unit_id}, label={label}")

            if label == 1:
                positive_count += 1
            if unit_id == positive_unit_id and label == 1:
                positive_in_candidates = True

        if positive_count <= 0:
            raise ValueError(f"teacher 样本没有正例: qid={qid}")
        if not positive_in_candidates:
            raise ValueError(f"positive_unit_id 不在正例候选中: qid={qid}, positive_unit_id={positive_unit_id}")

        teacher[qid] = {
            "qid": qid,
            "positive_unit_id": positive_unit_id,
        }

    return teacher


def load_memory_map(memory_path: Path) -> Dict[str, dict]:
    memory_map = {}
    for row_idx, record in enumerate(read_jsonl(memory_path), start=1):
        required_fields = ["qid", "unit_id", "title", "sent_id", "text"]
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"memory 记录缺少字段: file={memory_path}, row={row_idx}, field={field}"
                )

        unit_id = str(record["unit_id"])
        if unit_id in memory_map:
            raise ValueError(f"memory 中发现重复 unit_id: file={memory_path}, unit_id={unit_id}")

        text = str(record["text"]).strip()
        if not text:
            raise ValueError(f"memory 中 text 为空: file={memory_path}, unit_id={unit_id}")

        memory_map[unit_id] = {
            "qid": str(record["qid"]),
            "unit_id": unit_id,
            "title": str(record["title"]),
            "sent_id": int(record["sent_id"]),
            "text": text,
        }

    return memory_map


def format_notebook_line(memory_item: dict) -> str:
    return f"{memory_item['title']} [{memory_item['sent_id']}] {memory_item['text']}"


def build_state_step1_record(init_record: dict, teacher_record: dict, memory_map: Dict[str, dict]) -> dict:
    qid = init_record["qid"]
    h0 = init_record["H0"]
    positive_unit_id = teacher_record["positive_unit_id"]

    if positive_unit_id in h0:
        raise ValueError(f"positive_unit_id 已经存在于 H0 中: qid={qid}, unit_id={positive_unit_id}")

    h1 = list(h0) + [positive_unit_id]

    seen = set()
    for unit_id in h1:
        if unit_id in seen:
            raise ValueError(f"H1 中出现重复 unit_id: qid={qid}, unit_id={unit_id}")
        seen.add(unit_id)

    notebook_lines = []
    for unit_id in h1:
        if unit_id not in memory_map:
            raise ValueError(f"unit_id 不存在于 memory 中: qid={qid}, unit_id={unit_id}")

        memory_item = memory_map[unit_id]
        if memory_item["qid"] != qid:
            raise ValueError(
                f"unit_id 与 qid 不匹配: qid={qid}, unit_id={unit_id}, memory_qid={memory_item['qid']}"
            )

        notebook_lines.append(format_notebook_line(memory_item))

    k1 = "\n".join(notebook_lines)

    return {
        "qid": qid,
        "t": 1,
        "H_t": h1,
        "S_t": {
            "unit_ids": list(h1)
        },
        "K_t": k1,
    }


def build_state_split(init_state_path: Path, teacher_path: Path, memory_path: Path, state_path: Path) -> Dict[str, int]:
    init_states = load_init_state(init_state_path)
    teacher = load_teacher_step0(teacher_path)
    memory_map = load_memory_map(memory_path)

    total_init = 0
    total_written = 0
    total_units = 0

    def record_generator():
        nonlocal total_init, total_written, total_units

        qids = sorted(teacher.keys())
        for qid in qids:
            if qid not in init_states:
                raise ValueError(f"init_state 中找不到 qid: {qid}")

            total_init += 1
            record = build_state_step1_record(
                init_record=init_states[qid],
                teacher_record=teacher[qid],
                memory_map=memory_map,
            )
            total_written += 1
            total_units += len(record["H_t"])
            yield record

    written = write_jsonl(record_generator(), state_path)
    if written != total_written:
        raise RuntimeError(
            f"写入条数异常: file={state_path}, written={written}, expected={total_written}"
        )

    return {
        "input_teacher_records": len(teacher),
        "written_state_records": total_written,
        "total_units": total_units,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    init_state_dir = base_dir / "init_state"
    teacher_dir = base_dir / "teacher"
    memory_dir = base_dir / "memory"
    states_dir = base_dir / "states"

    init_state_files = {
        "train": init_state_dir / "train.jsonl",
        "val": init_state_dir / "val.jsonl",
        "test": init_state_dir / "test.jsonl",
    }
    teacher_files = {
        "train": teacher_dir / "step0_train.jsonl",
        "val": teacher_dir / "step0_val.jsonl",
        "test": teacher_dir / "step0_test.jsonl",
    }
    memory_files = {
        "train": memory_dir / "train.jsonl",
        "val": memory_dir / "val.jsonl",
        "test": memory_dir / "test.jsonl",
    }
    state_files = {
        "train": states_dir / "step1_train.jsonl",
        "val": states_dir / "step1_val.jsonl",
        "test": states_dir / "step1_test.jsonl",
    }

    for split in ["train", "val", "test"]:
        if not init_state_files[split].exists():
            raise FileNotFoundError(f"找不到 init_state 文件: {init_state_files[split]}")
        if not teacher_files[split].exists():
            raise FileNotFoundError(f"找不到 teacher 文件: {teacher_files[split]}")
        if not memory_files[split].exists():
            raise FileNotFoundError(f"找不到 memory 文件: {memory_files[split]}")

    states_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = build_state_split(
            init_state_path=init_state_files[split],
            teacher_path=teacher_files[split],
            memory_path=memory_files[split],
            state_path=state_files[split],
        )
        all_stats[split] = stats

    print("HotpotQA state step1 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"input_teacher_records={stats['input_teacher_records']}, "
            f"written_state_records={stats['written_state_records']}, "
            f"total_units={stats['total_units']}, "
            f"output={state_files[split]}"
        )


if __name__ == "__main__":
    main()