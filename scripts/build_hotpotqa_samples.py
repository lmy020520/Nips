import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def load_questions(processed_path: Path) -> Dict[str, str]:
    questions = {}
    for row_idx, record in enumerate(read_jsonl(processed_path), start=1):
        for field in ["qid", "question"]:
            if field not in record:
                raise ValueError(
                    f"processed 记录缺少字段: file={processed_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        question = str(record["question"]).strip()

        if not question:
            raise ValueError(f"question 为空: qid={qid}")
        if qid in questions:
            raise ValueError(f"processed 中发现重复 qid: {qid}")

        questions[qid] = question
    return questions


def load_init_state(init_state_path: Path) -> Dict[str, dict]:
    states = {}
    for row_idx, record in enumerate(read_jsonl(init_state_path), start=1):
        for field in ["qid", "H0", "K0"]:
            if field not in record:
                raise ValueError(
                    f"init_state 缺少字段: file={init_state_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in states:
            raise ValueError(f"init_state 中发现重复 qid: {qid}")

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
            "H_t": normalized_h0,
            "K_t": str(record["K0"]),
        }
    return states


def load_state_t(state_path: Path, expected_t: int) -> Dict[str, dict]:
    states = {}
    for row_idx, record in enumerate(read_jsonl(state_path), start=1):
        for field in ["qid", "t", "H_t", "K_t"]:
            if field not in record:
                raise ValueError(
                    f"state 缺少字段: file={state_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in states:
            raise ValueError(f"state 中发现重复 qid: {qid}")

        t = int(record["t"])
        if t != expected_t:
            raise ValueError(f"期望 t={expected_t}，但发现 t={t}: qid={qid}")

        h_t = record["H_t"]
        if not isinstance(h_t, list):
            raise ValueError(f"H_t 必须是 list: qid={qid}")

        normalized_h_t = []
        seen = set()
        for unit_id in h_t:
            unit_id = str(unit_id)
            if unit_id in seen:
                raise ValueError(f"H_t 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen.add(unit_id)
            normalized_h_t.append(unit_id)

        states[qid] = {
            "qid": qid,
            "H_t": normalized_h_t,
            "K_t": str(record["K_t"]),
        }
    return states


def load_rollout_t(rollout_path: Path, expected_t: int) -> Dict[str, dict]:
    rollout = {}
    for row_idx, record in enumerate(read_jsonl(rollout_path), start=1):
        for field in ["qid", "t", "H_t", "R_t"]:
            if field not in record:
                raise ValueError(
                    f"rollout 缺少字段: file={rollout_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in rollout:
            raise ValueError(f"rollout 中发现重复 qid: {qid}")

        t = int(record["t"])
        if t != expected_t:
            raise ValueError(f"期望 rollout t={expected_t}，但发现 t={t}: qid={qid}")

        h_t = record["H_t"]
        if not isinstance(h_t, list):
            raise ValueError(f"H_t 必须是 list: qid={qid}")

        normalized_h_t = [str(x) for x in h_t]

        r_t = record["R_t"]
        if not isinstance(r_t, list):
            raise ValueError(f"R_t 必须是 list: qid={qid}")

        candidate_items = []
        seen_unit_ids = set()
        seen_ranks = set()
        for i, item in enumerate(r_t):
            for field in ["unit_id", "rank"]:
                if field not in item:
                    raise ValueError(f"R_t[{i}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item["unit_id"])
            rank = int(item["rank"])

            if unit_id in seen_unit_ids:
                raise ValueError(f"R_t 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            if rank in seen_ranks:
                raise ValueError(f"R_t 中重复 rank: qid={qid}, rank={rank}")

            seen_unit_ids.add(unit_id)
            seen_ranks.add(rank)
            candidate_items.append({"unit_id": unit_id, "rank": rank})

        candidate_items.sort(key=lambda x: x["rank"])
        expected_ranks = list(range(1, len(candidate_items) + 1))
        actual_ranks = [x["rank"] for x in candidate_items]
        if actual_ranks != expected_ranks:
            raise ValueError(
                f"R_t rank 不连续: qid={qid}, actual={actual_ranks}, expected={expected_ranks}"
            )

        rollout[qid] = {
            "qid": qid,
            "H_t": normalized_h_t,
            "candidates": [x["unit_id"] for x in candidate_items],
        }
    return rollout


def load_teacher_t(teacher_path: Path, expected_t: int) -> Dict[str, dict]:
    teacher = {}
    for row_idx, record in enumerate(read_jsonl(teacher_path), start=1):
        for field in ["qid", "t", "positive_unit_id", "candidate_labels"]:
            if field not in record:
                raise ValueError(
                    f"teacher 缺少字段: file={teacher_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in teacher:
            raise ValueError(f"teacher 中发现重复 qid: {qid}")

        t = int(record["t"])
        if t != expected_t:
            raise ValueError(f"期望 teacher t={expected_t}，但发现 t={t}: qid={qid}")

        positive_unit_id = str(record["positive_unit_id"]).strip()
        if not positive_unit_id:
            raise ValueError(f"positive_unit_id 为空: qid={qid}")

        candidate_labels = record["candidate_labels"]
        if not isinstance(candidate_labels, list):
            raise ValueError(f"candidate_labels 必须是 list: qid={qid}")

        labels = {}
        positive_count = 0
        for i, item in enumerate(candidate_labels):
            if "unit_id" not in item or "label" not in item:
                raise ValueError(f"candidate_labels[{i}] 缺少 unit_id 或 label: qid={qid}")

            unit_id = str(item["unit_id"])
            label = int(item["label"])
            if label not in (0, 1):
                raise ValueError(f"label 只能是 0/1: qid={qid}, unit_id={unit_id}, label={label}")
            if unit_id in labels:
                raise ValueError(f"candidate_labels 中重复 unit_id: qid={qid}, unit_id={unit_id}")

            labels[unit_id] = label
            if label == 1:
                positive_count += 1

        if positive_count <= 0:
            raise ValueError(f"teacher 样本没有正例: qid={qid}")
        if labels.get(positive_unit_id) != 1:
            raise ValueError(f"positive_unit_id 不是正例: qid={qid}, positive_unit_id={positive_unit_id}")

        teacher[qid] = {
            "qid": qid,
            "positive_unit_id": positive_unit_id,
            "candidate_labels": labels,
        }
    return teacher


def load_terminal(terminal_path: Path) -> Dict[str, dict]:
    terminal = {}
    for row_idx, record in enumerate(read_jsonl(terminal_path), start=1):
        for field in ["qid", "terminal_t", "stop_label", "final_answer"]:
            if field not in record:
                raise ValueError(
                    f"terminal 缺少字段: file={terminal_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in terminal:
            raise ValueError(f"terminal 中发现重复 qid: {qid}")

        terminal[qid] = {
            "qid": qid,
            "terminal_t": int(record["terminal_t"]),
            "stop_label": int(record["stop_label"]),
            "final_answer": str(record["final_answer"]),
        }
    return terminal


def build_one_sample(
    qid: str,
    t: int,
    question: str,
    state_record: dict,
    rollout_record: dict,
    teacher_record: dict,
    terminal_record: dict,
) -> dict:
    state_h = state_record["H_t"]
    rollout_h = rollout_record["H_t"]

    if state_h != rollout_h:
        raise ValueError(f"state.H_t 与 rollout.H_t 不一致: qid={qid}, t={t}")

    candidates = rollout_record["candidates"]
    positive_unit_id = teacher_record["positive_unit_id"]

    if positive_unit_id not in candidates:
        raise ValueError(f"positive_unit_id 不在 candidates 中: qid={qid}, t={t}")

    terminal_t = terminal_record["terminal_t"]
    stop_label = 1 if terminal_t == (t + 1) else 0

    return {
        "qid": qid,
        "t": t,
        "question": question,
        "H_t": list(state_h),
        "K_t": state_record["K_t"],
        "candidates": candidates,
        "positive_unit_id": positive_unit_id,
        "stop_label": stop_label,
    }


def build_samples_split(
    processed_path: Path,
    init_state_path: Path,
    step1_state_path: Path,
    step0_rollout_path: Path,
    step1_rollout_path: Path,
    step0_teacher_path: Path,
    step1_teacher_path: Path,
    terminal_path: Path,
    output_path: Path,
) -> Dict[str, int]:
    questions = load_questions(processed_path)
    init_states = load_init_state(init_state_path)
    step1_states = load_state_t(step1_state_path, expected_t=1) if step1_state_path.exists() else {}
    step0_rollout = load_rollout_t(step0_rollout_path, expected_t=0)
    step1_rollout = load_rollout_t(step1_rollout_path, expected_t=1) if step1_rollout_path.exists() else {}
    step0_teacher = load_teacher_t(step0_teacher_path, expected_t=0)
    step1_teacher = load_teacher_t(step1_teacher_path, expected_t=1) if step1_teacher_path.exists() else {}
    terminal = load_terminal(terminal_path)

    records: List[dict] = []
    count_t0 = 0
    count_t1 = 0
    stop1 = 0
    stop0 = 0

    # t = 0 samples
    for qid in sorted(step0_teacher.keys()):
        if qid not in questions:
            raise ValueError(f"processed 中找不到 qid: {qid}")
        if qid not in init_states:
            raise ValueError(f"init_state 中找不到 qid: {qid}")
        if qid not in step0_rollout:
            raise ValueError(f"step0 rollout 中找不到 qid: {qid}")
        if qid not in terminal:
            raise ValueError(f"terminal 中找不到 qid: {qid}")

        sample = build_one_sample(
            qid=qid,
            t=0,
            question=questions[qid],
            state_record=init_states[qid],
            rollout_record=step0_rollout[qid],
            teacher_record=step0_teacher[qid],
            terminal_record=terminal[qid],
        )
        records.append(sample)
        count_t0 += 1
        if sample["stop_label"] == 1:
            stop1 += 1
        else:
            stop0 += 1

    # t = 1 samples
    for qid in sorted(step1_teacher.keys()):
        if qid not in questions:
            raise ValueError(f"processed 中找不到 qid: {qid}")
        if qid not in step1_states:
            raise ValueError(f"step1 state 中找不到 qid: {qid}")
        if qid not in step1_rollout:
            raise ValueError(f"step1 rollout 中找不到 qid: {qid}")
        if qid not in terminal:
            raise ValueError(f"terminal 中找不到 qid: {qid}")

        sample = build_one_sample(
            qid=qid,
            t=1,
            question=questions[qid],
            state_record=step1_states[qid],
            rollout_record=step1_rollout[qid],
            teacher_record=step1_teacher[qid],
            terminal_record=terminal[qid],
        )
        records.append(sample)
        count_t1 += 1
        if sample["stop_label"] == 1:
            stop1 += 1
        else:
            stop0 += 1

    records.sort(key=lambda x: (x["qid"], x["t"]))

    written = write_jsonl(records, output_path)
    if written != len(records):
        raise RuntimeError(
            f"写入条数异常: file={output_path}, written={written}, expected={len(records)}"
        )

    return {
        "samples": len(records),
        "t0_samples": count_t0,
        "t1_samples": count_t1,
        "stop_label_1": stop1,
        "stop_label_0": stop0,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    processed_dir = base_dir / "processed"
    init_state_dir = base_dir / "init_state"
    states_dir = base_dir / "states"
    rollout_dir = base_dir / "rollout"
    teacher_dir = base_dir / "teacher"
    terminal_dir = base_dir / "terminal"
    samples_dir = base_dir / "samples"

    processed_files = {
        "train": processed_dir / "train.jsonl",
        "val": processed_dir / "val.jsonl",
        "test": processed_dir / "test.jsonl",
    }
    init_state_files = {
        "train": init_state_dir / "train.jsonl",
        "val": init_state_dir / "val.jsonl",
        "test": init_state_dir / "test.jsonl",
    }
    step1_state_files = {
        "train": states_dir / "step1_train.jsonl",
        "val": states_dir / "step1_val.jsonl",
        "test": states_dir / "step1_test.jsonl",
    }
    step0_rollout_files = {
        "train": rollout_dir / "step0_train.jsonl",
        "val": rollout_dir / "step0_val.jsonl",
        "test": rollout_dir / "step0_test.jsonl",
    }
    step1_rollout_files = {
        "train": rollout_dir / "step1_train.jsonl",
        "val": rollout_dir / "step1_val.jsonl",
        "test": rollout_dir / "step1_test.jsonl",
    }
    step0_teacher_files = {
        "train": teacher_dir / "step0_train.jsonl",
        "val": teacher_dir / "step0_val.jsonl",
        "test": teacher_dir / "step0_test.jsonl",
    }
    step1_teacher_files = {
        "train": teacher_dir / "step1_train.jsonl",
        "val": teacher_dir / "step1_val.jsonl",
        "test": teacher_dir / "step1_test.jsonl",
    }
    terminal_files = {
        "train": terminal_dir / "train.jsonl",
        "val": terminal_dir / "val.jsonl",
        "test": terminal_dir / "test.jsonl",
    }
    output_files = {
        "train": samples_dir / "train.jsonl",
        "val": samples_dir / "val.jsonl",
        "test": samples_dir / "test.jsonl",
    }

    for split in ["train", "val", "test"]:
        required = [
            processed_files[split],
            init_state_files[split],
            step0_rollout_files[split],
            step0_teacher_files[split],
            terminal_files[split],
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

    samples_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = build_samples_split(
            processed_path=processed_files[split],
            init_state_path=init_state_files[split],
            step1_state_path=step1_state_files[split],
            step0_rollout_path=step0_rollout_files[split],
            step1_rollout_path=step1_rollout_files[split],
            step0_teacher_path=step0_teacher_files[split],
            step1_teacher_path=step1_teacher_files[split],
            terminal_path=terminal_files[split],
            output_path=output_files[split],
        )
        all_stats[split] = stats

    print("HotpotQA final prefix samples 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"samples={stats['samples']}, "
            f"t0_samples={stats['t0_samples']}, "
            f"t1_samples={stats['t1_samples']}, "
            f"stop_label_1={stats['stop_label_1']}, "
            f"stop_label_0={stats['stop_label_0']}, "
            f"output={output_files[split]}"
        )


if __name__ == "__main__":
    main()