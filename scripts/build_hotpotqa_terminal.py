import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


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


def load_processed_answers(processed_path: Path) -> Dict[str, str]:
    answers = {}
    for row_idx, record in enumerate(read_jsonl(processed_path), start=1):
        for field in ["qid", "answer"]:
            if field not in record:
                raise ValueError(
                    f"processed 记录缺少字段: file={processed_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        answer = str(record["answer"]).strip()

        if not answer:
            raise ValueError(f"answer 为空: file={processed_path}, row={row_idx}, qid={qid}")
        if qid in answers:
            raise ValueError(f"processed 中发现重复 qid: file={processed_path}, qid={qid}")

        answers[qid] = answer

    return answers


def load_targets(targets_path: Path) -> Dict[str, Set[str]]:
    targets = {}
    for row_idx, record in enumerate(read_jsonl(targets_path), start=1):
        for field in ["qid", "raw_targets"]:
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

        unit_ids = set()
        for i, item in enumerate(raw_targets):
            if not isinstance(item, dict):
                raise ValueError(f"raw_targets[{i}] 必须是 dict: qid={qid}")
            if "unit_id" not in item:
                raise ValueError(f"raw_targets[{i}] 缺少 unit_id: qid={qid}")

            unit_id = str(item["unit_id"])
            if unit_id in unit_ids:
                raise ValueError(f"raw_targets 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            unit_ids.add(unit_id)

        if not unit_ids:
            raise ValueError(f"raw_targets 为空: qid={qid}")

        targets[qid] = unit_ids

    return targets


def load_init_state_h0(init_state_path: Path) -> Dict[str, List[str]]:
    states = {}
    for row_idx, record in enumerate(read_jsonl(init_state_path), start=1):
        for field in ["qid", "H0"]:
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

        normalized = []
        seen = set()
        for unit_id in h0:
            unit_id = str(unit_id)
            if unit_id in seen:
                raise ValueError(f"H0 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen.add(unit_id)
            normalized.append(unit_id)

        states[qid] = normalized

    return states


def load_state_ht(state_path: Path, expected_t: int) -> Dict[str, List[str]]:
    states = {}
    for row_idx, record in enumerate(read_jsonl(state_path), start=1):
        for field in ["qid", "t", "H_t"]:
            if field not in record:
                raise ValueError(
                    f"state 记录缺少字段: file={state_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in states:
            raise ValueError(f"states 中发现重复 qid: file={state_path}, qid={qid}")

        t = int(record["t"])
        if t != expected_t:
            raise ValueError(f"期望 t={expected_t}，但发现 t={t}: qid={qid}")

        h_t = record["H_t"]
        if not isinstance(h_t, list):
            raise ValueError(f"H_t 必须是 list: qid={qid}")

        normalized = []
        seen = set()
        for unit_id in h_t:
            unit_id = str(unit_id)
            if unit_id in seen:
                raise ValueError(f"H_t 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen.add(unit_id)
            normalized.append(unit_id)

        states[qid] = normalized

    return states


def find_terminal_t(
    qid: str,
    target_set: Set[str],
    h0_map: Dict[str, List[str]],
    h1_map: Dict[str, List[str]],
    h2_map: Dict[str, List[str]],
) -> Optional[int]:
    candidate_states = [
        (0, h0_map.get(qid)),
        (1, h1_map.get(qid)),
        (2, h2_map.get(qid)),
    ]

    for t, h in candidate_states:
        if h is None:
            continue
        if target_set.issubset(set(h)):
            return t

    return None


def build_terminal_split(
    processed_path: Path,
    targets_path: Path,
    init_state_path: Path,
    step1_state_path: Path,
    step2_state_path: Path,
    output_path: Path,
) -> Dict[str, int]:
    answers = load_processed_answers(processed_path)
    targets = load_targets(targets_path)
    h0_map = load_init_state_h0(init_state_path)
    h1_map = load_state_ht(step1_state_path, expected_t=1) if step1_state_path.exists() else {}
    h2_map = load_state_ht(step2_state_path, expected_t=2) if step2_state_path.exists() else {}

    total = 0
    stop_1 = 0
    stop_0 = 0

    def record_generator():
        nonlocal total, stop_1, stop_0

        qids = sorted(targets.keys())
        for qid in qids:
            if qid not in answers:
                raise ValueError(f"processed 中找不到 qid: {qid}")
            if qid not in h0_map:
                raise ValueError(f"init_state 中找不到 qid: {qid}")

            terminal_t = find_terminal_t(
                qid=qid,
                target_set=targets[qid],
                h0_map=h0_map,
                h1_map=h1_map,
                h2_map=h2_map,
            )

            stop_label = 1 if terminal_t is not None else 0
            if stop_label == 1:
                stop_1 += 1
            else:
                stop_0 += 1

            total += 1
            yield {
                "qid": qid,
                "terminal_t": terminal_t if terminal_t is not None else -1,
                "stop_label": stop_label,
                "final_answer": answers[qid],
            }

    written = write_jsonl(record_generator(), output_path)
    if written != total:
        raise RuntimeError(
            f"写入条数异常: file={output_path}, written={written}, expected={total}"
        )

    return {
        "records": total,
        "stop_label_1": stop_1,
        "stop_label_0": stop_0,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    processed_dir = base_dir / "processed"
    targets_dir = base_dir / "targets"
    init_state_dir = base_dir / "init_state"
    states_dir = base_dir / "states"
    terminal_dir = base_dir / "terminal"

    processed_files = {
        "train": processed_dir / "train.jsonl",
        "val": processed_dir / "val.jsonl",
        "test": processed_dir / "test.jsonl",
    }
    targets_files = {
        "train": targets_dir / "train.jsonl",
        "val": targets_dir / "val.jsonl",
        "test": targets_dir / "test.jsonl",
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
    step2_state_files = {
        "train": states_dir / "step2_train.jsonl",
        "val": states_dir / "step2_val.jsonl",
        "test": states_dir / "step2_test.jsonl",
    }
    output_files = {
        "train": terminal_dir / "train.jsonl",
        "val": terminal_dir / "val.jsonl",
        "test": terminal_dir / "test.jsonl",
    }

    for split in ["train", "val", "test"]:
        if not processed_files[split].exists():
            raise FileNotFoundError(f"找不到 processed 文件: {processed_files[split]}")
        if not targets_files[split].exists():
            raise FileNotFoundError(f"找不到 targets 文件: {targets_files[split]}")
        if not init_state_files[split].exists():
            raise FileNotFoundError(f"找不到 init_state 文件: {init_state_files[split]}")

    terminal_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = build_terminal_split(
            processed_path=processed_files[split],
            targets_path=targets_files[split],
            init_state_path=init_state_files[split],
            step1_state_path=step1_state_files[split],
            step2_state_path=step2_state_files[split],
            output_path=output_files[split],
        )
        all_stats[split] = stats

    print("HotpotQA terminal 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"records={stats['records']}, "
            f"stop_label_1={stats['stop_label_1']}, "
            f"stop_label_0={stats['stop_label_0']}, "
            f"output={output_files[split]}"
        )


if __name__ == "__main__":
    main()