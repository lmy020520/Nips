import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v4"
OUTPUT_NAME = "stalled_analysis_v3.jsonl"
TAIL_N = 3
REPEAT_RATIO_THRESHOLD = 0.75

FOCUS_QIDS = {
    "val": {
        "5adce4d35542992c1e3a2473",
        "5ae536065542990ba0bbb227",
        "5ae690fd55429908198fa624",
    },
    "test": {
        "5ae5fc345542993aec5ec1f8",
    },
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: List[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def is_stalled_record(record: dict) -> bool:
    terminal_status = str(record.get("terminal_status", "")).strip().lower()
    abort_reason = str(record.get("abort_reason", "")).strip().lower()
    if terminal_status in {"failed_stalled", "stalled"}:
        return True
    if terminal_status == "abort" and any(k in abort_reason for k in ["stalled", "no_progress", "oscillation"]):
        return True
    return False


def get_nested(step: dict, *paths: Tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = step
        ok = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                ok = False
                break
            current = current[key]
        if ok:
            return current
    return None


def tail_values(steps: List[dict], getter, n: int = TAIL_N) -> List[Any]:
    values = []
    for step in steps[-n:]:
        if not isinstance(step, dict):
            continue
        values.append(getter(step))
    return values


def to_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def to_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def last_positive_ids(steps: List[dict]) -> List[str]:
    return tail_values(steps, lambda step: str(step.get("positive_unit_id", "")))


def last_delta_covered_targets(steps: List[dict]) -> List[int]:
    values = []
    for value in tail_values(
        steps,
        lambda step: get_nested(
            step,
            ("delta_covered_targets",),
            ("coverage_debug", "delta_covered_targets"),
        ),
    ):
        values.append(int(value) if isinstance(value, int) and not isinstance(value, bool) else 0)
    return values


def last_retrieval_repeat_ratio(steps: List[dict]) -> List[Optional[float]]:
    return tail_values(
        steps,
        lambda step: to_float(
            get_nested(
                step,
                ("candidate_debug", "retrieval_repeat_ratio"),
                ("retrieval_repeat_ratio",),
                ("stop_debug", "retrieval_repeat_ratio"),
            )
        ),
    )


def last_need_derived(steps: List[dict]) -> List[bool]:
    return tail_values(
        steps,
        lambda step: to_bool(
            get_nested(
                step,
                ("candidate_debug", "need_derived"),
                ("derived_debug", "need_derived"),
                ("need_derived",),
            )
        ),
    )


def last_triggered_propose_derived(steps: List[dict]) -> List[bool]:
    return tail_values(
        steps,
        lambda step: to_bool(
            get_nested(
                step,
                ("candidate_debug", "triggered_propose_derived"),
                ("derived_debug", "triggered_propose_derived"),
                ("triggered_propose_derived",),
            )
        ),
    )


def tail_covered_counts(steps: List[dict]) -> List[int]:
    values = []
    for value in tail_values(
        steps,
        lambda step: get_nested(
            step,
            ("covered_target_count",),
            ("coverage_debug", "covered_target_count"),
        ),
    ):
        values.append(int(value) if isinstance(value, int) and not isinstance(value, bool) else 0)
    return values


def count_stop_candidate_steps(steps: List[dict]) -> int:
    count = 0
    for step in steps:
        value = get_nested(step, ("stop_debug", "stop_candidate"), ("stop_candidate",))
        if isinstance(value, bool) and value:
            count += 1
    return count


def compute_stop_probe_count(record: dict, steps: List[dict]) -> int:
    if isinstance(record.get("stop_probe_count"), int):
        return int(record["stop_probe_count"])
    count = 0
    for step in steps:
        probe = get_nested(step, ("stop_debug", "probe"))
        if isinstance(probe, dict):
            count += 1
    return count


def compute_false_stop_count(record: dict, steps: List[dict]) -> int:
    if isinstance(record.get("final_false_stop_count"), int):
        return int(record["final_false_stop_count"])
    last_value = 0
    for step in steps:
        value = get_nested(step, ("stop_debug", "false_stop_count"), ("stop_debug", "false_stop_count_after"))
        if isinstance(value, int) and not isinstance(value, bool):
            last_value = int(value)
    return last_value


def positive_repetition(last_positive: List[str]) -> bool:
    positives = [x for x in last_positive if x]
    if len(positives) < 2:
        return False
    return len(set(positives)) < len(positives)


def high_repeat(last_ratios: List[Optional[float]]) -> bool:
    return any(value is not None and value >= REPEAT_RATIO_THRESHOLD for value in last_ratios)


def plateau(last_delta: List[int]) -> bool:
    return bool(last_delta) and all(value <= 0 for value in last_delta)


def coverage_ratio(record: dict, covered_counts: List[int]) -> Optional[float]:
    target_count = record.get("target_count")
    if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count <= 0:
        return None
    if not covered_counts:
        return 0.0
    return covered_counts[-1] / target_count


def classify_stalled(record: dict) -> Tuple[str, str, Dict[str, Any]]:
    steps = [step for step in record.get("steps", []) if isinstance(step, dict)]
    ever_progress = bool(record.get("ever_progress", False))
    stop_probe_count = compute_stop_probe_count(record, steps)
    false_stop_count = compute_false_stop_count(record, steps)
    stop_candidate_count = count_stop_candidate_steps(steps)
    last_positive = last_positive_ids(steps)
    last_delta = last_delta_covered_targets(steps)
    last_repeat = last_retrieval_repeat_ratio(steps)
    last_need = last_need_derived(steps)
    last_triggered = last_triggered_propose_derived(steps)
    covered_counts = tail_covered_counts(steps)
    cov_ratio = coverage_ratio(record, covered_counts)

    tail_plateau = plateau(last_delta)
    tail_high_repeat = high_repeat(last_repeat)
    tail_positive_repeat = positive_repetition(last_positive)
    need_derived_opened = any(last_need)
    derived_triggered = any(last_triggered)

    if not ever_progress:
        cause = "stalled_without_progress"
        reason = "Coverage never started to grow, so the retrieval/selection chain failed before terminal gating mattered."
    elif tail_plateau and derived_triggered:
        cause = "stalled_need_derived_opened_but_no_effect"
        reason = "Derived proposals were opened, but the tail still had zero coverage gain and the positive choices did not recover progress."
    elif (
        tail_plateau
        and stop_probe_count == 0
        and stop_candidate_count == 0
        and cov_ratio is not None
        and cov_ratio >= 0.85
    ):
        cause = "stalled_stop_gate_too_conservative"
        reason = "Coverage was already close to terminal, but stop_candidate never fired, so the stop gate stayed too conservative."
    elif (
        tail_plateau
        and stop_probe_count == 0
        and not need_derived_opened
        and cov_ratio is not None
        and cov_ratio >= 0.67
        and (not tail_positive_repeat)
    ):
        cause = "stalled_need_derived_not_opened"
        reason = "Coverage reached a late plateau, yet need_derived stayed closed and no stop probe was ever attempted."
    else:
        cause = "stalled_after_progress"
        reason = "The trajectory made progress earlier, then coverage flattened while retrieval and positive selection became repetitive."

    extras = {
        "stop_probe_count": stop_probe_count,
        "false_stop_count": false_stop_count,
        "stop_candidate_count": stop_candidate_count,
        "last_positive_ids": last_positive,
        "last_delta_covered": last_delta,
        "last_retrieval_repeat_ratio": last_repeat,
        "last_need_derived": last_need,
        "last_triggered_propose_derived": last_triggered,
        "tail_covered_counts": covered_counts,
        "coverage_ratio": cov_ratio,
        "tail_plateau": tail_plateau,
        "tail_high_repeat": tail_high_repeat,
        "tail_positive_repeat": tail_positive_repeat,
    }
    return cause, reason, extras


def analyze_split(split: str, path: Path) -> List[dict]:
    rows = []
    for record in read_jsonl(path):
        if not is_stalled_record(record):
            continue
        qid = str(record.get("qid", ""))
        cause, reason, extras = classify_stalled(record)
        rows.append(
            {
                "split": split,
                "qid": qid,
                "focus_qid": qid in FOCUS_QIDS.get(split, set()),
                "ever_progress": bool(record.get("ever_progress", False)),
                "final_status": str(record.get("terminal_status", "")),
                "abort_reason": record.get("abort_reason"),
                "main_cause": cause,
                "reason": reason,
                **extras,
            }
        )
    rows.sort(key=lambda row: (0 if row["focus_qid"] else 1, row["split"], row["qid"]))
    return rows


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    trajectories_dir = project_root / DEFAULT_BASE / "trajectories"
    output_path = trajectories_dir / OUTPUT_NAME

    all_rows: List[dict] = []
    for split in SPLITS:
        full_path = trajectories_dir / f"full_{split}.jsonl"
        if not full_path.exists():
            raise FileNotFoundError(f"找不到 full trajectory 文件: {full_path}")
        all_rows.extend(analyze_split(split, full_path))

    write_jsonl(all_rows, output_path)

    print(f"stalled analysis written to {output_path}")
    for row in all_rows:
        print(f"[{row['split']}] qid={row['qid']}")
        print(f"  ever_progress={row['ever_progress']}")
        print(f"  stop_probe_count={row['stop_probe_count']}")
        print(f"  false_stop_count={row['false_stop_count']}")
        print(f"  last_positive_ids={row['last_positive_ids']}")
        print(f"  last_delta_covered={row['last_delta_covered']}")
        print(f"  last_retrieval_repeat_ratio={row['last_retrieval_repeat_ratio']}")
        print(f"  last_need_derived={row['last_need_derived']}")
        print(f"  last_triggered_propose_derived={row['last_triggered_propose_derived']}")
        print(f"  final_cause={row['main_cause']}")
        print(f"  reason={row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
