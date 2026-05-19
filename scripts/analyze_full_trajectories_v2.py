import json
import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v2"
OUTPUT_NAME = "full_trajectory_analysis_v2.json"
REPEAT_RATIO_THRESHOLD = 0.75
TAIL_N = 3


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_run_id_from_records(records: List[dict], *, source_name: str, split: str) -> str:
    run_ids = set()
    for row in records:
        meta = row.get("build_meta")
        if not isinstance(meta, dict) or not isinstance(meta.get("run_id"), str) or not meta["run_id"].strip():
            raise ValueError(f"{source_name} 缺少 build_meta.run_id: split={split}, qid={row.get('qid')}")
        run_ids.add(meta["run_id"].strip())
    if len(run_ids) != 1:
        raise ValueError(f"mixed_run_id_across_outputs: source={source_name}, split={split}, run_ids={sorted(run_ids)}")
    return next(iter(run_ids))


def is_stalled_record(record: dict) -> bool:
    terminal_status = str(record.get("terminal_status", "")).strip().lower()
    abort_reason = str(record.get("abort_reason", "")).strip().lower()
    if terminal_status in {"failed_stalled", "stalled"}:
        return True
    if terminal_status == "abort" and any(k in abort_reason for k in ["stalled", "no_progress", "oscillation"]):
        return True
    return False


def trajectory_bucket(record: dict) -> str:
    terminal_status = str(record.get("terminal_status", "")).strip().lower()
    if terminal_status == "terminal":
        return "terminal"
    if is_stalled_record(record):
        return "stalled"
    return "abort"


def normalize_answer_match_rule(value: Any) -> str:
    x = str(value or "").strip().lower()
    if x in {"normalized_exact", "context_exact", "none"}:
        return x
    return "none"


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


def tail_values(steps: List[dict], getter, n: int = TAIL_N) -> List[Any]:
    return [getter(step) for step in steps[-n:] if isinstance(step, dict)]


def get_probe_records(record: dict) -> List[dict]:
    history = record.get("stop_probe_history")
    if isinstance(history, list) and history:
        return [probe for probe in history if isinstance(probe, dict)]
    terminal_probe = record.get("terminal_probe")
    if isinstance(terminal_probe, dict):
        return [terminal_probe]
    return []


def count_stop_candidate_steps(steps: List[dict]) -> int:
    count = 0
    for step in steps:
        value = get_nested(step, ("stop_candidate",), ("stop_debug", "stop_candidate"))
        if isinstance(value, bool) and value:
            count += 1
    return count


def classify_stalled(record: dict) -> Tuple[str, str, Dict[str, Any]]:
    steps = [step for step in record.get("steps", []) if isinstance(step, dict)]
    ever_progress = bool(record.get("ever_progress", False))
    stop_probe_count = int(record.get("stop_probe_count", 0) or 0)
    false_stop_count = int(record.get("final_false_stop_count", 0) or 0)
    stop_candidate_count = count_stop_candidate_steps(steps)
    last_positive_ids = tail_values(steps, lambda step: str(step.get("positive_unit_id", "")))
    last_delta_covered = tail_values(
        steps,
        lambda step: int(
            get_nested(step, ("delta_covered_targets",), ("coverage_debug", "delta_covered_targets")) or 0
        ),
    )
    last_retrieval_repeat_ratio = tail_values(
        steps,
        lambda step: to_float(
            get_nested(
                step,
                ("retrieval_repeat_ratio",),
                ("candidate_debug", "retrieval_repeat_ratio"),
                ("stop_debug", "retrieval_repeat_ratio"),
            )
        ),
    )
    last_need_derived = tail_values(
        steps,
        lambda step: to_bool(
            get_nested(
                step,
                ("need_derived",),
                ("candidate_debug", "need_derived"),
            )
        ),
    )
    last_triggered = tail_values(
        steps,
        lambda step: to_bool(
            get_nested(
                step,
                ("triggered_propose_derived",),
                ("candidate_debug", "triggered_propose_derived"),
            )
        ),
    )
    last_stop_candidate = tail_values(
        steps,
        lambda step: to_bool(get_nested(step, ("stop_candidate",), ("stop_debug", "stop_candidate"))),
    )
    covered_counts = tail_values(
        steps,
        lambda step: int(
            get_nested(
                step,
                ("covered_target_count",),
                ("coverage_debug", "covered_target_count"),
            )
            or 0
        ),
    )
    target_count = to_int(record.get("target_count"))
    coverage_ratio = None
    if target_count and covered_counts:
        coverage_ratio = covered_counts[-1] / target_count

    tail_plateau = bool(last_delta_covered) and all(value <= 0 for value in last_delta_covered)
    tail_high_repeat = any(value is not None and value >= REPEAT_RATIO_THRESHOLD for value in last_retrieval_repeat_ratio)
    tail_positive_repeat = len(set(x for x in last_positive_ids if x)) < len([x for x in last_positive_ids if x])
    need_derived_opened = any(last_need_derived)
    derived_triggered = any(last_triggered)
    stop_candidate_closed = (not last_stop_candidate) or (not any(last_stop_candidate))

    if not ever_progress:
        cause = "stalled_without_progress"
        reason = "Coverage never started to grow, so the issue is in early retrieval or teacher selection."
    elif tail_plateau and tail_high_repeat and tail_positive_repeat:
        cause = "stalled_retrieval_looping"
        reason = "Coverage plateaued while retrieval and positive selection repeated highly similar candidates."
    elif (
        tail_plateau
        and stop_probe_count == 0
        and stop_candidate_closed
        and coverage_ratio is not None
        and coverage_ratio >= 0.8
    ):
        cause = "stalled_stop_gate_too_conservative"
        reason = "Coverage was already high, but stop_candidate still never opened."
    elif tail_plateau and stop_probe_count == 0 and stop_candidate_closed and not need_derived_opened:
        cause = "stalled_need_derived_not_opened"
        reason = "Coverage plateaued, stop never opened, and need_derived stayed closed."
    elif tail_plateau and derived_triggered:
        cause = "stalled_need_derived_opened_but_no_effect"
        reason = "Derived proposals were triggered, but the tail still showed no new coverage."
    else:
        cause = "stalled_after_progress"
        reason = "The trajectory made progress earlier, then flattened before reaching a valid terminal stop."

    extras = {
        "ever_progress": ever_progress,
        "stop_probe_count": stop_probe_count,
        "false_stop_count": false_stop_count,
        "stop_candidate_count": stop_candidate_count,
        "last_positive_unit_ids": last_positive_ids,
        "last_delta_covered_targets": last_delta_covered,
        "last_retrieval_repeat_ratio": last_retrieval_repeat_ratio,
        "last_need_derived": last_need_derived,
        "last_triggered_propose_derived": last_triggered,
        "coverage_ratio": coverage_ratio,
    }
    return cause, reason, extras


def summarize_split(split: str, records: List[dict]) -> Tuple[dict, List[dict]]:
    total = len(records)
    bucket_counts = Counter(trajectory_bucket(record) for record in records)
    terminal_answer_match_counts = Counter()
    all_probe_answer_match_counts = Counter()
    stalled_rows: List[dict] = []
    repair_continuation_stats = Counter()

    for record in records:
        if trajectory_bucket(record) == "terminal":
            terminal_probe = record.get("terminal_probe")
            if isinstance(terminal_probe, dict):
                terminal_answer_match_counts[normalize_answer_match_rule(terminal_probe.get("answer_match_rule"))] += 1
        for probe in get_probe_records(record):
            all_probe_answer_match_counts[normalize_answer_match_rule(probe.get("answer_match_rule"))] += 1
        for step in record.get("steps", []):
            if not isinstance(step, dict):
                continue
            candidate_debug = step.get("candidate_debug", {})
            if not isinstance(candidate_debug, dict):
                continue
            if candidate_debug.get("closure_continuation_active", False):
                repair_continuation_stats["closure_continuation_steps"] += 1
            if candidate_debug.get("candidate_pool_had_repair_linked_options", False):
                repair_continuation_stats["steps_with_repair_linked_pool"] += 1
            if candidate_debug.get("repair_linked_candidate_dropped_from_pool", False):
                repair_continuation_stats["repair_linked_dropped_from_pool"] += 1
            if candidate_debug.get("closure_stop_reopened", False):
                repair_continuation_stats["closure_stop_reopened"] += 1
        if is_stalled_record(record):
            cause, reason, extras = classify_stalled(record)
            stalled_rows.append(
                {
                    "split": split,
                    "qid": str(record.get("qid", "")),
                    "main_cause": cause,
                    "reason": reason,
                    **extras,
                }
            )

    summary = {
        "split": split,
        "total": total,
        "status_counts": {
            "terminal": bucket_counts.get("terminal", 0),
            "abort": bucket_counts.get("abort", 0),
            "stalled": bucket_counts.get("stalled", 0),
        },
        "status_ratios": {
            "terminal": bucket_counts.get("terminal", 0) / total if total else 0.0,
            "abort": bucket_counts.get("abort", 0) / total if total else 0.0,
            "stalled": bucket_counts.get("stalled", 0) / total if total else 0.0,
        },
        "answer_match_rule": {
            "terminal_only": {
                "normalized_exact": terminal_answer_match_counts.get("normalized_exact", 0),
                "context_exact": terminal_answer_match_counts.get("context_exact", 0),
                "none": terminal_answer_match_counts.get("none", 0),
            },
            "all_stop_probes": {
                "normalized_exact": all_probe_answer_match_counts.get("normalized_exact", 0),
                "context_exact": all_probe_answer_match_counts.get("context_exact", 0),
                "none": all_probe_answer_match_counts.get("none", 0),
            },
        },
        "repair_continuation_stats": {
            "closure_continuation_steps": repair_continuation_stats.get("closure_continuation_steps", 0),
            "steps_with_repair_linked_pool": repair_continuation_stats.get("steps_with_repair_linked_pool", 0),
            "repair_linked_dropped_from_pool": repair_continuation_stats.get("repair_linked_dropped_from_pool", 0),
            "closure_stop_reopened": repair_continuation_stats.get("closure_stop_reopened", 0),
        },
    }
    return summary, stalled_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze hotpotqa full trajectories v2")
    parser.add_argument("--force", action="store_true", help="Accepted for compatibility; overwrites the fixed analysis output file")
    parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    trajectories_dir = project_root / DEFAULT_BASE / "trajectories"
    debug_dir = project_root / DEFAULT_BASE / "debug"
    output_path = debug_dir / OUTPUT_NAME
    debug_manifest_path = debug_dir / "build_manifest_v2.json"

    split_summaries: List[dict] = []
    stalled_rows: List[dict] = []
    success_debug_summaries: List[dict] = []
    full_run_ids: Dict[str, str] = {}
    success_counts_by_split: Dict[str, int] = {}
    failure_counts_by_split: Dict[str, int] = {}

    if not debug_manifest_path.exists():
        raise FileNotFoundError(f"analysis_input_not_from_same_full_batch: 缺少 debug manifest: {debug_manifest_path}")
    debug_manifest = read_json(debug_manifest_path)
    debug_types = debug_manifest.get("debug_types", {})
    if not isinstance(debug_types, dict):
        raise ValueError("analysis_input_not_from_same_full_batch: debug manifest 缺少 debug_types")

    for split in SPLITS:
        full_path = trajectories_dir / f"full_{split}.jsonl"
        success_debug_path = debug_dir / f"success_semantic_debug_{split}.jsonl"
        failure_debug_path = debug_dir / f"failure_semantic_debug_{split}.jsonl"
        if not full_path.exists():
            raise FileNotFoundError(f"找不到 full trajectory 文件: {full_path}")
        if not success_debug_path.exists():
            raise FileNotFoundError(f"找不到 success debug 文件: {success_debug_path}")
        if not failure_debug_path.exists():
            raise FileNotFoundError(f"找不到 failure debug 文件: {failure_debug_path}")

        records = list(read_jsonl(full_path))
        full_run_ids[split] = extract_run_id_from_records(records, source_name="full", split=split)
        summary, stalled = summarize_split(split, records)
        split_summaries.append(summary)
        stalled_rows.extend(stalled)

        success_records = list(read_jsonl(success_debug_path))
        failure_records = list(read_jsonl(failure_debug_path))
        success_counts_by_split[split] = len(success_records)
        failure_counts_by_split[split] = len(failure_records)

        success_manifest = debug_types.get("success", {})
        failure_manifest = debug_types.get("failure", {})
        success_manifest_run_id = (((success_manifest if isinstance(success_manifest, dict) else {}).get("run_id_by_split", {}) or {}).get(split))
        failure_manifest_run_id = (((failure_manifest if isinstance(failure_manifest, dict) else {}).get("run_id_by_split", {}) or {}).get(split))
        if success_manifest_run_id != full_run_ids[split]:
            raise ValueError(
                f"analysis_input_not_from_same_full_batch: split={split}, full_run_id={full_run_ids[split]}, "
                f"success_debug_run_id={success_manifest_run_id}"
            )
        if failure_manifest_run_id != full_run_ids[split]:
            raise ValueError(
                f"analysis_input_not_from_same_full_batch: split={split}, full_run_id={full_run_ids[split]}, "
                f"failure_debug_run_id={failure_manifest_run_id}"
            )

        if success_records:
            success_run_id = extract_run_id_from_records(success_records, source_name="success_debug", split=split)
            if success_run_id != full_run_ids[split]:
                raise ValueError(
                    f"mixed_run_id_across_outputs: split={split}, full_run_id={full_run_ids[split]}, success_run_id={success_run_id}"
                )
        if failure_records:
            failure_run_id = extract_run_id_from_records(failure_records, source_name="failure_debug", split=split)
            if failure_run_id != full_run_ids[split]:
                raise ValueError(
                    f"mixed_run_id_across_outputs: split={split}, full_run_id={full_run_ids[split]}, failure_run_id={failure_run_id}"
                )

        expected_success = summary["status_counts"]["terminal"]
        expected_failure = summary["total"] - expected_success
        if len(success_records) != expected_success:
            raise ValueError(
                f"success_count_mismatch: split={split}, full_terminal={expected_success}, success_debug={len(success_records)}"
            )
        if len(failure_records) != expected_failure:
            raise ValueError(
                f"failure_count_mismatch: split={split}, full_nonterminal={expected_failure}, failure_debug={len(failure_records)}"
            )
        if summary["status_counts"]["stalled"] > 0 and len(success_records) == summary["total"]:
            raise ValueError(
                f"terminal_analysis_conflicts_with_debug: split={split}, total={summary['total']}, stalled={summary['status_counts']['stalled']}, success_debug={len(success_records)}"
            )
        if summary["status_counts"]["stalled"] == 0 and len(failure_records) == summary["total"]:
            raise ValueError(
                f"stalled_analysis_conflicts_with_debug: split={split}, total={summary['total']}, failure_debug={len(failure_records)}"
            )

        success_debug_summaries.append(
            {
                "split": split,
                "run_id": full_run_ids[split],
                "success_records": len(success_records),
                "failure_records": len(failure_records),
                "warning_count": sum(1 for row in success_records if row.get("debug_warnings")),
            }
        )

    unique_full_run_ids = set(full_run_ids.values())
    if len(unique_full_run_ids) != 1:
        raise ValueError(f"mixed_run_id_across_outputs: full run_ids across splits do not match: {sorted(unique_full_run_ids)}")
    run_id = next(iter(unique_full_run_ids))

    report = {
        "build_meta": {
            "run_id": run_id,
            "source": "analyze_hotpotqa_full_trajectories_v2.py",
        },
        "split_summaries": split_summaries,
        "stalled_cases": stalled_rows,
        "success_debug_summaries": success_debug_summaries,
    }
    write_json(report, output_path)

    stale_output = trajectories_dir / OUTPUT_NAME
    if stale_output.exists():
        stale_output.unlink()

    print(f"full trajectory analysis written to {output_path}")
    print(f"run_id={run_id}")
    for summary in split_summaries:
        counts = summary["status_counts"]
        ratios = summary["status_ratios"]
        match = summary["answer_match_rule"]
        print(
            f"[{summary['split']}] total={summary['total']} "
            f"terminal={counts['terminal']} ({ratios['terminal']:.2%}) "
            f"abort={counts['abort']} ({ratios['abort']:.2%}) "
            f"stalled={counts['stalled']} ({ratios['stalled']:.2%})"
        )
        print(f"  answer_match_rule.terminal_only={match['terminal_only']}")
        print(f"  answer_match_rule.all_stop_probes={match['all_stop_probes']}")

    for row in stalled_rows:
        print(f"[{row['split']}] qid={row['qid']}")
        print(f"  ever_progress={row['ever_progress']}")
        print(f"  stop_probe_count={row['stop_probe_count']}")
        print(f"  false_stop_count={row['false_stop_count']}")
        print(f"  last_positive_unit_ids={row['last_positive_unit_ids']}")
        print(f"  last_delta_covered_targets={row['last_delta_covered_targets']}")
        print(f"  last_retrieval_repeat_ratio={row['last_retrieval_repeat_ratio']}")
        print(f"  last_need_derived={row['last_need_derived']}")
        print(f"  last_triggered_propose_derived={row['last_triggered_propose_derived']}")
        print(f"  final_cause={row['main_cause']}")
        print(f"  reason={row['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
