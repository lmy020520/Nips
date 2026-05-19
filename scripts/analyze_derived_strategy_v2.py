import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_BASE = PROJECT_DIR / "data" / "hotpotqa_distractor_v2"
OUTPUT_JSON = "derived_strategy_analysis_v2.json"
OUTPUT_MD = "derived_strategy_analysis_v2.md"

HISTORICAL_FAILURE_TO_SUCCESS_QIDS = {
    "5a7630cb554299109176e6af",
    "5a82a36a55429954d2e2eb8d",
    "5a7bbe76554299042af8f7d3",
    "5adbcbca5542996e6852523b",
    "5ae0c9f355429906c02dab51",
    "5ae536065542990ba0bbb227",
    "5a8b42be55429949d91db515",
    "5ab678f455429954757d32d3",
    "5ac2e9c45542990b17b154a0",
    "5ae5fc345542993aec5ec1f8",
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def is_derived_unit_id(unit_id: Any) -> bool:
    return "::derived::" in str(unit_id)


def normalize_note_type(value: Any) -> str:
    x = str(value or "").strip()
    return x if x else "unknown"


def position_bucket(t: int, terminal_t: Optional[int]) -> str:
    if t == 0:
        return "early"
    if terminal_t is not None and t == terminal_t:
        return "last"
    return "middle_late"


def load_queries(base_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for split in SPLITS:
        path = base_dir / "queries" / f"{split}.jsonl"
        for row in read_jsonl(path):
            out[str(row["qid"])] = {
                "split": split,
                "question": str(row.get("question", "")).strip(),
                "answer": str(row.get("answer", "")).strip(),
            }
    return out


def load_full_records(base_dir: Path) -> Tuple[Dict[str, dict], str]:
    out: Dict[str, dict] = {}
    run_ids = set()
    for split in SPLITS:
        path = base_dir / "trajectories" / f"full_{split}.jsonl"
        for row in read_jsonl(path):
            qid = str(row["qid"])
            row["_split"] = split
            out[qid] = row
            meta = row.get("build_meta", {})
            run_id = str(meta.get("run_id", "")).strip()
            if run_id:
                run_ids.add(run_id)
    if len(run_ids) != 1:
        raise ValueError(f"full trajectories run_id 不一致: {sorted(run_ids)}")
    return out, next(iter(run_ids))


def load_samples(base_dir: Path) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for split in SPLITS:
        path = base_dir / "samples" / f"{split}.jsonl"
        for row in read_jsonl(path):
            key = (str(row["qid"]), int(row["t"]))
            out[key] = row
    return out


def load_success_debug(base_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for split in SPLITS:
        path = base_dir / "debug" / f"success_semantic_debug_{split}.jsonl"
        for row in read_jsonl(path):
            out[str(row["qid"])] = row
    return out


def resolve_positive_derived_payload(
    *,
    qid: str,
    t: int,
    positive_unit_id: str,
    samples_by_key: Dict[Tuple[str, int], dict],
) -> Dict[str, Any]:
    row = samples_by_key.get((qid, t))
    if not isinstance(row, dict):
        return {}
    payloads = row.get("derived_payloads", {})
    if not isinstance(payloads, dict):
        return {}
    payload = payloads.get(positive_unit_id)
    return payload if isinstance(payload, dict) else {}


def classify_case_mechanism(
    *,
    derived_positive_events: List[dict],
    final_answer_source: str,
) -> str:
    if not derived_positive_events:
        return "raw_or_stop_closure"
    if any(
        event["position_bucket"] == "early" and event["note_type"] == "bridge_note"
        for event in derived_positive_events
    ):
        return "early_bridge_scaffold"
    if all(event["note_type"] == "verification_note" for event in derived_positive_events):
        return "late_verification_repair"
    if "fallback" in final_answer_source:
        return "mixed_with_stop_fallback"
    return "mixed_or_unknown"


def build_event_record(
    *,
    qid: str,
    split: str,
    t: int,
    terminal_t: Optional[int],
    positive_unit_id: str,
    step: dict,
    steps: List[dict],
    payload: Dict[str, Any],
) -> dict:
    note_type = normalize_note_type(payload.get("type"))
    next_step = steps[t + 1] if t + 1 < len(steps) else None
    future_steps = steps[t + 1 :]

    future_raw_positive_ids = [
        str(s.get("positive_unit_id"))
        for s in future_steps
        if not is_derived_unit_id(str(s.get("positive_unit_id", "")))
    ]
    next_raw_positive_id = None
    for s in future_steps:
        candidate = str(s.get("positive_unit_id", ""))
        if candidate and not is_derived_unit_id(candidate):
            next_raw_positive_id = candidate
            break

    current_c_t = [str(x) for x in step.get("C_t", [])]
    current_r_t = [str(x) for x in step.get("R_t", [])]

    next_delta = None
    if isinstance(next_step, dict):
        next_delta = int(next_step.get("coverage_debug", {}).get("delta_covered_targets", 0) or 0)

    return {
        "qid": qid,
        "split": split,
        "t": t,
        "terminal_t": terminal_t,
        "position_bucket": position_bucket(t, terminal_t),
        "positive_unit_id": positive_unit_id,
        "note_type": note_type,
        "note_text": str(payload.get("text", "")).strip() or None,
        "source_unit_ids": [str(x) for x in payload.get("source_unit_ids", []) if str(x).strip()],
        "derive_goal": str(step.get("gate_trace", {}).get("derive_goal", "")).strip() or None,
        "derive_mode": str(step.get("proposer_trace", {}).get("derive_mode", "")).strip() or None,
        "proposer_reason": str(step.get("proposer_trace", {}).get("reason", "")).strip() or None,
        "harvest_count": int(step.get("proposer_trace", {}).get("harvest_count", 0) or 0),
        "current_r_t_size": len(current_r_t),
        "current_c_t_size": len(current_c_t),
        "next_is_raw": bool(next_step and next_raw_positive_id == str(next_step.get("positive_unit_id"))),
        "next_raw_positive_id": next_raw_positive_id,
        "next_raw_positive_in_current_c_t": bool(next_raw_positive_id and next_raw_positive_id in current_c_t),
        "next_raw_positive_in_current_r_t": bool(next_raw_positive_id and next_raw_positive_id in current_r_t),
        "future_raw_positive_ids": future_raw_positive_ids,
        "future_raw_positive_count": len(future_raw_positive_ids),
        "next_delta_covered_targets": next_delta,
    }


def analyze_records(
    *,
    full_by_qid: Dict[str, dict],
    queries_by_qid: Dict[str, dict],
    samples_by_key: Dict[Tuple[str, int], dict],
    success_debug_by_qid: Dict[str, dict],
) -> dict:
    overall_position_counts = Counter()
    overall_type_counts = Counter()
    overall_type_by_position = defaultdict(Counter)
    overall_followup_counts = Counter()
    triggered_but_not_selected_counts = Counter()

    historical_position_counts = Counter()
    historical_type_counts = Counter()
    historical_followup_counts = Counter()

    per_qid_records: List[dict] = []
    historical_case_records: List[dict] = []

    for qid, row in sorted(full_by_qid.items()):
        split = str(row["_split"])
        question = queries_by_qid[qid]["question"]
        gold_answer = queries_by_qid[qid]["answer"]
        steps = [step for step in row.get("steps", []) if isinstance(step, dict)]
        terminal_t = row.get("terminal_t")
        if isinstance(terminal_t, bool) or not isinstance(terminal_t, int):
            terminal_t = None

        derived_positive_events: List[dict] = []
        derived_trigger_steps: List[dict] = []

        for step in steps:
            t = int(step["t"])
            step_positive = str(step.get("positive_unit_id", ""))
            triggered = bool(step.get("derived_debug", {}).get("triggered_propose_derived", False))
            harvest_count = int(step.get("proposer_trace", {}).get("harvest_count", 0) or 0)
            if triggered:
                derived_trigger_steps.append(
                    {
                        "t": t,
                        "harvest_count": harvest_count,
                        "derive_goal": str(step.get("gate_trace", {}).get("derive_goal", "")).strip() or None,
                        "derive_mode": str(step.get("proposer_trace", {}).get("derive_mode", "")).strip() or None,
                        "reason": str(step.get("proposer_trace", {}).get("reason", "")).strip() or None,
                    }
                )
            if not is_derived_unit_id(step_positive):
                continue

            payload = resolve_positive_derived_payload(
                qid=qid,
                t=t,
                positive_unit_id=step_positive,
                samples_by_key=samples_by_key,
            )
            event = build_event_record(
                qid=qid,
                split=split,
                t=t,
                terminal_t=terminal_t,
                positive_unit_id=step_positive,
                step=step,
                steps=steps,
                payload=payload,
            )
            derived_positive_events.append(event)

            overall_position_counts[event["position_bucket"]] += 1
            overall_type_counts[event["note_type"]] += 1
            overall_type_by_position[event["position_bucket"]][event["note_type"]] += 1

            if event["next_is_raw"]:
                overall_followup_counts["derived_then_immediate_raw"] += 1
            if event["future_raw_positive_count"] > 0:
                overall_followup_counts["derived_then_future_raw"] += 1
            if event["next_raw_positive_in_current_c_t"]:
                overall_followup_counts["next_raw_already_visible_in_c_t"] += 1
            else:
                overall_followup_counts["next_raw_not_yet_visible_in_c_t"] += 1

        if derived_trigger_steps and not derived_positive_events:
            triggered_but_not_selected_counts["triggered_but_no_positive_derived"] += 1

        final_answer_source = str(row.get("terminal_probe", {}).get("answer_source", "")).strip() or "unknown"
        final_support_rule = str(row.get("terminal_probe", {}).get("support_rule", "")).strip() or "unknown"
        final_k_t = str(success_debug_by_qid.get(qid, {}).get("terminal_state", {}).get("K_t", ""))
        final_k_t_has_derived = "[derived_note]" in final_k_t or "\nNotes:\n" in final_k_t

        record = {
            "qid": qid,
            "split": split,
            "question": question,
            "gold_answer": gold_answer,
            "terminal_t": terminal_t,
            "derived_positive_count": len(derived_positive_events),
            "derived_positive_ts": [event["t"] for event in derived_positive_events],
            "derived_positive_events": derived_positive_events,
            "derived_trigger_step_count": len(derived_trigger_steps),
            "derived_trigger_steps": derived_trigger_steps,
            "final_answer_source": final_answer_source,
            "final_support_rule": final_support_rule,
            "final_k_t_has_derived": final_k_t_has_derived,
            "case_mechanism": classify_case_mechanism(
                derived_positive_events=derived_positive_events,
                final_answer_source=final_answer_source,
            ),
        }
        per_qid_records.append(record)

        if qid in HISTORICAL_FAILURE_TO_SUCCESS_QIDS:
            historical_case_records.append(record)
            for event in derived_positive_events:
                historical_position_counts[event["position_bucket"]] += 1
                historical_type_counts[event["note_type"]] += 1
                if event["next_is_raw"]:
                    historical_followup_counts["derived_then_immediate_raw"] += 1
                if event["future_raw_positive_count"] > 0:
                    historical_followup_counts["derived_then_future_raw"] += 1
                if event["next_raw_positive_in_current_c_t"]:
                    historical_followup_counts["next_raw_already_visible_in_c_t"] += 1
                else:
                    historical_followup_counts["next_raw_not_yet_visible_in_c_t"] += 1

    return {
        "overall": {
            "trajectory_count": len(per_qid_records),
            "trajectories_with_derived_positive": sum(1 for x in per_qid_records if x["derived_positive_count"] > 0),
            "trajectories_triggered_but_no_positive_derived": triggered_but_not_selected_counts[
                "triggered_but_no_positive_derived"
            ],
            "derived_positive_position_counts": dict(overall_position_counts),
            "derived_positive_type_counts": dict(overall_type_counts),
            "derived_positive_type_by_position": {
                key: dict(counter) for key, counter in overall_type_by_position.items()
            },
            "derived_followup_counts": dict(overall_followup_counts),
        },
        "historical_failure_to_success": {
            "qid_count": len(historical_case_records),
            "derived_positive_position_counts": dict(historical_position_counts),
            "derived_positive_type_counts": dict(historical_type_counts),
            "derived_followup_counts": dict(historical_followup_counts),
            "case_records": historical_case_records,
        },
        "all_case_records": per_qid_records,
    }


def safe_ratio(numer: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return round(numer / denom, 4)


def build_hypothesis_checks(analysis: dict) -> dict:
    overall = analysis["overall"]
    historical = analysis["historical_failure_to_success"]

    overall_pos = Counter(overall["derived_positive_position_counts"])
    overall_types = Counter(overall["derived_positive_type_counts"])
    overall_follow = Counter(overall["derived_followup_counts"])
    overall_total = sum(overall_pos.values())

    historical_pos = Counter(historical["derived_positive_position_counts"])
    historical_types = Counter(historical["derived_positive_type_counts"])
    historical_follow = Counter(historical["derived_followup_counts"])
    historical_total = sum(historical_pos.values())

    overall_middle_late = overall_pos.get("middle_late", 0)
    historical_middle_late = historical_pos.get("middle_late", 0)
    historical_verification = historical_types.get("verification_note", 0)
    historical_bridge = historical_types.get("bridge_note", 0)

    return {
        "overall": {
            "derived_is_usually_middle_late": {
                "supported": overall_middle_late > overall_pos.get("early", 0) + overall_pos.get("last", 0),
                "ratio": safe_ratio(overall_middle_late, overall_total),
                "evidence": {
                    "middle_late": overall_middle_late,
                    "total": overall_total,
                },
            },
            "derived_often_followed_by_raw": {
                "supported": overall_follow.get("derived_then_immediate_raw", 0) > 0,
                "ratio": safe_ratio(
                    overall_follow.get("derived_then_immediate_raw", 0),
                    overall_total,
                ),
                "evidence": {
                    "derived_then_immediate_raw": overall_follow.get("derived_then_immediate_raw", 0),
                    "total": overall_total,
                },
            },
            "next_raw_not_always_visible_before_derived": {
                "supported": overall_follow.get("next_raw_not_yet_visible_in_c_t", 0) > 0,
                "ratio": safe_ratio(
                    overall_follow.get("next_raw_not_yet_visible_in_c_t", 0),
                    overall_total,
                ),
                "evidence": {
                    "next_raw_not_yet_visible_in_c_t": overall_follow.get("next_raw_not_yet_visible_in_c_t", 0),
                    "total": overall_total,
                },
            },
        },
        "historical_failure_to_success": {
            "derived_is_usually_middle_late": {
                "supported": historical_middle_late > historical_pos.get("early", 0) + historical_pos.get("last", 0),
                "ratio": safe_ratio(historical_middle_late, historical_total),
                "evidence": {
                    "middle_late": historical_middle_late,
                    "total": historical_total,
                },
            },
            "verification_dominates_when_derived_enters_winning_path": {
                "supported": historical_verification > historical_bridge,
                "ratio": safe_ratio(historical_verification, historical_total),
                "evidence": {
                    "verification_note": historical_verification,
                    "bridge_note": historical_bridge,
                    "total": historical_total,
                },
            },
            "early_bridge_exists_but_is_not_mainstream": {
                "supported": historical_pos.get("early", 0) > 0 and historical_pos.get("early", 0) < historical_middle_late,
                "ratio": safe_ratio(historical_pos.get("early", 0), historical_total),
                "evidence": {
                    "early": historical_pos.get("early", 0),
                    "middle_late": historical_middle_late,
                    "total": historical_total,
                },
            },
            "some_derived_steps_are_followed_by_new_raw_discovery": {
                "supported": historical_follow.get("next_raw_not_yet_visible_in_c_t", 0) > 0,
                "ratio": safe_ratio(
                    historical_follow.get("next_raw_not_yet_visible_in_c_t", 0),
                    historical_total,
                ),
                "evidence": {
                    "next_raw_not_yet_visible_in_c_t": historical_follow.get("next_raw_not_yet_visible_in_c_t", 0),
                    "total": historical_total,
                },
            },
        },
    }


def build_markdown_report(*, run_id: str, analysis: dict) -> str:
    overall = analysis["overall"]
    historical = analysis["historical_failure_to_success"]
    hypothesis_checks = analysis["hypothesis_checks"]

    lines: List[str] = []
    lines.append("# Derived Strategy Analysis v2")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- trajectory_count: `{overall['trajectory_count']}`")
    lines.append(
        f"- trajectories_with_derived_positive: `{overall['trajectories_with_derived_positive']}`"
    )
    lines.append(
        f"- trajectories_triggered_but_no_positive_derived: `{overall['trajectories_triggered_but_no_positive_derived']}`"
    )
    lines.append("")

    lines.append("## Overall Stats")
    lines.append("")
    lines.append(
        f"- derived_positive_position_counts: `{json.dumps(overall['derived_positive_position_counts'], ensure_ascii=False)}`"
    )
    lines.append(
        f"- derived_positive_type_counts: `{json.dumps(overall['derived_positive_type_counts'], ensure_ascii=False)}`"
    )
    lines.append(
        f"- derived_followup_counts: `{json.dumps(overall['derived_followup_counts'], ensure_ascii=False)}`"
    )
    lines.append("")
    lines.append("## Hypothesis Checks")
    lines.append("")
    for scope in ["overall", "historical_failure_to_success"]:
        lines.append(f"### {scope}")
        for name, payload in hypothesis_checks[scope].items():
            lines.append(
                f"- {name}: supported=`{payload['supported']}`, ratio=`{payload['ratio']}`, evidence=`{json.dumps(payload['evidence'], ensure_ascii=False)}`"
            )
        lines.append("")

    lines.append("## Historical Failure -> Success Focus Set")
    lines.append("")
    lines.append(f"- qid_count: `{historical['qid_count']}`")
    lines.append(
        f"- derived_positive_position_counts: `{json.dumps(historical['derived_positive_position_counts'], ensure_ascii=False)}`"
    )
    lines.append(
        f"- derived_positive_type_counts: `{json.dumps(historical['derived_positive_type_counts'], ensure_ascii=False)}`"
    )
    lines.append(
        f"- derived_followup_counts: `{json.dumps(historical['derived_followup_counts'], ensure_ascii=False)}`"
    )
    lines.append("")

    lines.append("## Case Notes")
    lines.append("")
    for record in historical["case_records"]:
        lines.append(f"### {record['split']} / {record['qid']}")
        lines.append(f"- question: {record['question']}")
        lines.append(f"- case_mechanism: `{record['case_mechanism']}`")
        lines.append(f"- terminal_t: `{record['terminal_t']}`")
        lines.append(f"- final_answer_source: `{record['final_answer_source']}`")
        lines.append(f"- final_support_rule: `{record['final_support_rule']}`")
        lines.append(f"- final_k_t_has_derived: `{record['final_k_t_has_derived']}`")
        lines.append(f"- derived_positive_ts: `{record['derived_positive_ts']}`")
        if record["derived_positive_events"]:
            for event in record["derived_positive_events"]:
                lines.append(
                    f"- derived@t={event['t']}: type=`{event['note_type']}`, position=`{event['position_bucket']}`, "
                    f"goal=`{event['derive_goal']}`, next_is_raw=`{event['next_is_raw']}`, "
                    f"next_raw_in_current_c_t=`{event['next_raw_positive_in_current_c_t']}`"
                )
                if event["note_text"]:
                    lines.append(f"- note_text: {event['note_text']}")
        else:
            lines.append("- derived_positive: none")
        lines.append(f"- derived_trigger_step_count: `{record['derived_trigger_step_count']}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    queries_by_qid = load_queries(base_dir)
    full_by_qid, run_id = load_full_records(base_dir)
    samples_by_key = load_samples(base_dir)
    success_debug_by_qid = load_success_debug(base_dir)

    analysis = analyze_records(
        full_by_qid=full_by_qid,
        queries_by_qid=queries_by_qid,
        samples_by_key=samples_by_key,
        success_debug_by_qid=success_debug_by_qid,
    )
    analysis["hypothesis_checks"] = build_hypothesis_checks(analysis)
    output = {
        "build_meta": {
            "run_id": run_id,
            "source": "analyze_derived_strategy_v2.py",
        },
        **analysis,
    }

    debug_dir = base_dir / "debug"
    json_path = debug_dir / OUTPUT_JSON
    md_path = debug_dir / OUTPUT_MD
    write_json(json_path, output)
    write_text(md_path, build_markdown_report(run_id=run_id, analysis=analysis))
    print(f"derived strategy analysis written: {json_path}")
    print(f"derived strategy analysis written: {md_path}")


if __name__ == "__main__":
    main()
