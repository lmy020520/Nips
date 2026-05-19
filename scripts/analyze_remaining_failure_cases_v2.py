import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_BASE = PROJECT_DIR / "data" / "hotpotqa_distractor_v2"


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_queries(path: Path) -> Dict[str, dict]:
    out = {}
    for record in read_jsonl(path):
        out[str(record["qid"])] = {
            "question": str(record["question"]).strip(),
            "answer": str(record["answer"]).strip(),
        }
    return out


def load_full_index(path: Path) -> Dict[str, dict]:
    return {str(record["qid"]): record for record in read_jsonl(path)}


def load_failure_debug(path: Path) -> Dict[str, dict]:
    return {str(record["qid"]): record for record in read_jsonl(path)}


def infer_case_bottleneck(case: dict) -> Tuple[str, str]:
    attempts = case["repair_attempts"]
    if not attempts:
        if case["failure_type"] == "stop_gate_too_conservative":
            return "stop_gate_not_reopened", "allow_one_more_stop_attempt"
        if case["failure_type"] == "answer_wrong_after_sufficient_evidence":
            return "answer_focus_not_fixed", "strengthen_answer_focus_verification"
        return "no_repair_attempt", "inspect_trigger_path"

    saw_harvest = any(int(a["harvest_candidate_count"]) > 0 for a in attempts)
    saw_legal = any(int(a["legal_candidate_count"]) > 0 for a in attempts)
    saw_final = any(int(a["final_retained_count"]) > 0 for a in attempts)
    saw_considered = any(bool(a["repair_note_considered_in_teacher_select"]) for a in attempts)
    saw_selected = any(bool(a["repair_note_selected_later"]) for a in attempts)
    any_delta = any(int(a["delta_covered_targets_after_repair"]) > 0 for a in attempts)
    dropped_from_pool = any(bool(a.get("repair_linked_candidate_dropped_from_pool")) for a in attempts)
    continuation_ran = any(bool(a.get("did_closure_continuation_run")) for a in attempts)
    continuation_kept_options = any(bool(a.get("did_candidate_pool_keep_repair_linked_options")) for a in attempts)

    if not saw_harvest or not saw_legal:
        return "repair_note_low_closure_value", "improve_repair_prompt_answer_facing"
    if saw_legal and not saw_final:
        return "repair_note_good_but_not_retained", "increase_repair_retained_priority"
    if saw_final and not saw_considered:
        return "repair_note_retained_but_not_used", "inject_repair_note_into_state_summary_and_k_t"
    if dropped_from_pool:
        return "repair_linked_candidate_dropped_from_pool", "preserve_repair_linked_candidates_for_two_steps"
    if saw_considered and not saw_selected:
        return "teacher_select_not_using_repair", "increase_answer_facing_use_of_retained_repair_note"
    if continuation_ran and continuation_kept_options and not any_delta:
        return "closure_continuation_not_catching_answer_hop", "keep_answer_hop_candidates_alive_after_repair_selection"
    if saw_selected and not any_delta:
        return "repair_note_selected_but_no_closure_gain", "improve_bridge_to_answer_candidate"
    return "repair_note_low_closure_value", "improve_repair_prompt_answer_facing"


def infer_repair_use_bottleneck(case: dict) -> str:
    attempts = case["repair_attempts"]
    if not attempts:
        return "not_considered"

    saw_retained = any(bool(a.get("repair_note_retained")) for a in attempts)
    saw_rendered = any(bool(a.get("repair_note_rendered_in_K_t")) for a in attempts)
    saw_summary = any(bool(a.get("repair_note_present_in_state_summary")) for a in attempts)
    saw_considered = any(bool(a.get("repair_note_considered_in_teacher_select")) for a in attempts)

    if not saw_retained or not saw_rendered:
        return "not_rendered"
    if not saw_summary:
        return "not_in_summary"
    if not saw_considered:
        return "not_considered"
    return "considered_but_not_selected"


def build_case_record(
    *,
    split: str,
    qid: str,
    question: str,
    gold_answer: str,
    full_rec: dict,
    failure_rec: dict,
) -> dict:
    steps = list(full_rec.get("steps", []))
    repair_attempts: List[dict] = []
    for idx, step in enumerate(steps):
        repair_debug = step.get("repair_debug", {})
        proposer_trace = step.get("proposer_trace", {})
        candidate_debug = step.get("candidate_debug", {})
        teacher_select_debug = candidate_debug.get("teacher_select_debug", {})
        is_repair = bool(repair_debug.get("repair_attempt_active")) or str(proposer_trace.get("derive_mode", "")) == "repair_after_false_stop"
        if not is_repair:
            continue

        harvest_candidates = list(proposer_trace.get("harvest_candidates", []))
        final_retained = list(candidate_debug.get("G_t_final", []))
        aux_retained = list(candidate_debug.get("G_t_aux", []))
        illegal = list(candidate_debug.get("G_t_illegal", []))
        harvest_candidate_count = int(proposer_trace.get("harvest_count", len(harvest_candidates)) or 0)
        legal_candidate_count = int(
            proposer_trace.get(
                "legal_count",
                candidate_debug.get("legal_candidate_count", len(final_retained) + len(aux_retained)),
            )
            or 0
        )
        final_retained_count = int(candidate_debug.get("final_retained_count", len(final_retained)) or 0)
        aux_retained_count = int(candidate_debug.get("aux_retained_count", len(aux_retained)) or 0)
        illegal_count = int(candidate_debug.get("illegal_count", len(illegal)) or 0)
        next_steps = steps[idx + 1 :]
        next_step = steps[idx + 1] if idx + 1 < len(steps) else {}
        second_next_step = steps[idx + 2] if idx + 2 < len(steps) else {}
        next_selected_ids = [str(s.get("positive_unit_id")) for s in next_steps if s.get("positive_unit_id")]
        selected_later = any(unit_id in next_selected_ids for unit_id in final_retained)
        next_delta = 0
        if idx + 1 < len(steps):
            next_delta = int(next_step.get("coverage_debug", {}).get("delta_covered_targets", 0))
        did_stop_reopen = any(bool(s.get("stop_candidate", False)) for s in next_steps[:2])
        did_answer_focus_improve = any(
            bool((s.get("stop_debug", {}).get("probe") or {}).get("AnswerCorrect_t", False))
            for s in next_steps[:2]
        )
        next_candidate_debug = next_step.get("candidate_debug", {}) if isinstance(next_step, dict) else {}
        second_candidate_debug = second_next_step.get("candidate_debug", {}) if isinstance(second_next_step, dict) else {}
        next_repair_linked = list(next_candidate_debug.get("repair_linked_carryover_candidates", []))
        second_repair_linked = list(second_candidate_debug.get("repair_linked_carryover_candidates", []))
        next_teacher_select = next_candidate_debug.get("teacher_select_debug", {}) if isinstance(next_candidate_debug, dict) else {}

        repair_attempts.append(
            {
                "t": int(step["t"]),
                "derive_mode": str(proposer_trace.get("derive_mode")),
                "derive_goal": str(proposer_trace.get("derive_goal")),
                "top_raw_candidates": list(candidate_debug.get("R_t", []))[:5],
                "bridge_anchors": list(proposer_trace.get("bridge_anchors", [])),
                "harvest_candidate_count": harvest_candidate_count,
                "legal_candidate_count": legal_candidate_count,
                "final_retained_count": final_retained_count,
                "aux_retained_count": aux_retained_count,
                "illegal_count": illegal_count,
                "harvest_candidates": harvest_candidates,
                "final_retained": final_retained,
                "aux_retained": aux_retained,
                "illegal": illegal,
                "repair_note_retained": bool(repair_debug.get("repair_note_retained", bool(final_retained))),
                "repair_note_used_in_next_state": bool(final_retained),
                "repair_note_rendered_in_K_t": bool(repair_debug.get("repair_note_rendered_in_K_t", False)),
                "repair_note_present_in_state_summary": bool(repair_debug.get("repair_note_present_in_state_summary", False)),
                "repair_note_considered_in_teacher_select": bool(repair_debug.get("repair_note_considered_in_teacher_select", False)),
                "repair_note_selected_later": bool(selected_later),
                "teacher_select_bias_to_repair_applied": bool(
                    repair_debug.get(
                        "teacher_select_bias_to_repair_applied",
                        candidate_debug.get("teacher_select_bias_to_repair_applied", False),
                    )
                ),
                "top_candidates_before_repair_bias": list(
                    teacher_select_debug.get("top_candidates_before_repair_bias", [])
                ),
                "top_candidates_after_repair_bias": list(
                    teacher_select_debug.get("top_candidates_after_repair_bias", [])
                ),
                "selected_candidate": teacher_select_debug.get("selected_candidate"),
                "highest_repair_linked_candidate": teacher_select_debug.get("highest_repair_linked_candidate"),
                "repair_linked_candidate_was_selected": bool(
                    teacher_select_debug.get("repair_linked_candidate_was_selected", False)
                ),
                "selected_repair_candidate": repair_debug.get("selected_repair_candidate"),
                "repair_focus_set_size": int(candidate_debug.get("repair_focus_set_size", 0) or 0),
                "repair_linked_candidates_next_step": next_repair_linked,
                "repair_linked_candidates_next_step_count": len(next_repair_linked),
                "repair_linked_candidates_selected_next_step": bool(
                    next_teacher_select.get("selected_candidate") in set(next_repair_linked)
                ),
                "did_candidate_pool_keep_repair_linked_options": bool(
                    next_candidate_debug.get("candidate_pool_had_repair_linked_options", False)
                ),
                "repair_linked_candidate_survived_next_step": bool(next_repair_linked),
                "repair_linked_candidate_survived_two_steps": bool(second_repair_linked),
                "repair_linked_candidate_dropped_from_pool": bool(
                    next_candidate_debug.get("repair_linked_candidate_dropped_from_pool", False)
                ),
                "did_closure_continuation_run": bool(
                    next_candidate_debug.get("closure_continuation_active", False)
                    or next_step.get("repair_debug", {}).get("closure_continuation_applied", False)
                ),
                "closure_stop_reopened": bool(
                    next_candidate_debug.get("closure_stop_reopened", False)
                    or next_step.get("repair_debug", {}).get("closure_stop_reopened", False)
                ),
                "closure_continuation_failed_reason": str(
                    next_step.get("repair_debug", {}).get("repair_failure_reason", "") or ""
                ),
                "delta_covered_targets_after_repair": int(next_delta),
                "did_stop_candidate_reopen": bool(did_stop_reopen),
                "did_answer_focus_improve": bool(did_answer_focus_improve),
            }
        )

    diagnosis_bottleneck, suggested_fix = infer_case_bottleneck(
        {
            "failure_type": failure_rec["failure_semantic"]["failure_type"],
            "repair_attempts": repair_attempts,
        }
    )

    return {
        "qid": qid,
        "split": split,
        "failure_type": failure_rec["failure_semantic"]["failure_type"],
        "question": question,
        "gold_answer": gold_answer,
        "repair_attempts": repair_attempts,
        "diagnosis": {
            "main_bottleneck": diagnosis_bottleneck,
            "repair_use_bottleneck": infer_repair_use_bottleneck(
                {
                    "failure_type": failure_rec["failure_semantic"]["failure_type"],
                    "repair_attempts": repair_attempts,
                }
            ),
            "suggested_fix": suggested_fix,
        },
    }


def write_markdown(path: Path, records: List[dict], run_id: Optional[str]) -> None:
    lines = ["# Remaining Failure Case Analysis", ""]
    if run_id:
        lines.append(f"- run_id: `{run_id}`")
        lines.append("")
    for record in records:
        lines.append(f"## {record['split']} / {record['qid']}")
        lines.append(f"- failure_type: `{record['failure_type']}`")
        lines.append(f"- question: {record['question']}")
        lines.append(f"- gold_answer: `{record['gold_answer']}`")
        lines.append(f"- bottleneck: `{record['diagnosis']['main_bottleneck']}`")
        lines.append(f"- repair_use_bottleneck: `{record['diagnosis']['repair_use_bottleneck']}`")
        lines.append(f"- suggested_fix: `{record['diagnosis']['suggested_fix']}`")
        if record["repair_attempts"]:
            lines.append("- repair_attempts:")
            for attempt in record["repair_attempts"]:
                lines.append(
                    f"  - t={attempt['t']} mode={attempt['derive_mode']} goal={attempt['derive_goal']} "
                    f"harvest={attempt['harvest_candidate_count']} legal={attempt['legal_candidate_count']} "
                    f"final={attempt['final_retained_count']} aux={attempt['aux_retained_count']} "
                    f"illegal={attempt['illegal_count']} delta_after={attempt['delta_covered_targets_after_repair']} "
                    f"retained={attempt['repair_note_retained']} rendered={attempt['repair_note_rendered_in_K_t']} "
                    f"summary={attempt['repair_note_present_in_state_summary']} "
                    f"considered={attempt['repair_note_considered_in_teacher_select']} "
                    f"bias={attempt['teacher_select_bias_to_repair_applied']} "
                    f"selected_later={attempt['repair_note_selected_later']} "
                    f"selected={attempt['selected_candidate']} "
                    f"best_repair_linked={attempt['highest_repair_linked_candidate']} "
                    f"repair_linked_selected={attempt['repair_linked_candidate_was_selected']} "
                    f"next_pool={attempt['repair_linked_candidates_next_step_count']} "
                    f"next_selected={attempt['repair_linked_candidates_selected_next_step']} "
                    f"continuation={attempt['did_closure_continuation_run']} "
                    f"dropped={attempt['repair_linked_candidate_dropped_from_pool']} "
                    f"closure_stop={attempt['closure_stop_reopened']}"
                )
        else:
            lines.append("- repair_attempts: none")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    queries_dir = base_dir / "queries"
    trajectories_dir = base_dir / "trajectories"
    debug_dir = base_dir / "debug"

    records: List[dict] = []
    run_id: Optional[str] = None

    for split in SPLITS:
        queries = load_queries(queries_dir / f"{split}.jsonl")
        full_index = load_full_index(trajectories_dir / f"full_{split}.jsonl")
        failure_index = load_failure_debug(debug_dir / f"failure_semantic_debug_{split}.jsonl")
        for qid, failure_rec in sorted(failure_index.items()):
            full_rec = full_index[qid]
            query = queries[qid]
            current_run_id = str(full_rec.get("build_meta", {}).get("run_id", "")).strip()
            if run_id is None:
                run_id = current_run_id
            records.append(
                build_case_record(
                    split=split,
                    qid=qid,
                    question=query["question"],
                    gold_answer=query["answer"],
                    full_rec=full_rec,
                    failure_rec=failure_rec,
                )
            )

    jsonl_path = debug_dir / "remaining_failure_case_analysis.jsonl"
    md_path = debug_dir / "remaining_failure_case_analysis.md"
    write_jsonl(jsonl_path, records)
    write_markdown(md_path, records, run_id)
    print(f"remaining failure analysis written: {jsonl_path}")
    print(f"remaining failure analysis written: {md_path}")
    if run_id:
        print(f"run_id={run_id}")
    print(f"cases={len(records)}")


if __name__ == "__main__":
    main()
