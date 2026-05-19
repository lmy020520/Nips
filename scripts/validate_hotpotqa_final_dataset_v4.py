import os
import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = Path(os.environ.get("HOTPOTQA_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "hotpotqa_distractor_v4")))
ALLOWED_STOP_LABEL_TYPES = {"continue", "near-terminal", "terminal", "false-stop", "stop"}
BRIDGE_SUBTYPES = {
    "early_bridge",
    "bridge_scaffold",
    "bridge_contextualization",
    "bridge_scaffold_for_progress",
    "bridge_scaffold_for_new_raw",
}
VERIFICATION_SUBTYPES = {
    "late_verification",
    "answer_facing_verification",
    "abstractive_verification",
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON decode failed: file={path} line={line_idx}") from exc


def is_derived(unit_id: Any) -> bool:
    return isinstance(unit_id, str) and "::derived::" in unit_id


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_run_id_datetime(run_id: str) -> datetime | None:
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_", run_id)
    if not match:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def parse_build_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            pass
    return None


def add_issue(
    issues: List[dict],
    severity: str,
    *,
    split: str,
    qid: str = "",
    t: Any = None,
    issue: str,
    extra: Dict[str, Any] | None = None,
) -> None:
    item = {"severity": severity, "split": split, "qid": qid, "t": t, "issue": issue}
    if extra:
        item.update(extra)
    issues.append(item)


def normalize_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"success", "terminal"}:
        return "terminal"
    if raw in {"abort", "failed", "failed_stalled", "stalled"}:
        return "abort"
    return raw


def subtype_type_consistent(subtype: Any, payload_type: Any) -> bool:
    subtype = str(subtype or "").strip()
    payload_type = str(payload_type or "").strip()
    if payload_type == "bridge_note":
        return subtype in BRIDGE_SUBTYPES
    if payload_type == "verification_note":
        return subtype in VERIFICATION_SUBTYPES
    return True


def load_raw_texts(base: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for split in SPLITS:
        path = base / "unit_registry" / f"raw_units_{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            unit_id = row.get("unit_id")
            text = row.get("text")
            if isinstance(unit_id, str) and isinstance(text, str):
                out[unit_id] = text
    return out


def load_full(base: Path, issues: List[dict]) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, dict], str, str]:
    full_by_split: Dict[str, Dict[str, dict]] = {}
    summary: Dict[str, dict] = {}
    run_ids = set()
    build_times = set()
    required = ["qid", "build_meta", "terminal_status", "terminal_t", "abort_reason", "steps"]
    for split in SPLITS:
        path = base / "trajectories" / f"full_{split}.jsonl"
        if not path.exists():
            add_issue(issues, "CRITICAL", split=split, issue="missing_full_file", extra={"path": str(path)})
            full_by_split[split] = {}
            summary[split] = {"total": 0, "terminal": 0, "abort": 0, "stalled": 0}
            continue
        rows = list(read_jsonl(path))
        full_by_split[split] = {str(row.get("qid")): row for row in rows}
        summary[split] = {
            "total": len(rows),
            "terminal": sum(1 for row in rows if row.get("terminal_status") == "terminal"),
            "abort": sum(1 for row in rows if row.get("terminal_status") == "abort"),
            "stalled": sum(1 for row in rows if row.get("abort_reason") == "stalled"),
        }
        for row in rows:
            qid = str(row.get("qid", ""))
            for key in required:
                if key not in row:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="full_missing_field", extra={"field": key})
            run_id = str((row.get("build_meta") or {}).get("run_id", "")).strip()
            build_time = str((row.get("build_meta") or {}).get("build_time", "")).strip()
            if run_id:
                run_ids.add(run_id)
            if build_time:
                build_times.add(build_time)
            if row.get("terminal_status") != "terminal":
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="full_not_terminal")
            if row.get("abort_reason") is not None:
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="full_abort_reason_not_null")
    if len(run_ids) != 1:
        add_issue(issues, "CRITICAL", split="all", issue="run_id_inconsistent", extra={"run_ids": sorted(run_ids)})
    run_id = next(iter(run_ids)) if len(run_ids) == 1 else ""
    build_time = next(iter(sorted(build_times))) if len(build_times) == 1 else ""
    run_dt = parse_run_id_datetime(run_id)
    build_dt = parse_build_time(build_time)
    if not build_time:
        add_issue(issues, "CRITICAL", split="all", issue="build_time_missing_or_inconsistent")
    elif build_dt is None:
        add_issue(issues, "WARNING", split="all", issue="build_time_invalid", extra={"build_time": build_time})
    elif run_dt is not None and build_dt.date() < run_dt.date():
        add_issue(issues, "WARNING", split="all", issue="build_time_older_than_run_id", extra={"build_time": build_time})
    return full_by_split, summary, run_id, build_time


def load_states(base: Path, issues: List[dict]) -> Dict[str, Dict[Tuple[str, int], dict]]:
    out: Dict[str, Dict[Tuple[str, int], dict]] = {}
    for split in SPLITS:
        path = base / "trajectories" / f"states_{split}.jsonl"
        split_states: Dict[Tuple[str, int], dict] = {}
        if not path.exists():
            add_issue(issues, "CRITICAL", split=split, issue="missing_states_file", extra={"path": str(path)})
            out[split] = split_states
            continue
        for row in read_jsonl(path):
            split_states[(str(row.get("qid")), int(row.get("t")))] = row
        out[split] = split_states
    return out


def load_samples(base: Path, issues: List[dict]) -> Dict[str, Dict[str, List[dict]]]:
    out: Dict[str, Dict[str, List[dict]]] = {}
    for split in SPLITS:
        path = base / "samples" / f"{split}.jsonl"
        by_qid: Dict[str, List[dict]] = defaultdict(list)
        if not path.exists():
            add_issue(issues, "CRITICAL", split=split, issue="missing_sample_file", extra={"path": str(path)})
            out[split] = by_qid
            continue
        for row in read_jsonl(path):
            by_qid[str(row.get("qid"))].append(row)
        out[split] = by_qid
    return out


def load_debug(base: Path, issues: List[dict]) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for split in SPLITS:
        path = base / "debug" / f"success_semantic_debug_{split}.jsonl"
        if not path.exists():
            add_issue(issues, "CRITICAL", split=split, issue="missing_success_debug_file", extra={"path": str(path)})
            out[split] = {}
            continue
        rows = list(read_jsonl(path))
        out[split] = {str(row.get("qid")): row for row in rows}
    return out


def sorted_h_units(state: dict) -> List[str]:
    h_t = state.get("H_t") if isinstance(state, dict) else []
    if not isinstance(h_t, list):
        return []
    return [
        str(item.get("unit_id"))
        for item in sorted(h_t, key=lambda x: int(x.get("step_id", 0)))
        if isinstance(item, dict)
    ]


def step_payload_type(step: dict, unit_id: str) -> str:
    for item in (step.get("proposer_trace") or {}).get("harvest_candidates", []):
        if isinstance(item, dict) and item.get("unit_id") == unit_id:
            return str(item.get("type", "")).strip()
    return ""


def unit_text(unit_id: str, raw_texts: Dict[str, str], derived_payloads: Dict[str, dict]) -> str:
    if is_derived(unit_id):
        payload = derived_payloads.get(unit_id) or {}
        return str(payload.get("text", ""))
    return raw_texts.get(unit_id, "")


def question_terms(question: str) -> List[str]:
    stop = {
        "what", "which", "when", "where", "who", "whom", "whose", "does", "did", "was", "were",
        "are", "is", "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "that",
        "both", "type", "types", "name", "called", "made",
    }
    return [x for x in re.findall(r"[A-Za-z][A-Za-z]+", question.lower()) if x not in stop]


def relevance_score(question: str, answer: str, text: str) -> int:
    blob = text.lower()
    terms = question_terms(question)
    score = sum(1 for term in terms if term in blob)
    if answer and str(answer).lower() in blob:
        score += 3
    return score


def validate_full_and_path(
    *,
    full_by_split: Dict[str, Dict[str, dict]],
    states_by_split: Dict[str, Dict[Tuple[str, int], dict]],
    raw_texts: Dict[str, str],
    issues: List[dict],
) -> Counter:
    stats = Counter()
    for split, rows in full_by_split.items():
        states = states_by_split.get(split, {})
        for qid, row in rows.items():
            steps = sorted(row.get("steps", []), key=lambda x: int(x.get("t", 0)))
            terminal_t = row.get("terminal_t")
            if len(steps) != terminal_t:
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="full_steps_not_equal_terminal_t")
            previous_covered = -1
            selected_raw = 0
            selected_derived = 0
            positive_units: List[str] = []
            for step in steps:
                t = step.get("t")
                positive = step.get("positive_unit_id")
                positive_units.append(str(positive))
                gate_trace = step.get("gate_trace") or {}
                trigger_derived = bool(gate_trace.get("trigger_derived", False))
                g_final_derived = [unit_id for unit_id in step.get("G_t_final", []) if is_derived(unit_id)]
                if is_derived(positive):
                    selected_derived += 1
                    stats["derived_positive_total"] += 1
                    if not trigger_derived:
                        stats["derived_positive_with_trigger_false"] += 1
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="trigger_false_but_derived_positive")
                    if gate_trace.get("derive_subtype") == "trigger_only_candidate":
                        stats["trigger_only_positive"] += 1
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="trigger_only_candidate_but_positive")
                    payload_type = step_payload_type(step, str(positive))
                    if not subtype_type_consistent(gate_trace.get("derive_subtype"), payload_type):
                        stats["type_subtype_mismatch"] += 1
                        add_issue(
                            issues,
                            "CRITICAL",
                            split=split,
                            qid=qid,
                            t=t,
                            issue="derived_subtype_type_mismatch",
                            extra={"subtype": gate_trace.get("derive_subtype"), "payload_type": payload_type},
                        )
                    if gate_trace.get("derive_subtype") == "early_bridge" and isinstance(t, int) and t > 1:
                        add_issue(issues, "WARNING", split=split, qid=qid, t=t, issue="early_bridge_after_t1")
                    if gate_trace.get("derive_subtype") in VERIFICATION_SUBTYPES and isinstance(t, int) and t <= 1:
                        add_issue(issues, "WARNING", split=split, qid=qid, t=t, issue="verification_too_early")
                else:
                    selected_raw += 1
                if g_final_derived and not trigger_derived:
                    stats["g_final_derived_with_trigger_false"] += len(g_final_derived)
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="G_t_final_derived_with_trigger_false")
                if step.get("need_derived") != gate_trace.get("derived_need"):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="need_derived_gate_mismatch")
                if step.get("triggered_propose_derived") != gate_trace.get("trigger_derived"):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="triggered_propose_gate_mismatch")
                covered = step.get("covered_target_count")
                if isinstance(covered, int):
                    if covered < previous_covered:
                        add_issue(issues, "WARNING", split=split, qid=qid, t=t, issue="covered_target_count_decreased")
                    previous_covered = max(previous_covered, covered)
                before = states.get((qid, int(t))) if isinstance(t, int) else None
                after = states.get((qid, int(t) + 1)) if isinstance(t, int) else None
                if before is not None and after is not None:
                    before_units = sorted_h_units(before)
                    after_units = sorted_h_units(after)
                    if after_units[: len(before_units)] != before_units:
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="state_history_prefix_not_preserved")
                    elif len(after_units) <= len(before_units) or after_units[len(before_units)] != positive:
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="positive_not_appended_to_state")
            stats["selected_raw_total"] += selected_raw
            stats["selected_derived_total"] += selected_derived
            stats[f"path_len_{len(steps)}"] += 1
            if selected_derived and selected_raw == 0:
                add_issue(issues, "WARNING", split=split, qid=qid, issue="path_has_derived_but_no_raw")
            if len(positive_units) != len(set(positive_units)):
                add_issue(issues, "WARNING", split=split, qid=qid, issue="duplicate_positive_in_path")
            terminal_state = states.get((qid, int(terminal_t))) if isinstance(terminal_t, int) else None
            if terminal_state is not None:
                terminal_units = sorted_h_units(terminal_state)
                # Terminal H_t includes the initial seed evidence prefix; the generated path
                # should match the suffix appended by teacher steps.
                if len(terminal_units) < len(positive_units) or terminal_units[-len(positive_units):] != positive_units:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="terminal_state_path_mismatch")
    return stats


def validate_samples(
    *,
    base: Path,
    full_by_split: Dict[str, Dict[str, dict]],
    samples_by_split: Dict[str, Dict[str, List[dict]]],
    run_id: str,
    issues: List[dict],
) -> Tuple[Dict[str, dict], Counter]:
    summary: Dict[str, dict] = {}
    stats = Counter()
    for split in SPLITS:
        by_qid = samples_by_split.get(split, {})
        rows = [row for group in by_qid.values() for row in group]
        summary[split] = {"rows": len(rows)}
        for qid, full_row in full_by_split.get(split, {}).items():
            terminal_t = int(full_row.get("terminal_t", -1))
            sample_rows = sorted(by_qid.get(qid, []), key=lambda x: int(x.get("t", -1)))
            if len(sample_rows) != terminal_t:
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="sample_count_not_equal_terminal_t")
            if [int(row.get("t", -1)) for row in sample_rows] != list(range(len(sample_rows))):
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="sample_t_not_continuous")
            full_steps = {step.get("t"): step for step in full_row.get("steps", []) if isinstance(step, dict)}
            for row in sample_rows:
                t = int(row.get("t"))
                if str((row.get("build_meta") or {}).get("run_id", "")) != run_id:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="sample_run_id_mismatch")
                labels = row.get("labels") or {}
                candidates = row.get("candidates") or {}
                ranking = labels.get("ranking_label") or {}
                u_plus = labels.get("u_t_plus") or {}
                positive = ranking.get("positive_unit_id")
                c_t = list(candidates.get("C_t") or [])
                negatives = list(ranking.get("negative_unit_ids") or [])
                if positive != u_plus.get("unit_id"):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="sample_positive_u_plus_mismatch")
                if full_steps.get(t, {}).get("positive_unit_id") != positive:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="sample_positive_full_mismatch")
                if positive not in c_t:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="positive_not_in_C_t")
                if len(c_t) != len(set(c_t)):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="C_t_has_duplicates")
                if negatives != [unit_id for unit_id in c_t if unit_id != positive]:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="negative_ids_not_C_t_minus_positive")
                provenance = candidates.get("candidate_provenance") or {}
                if set(provenance.keys()) != set(c_t):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="candidate_provenance_scope_invalid")
                aux_prov = candidates.get("aux_candidate_provenance") or {}
                expected_aux = set((candidates.get("G_t_aux") or []) + (candidates.get("G_t_illegal") or []))
                if set(aux_prov.keys()) != expected_aux:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="aux_candidate_provenance_scope_invalid")
                for block_name, keys in {
                    "d_t_star": ["d_raw", "d_br", "d_dis", "d_sup", "d_der"],
                    "c_t_star": ["c_raw", "c_br", "c_dis", "c_sup", "c_der"],
                }.items():
                    block = labels.get(block_name) or {}
                    for key in keys:
                        if key not in block or not is_number(block.get(key)):
                            add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue=f"{block_name}_{key}_invalid")
                stop_label = labels.get("stop_label") or {}
                if not isinstance(stop_label.get("should_stop"), bool):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="stop_label_should_stop_invalid")
                if stop_label.get("label_type") not in ALLOWED_STOP_LABEL_TYPES:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="stop_label_type_invalid")
                payloads = row.get("derived_payloads") or {}
                for unit_id, payload in payloads.items():
                    if not is_derived(unit_id):
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_key_not_derived")
                    if not isinstance(payload, dict):
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_not_dict")
                        continue
                    if not isinstance(payload.get("text"), str) or not payload["text"].strip():
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_text_missing")
                    if not isinstance(payload.get("type"), str) or payload.get("type") not in {"bridge_note", "verification_note"}:
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_type_invalid")
                    if not isinstance(payload.get("source_unit_ids"), list) or not payload["source_unit_ids"]:
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_sources_missing")
                if is_derived(positive):
                    stats["sample_derived_positive"] += 1
    return summary, stats


def validate_debug_and_semantics(
    *,
    full_by_split: Dict[str, Dict[str, dict]],
    debug_by_split: Dict[str, Dict[str, dict]],
    raw_texts: Dict[str, str],
    issues: List[dict],
) -> Tuple[Dict[str, dict], Counter]:
    summary: Dict[str, dict] = {}
    stats = Counter()
    for split in SPLITS:
        rows = debug_by_split.get(split, {})
        summary[split] = {"records": len(rows), "warnings": 0, "selected_derived_records": 0}
        for qid, full_row in full_by_split.get(split, {}).items():
            row = rows.get(qid)
            if row is None:
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="missing_success_debug")
                continue
            warnings = row.get("debug_warnings") or []
            if warnings:
                summary[split]["warnings"] += 1
                add_issue(issues, "WARNING", split=split, qid=qid, issue="debug_warnings_non_empty", extra={"warnings": warnings})
            answer_probe = row.get("answer_probe") or {}
            if not answer_probe.get("AnswerCorrect_t"):
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="answer_not_correct")
            else:
                stats["answer_correct"] += 1
            if not answer_probe.get("SupportSufficient_t"):
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="support_not_sufficient")
            else:
                stats["support_sufficient"] += 1
            if not answer_probe.get("pred_answer") or not answer_probe.get("gold_answer"):
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="answer_probe_missing_answer")
            if not answer_probe.get("answer_match_rule") or answer_probe.get("answer_match_rule") == "none":
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="answer_match_rule_invalid")
            if not answer_probe.get("support_rule"):
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="support_rule_missing")
            terminal_state = row.get("terminal_state") or {}
            selected_unit_ids = list(terminal_state.get("selected_unit_ids") or [])
            selected_derived_units = row.get("selected_derived_units") or []
            selected_derived_by_unit = {
                str(item.get("unit_id")): item
                for item in selected_derived_units
                if isinstance(item, dict) and is_derived(item.get("unit_id"))
            }
            if any(is_derived(unit_id) for unit_id in selected_unit_ids):
                summary[split]["selected_derived_records"] += 1
            for unit_id in selected_unit_ids:
                if not is_derived(unit_id):
                    continue
                item = selected_derived_by_unit.get(str(unit_id))
                if item is None:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_missing_detail", extra={"unit_id": unit_id})
                    continue
                for key in ["text", "type", "source_unit_ids", "source_unit_texts", "selected_t", "selected_h_step_id"]:
                    if key not in item:
                        add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_missing_field", extra={"unit_id": unit_id, "field": key})
                text = str(item.get("text", ""))
                if relevance_score(row.get("question", ""), row.get("gold_answer", ""), text) == 0:
                    add_issue(issues, "WARNING", split=split, qid=qid, issue="selected_derived_low_question_relevance", extra={"unit_id": unit_id})
            for h_item in terminal_state.get("H_t") or []:
                if not isinstance(h_item, dict) or not is_derived(h_item.get("unit_id")):
                    continue
                unit_id = str(h_item.get("unit_id"))
                h_text = h_item.get("unit_text")
                selected_text = (selected_derived_by_unit.get(unit_id) or {}).get("text")
                if not isinstance(h_text, str) or not h_text.strip():
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="H_t_derived_unit_text_missing", extra={"unit_id": unit_id})
                elif isinstance(selected_text, str) and selected_text.strip() and h_text.strip() != selected_text.strip():
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="H_t_derived_unit_text_mismatch", extra={"unit_id": unit_id})
            question = str(row.get("question", ""))
            answer = str(row.get("gold_answer", ""))
            if not question.lower().startswith(("are ", "is ", "was ", "were ")) and answer:
                k_t = str((terminal_state or {}).get("K_t", ""))
                chunks_text = "\n".join(
                    str(chunk.get("full_chunk_text", ""))
                    for chunk in row.get("chunks", [])
                    if isinstance(chunk, dict)
                )
                if answer.lower() not in k_t.lower() and answer.lower() not in chunks_text.lower():
                    add_issue(issues, "WARNING", split=split, qid=qid, issue="gold_answer_not_visible_in_context")
            full_terminal_t = full_row.get("terminal_t")
            debug_terminal_t = (row.get("trajectory_status") or {}).get("terminal_step")
            if full_terminal_t != debug_terminal_t:
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="debug_terminal_step_mismatch")
    return summary, stats


def print_summary(
    *,
    run_id: str,
    build_time: str,
    full_summary: Dict[str, dict],
    sample_summary: Dict[str, dict],
    debug_summary: Dict[str, dict],
    path_stats: Counter,
    sample_stats: Counter,
    semantic_stats: Counter,
    issues: List[dict],
) -> None:
    critical = sum(1 for issue in issues if issue["severity"] == "CRITICAL")
    warning = sum(1 for issue in issues if issue["severity"] == "WARNING")
    print("FINAL DATASET VALIDATION SUMMARY")
    print(f"run_id: {run_id}")
    print(f"build_time: {build_time}")
    print("full:")
    for split in SPLITS:
        s = full_summary.get(split, {"total": 0, "terminal": 0, "abort": 0, "stalled": 0})
        print(f"  {split}: total={s['total']} terminal={s['terminal']} abort={s['abort']} stalled={s['stalled']}")
    print("samples:")
    for split in SPLITS:
        print(f"  {split}: rows={sample_summary.get(split, {}).get('rows', 0)}")
    print("debug:")
    for split in SPLITS:
        s = debug_summary.get(split, {"records": 0, "warnings": 0, "selected_derived_records": 0})
        print(f"  {split}: records={s['records']} warnings={s['warnings']} selected_derived_records={s['selected_derived_records']}")
    print("derived:")
    print(f"  positive_derived_total={path_stats.get('derived_positive_total', 0)}")
    print(f"  derived_positive_with_trigger_false={path_stats.get('derived_positive_with_trigger_false', 0)}")
    print(f"  trigger_only_positive={path_stats.get('trigger_only_positive', 0)}")
    print(f"  type_subtype_mismatch={path_stats.get('type_subtype_mismatch', 0)}")
    print(f"  G_t_final_derived_with_trigger_false={path_stats.get('g_final_derived_with_trigger_false', 0)}")
    print("path_quality:")
    print(f"  selected_raw_total={path_stats.get('selected_raw_total', 0)}")
    print(f"  selected_derived_total={path_stats.get('selected_derived_total', 0)}")
    path_lens = {
        key.removeprefix("path_len_"): value
        for key, value in path_stats.items()
        if key.startswith("path_len_")
    }
    print(f"  terminal_step_distribution={dict(sorted(path_lens.items(), key=lambda x: int(x[0])))}")
    print("semantic:")
    expected_records = sum(s.get("total", 0) for s in full_summary.values())
    print(f"  answer_correct={semantic_stats.get('answer_correct', 0)}/{expected_records}")
    print(f"  support_sufficient={semantic_stats.get('support_sufficient', 0)}/{expected_records}")
    print("issues:")
    print(f"  critical={critical}")
    print(f"  warning={warning}")
    for issue in issues:
        extra = " ".join(f"{k}={v}" for k, v in issue.items() if k not in {"severity", "split", "qid", "t", "issue"})
        suffix = f" {extra}" if extra else ""
        print(f"[{issue['severity']}] split={issue['split']} qid={issue.get('qid','')} t={issue.get('t')} issue={issue['issue']}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final HotpotQA v4 dataset accuracy, compliance, and path quality.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    issues: List[dict] = []
    base = args.base_dir
    raw_texts = load_raw_texts(base)
    full_by_split, full_summary, run_id, build_time = load_full(base, issues)
    states_by_split = load_states(base, issues)
    samples_by_split = load_samples(base, issues)
    debug_by_split = load_debug(base, issues)

    path_stats = validate_full_and_path(
        full_by_split=full_by_split,
        states_by_split=states_by_split,
        raw_texts=raw_texts,
        issues=issues,
    )
    sample_summary, sample_stats = validate_samples(
        base=base,
        full_by_split=full_by_split,
        samples_by_split=samples_by_split,
        run_id=run_id,
        issues=issues,
    )
    debug_summary, semantic_stats = validate_debug_and_semantics(
        full_by_split=full_by_split,
        debug_by_split=debug_by_split,
        raw_texts=raw_texts,
        issues=issues,
    )
    print_summary(
        run_id=run_id,
        build_time=build_time,
        full_summary=full_summary,
        sample_summary=sample_summary,
        debug_summary=debug_summary,
        path_stats=path_stats,
        sample_stats=sample_stats,
        semantic_stats=semantic_stats,
        issues=issues,
    )
    if any(issue["severity"] == "CRITICAL" for issue in issues):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
