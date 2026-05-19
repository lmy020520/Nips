import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = Path(__file__).resolve().parents[1] / "data" / "hotpotqa_distractor_v4"
ALLOWED_STOP_LABEL_TYPES = {"continue", "near-terminal", "terminal", "false-stop", "stop"}
ALLOWED_DEBUG_WARNING_TYPES = {
    "missing_selected_derived_payload",
    "missing_selected_t_for_derived",
    "missing_source_unit_text_for_selected_derived",
    "missing_h_t_derived_unit_text",
    "mismatched_h_t_derived_unit_text",
}
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
                raise ValueError(f"json decode error: file={path} line={line_idx}") from exc


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_derived(unit_id: Any) -> bool:
    return isinstance(unit_id, str) and "::derived::" in unit_id


def normalize_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"terminal", "success"}:
        return "terminal"
    if raw in {"abort", "failed", "failed_stalled", "stalled"}:
        return "abort"
    return raw


def parse_run_id_datetime(run_id: str) -> datetime | None:
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_", run_id)
    if not match:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def parse_build_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def add_issue(issues: List[dict], severity: str, *, split: str, qid: str = "", t: Any = None, issue: str, extra: Dict[str, Any] | None = None) -> None:
    payload = {"severity": severity, "split": split, "qid": qid, "t": t, "issue": issue}
    if extra:
        payload.update(extra)
    issues.append(payload)


def load_full_by_split(base: Path) -> Tuple[Dict[str, Dict[str, dict]], Dict[str, dict], str, str]:
    full_by_split: Dict[str, Dict[str, dict]] = {}
    summary: Dict[str, dict] = {}
    run_ids = set()
    build_times = set()
    for split in SPLITS:
        rows = list(read_jsonl(base / "trajectories" / f"full_{split}.jsonl"))
        full_by_split[split] = {str(row["qid"]): row for row in rows}
        terminal = sum(1 for row in rows if row.get("terminal_status") == "terminal")
        abort = sum(1 for row in rows if row.get("terminal_status") == "abort")
        stalled = sum(1 for row in rows if row.get("abort_reason") == "stalled")
        summary[split] = {"total": len(rows), "terminal": terminal, "abort": abort, "stalled": stalled}
        for row in rows:
            run_id = (((row.get("build_meta") or {}).get("run_id")) or "").strip()
            if run_id:
                run_ids.add(run_id)
            build_time = (((row.get("build_meta") or {}).get("build_time")) or "").strip()
            if build_time:
                build_times.add(build_time)
    if len(run_ids) != 1:
        raise ValueError(f"run_id inconsistent across full files: {sorted(run_ids)}")
    return full_by_split, summary, next(iter(run_ids)), next(iter(sorted(build_times))) if len(build_times) == 1 else ""


def load_states_by_split(base: Path) -> Dict[str, Dict[Tuple[str, int], dict]]:
    out: Dict[str, Dict[Tuple[str, int], dict]] = {}
    for split in SPLITS:
        split_states: Dict[Tuple[str, int], dict] = {}
        path = base / "trajectories" / f"states_{split}.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                split_states[(str(row.get("qid")), int(row.get("t")))] = row
        out[split] = split_states
    return out


def history_unit_ids(state: dict) -> List[str]:
    h_t = state.get("H_t") if isinstance(state, dict) else []
    if not isinstance(h_t, list):
        return []
    return [str(item.get("unit_id")) for item in sorted(h_t, key=lambda x: int(x.get("step_id", 0))) if isinstance(item, dict)]


def positive_derived_payload_type(step: dict, unit_id: str) -> str:
    for item in (step.get("proposer_trace") or {}).get("harvest_candidates", []):
        if isinstance(item, dict) and item.get("unit_id") == unit_id:
            return str(item.get("type", "")).strip()
    return ""


def subtype_type_consistent(subtype: Any, payload_type: str) -> bool:
    subtype = str(subtype or "").strip()
    payload_type = str(payload_type or "").strip()
    if payload_type == "bridge_note":
        return subtype in BRIDGE_SUBTYPES
    if payload_type == "verification_note":
        return subtype in VERIFICATION_SUBTYPES
    return True


def validate_full(
    full_by_split: Dict[str, Dict[str, dict]],
    states_by_split: Dict[str, Dict[Tuple[str, int], dict]],
    run_id: str,
    issues: List[dict],
) -> Dict[str, int]:
    typed_stats = Counter()
    run_dt = parse_run_id_datetime(run_id)
    build_time_formats = set()
    for split, rows in full_by_split.items():
        states = states_by_split.get(split, {})
        for qid, row in rows.items():
            build_meta = row.get("build_meta") or {}
            build_time = build_meta.get("build_time")
            parsed_build_time = parse_build_time(build_time)
            if not isinstance(build_time, str) or not build_time.strip():
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="missing_build_time")
            elif parsed_build_time is None:
                add_issue(issues, "WARNING", split=split, qid=qid, issue="build_time_format_invalid", extra={"build_time": build_time})
            else:
                build_time_formats.add("T" if "T" in build_time else "space")
                if run_dt is not None and parsed_build_time.date() < run_dt.date():
                    add_issue(issues, "WARNING", split=split, qid=qid, issue="build_time_older_than_run_id", extra={"build_time": build_time})
            if row.get("terminal_status") != "terminal":
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="full_not_terminal")
            for step in row.get("steps", []):
                gate_trace = step.get("gate_trace") or {}
                typed_stats["total_steps"] += 1
                trigger_derived = bool(gate_trace.get("trigger_derived", False))
                if trigger_derived:
                    typed_stats["trigger_true_steps"] += 1
                else:
                    typed_stats["trigger_false_steps"] += 1
                if step.get("need_derived") != gate_trace.get("derived_need"):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=step.get("t"), issue="full_need_derived_mismatch")
                if step.get("triggered_propose_derived") != gate_trace.get("trigger_derived"):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=step.get("t"), issue="full_triggered_propose_derived_mismatch")
                t = step.get("t")
                positive = step.get("positive_unit_id")
                g_final_derived = [unit_id for unit_id in step.get("G_t_final", []) if is_derived(unit_id)]
                g_aux_derived = [
                    unit_id
                    for unit_id in ((step.get("candidate_debug") or {}).get("G_t_aux", []))
                    if is_derived(unit_id)
                ]
                if is_derived(positive):
                    typed_stats["derived_positive_total"] += 1
                    if trigger_derived:
                        typed_stats["derived_positive_with_trigger_true"] += 1
                    else:
                        typed_stats["derived_positive_with_trigger_false"] += 1
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="trigger_false_but_derived_positive", extra={"unit_id": positive})
                    if gate_trace.get("derive_subtype") == "trigger_only_candidate":
                        typed_stats["derived_positive_with_subtype_trigger_only_candidate"] += 1
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="trigger_only_candidate_but_derived_positive", extra={"unit_id": positive})
                    payload_type = positive_derived_payload_type(step, str(positive))
                    if not subtype_type_consistent(gate_trace.get("derive_subtype"), payload_type):
                        typed_stats["derived_positive_type_subtype_mismatch"] += 1
                        add_issue(
                            issues,
                            "CRITICAL",
                            split=split,
                            qid=qid,
                            t=t,
                            issue="derived_subtype_type_mismatch",
                            extra={
                                "unit_id": positive,
                                "subtype": gate_trace.get("derive_subtype"),
                                "payload_type": payload_type,
                            },
                        )
                if g_final_derived and not trigger_derived:
                    typed_stats["G_t_final_derived_with_trigger_false"] += len(g_final_derived)
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="G_t_final_derived_with_trigger_false", extra={"unit_ids": g_final_derived})
                if g_aux_derived and not trigger_derived:
                    typed_stats["shadow_derived_aux_count"] += len(g_aux_derived)
                if gate_trace.get("shadow_only") is True:
                    if g_final_derived:
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="shadow_only_but_G_t_final_has_derived", extra={"unit_ids": g_final_derived})
                    if is_derived(positive):
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="shadow_only_but_positive_is_derived", extra={"unit_id": positive})
                if isinstance(t, int):
                    before = states.get((qid, t))
                    after = states.get((qid, t + 1))
                    before_units = history_unit_ids(before or {})
                    after_units = history_unit_ids(after or {})
                    if before is not None and after is not None:
                        if len(after_units) <= len(before_units) or after_units[: len(before_units)] != before_units:
                            add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="state_history_prefix_not_preserved")
                        elif after_units[len(before_units)] != positive:
                            add_issue(
                                issues,
                                "CRITICAL",
                                split=split,
                                qid=qid,
                                t=t,
                                issue="positive_unit_not_matching_terminal_state",
                                extra={"positive": positive, "terminal_has": after_units[len(before_units)]},
                            )
    if len(build_time_formats) > 1:
        add_issue(issues, "WARNING", split="all", issue="build_time_format_inconsistent", extra={"formats": sorted(build_time_formats)})
    return dict(typed_stats)


def validate_samples(
    base: Path,
    full_by_split: Dict[str, Dict[str, dict]],
    run_id: str,
    typed_stats: Counter,
    issues: List[dict],
) -> Dict[str, dict]:
    sample_summary: Dict[str, dict] = {}
    for split in SPLITS:
        rows = list(read_jsonl(base / "samples" / f"{split}.jsonl"))
        sample_summary[split] = {"rows": len(rows)}
        by_qid: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            by_qid[str(row["qid"])].append(row)
            sample_run_id = (((row.get("build_meta") or {}).get("run_id")) or "").strip()
            if sample_run_id != run_id:
                add_issue(issues, "CRITICAL", split=split, qid=str(row.get("qid", "")), t=row.get("t"), issue="sample_run_id_mismatch")

        for qid, full_row in full_by_split[split].items():
            terminal_t = int(full_row.get("terminal_t", -1))
            sample_rows = sorted(by_qid.get(qid, []), key=lambda x: int(x["t"]))
            if len(sample_rows) != terminal_t:
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="sample_count_not_equal_terminal_t", extra={"expected": terminal_t, "actual": len(sample_rows)})
            expected_ts = list(range(len(sample_rows)))
            actual_ts = [int(row["t"]) for row in sample_rows]
            if actual_ts != expected_ts:
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="sample_t_not_continuous", extra={"actual_ts": actual_ts})
            for row in sample_rows:
                t = int(row["t"])
                labels = row.get("labels") or {}
                meta = row.get("meta") or {}
                traj = meta.get("trajectory_status") or {}
                if normalize_status(traj.get("status")) != normalize_status(full_row.get("terminal_status")):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="sample_status_mismatch")
                if traj.get("terminal_step") != terminal_t:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="sample_terminal_step_mismatch")
                if traj.get("abort_reason") != full_row.get("abort_reason"):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="sample_abort_reason_mismatch")

                u_plus = (labels.get("u_t_plus") or {})
                ranking = (labels.get("ranking_label") or {})
                positive = ranking.get("positive_unit_id")
                if positive != u_plus.get("unit_id"):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="positive_unit_id_mismatch")

                candidates = row.get("candidates") or {}
                c_t = list(candidates.get("C_t") or [])
                if positive not in c_t:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="positive_unit_not_in_C_t")
                if len(c_t) != len(set(c_t)):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="C_t_has_duplicates")
                negatives = list(ranking.get("negative_unit_ids") or [])
                if any(unit_id not in c_t for unit_id in negatives):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="negative_unit_not_in_C_t")
                expected_negatives = [unit_id for unit_id in c_t if unit_id != positive]
                if negatives != expected_negatives:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="negative_unit_ids_not_equal_C_t_minus_positive")
                provenance = candidates.get("candidate_provenance") or {}
                if set(provenance.keys()) != set(c_t):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="candidate_provenance_not_cover_C_t")
                extra_provenance_keys = set(provenance.keys()) - set(c_t)
                if extra_provenance_keys:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="candidate_provenance_extra_keys", extra={"keys": sorted(extra_provenance_keys)})
                aux_candidate_provenance = candidates.get("aux_candidate_provenance") or {}
                if not isinstance(aux_candidate_provenance, dict):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="aux_candidate_provenance_invalid")
                    aux_candidate_provenance = {}
                expected_aux_ids = set((candidates.get("G_t_aux") or []) + (candidates.get("G_t_illegal") or []))
                if set(aux_candidate_provenance.keys()) != expected_aux_ids:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="aux_candidate_provenance_keys_invalid")
                else:
                    typed_stats["aux_candidate_provenance_keys"] += len(aux_candidate_provenance)

                for label_name, expected_keys in {
                    "d_t_star": ["d_raw", "d_br", "d_dis", "d_sup", "d_der"],
                    "c_t_star": ["c_raw", "c_br", "c_dis", "c_sup", "c_der"],
                }.items():
                    label_block = labels.get(label_name) or {}
                    if list(label_block.keys()) != expected_keys:
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue=f"{label_name}_keys_invalid")
                    for key in expected_keys:
                        if not is_number(label_block.get(key)):
                            add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue=f"{label_name}_{key}_not_numeric")

                stop_label = labels.get("stop_label") or {}
                label_type = stop_label.get("label_type")
                should_stop = stop_label.get("should_stop")
                if not isinstance(should_stop, bool):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="stop_label_should_stop_not_bool")
                if label_type not in ALLOWED_STOP_LABEL_TYPES:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="stop_label_type_invalid", extra={"label_type": label_type})
                if label_type == "near-terminal":
                    if should_stop is not False:
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="near_terminal_should_stop_false_required")
                    if not (t < terminal_t):
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="near_terminal_window_invalid")

                derived_payloads = row.get("derived_payloads") or {}
                if is_derived(positive):
                    full_step = next(
                        (
                            step for step in full_row.get("steps", [])
                            if isinstance(step, dict) and step.get("t") == t
                        ),
                        {},
                    )
                    subtype = (full_step.get("gate_trace") or {}).get("derive_subtype") if isinstance(full_step, dict) else None
                    payload = derived_payloads.get(positive) if isinstance(derived_payloads, dict) else None
                    payload_type = str((payload or {}).get("type", "")).strip() if isinstance(payload, dict) else ""
                    if not subtype_type_consistent(subtype, payload_type):
                        add_issue(
                            issues,
                            "CRITICAL",
                            split=split,
                            qid=qid,
                            t=t,
                            issue="sample_derived_subtype_type_mismatch",
                            extra={"unit_id": positive, "subtype": subtype, "payload_type": payload_type},
                        )
                for unit_id, payload in derived_payloads.items():
                    if not is_derived(unit_id):
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_key_not_derived", extra={"unit_id": unit_id})
                    if not isinstance(payload, dict):
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_not_dict", extra={"unit_id": unit_id})
                        continue
                    if not isinstance(payload.get("text"), str) or not payload["text"].strip():
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_missing_text", extra={"unit_id": unit_id})
                    if not isinstance(payload.get("type"), str) or not payload["type"].strip():
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_missing_type", extra={"unit_id": unit_id})
                    source_unit_ids = payload.get("source_unit_ids")
                    if not (
                        isinstance(source_unit_ids, list)
                        and source_unit_ids
                        and all(isinstance(x, str) for x in source_unit_ids)
                    ):
                        add_issue(issues, "CRITICAL", split=split, qid=qid, t=t, issue="derived_payload_missing_source_unit_ids", extra={"unit_id": unit_id})
    return sample_summary


def validate_debug(
    base: Path,
    full_by_split: Dict[str, Dict[str, dict]],
    run_id: str,
    typed_stats: Counter,
    issues: List[dict],
) -> Dict[str, dict]:
    debug_summary: Dict[str, dict] = {}
    for split in SPLITS:
        rows = list(read_jsonl(base / "debug" / f"success_semantic_debug_{split}.jsonl"))
        debug_summary[split] = {"records": len(rows), "warnings": 0, "selected_derived_records": 0}
        by_qid = {str(row["qid"]): row for row in rows}
        for qid, full_row in full_by_split[split].items():
            if qid not in by_qid:
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="missing_success_debug_record")
                continue
            row = by_qid[qid]
            debug_run_id = (((row.get("build_meta") or {}).get("run_id")) or "").strip()
            if debug_run_id != run_id:
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="debug_run_id_mismatch")
            if not isinstance(row.get("gold_answer"), str) or not row["gold_answer"].strip():
                add_issue(issues, "CRITICAL", split=split, qid=qid, issue="debug_missing_gold_answer")
            answer_probe = row.get("answer_probe") or {}
            for key in ["pred_answer", "gold_answer", "answer_match_rule", "support_rule"]:
                if key not in answer_probe:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="debug_answer_probe_missing_field", extra={"field": key})
            warnings = row.get("debug_warnings") or []
            if warnings:
                debug_summary[split]["warnings"] += 1
            for warning in warnings:
                if isinstance(warning, str):
                    continue
                if not (isinstance(warning, dict) and warning.get("type") in ALLOWED_DEBUG_WARNING_TYPES):
                    add_issue(issues, "WARNING", split=split, qid=qid, issue="unexpected_debug_warning", extra={"warning": warning})

            terminal_state = row.get("terminal_state") or {}
            selected_unit_ids = list(terminal_state.get("selected_unit_ids") or [])
            selected_derived_ids = [unit_id for unit_id in selected_unit_ids if is_derived(unit_id)]
            selected_derived_units = row.get("selected_derived_units") or []
            if selected_derived_ids:
                debug_summary[split]["selected_derived_records"] += 1
                if not selected_derived_units:
                    typed_stats["selected_derived_units_missing"] += 1
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_units_missing")
            selected_unit_id_set = set(selected_unit_ids)
            selected_derived_by_unit = {
                str(item.get("unit_id")): item
                for item in selected_derived_units
                if isinstance(item, dict) and is_derived(item.get("unit_id"))
            }
            h_step_by_unit = {
                str(item.get("unit_id")): item.get("step_id")
                for item in (terminal_state.get("H_t") or [])
                if isinstance(item, dict)
            }
            positive_by_t = {
                step.get("t"): step.get("positive_unit_id")
                for step in full_row.get("steps", [])
                if isinstance(step, dict)
            }
            for h_item in terminal_state.get("H_t") or []:
                if not isinstance(h_item, dict):
                    continue
                h_unit_id = h_item.get("unit_id")
                if not is_derived(h_unit_id):
                    continue
                typed_stats["h_t_derived_units_total"] += 1
                h_text = h_item.get("unit_text")
                if not isinstance(h_text, str) or not h_text.strip():
                    typed_stats["h_t_derived_unit_text_missing"] += 1
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="missing_h_t_derived_unit_text", extra={"unit_id": h_unit_id})
                    continue
                selected_item = selected_derived_by_unit.get(str(h_unit_id))
                selected_text = selected_item.get("text") if isinstance(selected_item, dict) else None
                if isinstance(selected_text, str) and selected_text.strip() and h_text.strip() != selected_text.strip():
                    typed_stats["h_t_derived_unit_text_mismatch"] += 1
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="h_t_derived_unit_text_mismatch", extra={"unit_id": h_unit_id})
            for item in selected_derived_units:
                if not isinstance(item, dict):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_unit_not_dict")
                    continue
                unit_id = item.get("unit_id")
                if "selected_step" in item:
                    typed_stats["deprecated_selected_step_count"] += 1
                    add_issue(
                        issues,
                        "WARNING",
                        split=split,
                        qid=qid,
                        issue="selected_step is deprecated; use selected_h_step_id and selected_t",
                        extra={"unit_id": unit_id},
                    )
                if unit_id not in selected_unit_id_set:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_unit_id_not_in_selected_unit_ids", extra={"unit_id": unit_id})
                if not isinstance(item.get("text"), str) or not item["text"].strip():
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_unit_missing_text", extra={"unit_id": unit_id})
                if not isinstance(item.get("type"), str) or not item["type"].strip():
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_unit_missing_type", extra={"unit_id": unit_id})
                source_unit_ids = item.get("source_unit_ids")
                if not (
                    isinstance(source_unit_ids, list)
                    and source_unit_ids
                    and all(isinstance(x, str) for x in source_unit_ids)
                ):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_unit_missing_source_unit_ids", extra={"unit_id": unit_id})
                source_unit_texts = item.get("source_unit_texts")
                if not (
                    isinstance(source_unit_texts, list)
                    and source_unit_texts
                    and all(
                        isinstance(x, dict)
                        and isinstance(x.get("unit_id"), str)
                        and isinstance(x.get("text"), str)
                        and x.get("text").strip()
                        for x in source_unit_texts
                    )
                ):
                    typed_stats["selected_derived_source_text_missing"] += 1
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_source_text_missing", extra={"unit_id": unit_id})
                selected_h_step_id = item.get("selected_h_step_id")
                selected_t = item.get("selected_t")
                if not isinstance(selected_h_step_id, int):
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_unit_selected_h_step_id_invalid", extra={"unit_id": unit_id})
                elif h_step_by_unit.get(unit_id) != selected_h_step_id:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_h_step_id_not_matching_H_t", extra={"unit_id": unit_id})
                if not isinstance(selected_t, int):
                    typed_stats["selected_t_missing"] += 1
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_derived_unit_selected_t_invalid", extra={"unit_id": unit_id})
                elif positive_by_t.get(selected_t) != unit_id:
                    add_issue(issues, "CRITICAL", split=split, qid=qid, issue="selected_t_not_matching_full_positive", extra={"unit_id": unit_id, "selected_t": selected_t})
                if qid == "5adce4d35542992c1e3a2473":
                    semantic_blob = " ".join(
                        str(item.get(key, ""))
                        for key in ["unit_id", "text"]
                    ).lower()
                    if "whitemud" in semantic_blob or "clay" in semantic_blob:
                        add_issue(issues, "CRITICAL", split=split, qid=qid, issue="yes_no_unrelated_derived_selected", extra={"unit_id": unit_id})
    return debug_summary


def print_summary(
    run_id: str,
    build_time: str,
    full_summary: Dict[str, dict],
    sample_summary: Dict[str, dict],
    debug_summary: Dict[str, dict],
    typed_stats: Dict[str, int],
    issues: List[dict],
) -> None:
    critical = sum(1 for issue in issues if issue["severity"] == "CRITICAL")
    warning = sum(1 for issue in issues if issue["severity"] == "WARNING")
    print("FIELD COMPLIANCE SUMMARY")
    print(f"run_id: {run_id}")
    if build_time:
        print(f"build_time: {build_time}")
    print("full:")
    for split in SPLITS:
        s = full_summary[split]
        print(f"  {split}: total={s['total']} terminal={s['terminal']} abort={s['abort']} stalled={s['stalled']}")
    print("samples:")
    for split in SPLITS:
        s = sample_summary[split]
        print(f"  {split}: rows={s['rows']}")
    print("debug:")
    for split in SPLITS:
        s = debug_summary[split]
        print(f"  {split}: records={s['records']} warnings={s['warnings']} selected_derived_records={s['selected_derived_records']}")
    print("derived:")
    for key in [
        "derived_positive_total",
        "derived_positive_with_trigger_false",
        "derived_positive_with_subtype_trigger_only_candidate",
        "derived_positive_type_subtype_mismatch",
        "G_t_final_derived_with_trigger_false",
        "selected_derived_units_missing",
        "selected_t_missing",
        "selected_derived_source_text_missing",
        "deprecated_selected_step_count",
    ]:
        print(f"  {key}={int(typed_stats.get(key, 0))}")
    print("debug_h_t:")
    for key in [
        "h_t_derived_units_total",
        "h_t_derived_unit_text_missing",
        "h_t_derived_unit_text_mismatch",
    ]:
        print(f"  {key}={int(typed_stats.get(key, 0))}")
    print("provenance:")
    print(f"  candidate_provenance_extra_keys=0")
    print(f"  aux_candidate_provenance_keys={int(typed_stats.get('aux_candidate_provenance_keys', 0))}")
    print("typed_gate:")
    for key in [
        "total_steps",
        "trigger_true_steps",
        "trigger_false_steps",
        "derived_positive_with_trigger_true",
        "shadow_derived_aux_count",
    ]:
        print(f"  {key}={int(typed_stats.get(key, 0))}")
    print("issues:")
    print(f"  critical={critical}")
    print(f"  warning={warning}")
    for issue in issues:
        prefix = issue["severity"]
        print(f"[{prefix}] split={issue['split']} qid={issue.get('qid','')} t={issue.get('t')} issue={issue['issue']}")


def main() -> None:
    issues: List[dict] = []
    full_by_split, full_summary, run_id, build_time = load_full_by_split(DEFAULT_BASE)
    states_by_split = load_states_by_split(DEFAULT_BASE)
    typed_stats = Counter(validate_full(full_by_split, states_by_split, run_id, issues))
    sample_summary = validate_samples(DEFAULT_BASE, full_by_split, run_id, typed_stats, issues)
    debug_summary = validate_debug(DEFAULT_BASE, full_by_split, run_id, typed_stats, issues)
    print_summary(run_id, build_time, full_summary, sample_summary, debug_summary, typed_stats, issues)


if __name__ == "__main__":
    main()
