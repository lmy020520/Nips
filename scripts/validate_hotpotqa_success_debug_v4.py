import os
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = Path(os.environ.get("HOTPOTQA_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "hotpotqa_distractor_v4")))


def is_derived_unit_id(unit_id: Any) -> bool:
    return isinstance(unit_id, str) and "::derived::" in unit_id


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'“”‘’`.,;:!?()\[\]{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numeric_tokens(text: Any) -> List[str]:
    if text is None:
        return []
    return re.findall(r"\d+(?:[./:-]\d+)*", str(text))


def is_number_like_answer(text: Any) -> bool:
    if text is None:
        return False
    raw = str(text).strip()
    if not raw:
        return False
    if re.search(r"\d", raw):
        return True
    lowered = raw.lower()
    month_names = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    return any(month in lowered for month in month_names)


def answer_conflicts(pred_answer: Any, gold_answer: Any) -> bool:
    if pred_answer is None or gold_answer is None:
        return False
    norm_pred = normalize_text(pred_answer)
    norm_gold = normalize_text(gold_answer)
    if not norm_pred or not norm_gold or norm_pred == norm_gold:
        return False
    pred_numeric = extract_numeric_tokens(pred_answer)
    gold_numeric = extract_numeric_tokens(gold_answer)
    if pred_numeric or gold_numeric:
        return pred_numeric != gold_numeric
    if is_number_like_answer(pred_answer) or is_number_like_answer(gold_answer):
        return False
    return not (
        norm_pred.startswith(norm_gold + " ")
        or norm_pred.endswith(" " + norm_gold)
        or f" {norm_gold} " in f" {norm_pred} "
        or norm_gold.startswith(norm_pred + " ")
        or norm_gold.endswith(" " + norm_pred)
        or f" {norm_pred} " in f" {norm_gold} "
    )


def answers_equivalent(pred_answer: Any, gold_answer: Any) -> bool:
    if pred_answer is None or gold_answer is None:
        return False
    norm_pred = normalize_text(pred_answer)
    norm_gold = normalize_text(gold_answer)
    if not norm_pred or not norm_gold:
        return False
    if norm_pred == norm_gold:
        return True

    pred_numeric = extract_numeric_tokens(pred_answer)
    gold_numeric = extract_numeric_tokens(gold_answer)
    if pred_numeric and gold_numeric and pred_numeric == gold_numeric:
        return True

    if is_number_like_answer(pred_answer) or is_number_like_answer(gold_answer):
        return norm_gold in norm_pred or norm_pred in norm_gold

    return False


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"__json_error__": line_idx}


def add_error(
    errors: List[dict],
    *,
    split: str,
    qid: Optional[str],
    error_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {"split": split, "qid": qid, "error_type": error_type}
    if extra:
        payload.update(extra)
    errors.append(payload)


def validate_record(record: dict, *, split: str, errors: List[dict]) -> None:
    if "__json_error__" in record:
        add_error(errors, split=split, qid=None, error_type="json_decode_error", extra={"line": record["__json_error__"]})
        return

    qid = record.get("qid")
    if not isinstance(qid, str) or not qid:
        add_error(errors, split=split, qid=None, error_type="missing_qid")
        return

    if not isinstance(record.get("question"), str) or not record["question"].strip():
        add_error(errors, split=split, qid=qid, error_type="missing_question")
    if not isinstance(record.get("gold_answer"), str):
        add_error(errors, split=split, qid=qid, error_type="missing_gold_answer")

    trajectory_status = record.get("trajectory_status")
    if not isinstance(trajectory_status, dict):
        add_error(errors, split=split, qid=qid, error_type="missing_trajectory_status")
    else:
        if trajectory_status.get("status") != "success":
            add_error(errors, split=split, qid=qid, error_type="trajectory_status_not_success")
        terminal_step = trajectory_status.get("terminal_step")
        if isinstance(terminal_step, bool) or not isinstance(terminal_step, int):
            add_error(errors, split=split, qid=qid, error_type="missing_terminal_step")

    terminal_state = record.get("terminal_state")
    if not isinstance(terminal_state, dict):
        add_error(errors, split=split, qid=qid, error_type="missing_terminal_state")
    else:
        for key in ["t", "H_t", "K_t", "selected_unit_ids", "selected_chunk_ids"]:
            if key not in terminal_state:
                add_error(errors, split=split, qid=qid, error_type="terminal_state_missing_field", extra={"field": key})
        if isinstance(terminal_state.get("K_t"), str):
            if "[missing derived payload]" in terminal_state["K_t"]:
                add_error(errors, split=split, qid=qid, error_type="missing_derived_payload_in_K_t")
        else:
            add_error(errors, split=split, qid=qid, error_type="terminal_state_k_t_not_str")
        if (
            isinstance(record.get("trajectory_status"), dict)
            and isinstance(record["trajectory_status"].get("terminal_step"), int)
            and terminal_state.get("t") != record["trajectory_status"]["terminal_step"]
        ):
            add_error(errors, split=split, qid=qid, error_type="terminal_state_not_exact_terminal_step")
        if terminal_state.get("source_state_t") != terminal_state.get("t"):
            add_error(errors, split=split, qid=qid, error_type="terminal_state_not_exact_terminal_step")

    chunks = record.get("chunks")
    if not isinstance(chunks, list):
        add_error(errors, split=split, qid=qid, error_type="missing_chunks")
    else:
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                add_error(errors, split=split, qid=qid, error_type="chunk_item_not_dict", extra={"idx": idx})
                continue
            for key in ["chunk_id", "doc_id", "full_chunk_text"]:
                if key not in chunk:
                    add_error(errors, split=split, qid=qid, error_type="chunk_missing_field", extra={"idx": idx, "field": key})
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.startswith("rawchunk::"):
                add_error(errors, split=split, qid=qid, error_type="chunk_id_invalid", extra={"idx": idx})
            if not isinstance(chunk.get("doc_id"), str) or not chunk["doc_id"].strip():
                add_error(errors, split=split, qid=qid, error_type="doc_id_invalid", extra={"idx": idx})
            if not isinstance(chunk.get("full_chunk_text"), str) or not chunk["full_chunk_text"].strip():
                add_error(errors, split=split, qid=qid, error_type="missing_full_chunk_text", extra={"idx": idx})

    selected_derived_units = record.get("selected_derived_units")
    if not isinstance(selected_derived_units, list):
        add_error(errors, split=split, qid=qid, error_type="selected_derived_units_not_list")
        selected_derived_units = []
    else:
        selected_unit_ids = []
        if isinstance(terminal_state, dict) and isinstance(terminal_state.get("selected_unit_ids"), list):
            selected_unit_ids = [str(x) for x in terminal_state["selected_unit_ids"]]
        for idx, item in enumerate(selected_derived_units):
            if not isinstance(item, dict):
                add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_not_dict", extra={"idx": idx})
                continue
            for key in ["unit_id", "type", "text", "source_unit_ids", "source_unit_texts", "selected_h_step_id", "selected_t", "in_final_K_t"]:
                if key not in item:
                    add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_missing_field", extra={"idx": idx, "field": key})
            unit_id = item.get("unit_id")
            if not is_derived_unit_id(unit_id):
                add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_id_invalid", extra={"idx": idx})
            elif unit_id not in selected_unit_ids:
                add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_not_in_selected_unit_ids", extra={"idx": idx, "unit_id": unit_id})
            if item.get("text") is None or item.get("type") is None:
                add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_missing_payload", extra={"idx": idx, "unit_id": unit_id})
            if not (
                isinstance(item.get("source_unit_ids"), list)
                and all(isinstance(x, str) for x in item.get("source_unit_ids"))
            ):
                add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_source_unit_ids_invalid", extra={"idx": idx})
            source_unit_texts = item.get("source_unit_texts")
            if not (
                isinstance(source_unit_texts, list)
                and all(isinstance(x, dict) and isinstance(x.get("unit_id"), str) and isinstance(x.get("text"), str) and x.get("text").strip() for x in source_unit_texts)
            ):
                add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_source_unit_texts_invalid", extra={"idx": idx})
            if not isinstance(item.get("selected_h_step_id"), int):
                add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_selected_h_step_id_invalid", extra={"idx": idx})
            if not isinstance(item.get("selected_t"), int):
                add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_selected_t_invalid", extra={"idx": idx})
            if not isinstance(item.get("in_final_K_t"), bool):
                add_error(errors, split=split, qid=qid, error_type="selected_derived_unit_in_final_k_t_invalid", extra={"idx": idx})

    if isinstance(terminal_state, dict) and isinstance(terminal_state.get("selected_unit_ids"), list):
        selected_derived_ids = [str(x) for x in terminal_state["selected_unit_ids"] if is_derived_unit_id(x)]
        if selected_derived_ids and not selected_derived_units:
            add_error(errors, split=split, qid=qid, error_type="selected_derived_units_missing_for_terminal_state")

    answer_probe = record.get("answer_probe")
    if not isinstance(answer_probe, dict):
        add_error(errors, split=split, qid=qid, error_type="missing_answer_probe")
    else:
        required = [
            "pred_answer",
            "gold_answer",
            "normalized_pred",
            "normalized_gold",
            "answer_match_rule",
            "AnswerCorrect_t",
            "SupportSufficient_t",
            "support_rule",
            "support_evidence_summary",
            "missing_support_reasons",
            "TeacherStop_t",
        ]
        for key in required:
            if key not in answer_probe:
                add_error(errors, split=split, qid=qid, error_type="answer_probe_missing_field", extra={"field": key})
        if not isinstance(answer_probe.get("answer_match_rule"), str) or not answer_probe["answer_match_rule"].strip():
            add_error(errors, split=split, qid=qid, error_type="missing_answer_match_rule")
        for key in ["AnswerCorrect_t", "SupportSufficient_t", "TeacherStop_t"]:
            if not isinstance(answer_probe.get(key), bool):
                add_error(errors, split=split, qid=qid, error_type="answer_probe_flag_not_bool", extra={"field": key})
        pred_answer = answer_probe.get("pred_answer")
        gold_answer = answer_probe.get("gold_answer")
        teacher_stop = bool(answer_probe.get("TeacherStop_t", False))
        answer_correct = bool(answer_probe.get("AnswerCorrect_t", False))
        support_sufficient = bool(answer_probe.get("SupportSufficient_t", False))
        if teacher_stop and (not isinstance(pred_answer, str) or not pred_answer.strip()):
            add_error(errors, split=split, qid=qid, error_type="missing_pred_answer")
        if teacher_stop and (not isinstance(gold_answer, str) or not gold_answer.strip()):
            add_error(errors, split=split, qid=qid, error_type="missing_gold_answer")
        if teacher_stop and (
            not isinstance(answer_probe.get("answer_match_rule"), str)
            or not answer_probe["answer_match_rule"].strip()
        ):
            add_error(errors, split=split, qid=qid, error_type="missing_answer_match_rule")
        if teacher_stop and (
            not isinstance(answer_probe.get("support_rule"), str)
            or not answer_probe["support_rule"].strip()
        ):
            add_error(errors, split=split, qid=qid, error_type="missing_support_probe")
        if teacher_stop and (
            not isinstance(answer_probe.get("support_evidence_summary"), str)
            or not answer_probe["support_evidence_summary"].strip()
        ):
            add_error(errors, split=split, qid=qid, error_type="missing_support_probe")
        if not isinstance(answer_probe.get("missing_support_reasons"), list):
            add_error(errors, split=split, qid=qid, error_type="missing_support_probe")
        if (
            isinstance(pred_answer, str)
            and isinstance(gold_answer, str)
            and is_number_like_answer(pred_answer)
            and is_number_like_answer(gold_answer)
            and normalize_text(pred_answer) != normalize_text(gold_answer)
            and not answers_equivalent(pred_answer, gold_answer)
            and answer_correct
        ):
            add_error(errors, split=split, qid=qid, error_type="numeric_answer_conflict")
        if answer_conflicts(pred_answer, gold_answer) and answer_correct:
            add_error(errors, split=split, qid=qid, error_type="answer_conflict_but_marked_correct")
        if teacher_stop and not answer_correct:
            add_error(errors, split=split, qid=qid, error_type="teacher_stop_inconsistent")
        if teacher_stop and not support_sufficient:
            add_error(errors, split=split, qid=qid, error_type="support_insufficient_but_stopped")
        if (
            teacher_stop
            and isinstance(gold_answer, str)
            and is_number_like_answer(gold_answer)
            and isinstance(record.get("semantic_check"), dict)
            and not record["semantic_check"].get("has_gold_answer_in_K_t", False)
            and not record["semantic_check"].get("has_gold_answer_in_chunks", False)
        ):
            add_error(errors, split=split, qid=qid, error_type="gold_answer_not_supported_by_chunks")
        if (
            teacher_stop
            and isinstance(record.get("question"), str)
            and record["question"].strip().lower().startswith(("is ", "are ", "was ", "were "))
            and isinstance(terminal_state, dict)
            and len(terminal_state.get("selected_chunk_ids", [])) < 2
        ):
            add_error(errors, split=split, qid=qid, error_type="yes_no_missing_one_entity_support")
        missing_reasons = answer_probe.get("missing_support_reasons", [])
        if isinstance(missing_reasons, list):
            if "gold_answer_not_supported_by_chunks" in missing_reasons:
                add_error(errors, split=split, qid=qid, error_type="gold_answer_not_supported_by_chunks")
            if "missing_multihop_evidence" in missing_reasons:
                add_error(errors, split=split, qid=qid, error_type="missing_multihop_evidence")
            if "yes_no_missing_one_entity_support" in missing_reasons:
                add_error(errors, split=split, qid=qid, error_type="yes_no_missing_one_entity_support")

    semantic_check = record.get("semantic_check")
    if not isinstance(semantic_check, dict):
        add_error(errors, split=split, qid=qid, error_type="missing_semantic_check")
    else:
        for key in ["has_gold_answer_in_chunks", "has_gold_answer_in_K_t", "supporting_evidence_found", "support_rule", "support_evidence_summary", "missing_support_reasons", "notes"]:
            if key not in semantic_check:
                add_error(errors, split=split, qid=qid, error_type="semantic_check_missing_field", extra={"field": key})

    failure_signals = record.get("failure_signals")
    if not isinstance(failure_signals, dict):
        add_error(errors, split=split, qid=qid, error_type="missing_failure_signals")
    else:
        for key in [
            "recent_false_stop",
            "false_stop_count_recent",
            "last_delta_covered_targets",
            "last_retrieval_repeat_ratio",
            "last_probe_pred_answer",
            "last_probe_answer_correct",
            "answer_focus_mismatch",
            "stagnation",
        ]:
            if key not in failure_signals:
                add_error(errors, split=split, qid=qid, error_type="failure_signals_missing_field", extra={"field": key})

    gate_trace = record.get("gate_trace")
    if not isinstance(gate_trace, dict):
        add_error(errors, split=split, qid=qid, error_type="missing_gate_trace")
    else:
        for key in [
            "raw_complete",
            "no_obvious_derived_need",
            "no_recent_closure_failure",
            "stop_candidate",
            "repair_override",
            "bridgeable_raw",
            "has_recent_verification",
            "derived_need",
            "trigger_derived",
            "derive_mode",
            "derive_goal",
        ]:
            if key not in gate_trace:
                add_error(errors, split=split, qid=qid, error_type="gate_trace_missing_field", extra={"field": key})

    proposer_trace = record.get("proposer_trace")
    if not isinstance(proposer_trace, dict):
        add_error(errors, split=split, qid=qid, error_type="missing_proposer_trace")
    else:
        for key in [
            "derive_mode",
            "derive_goal",
            "bridge_anchors",
            "recent_probe_feedback",
            "harvest_count",
            "final_count",
            "aux_count",
            "illegal_count",
        ]:
            if key not in proposer_trace:
                add_error(errors, split=split, qid=qid, error_type="proposer_trace_missing_field", extra={"field": key})

    warnings = record.get("debug_warnings")
    if not isinstance(warnings, list):
        add_error(errors, split=split, qid=qid, error_type="debug_warnings_invalid")
    else:
        normalized_warning_types = []
        for warning in warnings:
            if isinstance(warning, str):
                normalized_warning_types.append(warning)
            elif isinstance(warning, dict) and isinstance(warning.get("type"), str):
                normalized_warning_types.append(warning["type"])
            else:
                add_error(errors, split=split, qid=qid, error_type="debug_warning_item_invalid")
        if "terminal_state_missing_exact_t" in normalized_warning_types:
            add_error(errors, split=split, qid=qid, error_type="terminal_state_not_exact_terminal_step")


def main() -> None:
    base_dir = DEFAULT_BASE
    debug_dir = base_dir / "debug"
    errors: List[dict] = []
    counts = Counter()
    for split in SPLITS:
        path = debug_dir / f"success_semantic_debug_{split}.jsonl"
        if not path.exists():
            add_error(errors, split=split, qid=None, error_type="missing_debug_file")
            continue
        for record in read_jsonl(path):
            validate_record(record, split=split, errors=errors)
        counts[split] = sum(1 for _ in read_jsonl(path))

    if errors:
        for err in errors[:50]:
            print(json.dumps(err, ensure_ascii=False))
    print("validate_hotpotqa_success_debug_v4")
    for split in SPLITS:
        if counts[split]:
            print(f"  {split}: {counts[split]}")
    print(f"errors={len(errors)}")


if __name__ == "__main__":
    main()
