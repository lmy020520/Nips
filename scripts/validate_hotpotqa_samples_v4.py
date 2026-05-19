import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v4"
DEFAULT_SAMPLES_BASE = "data/hotpotqa_distractor_v4/samples"

ALLOWED_TRAJECTORY_STATUS = {
    "success",
    "failed_but_progressive",
    "failed_stalled",
}
ALLOWED_PROGRESS_FLAG = {
    "progress",
    "stall",
}
ALLOWED_STOP_LABEL_TYPE = {
    "terminal",
    "false-stop",
    "near-terminal",
    "continue",
}
FORBIDDEN_DEBUG_KEYS = {
    "candidate_debug",
    "coverage_debug",
    "covered_target_count",
    "delta_covered_targets",
    "derived_debug",
    "gold_answer",
    "need_derived",
    "normalized_gold",
    "normalized_pred",
    "pred_answer",
    "probe",
    "stop_probe",
    "stop_candidate",
    "stop_debug",
    "probe_answer",
    "judge_output",
    "raw_response",
    "debug_info",
    "query_debug",
    "retrieval_repeat_ratio",
    "triggered_propose_derived",
    "failure_signals",
    "gate_trace",
    "proposer_trace",
    "derive_mode",
    "derive_goal",
    "bridge_anchors",
    "recent_probe_feedback",
}


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


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(item, str) for item in x)


def list_has_duplicates(xs: List[str]) -> bool:
    return len(xs) != len(set(xs))


def looks_like_raw_unit_id(value: str) -> bool:
    parts = str(value).split("::")
    return "::derived::" not in str(value) and len(parts) >= 3 and parts[-1].isdigit()


DERIVED_UNIT_ID_RE = re.compile(r"\b[0-9A-Za-z]+::derived::\d+\b")


def extract_derived_unit_ids_from_text(text: Any) -> List[str]:
    if not isinstance(text, str) or "::derived::" not in text:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for unit_id in DERIVED_UNIT_ID_RE.findall(text):
        if unit_id in seen:
            continue
        seen.add(unit_id)
        out.append(unit_id)
    return out


def normalize_text_for_chunk_identity(text: Any, *, name: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{name} 必须是 str")
    return " ".join(text.split())


def build_canonical_chunk_id(*, doc_id: str, chunk_text: str) -> str:
    payload = json.dumps(
        {
            "doc_id": normalize_text_for_chunk_identity(doc_id, name="canonical_chunk.doc_id"),
            "chunk_text": normalize_text_for_chunk_identity(chunk_text, name="canonical_chunk.chunk_text"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"rawchunk::{hashlib.sha1(payload.encode('utf-8')).hexdigest()}"


def is_canonical_chunk_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("rawchunk::") and len(value) == len("rawchunk::") + 40


def is_pseudo_chunk_id(value: Any, *, qid: Optional[str], unit_id: Optional[str]) -> bool:
    if not isinstance(value, str):
        return False
    if isinstance(qid, str) and value.startswith(f"{qid}::"):
        return True
    if isinstance(unit_id, str) and looks_like_raw_unit_id(unit_id) and value == unit_id.rsplit("::", 1)[0]:
        return True
    return False


def is_derived_unit_id(value: Any) -> bool:
    return isinstance(value, str) and "::derived::" in value


def load_raw_chunk_registry(
    raw_units_path: Path,
    chunks_path: Path,
) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    by_unit: Dict[str, Dict[str, Optional[str]]] = {}
    by_chunk_alias: Dict[str, Dict[str, Optional[str]]] = {}
    by_chunk_id: Dict[str, Dict[str, Optional[str]]] = {}

    for row_idx, record in enumerate(read_jsonl(chunks_path), start=1):
        if "__json_error__" in record:
            raise ValueError(f"json_decode_error: file={chunks_path}, row={row_idx}")
        for field in ["chunk_id", "doc_id", "chunk_text"]:
            if field not in record:
                raise ValueError(f"chunks 缺少字段: file={chunks_path}, row={row_idx}, field={field}")
        legacy_chunk_id = str(record["chunk_id"]).strip()
        doc_id = str(record["doc_id"]).strip()
        chunk_text = str(record["chunk_text"])
        canonical_chunk_id = build_canonical_chunk_id(doc_id=doc_id, chunk_text=chunk_text)
        entry = {
            "chunk_id": canonical_chunk_id,
            "doc_id": doc_id,
            "parent_chunk_id": canonical_chunk_id,
            "legacy_chunk_id": legacy_chunk_id,
        }
        by_chunk_alias[legacy_chunk_id] = entry
        by_chunk_id.setdefault(canonical_chunk_id, entry)

    for row_idx, record in enumerate(read_jsonl(raw_units_path), start=1):
        if "__json_error__" in record:
            raise ValueError(f"json_decode_error: file={raw_units_path}, row={row_idx}")
        for field in ["unit_id", "parent_chunk_id", "provenance"]:
            if field not in record:
                raise ValueError(f"raw_units 缺少字段: file={raw_units_path}, row={row_idx}, field={field}")
        if str(record["provenance"]) != "raw":
            continue
        unit_id = str(record["unit_id"]).strip()
        legacy_chunk_id = str(record["parent_chunk_id"]).strip()
        chunk_entry = by_chunk_alias.get(legacy_chunk_id)
        if chunk_entry is None:
            raise ValueError(
                f"raw_units 无法在 chunks registry 中定位真实 chunk_id: "
                f"file={raw_units_path}, row={row_idx}, unit_id={unit_id}, parent_chunk_id={legacy_chunk_id}"
            )
        by_unit[unit_id] = {
            "chunk_id": chunk_entry["chunk_id"],
            "doc_id": chunk_entry["doc_id"],
            "parent_chunk_id": chunk_entry["parent_chunk_id"],
            "legacy_chunk_id": legacy_chunk_id,
        }

    return {
        "by_unit": by_unit,
        "by_chunk_alias": by_chunk_alias,
        "by_chunk_id": by_chunk_id,
    }


def resolve_registry_entry(
    raw_id: Optional[str],
    registry: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> Optional[Dict[str, Optional[str]]]:
    if raw_id is None or is_derived_unit_id(raw_id):
        return None
    if raw_id in registry["by_unit"]:
        return registry["by_unit"][raw_id]
    if raw_id in registry["by_chunk_alias"]:
        return registry["by_chunk_alias"][raw_id]
    if raw_id in registry["by_chunk_id"]:
        return registry["by_chunk_id"][raw_id]
    return None


def validate_provenance_payload(
    errors: List[dict],
    split: str,
    qid: Optional[str],
    t: Optional[int],
    provenance: Any,
    *,
    unit_id: Optional[str],
    error_type: str,
    registry: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> None:
    if not isinstance(provenance, dict):
        add_error(errors, split, qid, t, error_type)
        return
    for key in ["chunk_id", "doc_id", "parent_chunk_id"]:
        if key not in provenance:
            add_error(errors, split, qid, t, error_type)
            return
    if is_derived_unit_id(unit_id):
        return
    chunk_id = provenance.get("chunk_id")
    parent_chunk_id = provenance.get("parent_chunk_id")
    if not isinstance(chunk_id, str) or not is_canonical_chunk_id(chunk_id):
        add_error(errors, split, qid, t, error_type)
        return
    if not isinstance(parent_chunk_id, str) or not is_canonical_chunk_id(parent_chunk_id):
        add_error(errors, split, qid, t, error_type)
        return
    doc_id = provenance.get("doc_id")
    if doc_id is not None and not isinstance(doc_id, str):
        add_error(errors, split, qid, t, error_type)
        return

    if is_pseudo_chunk_id(chunk_id, qid=qid, unit_id=unit_id) or is_pseudo_chunk_id(parent_chunk_id, qid=qid, unit_id=unit_id):
        add_error(errors, split, qid, t, "pseudo_chunk_id_not_allowed")
        return

    if chunk_id not in registry["by_chunk_id"] or parent_chunk_id not in registry["by_chunk_id"]:
        add_error(errors, split, qid, t, "chunk_id_not_found_in_raw_chunk_registry")
        return

    expected = resolve_registry_entry(unit_id, registry)
    if expected is None:
        add_error(errors, split, qid, t, "chunk_id_not_found_in_raw_chunk_registry")
        return
    if chunk_id != expected["chunk_id"] or parent_chunk_id != expected["parent_chunk_id"] or doc_id != expected["doc_id"]:
        add_error(errors, split, qid, t, "chunk_provenance_registry_mismatch")


def add_error(
    errors: List[dict],
    split: str,
    qid: Optional[str],
    t: Optional[int],
    error_type: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    error = {
        "split": split,
        "qid": qid,
        "t": t,
        "error_type": error_type,
    }
    if extra:
        error.update(extra)
    errors.append(error)


def collect_derived_unit_ids_from_record(record: dict) -> List[str]:
    out: List[str] = []
    state = record.get("state", {})
    candidates = record.get("candidates", {})
    labels = record.get("labels", {})
    h_t = state.get("H_t", []) if isinstance(state, dict) else []
    if isinstance(h_t, list):
        for item in h_t:
            if isinstance(item, dict) and is_derived_unit_id(item.get("unit_id")):
                out.append(item["unit_id"])
    if isinstance(candidates, dict):
        for key in ["G_t_final", "G_t_aux", "G_t_illegal", "C_t"]:
            value = candidates.get(key, [])
            if isinstance(value, list):
                out.extend([x for x in value if is_derived_unit_id(x)])
    s_t = state.get("S_t", {}) if isinstance(state, dict) else {}
    derived_refs = s_t.get("derived_refs", []) if isinstance(s_t, dict) else []
    if isinstance(derived_refs, list):
        for ref in derived_refs:
            if isinstance(ref, dict) and is_derived_unit_id(ref.get("unit_id")):
                out.append(ref["unit_id"])
    if isinstance(labels, dict):
        u_t_plus = labels.get("u_t_plus", {})
        if isinstance(u_t_plus, dict) and is_derived_unit_id(u_t_plus.get("unit_id")):
            out.append(u_t_plus["unit_id"])
        ranking_label = labels.get("ranking_label", {})
        if isinstance(ranking_label, dict):
            if is_derived_unit_id(ranking_label.get("positive_unit_id")):
                out.append(ranking_label["positive_unit_id"])
            negative_unit_ids = ranking_label.get("negative_unit_ids", [])
            if isinstance(negative_unit_ids, list):
                out.extend([x for x in negative_unit_ids if is_derived_unit_id(x)])
    if isinstance(state, dict):
        out.extend(extract_derived_unit_ids_from_text(state.get("K_t")))
    deduped: List[str] = []
    seen: Set[str] = set()
    for unit_id in out:
        if unit_id in seen:
            continue
        seen.add(unit_id)
        deduped.append(unit_id)
    return deduped


def normalize_stop_type(value: Any) -> str:
    x = str(value).strip().lower()
    mapping = {
        "terminal": "terminal",
        "terminal_positive": "terminal",
        "false-stop": "false-stop",
        "false_stop": "false-stop",
        "false_stop_negative": "false-stop",
        "near-terminal": "near-terminal",
        "near_terminal": "near-terminal",
        "near_terminal_negative": "near-terminal",
        "continue": "continue",
        "continue_negative": "continue",
    }
    return mapping.get(x, x)


def scan_forbidden_keys(obj: Any, path: str = "") -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else key
            if key in FORBIDDEN_DEBUG_KEYS:
                hits.append((key, next_path))
            hits.extend(scan_forbidden_keys(value, next_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            next_path = f"{path}[{idx}]"
            hits.extend(scan_forbidden_keys(item, next_path))
    return hits


def validate_record(
    split: str,
    record: dict,
    registry: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> List[dict]:
    errors: List[dict] = []
    qid = record.get("qid") if isinstance(record.get("qid"), str) else None
    t = record.get("t") if isinstance(record.get("t"), int) and not isinstance(record.get("t"), bool) else None

    if "__json_error__" in record:
        add_error(errors, split, None, None, "json_decode_error")
        return errors

    forbidden_hits = scan_forbidden_keys(record)
    for key, _ in forbidden_hits:
        add_error(errors, split, qid, t, f"forbidden_debug_field_{key}")

    expected_top = {"qid", "t", "build_meta", "question", "state", "candidates", "labels", "derived_payloads", "meta"}
    if set(record.keys()) != expected_top:
        add_error(errors, split, qid, t, "top_level_keys_invalid")

    if not isinstance(record.get("qid"), str):
        add_error(errors, split, qid, t, "qid_not_str")
    if not (isinstance(record.get("t"), int) and not isinstance(record.get("t"), bool)):
        add_error(errors, split, qid, t, "t_not_int")
    build_meta = record.get("build_meta")
    if not isinstance(build_meta, dict):
        add_error(errors, split, qid, t, "build_meta_not_dict")
        build_meta = {}
    run_id = build_meta.get("run_id")
    if not (isinstance(run_id, str) and run_id.strip()):
        add_error(errors, split, qid, t, "build_meta_run_id_invalid")
    source = build_meta.get("source")
    if not isinstance(source, str):
        add_error(errors, split, qid, t, "build_meta_source_not_str")
    elif source != "build_hotpotqa_samples_v4.py":
        add_error(errors, split, qid, t, "build_meta_source_invalid")
    meta_split = build_meta.get("split")
    if not isinstance(meta_split, str):
        add_error(errors, split, qid, t, "build_meta_split_not_str")
    elif meta_split != split:
        add_error(errors, split, qid, t, "build_meta_split_mismatch")
    if not isinstance(record.get("question"), str):
        add_error(errors, split, qid, t, "question_not_str")

    state = record.get("state")
    candidates = record.get("candidates")
    labels = record.get("labels")
    derived_payloads = record.get("derived_payloads")
    meta = record.get("meta")
    if not isinstance(state, dict):
        add_error(errors, split, qid, t, "state_not_dict")
        state = {}
    if not isinstance(candidates, dict):
        add_error(errors, split, qid, t, "candidates_not_dict")
        candidates = {}
    if not isinstance(labels, dict):
        add_error(errors, split, qid, t, "labels_not_dict")
        labels = {}
    if not isinstance(derived_payloads, dict):
        add_error(errors, split, qid, t, "derived_payloads_not_dict")
        derived_payloads = {}
    if not isinstance(meta, dict):
        add_error(errors, split, qid, t, "meta_not_dict")
        meta = {}

    h_t_len: Optional[int] = None
    c_t_set: Optional[Set[str]] = None

    expected_state = {"H_t", "A_t", "S_t", "K_t"}
    if set(state.keys()) != expected_state:
        add_error(errors, split, qid, t, "state_keys_invalid")

    h_t = state.get("H_t")
    if not isinstance(h_t, list):
        add_error(errors, split, qid, t, "state_h_t_not_list")
    else:
        h_t_len = len(h_t)
        for idx, item in enumerate(h_t):
            if not isinstance(item, dict):
                add_error(errors, split, qid, t, "state_h_t_item_not_dict")
                continue
            if set(item.keys()) != {"step_id", "unit_id", "chunk_id", "doc_id", "parent_chunk_id"}:
                add_error(errors, split, qid, t, "state_h_t_item_keys_invalid")
            step_id = item.get("step_id")
            unit_id = item.get("unit_id")
            if not (isinstance(step_id, int) and not isinstance(step_id, bool)):
                add_error(errors, split, qid, t, "state_h_t_step_id_not_int")
            elif step_id != idx:
                add_error(errors, split, qid, t, "state_h_t_step_id_not_contiguous")
            if not isinstance(unit_id, str):
                add_error(errors, split, qid, t, "state_h_t_unit_id_not_str")
            elif not is_derived_unit_id(unit_id):
                validate_provenance_payload(
                    errors,
                    split,
                    qid,
                    t,
                    {
                        "chunk_id": item.get("chunk_id"),
                        "doc_id": item.get("doc_id"),
                        "parent_chunk_id": item.get("parent_chunk_id"),
                    },
                    unit_id=unit_id,
                    error_type="missing_chunk_provenance_in_H_t",
                    registry=registry,
                )

    a_t = state.get("A_t")
    if not isinstance(a_t, dict):
        add_error(errors, split, qid, t, "state_a_t_not_dict")
    else:
        if set(a_t.keys()) != {"covered_target_ids", "k_bridge", "k_distinguish", "k_support", "coverage_trace"}:
            add_error(errors, split, qid, t, "state_a_t_keys_invalid")
        covered_target_ids = a_t.get("covered_target_ids")
        if not is_str_list(covered_target_ids):
            add_error(errors, split, qid, t, "state_a_t_covered_target_ids_invalid")
        elif list_has_duplicates(covered_target_ids):
            add_error(errors, split, qid, t, "state_a_t_covered_target_ids_duplicate")
        for key in ["k_bridge", "k_distinguish", "k_support"]:
            if not is_number(a_t.get(key)):
                add_error(errors, split, qid, t, f"state_a_t_{key}_not_number")
        coverage_trace = a_t.get("coverage_trace")
        if not isinstance(coverage_trace, dict):
            add_error(errors, split, qid, t, "state_a_t_coverage_trace_not_dict")
        else:
            for k, v in coverage_trace.items():
                if not isinstance(k, str):
                    add_error(errors, split, qid, t, "state_a_t_coverage_trace_entry_invalid")
                    break
                if not is_canonical_chunk_id(k):
                    add_error(errors, split, qid, t, "state_a_t_coverage_trace_not_chunk_level")
                    break
                if not isinstance(v, dict):
                    add_error(errors, split, qid, t, "missing_chunk_provenance_in_coverage_trace")
                    break
                if not isinstance(v.get("unit_id"), str):
                    add_error(errors, split, qid, t, "missing_chunk_provenance_in_coverage_trace")
                    break
                expected = resolve_registry_entry(v.get("unit_id"), registry)
                if expected is None:
                    add_error(errors, split, qid, t, "chunk_id_not_found_in_raw_chunk_registry")
                    break
                if k != expected["chunk_id"]:
                    add_error(errors, split, qid, t, "chunk_provenance_registry_mismatch")
                    break
                validate_provenance_payload(
                    errors,
                    split,
                    qid,
                    t,
                    v,
                    unit_id=v.get("unit_id"),
                    error_type="missing_chunk_provenance_in_coverage_trace",
                    registry=registry,
                )

    s_t = state.get("S_t")
    if not isinstance(s_t, dict):
        add_error(errors, split, qid, t, "state_s_t_not_dict")
    else:
        if set(s_t.keys()) != {"raw_refs", "derived_refs", "last_added_unit_id", "last_updated_step"}:
            add_error(errors, split, qid, t, "state_s_t_keys_invalid")
        raw_refs = s_t.get("raw_refs")
        if not isinstance(raw_refs, list):
            add_error(errors, split, qid, t, "state_s_t_raw_refs_not_list")
        else:
            seen_chunk_ids: Set[str] = set()
            for ref in raw_refs:
                if not isinstance(ref, dict):
                    add_error(errors, split, qid, t, "state_s_t_raw_refs_item_not_dict")
                    continue
                if set(ref.keys()) != {"unit_id", "chunk_id", "doc_id", "parent_chunk_id", "added_step", "used_in_summary_count", "selected_count"}:
                    add_error(errors, split, qid, t, "state_s_t_raw_refs_item_keys_invalid")
                unit_id = ref.get("unit_id")
                if not isinstance(unit_id, str):
                    add_error(errors, split, qid, t, "state_s_t_raw_refs_unit_id_not_str")
                chunk_id = ref.get("chunk_id")
                if not isinstance(chunk_id, str):
                    add_error(errors, split, qid, t, "state_s_t_raw_refs_chunk_id_not_str")
                elif not is_canonical_chunk_id(chunk_id):
                    add_error(errors, split, qid, t, "state_s_t_raw_refs_not_chunk_level")
                elif chunk_id in seen_chunk_ids:
                    add_error(errors, split, qid, t, "state_s_t_raw_refs_duplicate_chunk_id")
                else:
                    seen_chunk_ids.add(chunk_id)
                validate_provenance_payload(
                    errors,
                    split,
                    qid,
                    t,
                    {
                        "chunk_id": ref.get("chunk_id"),
                        "doc_id": ref.get("doc_id"),
                        "parent_chunk_id": ref.get("parent_chunk_id"),
                    },
                    unit_id=unit_id if isinstance(unit_id, str) else None,
                    error_type="missing_chunk_provenance_in_raw_refs",
                    registry=registry,
                )
                for key in ["added_step", "used_in_summary_count", "selected_count"]:
                    if not (isinstance(ref.get(key), int) and not isinstance(ref.get(key), bool)):
                        add_error(errors, split, qid, t, f"state_s_t_raw_refs_{key}_not_int")
        derived_refs = s_t.get("derived_refs")
        if not isinstance(derived_refs, list):
            add_error(errors, split, qid, t, "state_s_t_derived_refs_not_list")
        else:
            seen_unit_ids: Set[str] = set()
            for ref in derived_refs:
                if not isinstance(ref, dict):
                    add_error(errors, split, qid, t, "state_s_t_derived_refs_item_not_dict")
                    continue
                if set(ref.keys()) != {"unit_id", "added_step", "used_in_summary_count", "selected_count"}:
                    add_error(errors, split, qid, t, "state_s_t_derived_refs_item_keys_invalid")
                unit_id = ref.get("unit_id")
                if not isinstance(unit_id, str):
                    add_error(errors, split, qid, t, "state_s_t_derived_refs_unit_id_not_str")
                elif unit_id in seen_unit_ids:
                    add_error(errors, split, qid, t, "state_s_t_derived_refs_duplicate_unit_id")
                else:
                    seen_unit_ids.add(unit_id)
                for key in ["added_step", "used_in_summary_count", "selected_count"]:
                    if not (isinstance(ref.get(key), int) and not isinstance(ref.get(key), bool)):
                        add_error(errors, split, qid, t, f"state_s_t_derived_refs_{key}_not_int")
        last_added_unit_id = s_t.get("last_added_unit_id")
        if last_added_unit_id is not None and not isinstance(last_added_unit_id, str):
            add_error(errors, split, qid, t, "state_s_t_last_added_unit_id_invalid")
        if not (isinstance(s_t.get("last_updated_step"), int) and not isinstance(s_t.get("last_updated_step"), bool)):
            add_error(errors, split, qid, t, "state_s_t_last_updated_step_not_int")

    if not isinstance(state.get("K_t"), str):
        add_error(errors, split, qid, t, "state_k_t_not_str")
    elif "[missing derived payload]" in state.get("K_t", ""):
        add_error(errors, split, qid, t, "missing_derived_payload_in_K_t")

    expected_candidates = {"R_t", "G_t_final", "G_t_aux", "G_t_illegal", "C_t", "candidate_provenance", "aux_candidate_provenance"}
    if set(candidates.keys()) != expected_candidates:
        add_error(errors, split, qid, t, "candidates_keys_invalid")
    candidate_lists: Dict[str, List[str]] = {}
    for key in ["R_t", "G_t_final", "G_t_aux", "G_t_illegal", "C_t"]:
        value = candidates.get(key)
        if not is_str_list(value):
            add_error(errors, split, qid, t, f"candidates_{key}_invalid")
            candidate_lists[key] = []
        else:
            candidate_lists[key] = value
            if list_has_duplicates(value):
                add_error(errors, split, qid, t, f"candidates_{key}_duplicate")
    expected_c_t = []
    seen = set()
    for unit_id in candidate_lists["R_t"] + candidate_lists["G_t_final"]:
        if unit_id not in seen:
            seen.add(unit_id)
            expected_c_t.append(unit_id)
    if candidate_lists["C_t"] != expected_c_t:
        add_error(errors, split, qid, t, "candidates_c_t_not_union")
    c_t_set = set(candidate_lists["C_t"])
    candidate_provenance = candidates.get("candidate_provenance")
    if not isinstance(candidate_provenance, dict):
        add_error(errors, split, qid, t, "missing_chunk_provenance_in_candidate_provenance")
        candidate_provenance = {}
    if set(candidate_provenance.keys()) != c_t_set:
        add_error(errors, split, qid, t, "missing_chunk_provenance_in_candidate_provenance")
    for unit_id in c_t_set:
        if not is_derived_unit_id(unit_id):
            validate_provenance_payload(
                errors,
                split,
                qid,
                t,
                candidate_provenance.get(unit_id),
                unit_id=unit_id,
                error_type="missing_chunk_provenance_in_candidate_provenance",
                registry=registry,
            )
    aux_candidate_provenance = candidates.get("aux_candidate_provenance")
    if not isinstance(aux_candidate_provenance, dict):
        add_error(errors, split, qid, t, "aux_candidate_provenance_invalid")
        aux_candidate_provenance = {}
    expected_aux_ids = set(candidate_lists["G_t_aux"] + candidate_lists["G_t_illegal"])
    if set(aux_candidate_provenance.keys()) != expected_aux_ids:
        add_error(errors, split, qid, t, "aux_candidate_provenance_keys_invalid")
    for unit_id, provenance in aux_candidate_provenance.items():
        if not isinstance(provenance, dict):
            add_error(errors, split, qid, t, "aux_candidate_provenance_payload_invalid")
            continue
        if provenance.get("provenance_scope") != "aux":
            add_error(errors, split, qid, t, "aux_candidate_provenance_scope_invalid")
        if provenance.get("candidate_status") not in {"aux", "illegal"}:
            add_error(errors, split, qid, t, "aux_candidate_provenance_status_invalid")

    expected_labels = {"u_t_plus", "d_t_star", "c_t_star", "ranking_label", "stop_label"}
    if set(labels.keys()) != expected_labels:
        add_error(errors, split, qid, t, "labels_keys_invalid")

    u_t_plus = labels.get("u_t_plus")
    ranking_label = labels.get("ranking_label")
    d_t_star = labels.get("d_t_star")
    c_t_star = labels.get("c_t_star")
    stop_label = labels.get("stop_label")

    if not isinstance(u_t_plus, dict):
        add_error(errors, split, qid, t, "labels_u_t_plus_not_dict")
        u_t_plus = {}
    else:
        if set(u_t_plus.keys()) != {"step_id", "unit_id", "chunk_id", "doc_id", "parent_chunk_id"}:
            add_error(errors, split, qid, t, "labels_u_t_plus_keys_invalid")
        if not (isinstance(u_t_plus.get("step_id"), int) and not isinstance(u_t_plus.get("step_id"), bool)):
            add_error(errors, split, qid, t, "labels_u_t_plus_step_id_not_int")
        if not isinstance(u_t_plus.get("unit_id"), str):
            add_error(errors, split, qid, t, "labels_u_t_plus_unit_id_not_str")
        elif u_t_plus.get("unit_id") == "__STOP__":
            add_error(errors, split, qid, t, "labels_u_t_plus_stop_token_forbidden")
        elif not is_derived_unit_id(u_t_plus.get("unit_id")):
            validate_provenance_payload(
                errors,
                split,
                qid,
                t,
                {
                    "chunk_id": u_t_plus.get("chunk_id"),
                    "doc_id": u_t_plus.get("doc_id"),
                    "parent_chunk_id": u_t_plus.get("parent_chunk_id"),
                },
                unit_id=u_t_plus.get("unit_id"),
                error_type="missing_chunk_provenance_in_u_t_plus",
                registry=registry,
            )

    if not isinstance(d_t_star, dict):
        add_error(errors, split, qid, t, "labels_d_t_star_not_dict")
    else:
        if set(d_t_star.keys()) != {"d_raw", "d_br", "d_dis", "d_sup", "d_der"}:
            add_error(errors, split, qid, t, "labels_d_t_star_keys_invalid")
        for key in ["d_raw", "d_br", "d_dis", "d_sup", "d_der"]:
            if not is_number(d_t_star.get(key)):
                add_error(errors, split, qid, t, f"labels_d_t_star_{key}_invalid")

    if not isinstance(c_t_star, dict):
        add_error(errors, split, qid, t, "labels_c_t_star_not_dict")
    else:
        if set(c_t_star.keys()) != {"c_raw", "c_br", "c_dis", "c_sup", "c_der"}:
            add_error(errors, split, qid, t, "labels_c_t_star_keys_invalid")
        for key in ["c_raw", "c_br", "c_dis", "c_sup", "c_der"]:
            if not is_number(c_t_star.get(key)):
                add_error(errors, split, qid, t, f"labels_c_t_star_{key}_invalid")

    if not isinstance(ranking_label, dict):
        add_error(errors, split, qid, t, "labels_ranking_label_not_dict")
        ranking_label = {}
    else:
        if set(ranking_label.keys()) != {"positive_unit_id", "negative_unit_ids", "positive_provenance", "negative_provenance"}:
            add_error(errors, split, qid, t, "labels_ranking_label_keys_invalid")
        positive_unit_id = ranking_label.get("positive_unit_id")
        negative_unit_ids = ranking_label.get("negative_unit_ids")
        if not isinstance(positive_unit_id, str):
            add_error(errors, split, qid, t, "labels_ranking_label_positive_unit_id_not_str")
        elif positive_unit_id == "__STOP__":
            add_error(errors, split, qid, t, "labels_ranking_label_stop_token_forbidden")
        if not is_str_list(negative_unit_ids):
            add_error(errors, split, qid, t, "labels_ranking_label_negative_unit_ids_invalid")
            negative_unit_ids = []
        elif list_has_duplicates(negative_unit_ids):
            add_error(errors, split, qid, t, "labels_ranking_label_negative_unit_ids_duplicate")
        if not is_derived_unit_id(positive_unit_id):
            validate_provenance_payload(
                errors,
                split,
                qid,
                t,
                ranking_label.get("positive_provenance"),
                unit_id=positive_unit_id,
                error_type="missing_chunk_provenance_in_ranking_label",
                registry=registry,
            )
        negative_provenance = ranking_label.get("negative_provenance")
        if not isinstance(negative_provenance, dict):
            add_error(errors, split, qid, t, "missing_chunk_provenance_in_ranking_label")
        else:
            if set(negative_provenance.keys()) != set(negative_unit_ids):
                add_error(errors, split, qid, t, "missing_chunk_provenance_in_ranking_label")
            for unit_id in negative_unit_ids:
                if not is_derived_unit_id(unit_id):
                    validate_provenance_payload(
                        errors,
                        split,
                        qid,
                        t,
                        negative_provenance.get(unit_id),
                        unit_id=unit_id,
                        error_type="missing_chunk_provenance_in_ranking_label",
                        registry=registry,
                    )

    if not isinstance(stop_label, dict):
        add_error(errors, split, qid, t, "labels_stop_label_not_dict")
        stop_label = {}
    else:
        if set(stop_label.keys()) != {"should_stop", "label_type"}:
            add_error(errors, split, qid, t, "labels_stop_label_keys_invalid")
        if not isinstance(stop_label.get("should_stop"), bool):
            add_error(errors, split, qid, t, "labels_stop_label_should_stop_not_bool")
        label_type = normalize_stop_type(stop_label.get("label_type"))
        if not isinstance(stop_label.get("label_type"), str):
            add_error(errors, split, qid, t, "labels_stop_label_label_type_not_str")
        elif label_type not in ALLOWED_STOP_LABEL_TYPE:
            add_error(errors, split, qid, t, "labels_stop_label_label_type_invalid")
        elif stop_label.get("should_stop") is True and label_type != "terminal":
            add_error(errors, split, qid, t, "labels_stop_label_terminal_consistency_error")
        elif stop_label.get("should_stop") is False and label_type == "terminal":
            add_error(errors, split, qid, t, "labels_stop_label_terminal_consistency_error")
        elif label_type == "near-terminal":
            trajectory_status = meta.get("trajectory_status") if isinstance(meta, dict) else {}
            terminal_step = trajectory_status.get("terminal_step")
            if stop_label.get("should_stop") is not False:
                add_error(errors, split, qid, t, "labels_stop_label_near_terminal_should_not_stop")
            elif not (isinstance(terminal_step, int) and t < terminal_step):
                add_error(errors, split, qid, t, "labels_stop_label_near_terminal_window_invalid")

    derived_unit_ids = collect_derived_unit_ids_from_record(record)
    if set(derived_payloads.keys()) != set(derived_unit_ids):
        add_error(errors, split, qid, t, "missing_derived_payload")
    for unit_id in derived_unit_ids:
        payload = derived_payloads.get(unit_id)
        if not isinstance(payload, dict):
            add_error(errors, split, qid, t, "missing_derived_payload")
            continue
        text = payload.get("text")
        note_type = payload.get("type")
        source_unit_ids = payload.get("source_unit_ids")
        if not isinstance(text, str) or not text.strip():
            add_error(errors, split, qid, t, "empty_derived_text")
        if not isinstance(note_type, str) or not note_type.strip():
            add_error(errors, split, qid, t, "missing_derived_type")
        if not (isinstance(source_unit_ids, list) and source_unit_ids and all(isinstance(x, str) for x in source_unit_ids)):
            add_error(errors, split, qid, t, "missing_derived_source_unit_ids")
        else:
            bad = False
            for source_unit_id in source_unit_ids:
                if is_derived_unit_id(source_unit_id):
                    bad = True
                    break
                if resolve_registry_entry(source_unit_id, registry) is None:
                    bad = True
                    break
            if bad:
                add_error(errors, split, qid, t, "invalid_derived_source_unit_ids")

    expected_meta = {"trajectory_status", "progress_flag", "keep_prefix"}
    if set(meta.keys()) != expected_meta:
        add_error(errors, split, qid, t, "meta_keys_invalid")
    trajectory_status = meta.get("trajectory_status")
    if not isinstance(trajectory_status, dict):
        add_error(errors, split, qid, t, "meta_trajectory_status_not_dict")
    else:
        if set(trajectory_status.keys()) != {"status", "terminal_step", "abort_reason"}:
            add_error(errors, split, qid, t, "meta_trajectory_status_keys_invalid")
        status = trajectory_status.get("status")
        terminal_step = trajectory_status.get("terminal_step")
        abort_reason = trajectory_status.get("abort_reason")
        if not isinstance(status, str):
            add_error(errors, split, qid, t, "meta_trajectory_status_status_not_str")
        elif status not in ALLOWED_TRAJECTORY_STATUS:
            add_error(errors, split, qid, t, "meta_trajectory_status_status_invalid")
        if status == "success":
            if not (isinstance(terminal_step, int) and not isinstance(terminal_step, bool)):
                add_error(errors, split, qid, t, "meta_trajectory_status_terminal_step_invalid")
            elif isinstance(t, int) and not isinstance(t, bool) and t >= terminal_step:
                add_error(
                    errors,
                    split,
                    qid,
                    t,
                    "prefix_kept_at_or_after_terminal_step",
                    {"terminal_step": terminal_step},
                )
            if abort_reason is not None:
                add_error(errors, split, qid, t, "meta_trajectory_status_abort_reason_invalid")
        elif status in {"failed_but_progressive", "failed_stalled"}:
            if terminal_step is not None:
                add_error(errors, split, qid, t, "meta_trajectory_status_terminal_step_invalid")
            if not isinstance(abort_reason, str):
                add_error(errors, split, qid, t, "meta_trajectory_status_abort_reason_invalid")

    if not isinstance(meta.get("progress_flag"), str):
        add_error(errors, split, qid, t, "meta_progress_flag_not_str")
    elif meta.get("progress_flag") not in ALLOWED_PROGRESS_FLAG:
        add_error(errors, split, qid, t, "meta_progress_flag_invalid")
    if not isinstance(meta.get("keep_prefix"), bool):
        add_error(errors, split, qid, t, "meta_keep_prefix_not_bool")
    elif meta.get("keep_prefix") is not True:
        add_error(errors, split, qid, t, "meta_keep_prefix_not_true")

    if h_t_len is not None and isinstance(u_t_plus.get("step_id"), int) and not isinstance(u_t_plus.get("step_id"), bool):
        if u_t_plus["step_id"] != h_t_len:
            add_error(errors, split, qid, t, "cross_step_id_not_equal_len_h_t")
    if isinstance(u_t_plus.get("unit_id"), str) and isinstance(ranking_label.get("positive_unit_id"), str):
        if u_t_plus["unit_id"] != ranking_label["positive_unit_id"]:
            add_error(errors, split, qid, t, "cross_positive_unit_mismatch")
    positive_unit_id = ranking_label.get("positive_unit_id")
    negative_unit_ids = ranking_label.get("negative_unit_ids") if isinstance(ranking_label.get("negative_unit_ids"), list) else []
    if isinstance(positive_unit_id, str) and c_t_set is not None and positive_unit_id not in c_t_set:
        add_error(errors, split, qid, t, "cross_positive_not_in_c_t")
    if c_t_set is not None and isinstance(negative_unit_ids, list):
        for unit_id in negative_unit_ids:
            if unit_id not in c_t_set:
                add_error(errors, split, qid, t, "cross_negative_not_in_c_t")
                break
    if isinstance(positive_unit_id, str) and isinstance(negative_unit_ids, list) and positive_unit_id in negative_unit_ids:
        add_error(errors, split, qid, t, "cross_positive_in_negative")

    return errors


def validate_split(
    split: str,
    path: Path,
    registry: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> List[dict]:
    errors: List[dict] = []
    seen_keys: Set[Tuple[str, int]] = set()
    for record in read_jsonl(path):
        record_errors = validate_record(split, record, registry)
        errors.extend(record_errors)
        qid = record.get("qid") if isinstance(record.get("qid"), str) else None
        t = record.get("t") if isinstance(record.get("t"), int) and not isinstance(record.get("t"), bool) else None
        if qid is not None and t is not None:
            key = (qid, t)
            if key in seen_keys:
                add_error(errors, split, qid, t, "duplicate_qid_t")
            else:
                seen_keys.add(key)
    return errors


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    samples_base = project_root / DEFAULT_SAMPLES_BASE
    base_dir = project_root / DEFAULT_BASE
    unit_registry_dir = base_dir / "unit_registry"
    index_store_dir = base_dir / "index_store"

    all_errors: List[dict] = []
    for split in SPLITS:
        path = samples_base / f"{split}.jsonl"
        if not path.exists():
            all_errors.append(
                {
                    "split": split,
                    "qid": None,
                    "t": None,
                    "error_type": "file_missing",
                }
            )
            continue
        raw_units_path = unit_registry_dir / f"raw_units_{split}.jsonl"
        chunks_path = index_store_dir / f"chunks_{split}.jsonl"
        if not raw_units_path.exists():
            all_errors.append({"split": split, "qid": None, "t": None, "error_type": "raw_units_file_missing"})
            continue
        if not chunks_path.exists():
            all_errors.append({"split": split, "qid": None, "t": None, "error_type": "chunks_file_missing"})
            continue
        registry = load_raw_chunk_registry(raw_units_path, chunks_path)
        all_errors.extend(validate_split(split, path, registry))

    counter = Counter(error["error_type"] for error in all_errors)
    print("validate_hotpotqa_samples_v4")
    if not all_errors:
        print("errors=0")
        return 0

    print("error_counts:")
    for error_type in sorted(counter.keys()):
        print(f"  {error_type}: {counter[error_type]}")
    print("error_details:")
    for error in all_errors:
        terminal_step_suffix = ""
        if "terminal_step" in error:
            terminal_step_suffix = f" terminal_step={error['terminal_step']}"
        print(
            f"  split={error['split']} qid={error['qid']} t={error['t']}{terminal_step_suffix} "
            f"error_type={error['error_type']}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
