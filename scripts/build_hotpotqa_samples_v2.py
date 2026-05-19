import hashlib
import json
import re
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v2"
DEFAULT_SAMPLES_BASE = "data/hotpotqa_distractor_v2/samples"
TOP_RAW_J = 3
MAX_DERIVED_CANDIDATES = 4
ALLOWED_DERIVED_TYPES = {"bridge_note", "verification_note"}

ALLOWED_TRAJECTORY_STATUS = {
    "success",
    "failed_but_progressive",
    "failed_stalled",
}
ALLOWED_PROGRESS_FLAG = {
    "progress",
    "stall",
}
FORBIDDEN_DEBUG_KEYS = {
    "candidate_debug",
    "coverage_debug",
    "debug_info",
    "delta_covered_targets",
    "derived_debug",
    "gold_answer",
    "judge_output",
    "need_derived",
    "normalized_gold",
    "normalized_pred",
    "pred_answer",
    "probe",
    "probe_answer",
    "query_debug",
    "retrieval_repeat_ratio",
    "stop_candidate",
    "stop_debug",
    "stop_probe",
    "triggered_propose_derived",
    "covered_target_count",
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


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def expect_dict(x: Any, *, name: str) -> dict:
    if not isinstance(x, dict):
        raise ValueError(f"{name} 必须是 dict，当前得到: {type(x)}")
    return x


def expect_list(x: Any, *, name: str) -> list:
    if not isinstance(x, list):
        raise ValueError(f"{name} 必须是 list，当前得到: {type(x)}")
    return x


def expect_str(x: Any, *, name: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"{name} 必须是 str，当前得到: {type(x)}")
    return x


def expect_bool(x: Any, *, name: str) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"{name} 必须是 bool，当前得到: {type(x)}")
    return x


def expect_int(x: Any, *, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError(f"{name} 必须是 int，当前得到: {type(x)}")
    return x


def normalize_unit_id_list(xs: Any, *, name: str) -> List[str]:
    expect_list(xs, name=name)
    out: List[str] = []
    seen: Set[str] = set()
    for idx, item in enumerate(xs):
        if not isinstance(item, str):
            raise ValueError(f"{name}[{idx}] 必须是 str，当前得到: {type(item)}")
        if item in seen:
            raise ValueError(f"{name} 中出现重复 unit_id: {item}")
        seen.add(item)
        out.append(item)
    return out


def looks_like_raw_unit_id(value: str) -> bool:
    parts = str(value).split("::")
    return "::derived::" not in str(value) and len(parts) >= 3 and parts[-1].isdigit()


def normalize_text_for_chunk_identity(text: Any, *, name: str) -> str:
    return " ".join(expect_str(text, name=name).split())


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


def build_derived_unit_id(qid: str, idx: int) -> str:
    return f"{qid}::derived::{idx}"


def parse_derived_idx(unit_id: str) -> int:
    if not is_derived_unit_id(unit_id):
        raise ValueError(f"不是 derived unit_id: {unit_id}")
    try:
        return int(str(unit_id).rsplit("::", 1)[1])
    except Exception as e:
        raise ValueError(f"无法解析 derived idx: {unit_id}") from e


DERIVED_UNIT_ID_RE = re.compile(r"\b[0-9A-Za-z]+::derived::\d+\b")


def extract_derived_unit_ids_from_text(text: Any) -> List[str]:
    if not isinstance(text, str) or "::derived::" not in text:
        return []
    return merge_unique_in_order(DERIVED_UNIT_ID_RE.findall(text))


def shorten(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def extract_goal_from_question(question: str) -> str:
    q = question.strip().rstrip("?").lower()
    patterns = [
        (r"which magazine was started first", "publication founding time"),
        (r"which .* was started first", "founding time comparison"),
        (r"when was .* founded", "founding time"),
        (r"what year was .* founded", "founding time"),
        (r"which university did .* attend", "university attended"),
        (r"who acquired .*", "acquirer"),
        (r"who wrote .*", "author"),
        (r"where was .* born", "birthplace"),
        (r"when did .* die", "death time"),
    ]
    for pattern, goal in patterns:
        if re.search(pattern, q):
            return goal
    return "more evidence for the question"


def load_raw_provenance_index(
    raw_units_path: Path,
    chunks_path: Path,
) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    by_unit: Dict[str, Dict[str, Optional[str]]] = {}
    by_chunk_alias: Dict[str, Dict[str, Optional[str]]] = {}
    by_chunk_id: Dict[str, Dict[str, Optional[str]]] = {}

    for row_idx, record in enumerate(read_jsonl(chunks_path), start=1):
        required = ["chunk_id", "doc_id", "chunk_text"]
        for field in required:
            if field not in record:
                raise ValueError(f"chunks 缺少字段: file={chunks_path}, row={row_idx}, field={field}")
        legacy_chunk_id = expect_str(record["chunk_id"], name=f"chunks[{row_idx}].chunk_id").strip()
        doc_id = expect_str(record["doc_id"], name=f"chunks[{row_idx}].doc_id").strip()
        chunk_text = expect_str(record["chunk_text"], name=f"chunks[{row_idx}].chunk_text")
        canonical_chunk_id = build_canonical_chunk_id(doc_id=doc_id, chunk_text=chunk_text)
        entry = {
            "chunk_id": canonical_chunk_id,
            "doc_id": doc_id,
            "parent_chunk_id": canonical_chunk_id,
            "legacy_chunk_id": legacy_chunk_id,
            "full_chunk_text": chunk_text,
        }
        by_chunk_alias[legacy_chunk_id] = entry
        by_chunk_id.setdefault(canonical_chunk_id, entry)

    for row_idx, record in enumerate(read_jsonl(raw_units_path), start=1):
        required = ["unit_id", "doc_id", "parent_chunk_id", "provenance"]
        for field in required:
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
        entry = {
            "unit_id": unit_id,
            "chunk_id": chunk_entry["chunk_id"],
            "doc_id": chunk_entry["doc_id"],
            "parent_chunk_id": chunk_entry["parent_chunk_id"],
            "legacy_chunk_id": legacy_chunk_id,
            "text": None if record.get("text") is None else str(record.get("text")),
            "full_chunk_text": chunk_entry.get("full_chunk_text"),
        }
        by_unit[unit_id] = entry

    return {
        "by_unit": by_unit,
        "by_chunk_alias": by_chunk_alias,
        "by_chunk_id": by_chunk_id,
    }


def load_derived_harvest_index(path: Path) -> Dict[str, Any]:
    by_unit: Dict[str, Dict[str, Any]] = {}
    by_key: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "t" not in record or "G_t_harvest" not in record:
            raise ValueError(f"derived_harvest 缺少字段: file={path}, row={row_idx}")
        qid = str(record["qid"])
        t = int(record["t"])
        harvest = expect_list(record["G_t_harvest"], name=f"derived_harvest[{qid},{t}].G_t_harvest")
        payloads_for_key: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(harvest):
            item = expect_dict(item, name=f"derived_harvest[{qid},{t}][{idx}]")
            for field in ["unit_id", "text", "type", "source_unit_ids"]:
                if field not in item:
                    raise ValueError(
                        f"derived_harvest 缺少字段: file={path}, row={row_idx}, field={field}"
                    )
            unit_id = str(item["unit_id"])
            payload = {
                "text": str(item["text"]).strip(),
                "type": str(item["type"]).strip(),
                "source_unit_ids": [str(x) for x in expect_list(item["source_unit_ids"], name=f"{unit_id}.source_unit_ids")],
            }
            payloads_for_key[unit_id] = payload
            prev = by_unit.get(unit_id)
            if prev is not None and prev != payload:
                raise ValueError(f"derived_harvest 中同一 derived unit_id payload 不一致: {unit_id}")
            by_unit[unit_id] = payload
        by_key[(qid, t)] = payloads_for_key
    return {"by_unit": by_unit, "by_key": by_key}


def load_derived_cache_index(path: Path) -> Dict[str, List[dict]]:
    by_qid: Dict[str, List[dict]] = {}
    if not path.exists():
        return by_qid
    for cache_file in sorted(path.glob("*.json")):
        record = json.loads(cache_file.read_text(encoding="utf-8"))
        candidates = record.get("derived_candidates", [])
        if not isinstance(candidates, list) or not candidates:
            continue
        qids = set()
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            source_unit_ids = cand.get("source_unit_ids", [])
            if not isinstance(source_unit_ids, list):
                continue
            for unit_id in source_unit_ids:
                if looks_like_raw_unit_id(str(unit_id)):
                    qids.add(str(unit_id).split("::", 1)[0])
        if not qids:
            continue
        for qid in sorted(qids):
            by_qid.setdefault(qid, []).append(record)
    return by_qid


def resolve_raw_registry_entry(
    raw_id: str,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
    *,
    name: str,
) -> Dict[str, Optional[str]]:
    if is_derived_unit_id(raw_id):
        raise ValueError(f"{name} 不应传入 derived unit_id: {raw_id}")

    by_unit = provenance_index["by_unit"]
    by_chunk_alias = provenance_index["by_chunk_alias"]
    by_chunk_id = provenance_index["by_chunk_id"]

    unit_entry = by_unit.get(raw_id)
    if unit_entry is not None:
        return unit_entry

    chunk_entry = by_chunk_alias.get(raw_id)
    if chunk_entry is not None:
        return chunk_entry

    chunk_entry = by_chunk_id.get(raw_id)
    if chunk_entry is not None:
        return chunk_entry

    raise ValueError(f"{name} 无法从 raw chunk registry 解析真实 chunk_id: {raw_id}")


def build_raw_provenance(unit_id: str, provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]]) -> Dict[str, Optional[str]]:
    by_unit = provenance_index["by_unit"]

    if is_derived_unit_id(unit_id):
        return {
            "chunk_id": None,
            "doc_id": None,
            "parent_chunk_id": None,
        }

    unit_entry = by_unit.get(unit_id)
    if unit_entry is not None:
        return {
            "chunk_id": unit_entry["chunk_id"],
            "doc_id": unit_entry["doc_id"],
            "parent_chunk_id": unit_entry["parent_chunk_id"],
        }

    chunk_entry = resolve_raw_registry_entry(
        unit_id,
        provenance_index,
        name="raw_provenance.unit_id",
    )
    return {
        "chunk_id": chunk_entry["chunk_id"],
        "doc_id": chunk_entry["doc_id"],
        "parent_chunk_id": chunk_entry["parent_chunk_id"],
    }


def build_unit_with_provenance(unit_id: str, provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]]) -> Dict[str, Optional[str]]:
    provenance = build_raw_provenance(unit_id, provenance_index)
    return {
        "unit_id": unit_id,
        "chunk_id": provenance["chunk_id"],
        "doc_id": provenance["doc_id"],
        "parent_chunk_id": provenance["parent_chunk_id"],
    }


def build_candidate_provenance(
    candidate_lists: List[List[str]],
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> Dict[str, Dict[str, Optional[str]]]:
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for unit_id in merge_unique_in_order([x for xs in candidate_lists for x in xs]):
        out[unit_id] = build_raw_provenance(unit_id, provenance_index)
    return out


def build_negative_provenance(
    unit_ids: List[str],
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> Dict[str, Dict[str, Optional[str]]]:
    return {
        unit_id: build_raw_provenance(unit_id, provenance_index)
        for unit_id in unit_ids
    }


def get_recent_raw_texts_for_derived_cache(
    s_t: dict,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
    n: int = 2,
) -> List[str]:
    raw_refs = s_t.get("raw_refs", [])
    if not isinstance(raw_refs, list):
        return []
    recent = []
    for item in raw_refs[-n:]:
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit = provenance_index["by_unit"].get(str(item["unit_id"]))
        text = None if unit is None else unit.get("text")
        if isinstance(text, str) and text.strip():
            recent.append(text)
    return recent


def get_last_derived_text_for_derived_cache(
    s_t: dict,
    derived_runtime: Dict[str, Any],
) -> Optional[str]:
    derived_refs = s_t.get("derived_refs", [])
    if not isinstance(derived_refs, list) or not derived_refs:
        return None
    last_ref = derived_refs[-1]
    if not isinstance(last_ref, dict) or "unit_id" not in last_ref:
        return None
    payload = derived_runtime["resolved_by_unit"].get(str(last_ref["unit_id"]))
    if payload is None:
        return None
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return text


def build_state_summary_for_derived_cache(
    *,
    question: str,
    s_t: dict,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
    derived_runtime: Dict[str, Any],
) -> str:
    lines = []
    for text in get_recent_raw_texts_for_derived_cache(s_t, provenance_index, n=2)[:2]:
        lines.append("Evidence: " + shorten(text, 90))
    last_note = get_last_derived_text_for_derived_cache(s_t, derived_runtime)
    if last_note:
        lines.append("Note: " + shorten(last_note, 70))
    lines.append("Need: " + extract_goal_from_question(question))
    return "\n".join(lines)


def build_top_raw_candidates_for_derived_cache(
    r_t: List[str],
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> List[dict]:
    out = []
    for unit_id in r_t[:TOP_RAW_J]:
        unit = provenance_index["by_unit"].get(unit_id)
        if unit is None:
            continue
        text = unit.get("text")
        doc_id = unit.get("doc_id")
        if not isinstance(text, str) or not isinstance(doc_id, str):
            continue
        out.append({"unit_id": unit_id, "text": text, "doc_id": doc_id})
    return out


def collect_derived_unit_ids(
    *,
    state_block: dict,
    candidates_block: dict,
    labels_block: dict,
    k_t_text: Optional[str] = None,
) -> List[str]:
    collected: List[str] = []
    collected.extend([item["unit_id"] for item in state_block["H_t"] if is_derived_unit_id(item["unit_id"])])
    for key in ["G_t_final", "G_t_aux", "G_t_illegal", "C_t"]:
        collected.extend([x for x in candidates_block[key] if is_derived_unit_id(x)])
    collected.extend(
        [ref["unit_id"] for ref in state_block["S_t"]["derived_refs"] if is_derived_unit_id(ref["unit_id"])]
    )
    for unit_id in [
        labels_block["u_t_plus"]["unit_id"],
        labels_block["ranking_label"]["positive_unit_id"],
        *labels_block["ranking_label"]["negative_unit_ids"],
    ]:
        if is_derived_unit_id(unit_id):
            collected.append(unit_id)
    collected.extend(extract_derived_unit_ids_from_text(k_t_text))
    return merge_unique_in_order(collected)


def validate_cached_derived_candidates(
    *,
    qid: str,
    candidates: Any,
    visible_source_ids: Set[str],
    start_idx: int,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(candidates, list):
        return {}
    validated = []
    seen_unit_ids: Set[str] = set()
    next_idx = start_idx
    for idx, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            continue
        cand_type = str(cand.get("type", "")).strip()
        text = str(cand.get("text", "")).strip()
        source_unit_ids = cand.get("source_unit_ids", [])
        if cand_type not in ALLOWED_DERIVED_TYPES or not text or not isinstance(source_unit_ids, list):
            continue
        source_unit_ids = [str(x) for x in source_unit_ids]
        if not (1 <= len(source_unit_ids) <= 3):
            continue
        if len(source_unit_ids) != len(set(source_unit_ids)):
            continue
        if not set(source_unit_ids).issubset(visible_source_ids):
            continue
        unit_id = build_derived_unit_id(qid, next_idx)
        next_idx += 1
        if unit_id in seen_unit_ids:
            continue
        seen_unit_ids.add(unit_id)
        validated.append(
            {
                "unit_id": unit_id,
                "text": text,
                "type": cand_type,
                "source_unit_ids": source_unit_ids,
                "coarse_priority": int(cand.get("coarse_priority", idx)),
            }
        )
    validated.sort(key=lambda x: (x["coarse_priority"], x["unit_id"]))
    out: Dict[str, Dict[str, Any]] = {}
    for item in validated[:MAX_DERIVED_CANDIDATES]:
        out[item["unit_id"]] = {
            "text": item["text"],
            "type": item["type"],
            "source_unit_ids": item["source_unit_ids"],
        }
    return out


def resolve_derived_payloads_for_key(
    *,
    key: Tuple[str, int],
    query_info: dict,
    state_rec: dict,
    candidate_rec: dict,
    state_block: dict,
    candidates_block: dict,
    labels_block: dict,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
    derived_harvest_index: Dict[str, Any],
    derived_cache_index: Dict[str, List[dict]],
    derived_runtime: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    qid, _ = key
    derived_unit_ids = collect_derived_unit_ids(
        state_block=state_block,
        candidates_block=candidates_block,
        labels_block=labels_block,
        k_t_text=state_block.get("K_t"),
    )
    if not derived_unit_ids:
        return {}

    resolved: Dict[str, Dict[str, Any]] = {}
    for unit_id in derived_unit_ids:
        payload = derived_runtime["resolved_by_unit"].get(unit_id)
        if payload is None:
            payload = derived_harvest_index["by_unit"].get(unit_id)
        if payload is not None:
            derived_runtime["resolved_by_unit"][unit_id] = payload
            derived_runtime["max_idx_by_qid"][qid] = max(
                derived_runtime["max_idx_by_qid"].get(qid, -1),
                parse_derived_idx(unit_id),
            )
            resolved[unit_id] = payload

    missing = [unit_id for unit_id in derived_unit_ids if unit_id not in resolved]
    if missing:
        visible_source_ids = {
            item["unit_id"]
            for item in build_top_raw_candidates_for_derived_cache(
                [str(x) for x in expect_list(candidate_rec.get("R_t", []), name=f"candidate_rec[{qid}].R_t")],
                provenance_index,
            )
        }
        for item in expect_list(state_block.get("S_t", {}).get("raw_refs", []), name=f"state_block[{qid}].S_t.raw_refs"):
            if isinstance(item, dict) and item.get("unit_id"):
                visible_source_ids.add(str(item["unit_id"]))
        for cache_record in derived_cache_index.get(qid, []):
            start_idx = derived_runtime["max_idx_by_qid"].get(qid, -1) + 1
            candidate_map = validate_cached_derived_candidates(
                qid=qid,
                candidates=cache_record.get("derived_candidates", []),
                visible_source_ids=visible_source_ids,
                start_idx=start_idx,
            )
            for unit_id, payload in candidate_map.items():
                derived_runtime["resolved_by_unit"].setdefault(unit_id, payload)
                derived_runtime["max_idx_by_qid"][qid] = max(
                    derived_runtime["max_idx_by_qid"].get(qid, -1),
                    parse_derived_idx(unit_id),
                )
            missing = [unit_id for unit_id in derived_unit_ids if unit_id not in derived_runtime["resolved_by_unit"]]
            if not missing:
                break

    missing = [unit_id for unit_id in derived_unit_ids if unit_id not in derived_runtime["resolved_by_unit"]]
    if missing:
        raise ValueError(f"无法解析 derived payload: qid={qid}, missing={missing}")

    for unit_id in derived_unit_ids:
        resolved[unit_id] = derived_runtime["resolved_by_unit"][unit_id]
    return resolved


def render_k_t_with_derived_payloads(k_t: str, derived_payloads: Dict[str, Dict[str, Any]]) -> str:
    if "[missing derived payload]" not in k_t:
        return k_t

    rendered_lines: List[str] = []
    for line in k_t.splitlines():
        match = re.fullmatch(r"\[derived_note\] \[missing derived payload\] (.+)", line.strip())
        if not match:
            rendered_lines.append(line)
            continue
        unit_id = match.group(1).strip()
        payload = derived_payloads.get(unit_id)
        text = "" if payload is None else str(payload.get("text", "")).strip()
        if not text:
            rendered_lines.append(line)
            continue
        rendered_lines.append(f"[derived_note] {text}")
    return "\n".join(rendered_lines)


def merge_unique_in_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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
            hits.extend(scan_forbidden_keys(item, f"{path}[{idx}]"))
    return hits


def is_derived_unit_id(unit_id: str) -> bool:
    return "::derived::" in str(unit_id)


def normalize_stop_type(value: str) -> str:
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


def normalize_trajectory_status_name(value: str) -> str:
    x = str(value).strip().lower()
    if x in {"success", "terminal", "terminated"}:
        return "success"
    if x in {
        "failed_but_progressive",
        "failed-progressive",
        "failed_progress",
        "failed-progress",
        "progressive_failure",
    }:
        return "failed_but_progressive"
    if x in {
        "failed_stalled",
        "failed-stall",
        "failed_stall",
        "stall",
        "stalled",
    }:
        return "failed_stalled"
    raise ValueError(f"非法 trajectory status: {value}")


def infer_failed_status_from_abort_reason(abort_reason: Optional[str]) -> str:
    if abort_reason is not None:
        lowered = abort_reason.lower()
        if any(k in lowered for k in ["stall", "stalled", "oscillation", "no_progress"]):
            return "failed_stalled"
    return "failed_but_progressive"


def extract_terminal_step_from_full_record(record: dict) -> Optional[int]:
    terminal_step = record.get("terminal_step", record.get("terminal_t"))
    if isinstance(terminal_step, bool):
        terminal_step = None
    if isinstance(terminal_step, int):
        return terminal_step

    steps = record.get("steps", [])
    if isinstance(steps, list) and steps:
        ts = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            t = step.get("t")
            if isinstance(t, bool):
                continue
            if isinstance(t, int):
                ts.append(t)
        if ts:
            return max(ts)
    return None


def infer_trajectory_status_obj_from_record(record: dict) -> Dict[str, Any]:
    explicit = record.get("trajectory_status", record.get("terminal_status", record.get("status")))
    abort_reason_raw = record.get("abort_reason")
    abort_reason = None if abort_reason_raw is None else str(abort_reason_raw).strip() or None

    if explicit is not None:
        explicit_str = str(explicit).strip().lower()
        if explicit_str in {"abort", "aborted"}:
            status = infer_failed_status_from_abort_reason(abort_reason)
        else:
            status = normalize_trajectory_status_name(explicit_str)
    else:
        status = infer_failed_status_from_abort_reason(abort_reason)

    if status == "success":
        return {
            "status": "success",
            "terminal_step": extract_terminal_step_from_full_record(record),
            "abort_reason": None,
        }

    if abort_reason is None:
        abort_reason = "stalled" if status == "failed_stalled" else "unknown_failure"

    return {
        "status": status,
        "terminal_step": None,
        "abort_reason": abort_reason,
    }


def infer_progress_flag_from_status_obj(trajectory_status: Dict[str, Any]) -> str:
    if trajectory_status["status"] == "failed_stalled":
        return "stall"
    return "progress"


def load_queries(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "question" not in record:
            raise ValueError(f"queries 缺少字段: file={path}, row={row_idx}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"queries 中重复 qid: {qid}")

        out[qid] = {
            "qid": qid,
            "question": str(record["question"]).strip(),
            "answer": str(record.get("answer", "")).strip(),
        }
    return out


def load_prefix_records(path: Path, name: str) -> Dict[Tuple[str, int], dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "t" not in record:
            raise ValueError(f"{name} 缺少 qid 或 t: file={path}, row={row_idx}")
        key = (str(record["qid"]), int(record["t"]))
        if key in out:
            raise ValueError(f"{name} 出现重复 prefix key: file={path}, row={row_idx}, key={key}")
        out[key] = record
    return out


def load_trajectory_meta(
    path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Optional[bool]]]:
    out: Dict[str, Dict[str, Any]] = {}
    ever_progress_by_qid: Dict[str, Optional[bool]] = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record:
            raise ValueError(f"full trajectory 缺少 qid: file={path}, row={row_idx}")
        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"full trajectory 中重复 qid: file={path}, row={row_idx}, qid={qid}")
        out[qid] = infer_trajectory_status_obj_from_record(record)
        raw_ever_progress = record.get("ever_progress")
        if raw_ever_progress is None:
            ever_progress_by_qid[qid] = None
        elif isinstance(raw_ever_progress, bool):
            ever_progress_by_qid[qid] = raw_ever_progress
        else:
            raise ValueError(f"full trajectory ever_progress 必须是 bool/null: file={path}, row={row_idx}, qid={qid}")
    return out, ever_progress_by_qid


def load_full_records(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        qid = str(record.get("qid", "")).strip()
        if not qid:
            raise ValueError(f"full trajectory 缺少 qid: file={path}, row={row_idx}")
        if qid in out:
            raise ValueError(f"full trajectory 中重复 qid: file={path}, row={row_idx}, qid={qid}")
        out[qid] = record
    return out


def extract_run_id_from_full_records(full_by_qid: Dict[str, dict], *, split: str) -> str:
    run_ids: Set[str] = set()
    for qid, row in full_by_qid.items():
        meta = row.get("build_meta")
        if not isinstance(meta, dict) or not isinstance(meta.get("run_id"), str) or not meta["run_id"].strip():
            raise ValueError(f"full 缺少 build_meta.run_id: split={split}, qid={qid}")
        run_ids.add(meta["run_id"].strip())
    if len(run_ids) != 1:
        raise ValueError(f"full split 内 run_id 不一致: split={split}, run_ids={sorted(run_ids)}")
    return next(iter(run_ids))


def build_k_t_from_prefix_history(
    history: List[dict],
    *,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> str:
    lines: List[str] = ["Evidence:"]
    seen_chunk_ids: Set[str] = set()
    evidence_idx = 1
    derived_lines: List[str] = []
    for item in history:
        unit_id = str(item["unit_id"])
        if is_derived_unit_id(unit_id):
            derived_lines.append(f"[derived_note] [missing derived payload] {unit_id}")
            continue
        provenance = resolve_raw_registry_entry(unit_id, provenance_index, name=f"K_t[{unit_id}]")
        chunk_id = str(provenance["chunk_id"])
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        full_chunk_text = str(provenance.get("full_chunk_text") or "").strip()
        if full_chunk_text:
            lines.append(f"[{evidence_idx}] {full_chunk_text}")
        else:
            doc_id = str(provenance["doc_id"])
            lines.append(f"[{evidence_idx}] {doc_id}")
        evidence_idx += 1
    lines.extend(derived_lines)
    return "\n".join(lines)


def build_success_prefix_state_rec_from_full(
    qid: str,
    t: int,
    *,
    full_record: dict,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> dict:
    terminal_state = expect_dict(
        expect_dict(full_record.get("terminal_probe"), name=f"full[{qid}].terminal_probe").get("state_snapshot"),
        name=f"full[{qid}].terminal_probe.state_snapshot",
    )
    terminal_step = extract_terminal_step_from_full_record(full_record)
    if terminal_step is None:
        raise ValueError(f"success trajectory 缺少 terminal_step: qid={qid}")

    final_h_t = expect_list(terminal_state.get("H_t", []), name=f"full[{qid}].terminal_state.H_t")
    initial_prefix_len = len(final_h_t) - terminal_step
    if initial_prefix_len < 0:
        raise ValueError(f"无法从 terminal state 推断 prefix 长度: qid={qid}")
    prefix_len = initial_prefix_len + t
    if prefix_len < 0 or prefix_len > len(final_h_t):
        raise ValueError(f"非法 prefix 长度: qid={qid}, t={t}, prefix_len={prefix_len}")

    prefix_h_t = []
    for idx, item in enumerate(final_h_t[:prefix_len]):
        item = expect_dict(item, name=f"full[{qid}].terminal_state.H_t[{idx}]")
        prefix_h_t.append(
            {
                "step_id": idx,
                "unit_id": expect_str(item["unit_id"], name=f"full[{qid}].terminal_state.H_t[{idx}].unit_id"),
            }
        )

    if t == 0:
        covered_target_ids: List[str] = []
        role_scores = expect_dict(
            expect_dict(full_record["steps"][0], name=f"full[{qid}].steps[0]").get("coverage_debug", {}),
            name=f"full[{qid}].steps[0].coverage_debug",
        ).get("role_scores_before", {})
    else:
        prev_step = expect_dict(full_record["steps"][t - 1], name=f"full[{qid}].steps[{t - 1}]")
        coverage_debug = expect_dict(prev_step.get("coverage_debug", {}), name=f"full[{qid}].steps[{t - 1}].coverage_debug")
        covered_target_ids = [str(x) for x in expect_list(coverage_debug.get("covered_chunk_ids_after", []), name=f"full[{qid}].steps[{t - 1}].coverage_debug.covered_chunk_ids_after")]
        role_scores = expect_dict(coverage_debug.get("role_scores_after", {}), name=f"full[{qid}].steps[{t - 1}].coverage_debug.role_scores_after")

    raw_refs: List[dict] = []
    derived_refs: List[dict] = []
    seen_raw: Set[str] = set()
    seen_derived: Set[str] = set()
    selected_counts: Dict[str, int] = {}
    for item in prefix_h_t:
        unit_id = item["unit_id"]
        selected_counts[unit_id] = selected_counts.get(unit_id, 0) + 1
    for item in prefix_h_t:
        unit_id = item["unit_id"]
        if is_derived_unit_id(unit_id):
            if unit_id in seen_derived:
                continue
            seen_derived.add(unit_id)
            derived_refs.append(
                {
                    "unit_id": unit_id,
                    "added_step": int(item["step_id"]),
                    "used_in_summary_count": 0,
                    "selected_count": selected_counts[unit_id],
                }
            )
            continue
        if unit_id in seen_raw:
            continue
        seen_raw.add(unit_id)
        raw_refs.append(
            {
                "unit_id": unit_id,
                "added_step": int(item["step_id"]),
                "used_in_summary_count": 0,
                "selected_count": selected_counts[unit_id],
            }
        )

    return {
        "qid": qid,
        "t": t,
        "H_t": prefix_h_t,
        "A_t": {
            "covered_target_ids": covered_target_ids,
            "k_bridge": float(role_scores.get("bridge", 0.0)),
            "k_distinguish": float(role_scores.get("distinguish", 0.0)),
            "k_support": float(role_scores.get("support", 0.0)),
            "coverage_trace": {str(x): str(x) for x in covered_target_ids},
        },
        "S_t": {
            "raw_refs": raw_refs,
            "derived_refs": derived_refs,
            "last_added_unit_id": None if not prefix_h_t else str(prefix_h_t[-1]["unit_id"]),
            "last_updated_step": 0 if not prefix_h_t else int(prefix_h_t[-1]["step_id"]),
        },
        "K_t": build_k_t_from_prefix_history(prefix_h_t, provenance_index=provenance_index),
    }


def build_success_candidate_rec_from_full(qid: str, t: int, *, full_record: dict) -> dict:
    step = expect_dict(full_record["steps"][t], name=f"full[{qid}].steps[{t}]")
    candidate_debug = expect_dict(step.get("candidate_debug", {}), name=f"full[{qid}].steps[{t}].candidate_debug")
    return {
        "qid": qid,
        "t": t,
        "R_t": [str(x) for x in expect_list(candidate_debug.get("R_t", step.get("R_t", [])), name=f"full[{qid}].steps[{t}].R_t")],
        "G_t_final": [str(x) for x in expect_list(candidate_debug.get("G_t_final", step.get("G_t_final", [])), name=f"full[{qid}].steps[{t}].G_t_final")],
        "G_t_aux": [str(x) for x in expect_list(candidate_debug.get("G_t_aux", []), name=f"full[{qid}].steps[{t}].G_t_aux")],
        "G_t_illegal": [str(x) for x in expect_list(candidate_debug.get("G_t_illegal", []), name=f"full[{qid}].steps[{t}].G_t_illegal")],
        "C_t": [str(x) for x in expect_list(candidate_debug.get("C_t", step.get("C_t", [])), name=f"full[{qid}].steps[{t}].C_t")],
    }


def build_success_ranking_rec_from_full(qid: str, t: int, *, full_record: dict) -> dict:
    step = expect_dict(full_record["steps"][t], name=f"full[{qid}].steps[{t}]")
    positive_unit_id = expect_str(step["positive_unit_id"], name=f"full[{qid}].steps[{t}].positive_unit_id")
    c_t = [str(x) for x in expect_list(step.get("C_t", []), name=f"full[{qid}].steps[{t}].C_t")]
    negative_unit_ids = [unit_id for unit_id in c_t if unit_id != positive_unit_id]
    return {
        "qid": qid,
        "t": t,
        "positive_unit_id": positive_unit_id,
        "negative_unit_ids": merge_unique_in_order(negative_unit_ids),
    }


def build_success_stop_rec_from_full(qid: str, t: int, *, full_record: dict) -> dict:
    step = expect_dict(full_record["steps"][t], name=f"full[{qid}].steps[{t}]")
    stop_candidate = bool(step.get("stop_candidate", False))
    return {
        "qid": qid,
        "t": t,
        "stop_label": 0,
        "stop_type": "near-terminal" if stop_candidate else "continue",
    }


def clamp01(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def build_success_deficit_rec_from_full(qid: str, t: int, *, full_record: dict) -> dict:
    step = expect_dict(full_record["steps"][t], name=f"full[{qid}].steps[{t}]")
    coverage_debug = expect_dict(step.get("coverage_debug", {}), name=f"full[{qid}].steps[{t}].coverage_debug")
    role_scores_before = expect_dict(coverage_debug.get("role_scores_before", {}), name=f"full[{qid}].steps[{t}].coverage_debug.role_scores_before")
    required_roles = {str(x) for x in expect_list(full_record.get("required_roles", []), name=f"full[{qid}].required_roles")}
    return {
        "qid": qid,
        "t": t,
        "d_t_star": {
            "d_br_star": 0.0 if "bridge" not in required_roles else round(1.0 - clamp01(role_scores_before.get("bridge", 0.0)), 6),
            "d_dis_star": 0.0 if "distinguish" not in required_roles else round(1.0 - clamp01(role_scores_before.get("distinguish", 0.0)), 6),
            "d_sup_star": 0.0 if "support" not in required_roles else round(1.0 - clamp01(role_scores_before.get("support", 0.0)), 6),
            "d_der_star": 0.0,
        },
    }


def build_success_contribution_rec_from_full(qid: str, t: int, *, full_record: dict) -> dict:
    step = expect_dict(full_record["steps"][t], name=f"full[{qid}].steps[{t}]")
    coverage_debug = expect_dict(step.get("coverage_debug", {}), name=f"full[{qid}].steps[{t}].coverage_debug")
    role_scores_before = expect_dict(coverage_debug.get("role_scores_before", {}), name=f"full[{qid}].steps[{t}].coverage_debug.role_scores_before")
    role_scores_after = expect_dict(coverage_debug.get("role_scores_after", {}), name=f"full[{qid}].steps[{t}].coverage_debug.role_scores_after")
    return {
        "qid": qid,
        "t": t,
        "positive_unit_id": expect_str(step["positive_unit_id"], name=f"full[{qid}].steps[{t}].positive_unit_id"),
        "c_t_star": {
            "c_br_star": round(max(0.0, clamp01(role_scores_after.get("bridge", 0.0)) - clamp01(role_scores_before.get("bridge", 0.0))), 6),
            "c_dis_star": round(max(0.0, clamp01(role_scores_after.get("distinguish", 0.0)) - clamp01(role_scores_before.get("distinguish", 0.0))), 6),
            "c_sup_star": round(max(0.0, clamp01(role_scores_after.get("support", 0.0)) - clamp01(role_scores_before.get("support", 0.0))), 6),
            "c_der_star": 0.0,
        },
    }


def enrich_refs(refs: Any, *, name: str) -> List[dict]:
    refs = expect_list(refs, name=name)
    out = []
    seen = set()
    for idx, ref in enumerate(refs):
        ref = expect_dict(ref, name=f"{name}[{idx}]")
        if "unit_id" not in ref or "added_step" not in ref:
            raise ValueError(f"{name}[{idx}] 缺少 unit_id 或 added_step")
        unit_id = expect_str(ref["unit_id"], name=f"{name}[{idx}].unit_id")
        if unit_id in seen:
            raise ValueError(f"{name} 中出现重复 unit_id: {unit_id}")
        seen.add(unit_id)
        out.append(
            {
                "unit_id": unit_id,
                "added_step": int(ref["added_step"]),
                "used_in_summary_count": int(ref.get("used_in_summary_count", 0)),
                "selected_count": int(ref.get("selected_count", 1)),
            }
        )
    return out


def enrich_raw_chunk_refs(
    refs: Any,
    *,
    name: str,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> List[dict]:
    refs = expect_list(refs, name=name)
    out: List[dict] = []
    by_chunk_id: Dict[str, dict] = {}

    for idx, ref in enumerate(refs):
        ref = expect_dict(ref, name=f"{name}[{idx}]")
        if "added_step" not in ref:
            raise ValueError(f"{name}[{idx}] 缺少 added_step")
        raw_id = ref.get("chunk_id", ref.get("unit_id"))
        if raw_id is None:
            raise ValueError(f"{name}[{idx}] 缺少 chunk_id/unit_id")
        unit_id = expect_str(raw_id, name=f"{name}[{idx}]")
        chunk_id = expect_str(
            resolve_raw_registry_entry(
                unit_id,
                provenance_index,
                name=f"{name}[{idx}]",
            )["chunk_id"],
            name=f"{name}[{idx}].chunk_id",
        )
        added_step = int(ref["added_step"])
        used_in_summary_count = int(ref.get("used_in_summary_count", 0))
        selected_count = int(ref.get("selected_count", 1))
        provenance = build_raw_provenance(unit_id, provenance_index)

        existing = by_chunk_id.get(chunk_id)
        if existing is None:
            existing = {
                "unit_id": unit_id,
                "chunk_id": chunk_id,
                "doc_id": provenance["doc_id"],
                "parent_chunk_id": provenance["parent_chunk_id"],
                "added_step": added_step,
                "used_in_summary_count": used_in_summary_count,
                "selected_count": selected_count,
            }
            by_chunk_id[chunk_id] = existing
            out.append(existing)
            continue

        prev_added_step = existing["added_step"]
        existing["added_step"] = min(existing["added_step"], added_step)
        existing["used_in_summary_count"] += used_in_summary_count
        existing["selected_count"] += selected_count
        if added_step < prev_added_step:
            existing["unit_id"] = unit_id
            existing["doc_id"] = provenance["doc_id"]
            existing["parent_chunk_id"] = provenance["parent_chunk_id"]

    return out


def compute_progress_signal_by_qid(
    states_map: Dict[Tuple[str, int], dict],
    ranking_map: Dict[Tuple[str, int], dict],
) -> Dict[str, bool]:
    covered_counts_by_qid: Dict[str, List[int]] = {}
    derived_selected_by_qid: Dict[str, bool] = {}

    for (qid, _), state_rec in states_map.items():
        a_t = expect_dict(state_rec.get("A_t"), name=f"states[{qid}].A_t")
        covered_target_ids = normalize_unit_id_list(
            a_t.get("covered_target_ids", []),
            name=f"states[{qid}].A_t.covered_target_ids",
        )
        covered_counts_by_qid.setdefault(qid, []).append(len(covered_target_ids))

    for (qid, _), ranking_rec in ranking_map.items():
        positive_unit_id = str(ranking_rec.get("positive_unit_id", "")).strip()
        if is_derived_unit_id(positive_unit_id):
            derived_selected_by_qid[qid] = True
        else:
            derived_selected_by_qid.setdefault(qid, False)

    out: Dict[str, bool] = {}
    all_qids = set(covered_counts_by_qid.keys()) | set(derived_selected_by_qid.keys())
    for qid in all_qids:
        covered_counts = covered_counts_by_qid.get(qid, [])
        initial_covered = min(covered_counts) if covered_counts else 0
        max_covered = max(covered_counts) if covered_counts else 0
        out[qid] = bool(max_covered > initial_covered or derived_selected_by_qid.get(qid, False))
    return out


def resolve_trajectory_status_by_qid(
    base_status_by_qid: Dict[str, Dict[str, Any]],
    explicit_ever_progress_by_qid: Dict[str, Optional[bool]],
    inferred_ever_progress_by_qid: Dict[str, bool],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for qid, status_obj in base_status_by_qid.items():
        if status_obj["status"] == "success":
            out[qid] = status_obj
            continue

        explicit = explicit_ever_progress_by_qid.get(qid)
        ever_progress = explicit if explicit is not None else inferred_ever_progress_by_qid.get(qid, False)
        out[qid] = {
            "status": "failed_but_progressive" if ever_progress else "failed_stalled",
            "terminal_step": None,
            "abort_reason": status_obj["abort_reason"],
        }
    return out


def build_state_block(
    state_rec: dict,
    *,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> dict:
    h_t = expect_list(state_rec["H_t"], name="state_rec.H_t")
    normalized_h = []
    for idx, item in enumerate(h_t):
        item = expect_dict(item, name=f"state_rec.H_t[{idx}]")
        if "step_id" not in item or "unit_id" not in item:
            raise ValueError(f"state_rec.H_t[{idx}] 缺少 step_id 或 unit_id")
        unit_id = str(item["unit_id"])
        provenance = build_raw_provenance(unit_id, provenance_index)
        normalized_h.append(
            {
                "step_id": int(item["step_id"]),
                "unit_id": unit_id,
                "chunk_id": provenance["chunk_id"],
                "doc_id": provenance["doc_id"],
                "parent_chunk_id": provenance["parent_chunk_id"],
            }
        )

    a_t = expect_dict(state_rec["A_t"], name="state_rec.A_t")
    s_t = expect_dict(state_rec["S_t"], name="state_rec.S_t")
    covered_target_ids = normalize_unit_id_list(
        a_t.get("covered_target_ids", a_t.get("covered_chunk_ids", [])),
        name="state_rec.A_t.covered_target_ids",
    )
    covered_chunk_ids: List[str] = []
    seen_covered_chunk_ids: Set[str] = set()
    for idx, target_id in enumerate(covered_target_ids):
        chunk_id = expect_str(
            build_raw_provenance(target_id, provenance_index)["chunk_id"],
            name=f"state_rec.A_t.covered_target_ids[{idx}].chunk_id",
        )
        if chunk_id in seen_covered_chunk_ids:
            continue
        seen_covered_chunk_ids.add(chunk_id)
        covered_chunk_ids.append(chunk_id)
    raw_refs = enrich_raw_chunk_refs(
        s_t.get("raw_refs", []),
        name="state_rec.S_t.raw_refs",
        provenance_index=provenance_index,
    )
    raw_ref_by_chunk = {str(ref["chunk_id"]): ref for ref in raw_refs}
    raw_coverage_trace = a_t.get("coverage_trace", {})
    coverage_trace: Dict[str, dict] = {}
    if isinstance(raw_coverage_trace, dict) and raw_coverage_trace:
        for target_id, cover in raw_coverage_trace.items():
            target_chunk_id = expect_str(
                build_raw_provenance(str(target_id), provenance_index)["chunk_id"],
                name="state_rec.A_t.coverage_trace.target.chunk_id",
            )
            cover_unit_id = None
            if isinstance(cover, dict):
                raw_cover_id = cover.get("unit_id", cover.get("chunk_id"))
                if isinstance(raw_cover_id, str):
                    cover_unit_id = raw_cover_id
            elif isinstance(cover, str):
                cover_unit_id = cover
            if cover_unit_id is None:
                cover_unit_id = str(target_id)
            coverage_trace[target_chunk_id] = build_unit_with_provenance(cover_unit_id, provenance_index)
    for chunk_id in covered_chunk_ids:
        coverage_trace.setdefault(
            chunk_id,
            build_unit_with_provenance(raw_ref_by_chunk.get(chunk_id, {}).get("unit_id", chunk_id), provenance_index),
        )

    return {
        "H_t": normalized_h,
        "A_t": {
            "covered_target_ids": covered_target_ids,
            "k_bridge": float(a_t.get("k_bridge", a_t.get("k_br", 0.0))),
            "k_distinguish": float(a_t.get("k_distinguish", a_t.get("k_dis", 0.0))),
            "k_support": float(a_t.get("k_support", a_t.get("k_sup", 0.0))),
            "coverage_trace": coverage_trace,
        },
        "S_t": {
            "raw_refs": raw_refs,
            "derived_refs": enrich_refs(s_t.get("derived_refs", []), name="state_rec.S_t.derived_refs"),
            "last_added_unit_id": None if s_t.get("last_added_unit_id") is None else str(s_t.get("last_added_unit_id")),
            "last_updated_step": int(s_t.get("last_updated_step", 0)),
        },
        "K_t": str(state_rec["K_t"]),
    }


def build_candidates_block(
    candidate_rec: dict,
    *,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> dict:
    r_t = [str(x) for x in expect_list(candidate_rec.get("R_t", []), name="candidate_rec.R_t")]
    g_t_final = [str(x) for x in expect_list(candidate_rec.get("G_t_final", []), name="candidate_rec.G_t_final")]
    g_t_aux = [str(x) for x in expect_list(candidate_rec.get("G_t_aux", []), name="candidate_rec.G_t_aux")]
    g_t_illegal = [str(x) for x in expect_list(candidate_rec.get("G_t_illegal", []), name="candidate_rec.G_t_illegal")]
    c_t = [str(x) for x in expect_list(candidate_rec.get("C_t", []), name="candidate_rec.C_t")]
    return {
        "R_t": r_t,
        "G_t_final": g_t_final,
        "G_t_aux": g_t_aux,
        "G_t_illegal": g_t_illegal,
        "C_t": c_t,
        "candidate_provenance": build_candidate_provenance(
            [r_t, g_t_final, g_t_aux, g_t_illegal, c_t],
            provenance_index,
        ),
    }


def build_labels_block(
    state_block: dict,
    deficit_rec: dict,
    contribution_rec: dict,
    ranking_rec: dict,
    stop_rec: dict,
    *,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> dict:
    positive_unit_id = str(ranking_rec["positive_unit_id"])
    negative_unit_ids = [str(x) for x in expect_list(ranking_rec["negative_unit_ids"], name="ranking_rec.negative_unit_ids")]

    d_t_star = expect_dict(deficit_rec["d_t_star"], name="deficit_rec.d_t_star")
    c_t_star = expect_dict(contribution_rec["c_t_star"], name="contribution_rec.c_t_star")

    stop_label_value = stop_rec["stop_label"]
    if isinstance(stop_label_value, bool):
        should_stop = stop_label_value
    else:
        should_stop = bool(int(stop_label_value))

    return {
        "u_t_plus": {
            "step_id": len(state_block["H_t"]),
            "unit_id": positive_unit_id,
            "chunk_id": build_raw_provenance(positive_unit_id, provenance_index)["chunk_id"],
            "doc_id": build_raw_provenance(positive_unit_id, provenance_index)["doc_id"],
            "parent_chunk_id": build_raw_provenance(positive_unit_id, provenance_index)["parent_chunk_id"],
        },
        "d_t_star": {
            "d_raw": None,
            "d_br": float(d_t_star.get("d_br_star", 0.0)),
            "d_dis": float(d_t_star.get("d_dis_star", 0.0)),
            "d_sup": float(d_t_star.get("d_sup_star", 0.0)),
            "d_der": float(d_t_star.get("d_der_star", 0.0)),
        },
        "c_t_star": {
            "c_raw": None,
            "c_br": float(c_t_star.get("c_br_star", 0.0)),
            "c_dis": float(c_t_star.get("c_dis_star", 0.0)),
            "c_sup": float(c_t_star.get("c_sup_star", 0.0)),
            "c_der": float(c_t_star.get("c_der_star", 0.0)),
        },
        "ranking_label": {
            "positive_unit_id": positive_unit_id,
            "negative_unit_ids": merge_unique_in_order([x for x in negative_unit_ids if x != positive_unit_id]),
            "positive_provenance": build_raw_provenance(positive_unit_id, provenance_index),
            "negative_provenance": build_negative_provenance(
                merge_unique_in_order([x for x in negative_unit_ids if x != positive_unit_id]),
                provenance_index,
            ),
        },
        "stop_label": {
            "should_stop": should_stop,
            "label_type": normalize_stop_type(str(stop_rec["stop_type"])),
        },
    }


def build_meta_block(trajectory_status: Dict[str, Any]) -> dict:
    return {
        "trajectory_status": trajectory_status,
        "progress_flag": infer_progress_flag_from_status_obj(trajectory_status),
        "keep_prefix": True,
    }


def build_sample_for_key(
    key: Tuple[str, int],
    *,
    split: str,
    run_id: str,
    queries: Dict[str, dict],
    states_map: Dict[Tuple[str, int], dict],
    candidates_map: Dict[Tuple[str, int], dict],
    deficit_map: Dict[Tuple[str, int], dict],
    contribution_map: Dict[Tuple[str, int], dict],
    ranking_map: Dict[Tuple[str, int], dict],
    stop_map: Dict[Tuple[str, int], dict],
    trajectory_status_by_qid: Dict[str, Dict[str, Any]],
    full_records_by_qid: Dict[str, dict],
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
    derived_harvest_index: Dict[str, Any],
    derived_cache_index: Dict[str, List[dict]],
    derived_runtime: Dict[str, Any],
) -> dict:
    qid, t = key
    if qid not in queries:
        raise ValueError(f"queries 中找不到 qid: {qid}")
    if qid not in trajectory_status_by_qid:
        raise ValueError(f"full trajectory meta 中找不到 qid: {qid}")

    trajectory_status = trajectory_status_by_qid[qid]
    full_record = full_records_by_qid.get(qid)
    state_rec = states_map.get(key)
    candidate_rec = candidates_map.get(key)
    ranking_rec = ranking_map.get(key)
    stop_rec = stop_map.get(key)
    deficit_rec = deficit_map.get(key)
    contribution_rec = contribution_map.get(key)
    if (
        trajectory_status.get("status") == "success"
        and isinstance(trajectory_status.get("terminal_step"), int)
        and full_record is not None
        and full_record.get("terminal_probe", {}).get("state_snapshot") is not None
        and t < int(trajectory_status["terminal_step"])
    ):
        state_rec = build_success_prefix_state_rec_from_full(
            qid,
            t,
            full_record=full_record,
            provenance_index=provenance_index,
        )
        candidate_rec = build_success_candidate_rec_from_full(qid, t, full_record=full_record)
        ranking_rec = build_success_ranking_rec_from_full(qid, t, full_record=full_record)
        stop_rec = build_success_stop_rec_from_full(qid, t, full_record=full_record)
        if deficit_rec is None:
            deficit_rec = build_success_deficit_rec_from_full(qid, t, full_record=full_record)
        if contribution_rec is None:
            contribution_rec = build_success_contribution_rec_from_full(qid, t, full_record=full_record)

    if state_rec is None or candidate_rec is None:
        raise KeyError(f"state/candidate missing for key={key}")
    if ranking_rec is None or stop_rec is None or deficit_rec is None or contribution_rec is None:
        raise KeyError(f"label blocks missing for key={key}")

    state_block = build_state_block(state_rec, provenance_index=provenance_index)
    candidates_block = build_candidates_block(candidate_rec, provenance_index=provenance_index)
    labels_block = build_labels_block(
        state_block=state_block,
        deficit_rec=deficit_rec,
        contribution_rec=contribution_rec,
        ranking_rec=ranking_rec,
        stop_rec=stop_rec,
        provenance_index=provenance_index,
    )
    derived_payloads = resolve_derived_payloads_for_key(
        key=key,
        query_info=queries[qid],
        state_rec=state_rec,
        candidate_rec=candidate_rec,
        state_block=state_block,
        candidates_block=candidates_block,
        labels_block=labels_block,
        provenance_index=provenance_index,
        derived_harvest_index=derived_harvest_index,
        derived_cache_index=derived_cache_index,
        derived_runtime=derived_runtime,
    )
    state_block["K_t"] = render_k_t_with_derived_payloads(state_block["K_t"], derived_payloads)
    meta_block = build_meta_block(trajectory_status_by_qid[qid])

    return {
        "qid": qid,
        "t": t,
        "build_meta": {
            "run_id": run_id,
            "source": "build_hotpotqa_samples_v2.py",
            "split": split,
        },
        "question": queries[qid]["question"],
        "state": state_block,
        "candidates": candidates_block,
        "labels": labels_block,
        "derived_payloads": derived_payloads,
        "meta": meta_block,
    }


def validate_top_level(record: dict) -> Tuple[str, int]:
    if set(record.keys()) != {"qid", "t", "build_meta", "question", "state", "candidates", "labels", "derived_payloads", "meta"}:
        raise ValueError(f"顶层字段必须严格为 qid/t/build_meta/question/state/candidates/labels/derived_payloads/meta，当前得到: {sorted(record.keys())}")
    required = ["qid", "t", "build_meta", "question", "state", "candidates", "labels", "derived_payloads", "meta"]
    for key in required:
        if key not in record:
            raise ValueError(f"顶层缺少字段: {key}")

    qid = expect_str(record["qid"], name="qid")
    t = expect_int(record["t"], name="t")
    build_meta = expect_dict(record["build_meta"], name="build_meta")
    run_id = expect_str(build_meta.get("run_id"), name="build_meta.run_id")
    if not run_id.strip():
        raise ValueError("build_meta.run_id 不能为空")
    source = expect_str(build_meta.get("source"), name="build_meta.source")
    if source != "build_hotpotqa_samples_v2.py":
        raise ValueError(f"非法 build_meta.source: {source}")
    split = expect_str(build_meta.get("split"), name="build_meta.split")
    if split not in SPLITS:
        raise ValueError(f"非法 build_meta.split: {split}")
    expect_str(record["question"], name="question")
    expect_dict(record["state"], name="state")
    expect_dict(record["candidates"], name="candidates")
    expect_dict(record["labels"], name="labels")
    expect_dict(record["meta"], name="meta")
    return qid, t


def validate_resolved_raw_provenance(
    payload: dict,
    *,
    unit_id: str,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
    name: str,
) -> None:
    expected = build_raw_provenance(unit_id, provenance_index)
    for key in ["chunk_id", "doc_id", "parent_chunk_id"]:
        if key not in payload:
            raise ValueError(f"{name} 缺少字段: {key}")
    if expected["chunk_id"] is None:
        if payload["chunk_id"] is not None or payload["parent_chunk_id"] is not None:
            raise ValueError(f"{name} 不应为 derived 单元伪造 chunk provenance")
        return
    if payload["chunk_id"] != expected["chunk_id"]:
        raise ValueError(f"{name}.chunk_id 必须来自 raw chunk registry")
    if payload["parent_chunk_id"] != expected["parent_chunk_id"]:
        raise ValueError(f"{name}.parent_chunk_id 必须来自 raw chunk registry")
    if payload["doc_id"] != expected["doc_id"]:
        raise ValueError(f"{name}.doc_id 必须与 raw chunk registry 一致")


def validate_derived_payloads_block(record: dict) -> None:
    if "derived_payloads" not in record:
        raise ValueError("顶层缺少 derived_payloads")
    derived_payloads = expect_dict(record["derived_payloads"], name="derived_payloads")

    state = expect_dict(record["state"], name="state")
    candidates = expect_dict(record["candidates"], name="candidates")
    labels = expect_dict(record["labels"], name="labels")
    derived_unit_ids = collect_derived_unit_ids(
        state_block=state,
        candidates_block=candidates,
        labels_block=labels,
        k_t_text=state.get("K_t"),
    )
    if set(derived_payloads.keys()) != set(derived_unit_ids):
        raise ValueError("derived_payloads 必须严格覆盖当前 sample 中出现的全部 derived unit")

    for unit_id in derived_unit_ids:
        payload = expect_dict(derived_payloads[unit_id], name=f"derived_payloads[{unit_id}]")
        for key in ["text", "type", "source_unit_ids"]:
            if key not in payload:
                raise ValueError(f"derived_payloads[{unit_id}] 缺少字段: {key}")
        text = expect_str(payload["text"], name=f"derived_payloads[{unit_id}].text").strip()
        if not text:
            raise ValueError(f"derived_payloads[{unit_id}].text 不能为空")
        note_type = expect_str(payload["type"], name=f"derived_payloads[{unit_id}].type").strip()
        if not note_type:
            raise ValueError(f"derived_payloads[{unit_id}].type 不能为空")
        source_unit_ids = normalize_unit_id_list(
            payload["source_unit_ids"],
            name=f"derived_payloads[{unit_id}].source_unit_ids",
        )
        if not source_unit_ids:
            raise ValueError(f"derived_payloads[{unit_id}].source_unit_ids 不能为空")
        for source_unit_id in source_unit_ids:
            if is_derived_unit_id(source_unit_id):
                raise ValueError(f"derived_payloads[{unit_id}] 不允许使用 derived source_unit_id: {source_unit_id}")


def validate_state_block(
    state: dict,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> None:
    required = ["H_t", "A_t", "S_t", "K_t"]
    for key in required:
        if key not in state:
            raise ValueError(f"state 缺少字段: {key}")

    h_t = expect_list(state["H_t"], name="state.H_t")
    for i, item in enumerate(h_t):
        item = expect_dict(item, name=f"state.H_t[{i}]")
        for key in ["step_id", "unit_id", "chunk_id", "doc_id", "parent_chunk_id"]:
            if key not in item:
                raise ValueError(f"state.H_t[{i}] 缺少字段: {key}")
        step_id = expect_int(item["step_id"], name=f"state.H_t[{i}].step_id")
        if step_id != i:
            raise ValueError(f"state.H_t[*].step_id 必须从 0 开始连续递增: idx={i}, step_id={step_id}")
        unit_id = expect_str(item["unit_id"], name=f"state.H_t[{i}].unit_id")
        if is_derived_unit_id(unit_id):
            if item["chunk_id"] is not None or item["parent_chunk_id"] is not None:
                raise ValueError("derived H_t entry 不应伪造 chunk provenance")
        else:
            validate_resolved_raw_provenance(
                item,
                unit_id=unit_id,
                provenance_index=provenance_index,
                name=f"state.H_t[{i}]",
            )

    a_t = expect_dict(state["A_t"], name="state.A_t")
    for key in ["covered_target_ids", "k_bridge", "k_distinguish", "k_support", "coverage_trace"]:
        if key not in a_t:
            raise ValueError(f"state.A_t 缺少字段: {key}")
    normalize_unit_id_list(a_t["covered_target_ids"], name="state.A_t.covered_target_ids")
    if a_t.get("coverage_trace") is not None:
        coverage_trace = expect_dict(a_t["coverage_trace"], name="state.A_t.coverage_trace")
        for target_chunk_id, entry in coverage_trace.items():
            entry = expect_dict(entry, name=f"state.A_t.coverage_trace[{target_chunk_id}]")
            for key in ["unit_id", "chunk_id", "doc_id", "parent_chunk_id"]:
                if key not in entry:
                    raise ValueError(f"state.A_t.coverage_trace[{target_chunk_id}] 缺少字段: {key}")
            unit_id = expect_str(entry["unit_id"], name=f"state.A_t.coverage_trace[{target_chunk_id}].unit_id")
            validate_resolved_raw_provenance(
                entry,
                unit_id=unit_id,
                provenance_index=provenance_index,
                name=f"state.A_t.coverage_trace[{target_chunk_id}]",
            )
            expected_target_chunk_id = build_raw_provenance(unit_id, provenance_index)["chunk_id"]
            if target_chunk_id != expected_target_chunk_id:
                raise ValueError("state.A_t.coverage_trace 的 key 必须等于 registry 中解析出的真实 chunk_id")

    s_t = expect_dict(state["S_t"], name="state.S_t")
    for key in ["raw_refs", "derived_refs", "last_added_unit_id", "last_updated_step"]:
        if key not in s_t:
            raise ValueError(f"state.S_t 缺少字段: {key}")

    raw_refs = expect_list(s_t["raw_refs"], name="state.S_t.raw_refs")
    seen_chunk_ids: Set[str] = set()
    for i, ref in enumerate(raw_refs):
        ref = expect_dict(ref, name=f"state.S_t.raw_refs[{i}]")
        for key in ["unit_id", "chunk_id", "doc_id", "parent_chunk_id", "added_step", "used_in_summary_count", "selected_count"]:
            if key not in ref:
                raise ValueError(f"state.S_t.raw_refs[{i}] 缺少字段: {key}")
        unit_id = expect_str(ref["unit_id"], name=f"state.S_t.raw_refs[{i}].unit_id")
        validate_resolved_raw_provenance(
            ref,
            unit_id=unit_id,
            provenance_index=provenance_index,
            name=f"state.S_t.raw_refs[{i}]",
        )
        chunk_id = expect_str(ref["chunk_id"], name=f"state.S_t.raw_refs[{i}].chunk_id")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"state.S_t.raw_refs 中出现重复 chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)
        expect_int(ref["added_step"], name=f"state.S_t.raw_refs[{i}].added_step")
        expect_int(ref["used_in_summary_count"], name=f"state.S_t.raw_refs[{i}].used_in_summary_count")
        expect_int(ref["selected_count"], name=f"state.S_t.raw_refs[{i}].selected_count")

    derived_refs = expect_list(s_t["derived_refs"], name="state.S_t.derived_refs")
    seen_unit_ids: Set[str] = set()
    for i, ref in enumerate(derived_refs):
        ref = expect_dict(ref, name=f"state.S_t.derived_refs[{i}]")
        for key in ["unit_id", "added_step", "used_in_summary_count", "selected_count"]:
            if key not in ref:
                raise ValueError(f"state.S_t.derived_refs[{i}] 缺少字段: {key}")
        unit_id = expect_str(ref["unit_id"], name=f"state.S_t.derived_refs[{i}].unit_id")
        if unit_id in seen_unit_ids:
            raise ValueError(f"state.S_t.derived_refs 中出现重复 unit_id: {unit_id}")
        seen_unit_ids.add(unit_id)
        expect_int(ref["added_step"], name=f"state.S_t.derived_refs[{i}].added_step")
        expect_int(ref["used_in_summary_count"], name=f"state.S_t.derived_refs[{i}].used_in_summary_count")
        expect_int(ref["selected_count"], name=f"state.S_t.derived_refs[{i}].selected_count")

    if s_t["last_added_unit_id"] is not None:
        expect_str(s_t["last_added_unit_id"], name="state.S_t.last_added_unit_id")
    expect_int(s_t["last_updated_step"], name="state.S_t.last_updated_step")
    k_t = expect_str(state["K_t"], name="state.K_t")
    if "[missing derived payload]" in k_t:
        raise ValueError("state.K_t 不允许出现 [missing derived payload]")


def validate_candidates_block(
    candidates: dict,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    required = ["R_t", "G_t_final", "G_t_aux", "G_t_illegal", "C_t", "candidate_provenance"]
    for key in required:
        if key not in candidates:
            raise ValueError(f"candidates 缺少字段: {key}")

    r_t = normalize_unit_id_list(candidates["R_t"], name="candidates.R_t")
    g_t_final = normalize_unit_id_list(candidates["G_t_final"], name="candidates.G_t_final")
    g_t_aux = normalize_unit_id_list(candidates["G_t_aux"], name="candidates.G_t_aux")
    g_t_illegal = normalize_unit_id_list(candidates["G_t_illegal"], name="candidates.G_t_illegal")
    c_t = normalize_unit_id_list(candidates["C_t"], name="candidates.C_t")
    candidate_provenance = expect_dict(candidates["candidate_provenance"], name="candidates.candidate_provenance")

    expected_c_t = merge_unique_in_order(r_t + g_t_final)
    if c_t != expected_c_t:
        raise ValueError("candidates.C_t 必须严格等于 R_t ∪ G_t_final（保持顺序去重后）")

    expected_candidate_ids = set(merge_unique_in_order(r_t + g_t_final + g_t_aux + g_t_illegal + c_t))
    if set(candidate_provenance.keys()) != expected_candidate_ids:
        raise ValueError("candidates.candidate_provenance 必须覆盖全部 candidate unit_id")
    for unit_id, provenance in candidate_provenance.items():
        provenance = expect_dict(provenance, name=f"candidates.candidate_provenance[{unit_id}]")
        for key in ["chunk_id", "doc_id", "parent_chunk_id"]:
            if key not in provenance:
                raise ValueError(f"candidates.candidate_provenance[{unit_id}] 缺少字段: {key}")
        if is_derived_unit_id(unit_id):
            if provenance["chunk_id"] is not None or provenance["parent_chunk_id"] is not None:
                raise ValueError("derived candidate 不应伪造 chunk provenance")
        else:
            validate_resolved_raw_provenance(
                provenance,
                unit_id=unit_id,
                provenance_index=provenance_index,
                name=f"candidates.candidate_provenance[{unit_id}]",
            )

    return r_t, g_t_final, g_t_aux, g_t_illegal, c_t


def validate_labels_block(
    labels: dict,
    c_t: List[str],
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> None:
    required = ["u_t_plus", "d_t_star", "c_t_star", "ranking_label", "stop_label"]
    for key in required:
        if key not in labels:
            raise ValueError(f"labels 缺少字段: {key}")

    stop_label = expect_dict(labels["stop_label"], name="labels.stop_label")
    for key in ["should_stop", "label_type"]:
        if key not in stop_label:
            raise ValueError(f"labels.stop_label 缺少字段: {key}")
    should_stop = expect_bool(stop_label["should_stop"], name="labels.stop_label.should_stop")
    label_type = normalize_stop_type(expect_str(stop_label["label_type"], name="labels.stop_label.label_type"))
    if label_type not in {"terminal", "false-stop", "near-terminal", "continue"}:
        raise ValueError(f"非法 labels.stop_label.label_type: {label_type}")
    if should_stop and label_type != "terminal":
        raise ValueError("should_stop=true 时 label_type 必须是 terminal")
    if (not should_stop) and label_type == "terminal":
        raise ValueError("label_type=terminal 时 should_stop 必须为 true")

    u_t_plus = expect_dict(labels["u_t_plus"], name="labels.u_t_plus")
    for key in ["step_id", "unit_id", "chunk_id", "doc_id", "parent_chunk_id"]:
        if key not in u_t_plus:
            raise ValueError(f"labels.u_t_plus 缺少字段: {key}")
    expect_int(u_t_plus["step_id"], name="labels.u_t_plus.step_id")
    positive_unit_id_in_u = expect_str(u_t_plus["unit_id"], name="labels.u_t_plus.unit_id")
    if is_derived_unit_id(positive_unit_id_in_u):
        if u_t_plus["chunk_id"] is not None or u_t_plus["parent_chunk_id"] is not None:
            raise ValueError("derived u_t_plus 不应伪造 chunk provenance")
    else:
        validate_resolved_raw_provenance(
            u_t_plus,
            unit_id=positive_unit_id_in_u,
            provenance_index=provenance_index,
            name="labels.u_t_plus",
        )

    d_t_star = expect_dict(labels["d_t_star"], name="labels.d_t_star")
    for key in ["d_raw", "d_br", "d_dis", "d_sup", "d_der"]:
        if key not in d_t_star:
            raise ValueError(f"labels.d_t_star 缺少字段: {key}")

    c_t_star = expect_dict(labels["c_t_star"], name="labels.c_t_star")
    for key in ["c_raw", "c_br", "c_dis", "c_sup", "c_der"]:
        if key not in c_t_star:
            raise ValueError(f"labels.c_t_star 缺少字段: {key}")

    ranking_label = expect_dict(labels["ranking_label"], name="labels.ranking_label")
    for key in ["positive_unit_id", "negative_unit_ids", "positive_provenance", "negative_provenance"]:
        if key not in ranking_label:
            raise ValueError(f"labels.ranking_label 缺少字段: {key}")

    positive_unit_id = expect_str(
        ranking_label["positive_unit_id"],
        name="labels.ranking_label.positive_unit_id",
    )
    negative_unit_ids = normalize_unit_id_list(
        ranking_label["negative_unit_ids"],
        name="labels.ranking_label.negative_unit_ids",
    )
    positive_provenance = expect_dict(
        ranking_label["positive_provenance"],
        name="labels.ranking_label.positive_provenance",
    )
    negative_provenance = expect_dict(
        ranking_label["negative_provenance"],
        name="labels.ranking_label.negative_provenance",
    )
    if positive_unit_id != expect_str(u_t_plus["unit_id"], name="labels.u_t_plus.unit_id"):
        raise ValueError("labels.u_t_plus.unit_id 必须等于 labels.ranking_label.positive_unit_id")
    if expect_int(u_t_plus["step_id"], name="labels.u_t_plus.step_id") < 0:
        raise ValueError("labels.u_t_plus.step_id 必须非负")

    c_t_set = set(c_t)
    if positive_unit_id not in c_t_set:
        raise ValueError("labels.ranking_label.positive_unit_id 不在 candidates.C_t 中")
    if positive_unit_id in set(negative_unit_ids):
        raise ValueError("positive_unit_id 不能同时出现在 negative_unit_ids 中")

    bad_negatives = [x for x in negative_unit_ids if x not in c_t_set]
    if bad_negatives:
        raise ValueError(
            f"labels.ranking_label.negative_unit_ids 必须是 candidates.C_t 的子集，非法项: {bad_negatives[:5]}"
        )

    for key in ["chunk_id", "doc_id", "parent_chunk_id"]:
        if key not in positive_provenance:
            raise ValueError(f"labels.ranking_label.positive_provenance 缺少字段: {key}")
    if is_derived_unit_id(positive_unit_id):
        if positive_provenance["chunk_id"] is not None or positive_provenance["parent_chunk_id"] is not None:
            raise ValueError("derived positive 不应伪造 chunk provenance")
    else:
        validate_resolved_raw_provenance(
            positive_provenance,
            unit_id=positive_unit_id,
            provenance_index=provenance_index,
            name="labels.ranking_label.positive_provenance",
        )

    if set(negative_provenance.keys()) != set(negative_unit_ids):
        raise ValueError("labels.ranking_label.negative_provenance 必须覆盖全部 negative_unit_ids")
    for unit_id, provenance in negative_provenance.items():
        provenance = expect_dict(provenance, name=f"labels.ranking_label.negative_provenance[{unit_id}]")
        for key in ["chunk_id", "doc_id", "parent_chunk_id"]:
            if key not in provenance:
                raise ValueError(f"labels.ranking_label.negative_provenance[{unit_id}] 缺少字段: {key}")
        if is_derived_unit_id(unit_id):
            if provenance["chunk_id"] is not None or provenance["parent_chunk_id"] is not None:
                raise ValueError("derived negative 不应伪造 chunk provenance")
        else:
            validate_resolved_raw_provenance(
                provenance,
                unit_id=unit_id,
                provenance_index=provenance_index,
                name=f"labels.ranking_label.negative_provenance[{unit_id}]",
            )

    expect_bool(stop_label["should_stop"], name="labels.stop_label.should_stop")
    expect_str(stop_label["label_type"], name="labels.stop_label.label_type")


def validate_meta_block(meta: dict) -> None:
    if set(meta.keys()) != {"trajectory_status", "progress_flag", "keep_prefix"}:
        raise ValueError(f"meta 字段必须严格为 trajectory_status/progress_flag/keep_prefix，当前得到: {sorted(meta.keys())}")
    required = ["trajectory_status", "progress_flag", "keep_prefix"]
    for key in required:
        if key not in meta:
            raise ValueError(f"meta 缺少字段: {key}")

    traj = expect_dict(meta["trajectory_status"], name="meta.trajectory_status")
    for key in ["status", "terminal_step", "abort_reason"]:
        if key not in traj:
            raise ValueError(f"meta.trajectory_status 缺少字段: {key}")

    status = expect_str(traj["status"], name="meta.trajectory_status.status")
    if status not in ALLOWED_TRAJECTORY_STATUS:
        raise ValueError(f"非法 trajectory_status.status: {status}")

    if status == "success":
        if traj["terminal_step"] is None:
            raise ValueError("success 样本的 terminal_step 不能为空")
        expect_int(traj["terminal_step"], name="meta.trajectory_status.terminal_step")
        if traj["abort_reason"] is not None:
            raise ValueError("success 样本的 abort_reason 必须为 null")
    else:
        if traj["terminal_step"] is not None:
            raise ValueError("failed_* 样本的 terminal_step 必须为 null")
        if traj["abort_reason"] is None:
            raise ValueError("failed_* 样本的 abort_reason 不能为空")
        expect_str(traj["abort_reason"], name="meta.trajectory_status.abort_reason")

    progress_flag = expect_str(meta["progress_flag"], name="meta.progress_flag")
    if progress_flag not in ALLOWED_PROGRESS_FLAG:
        raise ValueError(f"非法 meta.progress_flag: {progress_flag}")

    keep_prefix = expect_bool(meta["keep_prefix"], name="meta.keep_prefix")
    if keep_prefix is not True:
        raise ValueError("最终 samples_v2 中只允许出现 keep_prefix=true 的样本")


def validate_record(
    record: dict,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> Tuple[str, int]:
    qid, t = validate_top_level(record)
    forbidden_hits = scan_forbidden_keys(record)
    if forbidden_hits:
        raise ValueError(f"最终 samples_v2 不能包含 debug 字段: {forbidden_hits[:5]}")
    validate_state_block(record["state"], provenance_index)
    _, _, _, _, c_t = validate_candidates_block(record["candidates"], provenance_index)
    validate_labels_block(record["labels"], c_t, provenance_index)
    validate_derived_payloads_block(record)
    if record["labels"]["u_t_plus"]["step_id"] != len(record["state"]["H_t"]):
        raise ValueError("labels.u_t_plus.step_id 必须等于 len(state.H_t)")
    validate_meta_block(record["meta"])
    return qid, t


def validate_split(
    path: Path,
    provenance_index: Dict[str, Dict[str, Dict[str, Optional[str]]]],
) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 samples 文件: {path}")

    stats = {
        "rows": 0,
        "unique_qids": 0,
        "max_t": -1,
    }
    seen_keys: Set[Tuple[str, int]] = set()
    qids: Set[str] = set()

    for row_idx, record in enumerate(read_jsonl(path), start=1):
        try:
            qid, t = validate_record(record, provenance_index)
        except Exception as e:
            raise ValueError(
                f"样本校验失败: file={path}, row={row_idx}, qid={record.get('qid', 'UNKNOWN')}, "
                f"t={record.get('t', 'UNKNOWN')}, error={e}"
            ) from e

        key = (qid, t)
        if key in seen_keys:
            raise ValueError(f"出现重复 prefix 主键: file={path}, key={key}")
        seen_keys.add(key)
        qids.add(qid)

        stats["rows"] += 1
        stats["max_t"] = max(stats["max_t"], t)

    stats["unique_qids"] = len(qids)
    return stats


def convert_split(
    split: str,
    *,
    queries_path: Path,
    states_path: Path,
    candidates_path: Path,
    deficit_path: Path,
    contribution_path: Path,
    ranking_path: Path,
    stop_path: Path,
    full_path: Path,
    raw_units_path: Path,
    chunks_path: Path,
    derived_harvest_path: Path,
    derived_cache_dir: Path,
    output_path: Path,
) -> Tuple[int, Dict[str, int], str]:
    queries = load_queries(queries_path)
    states_map = load_prefix_records(states_path, "states")
    candidates_map = load_prefix_records(candidates_path, "candidates")
    deficit_map = load_prefix_records(deficit_path, "deficit")
    contribution_map = load_prefix_records(contribution_path, "contribution")
    ranking_map = load_prefix_records(ranking_path, "ranking")
    stop_map = load_prefix_records(stop_path, "stop")
    provenance_index = load_raw_provenance_index(raw_units_path, chunks_path)
    derived_harvest_index = load_derived_harvest_index(derived_harvest_path)
    derived_cache_index = load_derived_cache_index(derived_cache_dir)
    derived_runtime = {"resolved_by_unit": {}, "max_idx_by_qid": {}}
    full_records_by_qid = load_full_records(full_path)
    run_id = extract_run_id_from_full_records(full_records_by_qid, split=split)
    trajectory_status_by_qid, explicit_ever_progress_by_qid = load_trajectory_meta(full_path)
    inferred_ever_progress_by_qid = compute_progress_signal_by_qid(states_map, ranking_map)
    trajectory_status_by_qid = resolve_trajectory_status_by_qid(
        trajectory_status_by_qid,
        explicit_ever_progress_by_qid,
        inferred_ever_progress_by_qid,
    )

    common_keys = (
        set(states_map.keys())
        & set(candidates_map.keys())
        & set(deficit_map.keys())
        & set(contribution_map.keys())
        & set(ranking_map.keys())
        & set(stop_map.keys())
    )
    success_prefix_keys: Set[Tuple[str, int]] = set()
    for qid, status in trajectory_status_by_qid.items():
        if status.get("status") != "success":
            continue
        terminal_step = status.get("terminal_step")
        if not isinstance(terminal_step, int) or isinstance(terminal_step, bool) or terminal_step <= 0:
            continue
        full_record = full_records_by_qid.get(qid)
        if full_record is None or full_record.get("terminal_probe", {}).get("state_snapshot") is None:
            continue
        for t in range(terminal_step):
            success_prefix_keys.add((qid, t))

    target_keys = common_keys | success_prefix_keys
    if not target_keys:
        raise RuntimeError(
            f"prefix 对齐失败: split={split}, "
            f"states={len(states_map)}, candidates={len(candidates_map)}, deficit={len(deficit_map)}, "
            f"contribution={len(contribution_map)}, ranking={len(ranking_map)}, stop={len(stop_map)}"
        )

    output_records = []
    for key in sorted(target_keys, key=lambda x: (x[0], x[1])):
        qid, t = key
        trajectory_status = trajectory_status_by_qid.get(qid)
        if trajectory_status is None:
            raise KeyError(f"trajectory_status missing for qid={qid}")
        if trajectory_status.get("status") == "success":
            terminal_step = trajectory_status.get("terminal_step")
            if isinstance(terminal_step, int) and not isinstance(terminal_step, bool) and t >= terminal_step:
                continue
        output_records.append(
            build_sample_for_key(
                key,
                split=split,
                run_id=run_id,
                queries=queries,
                states_map=states_map,
                candidates_map=candidates_map,
                deficit_map=deficit_map,
                contribution_map=contribution_map,
                ranking_map=ranking_map,
                stop_map=stop_map,
                trajectory_status_by_qid=trajectory_status_by_qid,
                full_records_by_qid=full_records_by_qid,
                provenance_index=provenance_index,
                derived_harvest_index=derived_harvest_index,
                derived_cache_index=derived_cache_index,
                derived_runtime=derived_runtime,
            )
        )

    output_records.sort(key=lambda x: (x["qid"], int(x["t"])))

    count = write_jsonl(output_records, output_path)
    validate_stats = validate_split(output_path, provenance_index)
    stats = {
        "intersection_keys": len(target_keys),
        "terminal_only_keys": 0,
        "built": count,
        "unique_qids": validate_stats["unique_qids"],
        "max_t": validate_stats["max_t"],
    }
    return count, stats, run_id


def main():
    parser = argparse.ArgumentParser(description="Build hotpotqa samples v2")
    parser.add_argument("--split", choices=SPLITS, help="Only rebuild one split")
    parser.add_argument("--force", action="store_true", help="Accepted for compatibility; rebuilds output file in place")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE
    samples_base = project_root / DEFAULT_SAMPLES_BASE
    samples_base.mkdir(parents=True, exist_ok=True)

    queries_dir = base_dir / "queries"
    trajectories_dir = base_dir / "trajectories"
    labels_dir = base_dir / "labels"
    index_store_dir = base_dir / "index_store"
    unit_registry_dir = base_dir / "unit_registry"

    all_stats: Dict[str, Dict[str, int]] = {}
    run_ids_by_split: Dict[str, str] = {}

    target_splits = [args.split] if args.split else SPLITS

    for split in target_splits:
        queries_path = queries_dir / f"{split}.jsonl"
        states_path = trajectories_dir / f"states_{split}.jsonl"
        candidates_path = trajectories_dir / f"candidates_{split}.jsonl"
        deficit_path = labels_dir / f"deficit_{split}.jsonl"
        contribution_path = labels_dir / f"contribution_{split}.jsonl"
        ranking_path = labels_dir / f"ranking_{split}.jsonl"
        stop_path = labels_dir / f"stop_{split}.jsonl"
        full_path = trajectories_dir / f"full_{split}.jsonl"
        raw_units_path = unit_registry_dir / f"raw_units_{split}.jsonl"
        chunks_path = index_store_dir / f"chunks_{split}.jsonl"
        derived_harvest_path = trajectories_dir / f"derived_harvest_{split}.jsonl"
        derived_cache_dir = base_dir / "cache" / "derived_cache" / split
        output_path = samples_base / f"{split}.jsonl"

        for path in [
            queries_path,
            states_path,
            candidates_path,
            deficit_path,
            contribution_path,
            ranking_path,
            stop_path,
            full_path,
            raw_units_path,
            chunks_path,
            derived_harvest_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        count, stats, run_id = convert_split(
            split,
            queries_path=queries_path,
            states_path=states_path,
            candidates_path=candidates_path,
            deficit_path=deficit_path,
            contribution_path=contribution_path,
            ranking_path=ranking_path,
            stop_path=stop_path,
            full_path=full_path,
            raw_units_path=raw_units_path,
            chunks_path=chunks_path,
            derived_harvest_path=derived_harvest_path,
            derived_cache_dir=derived_cache_dir,
            output_path=output_path,
        )
        run_ids_by_split[split] = run_id
        all_stats[split] = {
            "count": count,
            "intersection_keys": stats["intersection_keys"],
            "terminal_only_keys": stats["terminal_only_keys"],
            "unique_qids": stats["unique_qids"],
            "max_t": stats["max_t"],
        }

    write_json(
        {
            "source": "build_hotpotqa_samples_v2.py",
            "run_id_by_split": {split: run_ids_by_split[split] for split in target_splits},
        },
        samples_base / "build_manifest_v2.json",
    )

    print("samples v2 构建并校验完成：")
    for split in target_splits:
        info = all_stats[split]
        print(
            f"  {split}: {info['count']} -> {samples_base / f'{split}.jsonl'}"
        )
        print(
            f"    intersection={info['intersection_keys']}, "
            f"terminal_only={info['terminal_only_keys']}, "
            f"unique_qids={info['unique_qids']}, max_t={info['max_t']}"
        )


if __name__ == "__main__":
    main()
