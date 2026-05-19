import hashlib
import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = Path(__file__).resolve().parents[1] / "data" / "hotpotqa_distractor_v2"
DERIVED_UNIT_ID_RE = re.compile(r"\b[0-9A-Za-z]+::derived::\d+\b")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: file={path}, line={line_idx}, error={e}") from e


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


def merge_unique_in_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'“”‘’`.,;:!?()\[\]{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def context_contains_answer(gold_answer: str, text: str) -> bool:
    norm_gold = normalize_text(gold_answer)
    norm_text = normalize_text(text)
    if not norm_gold:
        return False
    return norm_gold in norm_text


def is_yes_no_question(question: str) -> bool:
    q = str(question).strip().lower()
    prefixes = (
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "should ",
        "would ",
        "has ",
        "have ",
        "had ",
    )
    return any(q.startswith(prefix) for prefix in prefixes)


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


def effective_answer_match_rule(
    stored_rule: Any,
    *,
    pred_answer: Any,
    gold_answer: Any,
    answer_correct: bool,
    context_exact_match: bool,
) -> str:
    rule = str(stored_rule or "").strip() or "none"
    if not answer_correct:
        return rule

    if answers_equivalent(pred_answer, gold_answer):
        norm_pred = normalize_text(pred_answer)
        norm_gold = normalize_text(gold_answer)
        if norm_pred == norm_gold:
            return "normalized_exact"
        if is_number_like_answer(pred_answer) or is_number_like_answer(gold_answer):
            return "numeric_equivalent"
        if context_exact_match:
            return "context_exact"

    return rule


def is_derived_unit_id(unit_id: Any) -> bool:
    return isinstance(unit_id, str) and "::derived::" in unit_id


def looks_like_raw_unit_id(unit_id: Any) -> bool:
    if not isinstance(unit_id, str) or is_derived_unit_id(unit_id):
        return False
    parts = unit_id.split("::")
    return len(parts) >= 3 and parts[-1].isdigit()


def normalize_text_for_chunk_identity(text: Any, *, name: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{name} 必须是 str，当前得到: {type(text)}")
    return re.sub(r"\s+", " ", text).strip()


def canonical_chunk_id(doc_id: str, chunk_text: str) -> str:
    payload = json.dumps(
        {
            "doc_id": str(doc_id).strip(),
            "chunk_text": normalize_text_for_chunk_identity(chunk_text, name="canonical_chunk.chunk_text"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"rawchunk::{hashlib.sha1(payload.encode('utf-8')).hexdigest()}"


def extract_derived_unit_ids_from_text(text: Any) -> List[str]:
    if not isinstance(text, str) or "::derived::" not in text:
        return []
    return merge_unique_in_order(DERIVED_UNIT_ID_RE.findall(text))


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


def find_existing(base: Path, candidates: List[Path]) -> Path:
    for path in candidates:
        resolved = path if path.is_absolute() else (base / path)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"找不到任何可用路径: {[str(p) for p in candidates]}")


def load_queries(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in read_jsonl(path):
        qid = str(row["qid"])
        out[qid] = {
            "question": str(row["question"]),
            "answer": str(row.get("answer", "")),
        }
    return out


def load_full(path: Path) -> Dict[str, dict]:
    return {str(row["qid"]): row for row in read_jsonl(path)}


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


def write_debug_manifest(base_dir: Path, *, debug_type: str, run_ids_by_split: Dict[str, str]) -> None:
    manifest_path = base_dir / "debug" / "build_manifest_v2.json"
    manifest: Dict[str, Any]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            manifest = {}
    else:
        manifest = {}
    manifest["run_id_by_split"] = {split: run_ids_by_split[split] for split in SPLITS}
    manifest["source"] = "build_hotpotqa_success_debug_v2.py" if debug_type == "success" else "build_hotpotqa_failure_debug_v2.py"
    manifest["debug_types"] = manifest.get("debug_types", {})
    manifest["debug_types"][debug_type] = {
        "run_id_by_split": {split: run_ids_by_split[split] for split in SPLITS},
    }
    write_json(manifest, manifest_path)


def load_states(path: Path) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for row in read_jsonl(path):
        out[(str(row["qid"]), int(row["t"]))] = row
    return out


def load_sample_derived_payloads(path: Path) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for row in read_jsonl(path):
        qid = str(row["qid"])
        payloads = row.get("derived_payloads", {})
        if not isinstance(payloads, dict):
            continue
        out[qid].update(payloads)
    return out


def load_derived_harvest(path: Path) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    if not path.exists():
        return out
    for row in read_jsonl(path):
        qid = str(row.get("qid"))
        harvested = row.get("G_t_harvest", [])
        if not isinstance(harvested, list):
            continue
        for item in harvested:
            if not isinstance(item, dict):
                continue
            unit_id = item.get("unit_id")
            if not isinstance(unit_id, str) or not is_derived_unit_id(unit_id):
                continue
            text = item.get("text")
            note_type = item.get("type")
            source_unit_ids = item.get("source_unit_ids", [])
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(note_type, str) or not note_type.strip():
                continue
            if not (isinstance(source_unit_ids, list) and source_unit_ids and all(isinstance(x, str) for x in source_unit_ids)):
                continue
            out[qid][unit_id] = {
                "text": text.strip(),
                "type": note_type.strip(),
                "source_unit_ids": [str(x) for x in source_unit_ids],
            }
    return out


def load_raw_chunk_registry(raw_units_path: Path, chunks_path: Path) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    chunk_by_old: Dict[str, dict] = {}
    for row in read_jsonl(chunks_path):
        old_chunk_id = str(row["chunk_id"])
        doc_id = str(row["doc_id"])
        chunk_text = str(row.get("chunk_text", ""))
        chunk_id = canonical_chunk_id(doc_id, chunk_text)
        chunk_by_old[old_chunk_id] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "parent_chunk_id": chunk_id,
            "full_chunk_text": chunk_text,
        }

    unit_registry: Dict[str, dict] = {}
    chunk_registry: Dict[str, dict] = {}
    for row in read_jsonl(raw_units_path):
        unit_id = str(row["unit_id"])
        doc_id = str(row["doc_id"])
        old_parent_chunk_id = str(row["parent_chunk_id"])
        chunk_info = chunk_by_old.get(old_parent_chunk_id)
        if chunk_info is None:
            fallback_chunk_text = str(row.get("text", ""))
            chunk_id = canonical_chunk_id(doc_id, fallback_chunk_text)
            chunk_info = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "parent_chunk_id": chunk_id,
                "full_chunk_text": fallback_chunk_text,
            }
        unit_registry[unit_id] = {
            "unit_id": unit_id,
            "text": str(row.get("text", "")),
            "doc_id": doc_id,
            "chunk_id": chunk_info["chunk_id"],
            "parent_chunk_id": chunk_info["parent_chunk_id"],
        }
        chunk_registry[chunk_info["chunk_id"]] = chunk_info
    return unit_registry, chunk_registry


def build_history_entry(item: dict, unit_registry: Dict[str, dict]) -> dict:
    unit_id = str(item["unit_id"])
    entry = {
        "step_id": int(item["step_id"]),
        "unit_id": unit_id,
    }
    if is_derived_unit_id(unit_id):
        entry.update(
            {
                "chunk_id": None,
                "doc_id": None,
                "parent_chunk_id": None,
                "unit_text": None,
            }
        )
        return entry
    raw = unit_registry.get(unit_id)
    if raw is None:
        entry.update(
            {
                "chunk_id": None,
                "doc_id": None,
                "parent_chunk_id": None,
                "unit_text": None,
            }
        )
        return entry
    entry.update(
        {
            "chunk_id": raw["chunk_id"],
            "doc_id": raw["doc_id"],
            "parent_chunk_id": raw["parent_chunk_id"],
            "unit_text": raw["text"],
        }
    )
    return entry


def resolve_answer_probe(full_rec: dict) -> dict:
    probe = full_rec.get("terminal_probe")
    if not isinstance(probe, dict):
        history = full_rec.get("stop_probe_history", [])
        if isinstance(history, list) and history:
            probe = history[-1]
        else:
            probe = {}
    pred_answer = probe.get("pred_answer")
    gold_answer = probe.get("gold_answer")
    answer_correct = bool(probe.get("AnswerCorrect_t", False))
    context_exact_match = bool(probe.get("context_exact_match", False))
    return {
        "pred_answer": pred_answer,
        "gold_answer": gold_answer,
        "normalized_pred": probe.get("normalized_pred"),
        "normalized_gold": probe.get("normalized_gold"),
        "exact_match": bool(probe.get("exact_match", False)),
        "context_exact_match": context_exact_match,
        "answer_match_rule": effective_answer_match_rule(
            probe.get("answer_match_rule", "none"),
            pred_answer=pred_answer,
            gold_answer=gold_answer,
            answer_correct=answer_correct,
            context_exact_match=context_exact_match,
        ),
        "AnswerCorrect_t": answer_correct,
        "SupportSufficient_t": bool(probe.get("SupportSufficient_t", False)),
        "support_rule": str(probe.get("support_rule", "")).strip(),
        "support_evidence_summary": str(probe.get("support_evidence_summary", "")).strip(),
        "missing_support_reasons": [
            str(x) for x in probe.get("missing_support_reasons", []) if str(x).strip()
        ] if isinstance(probe.get("missing_support_reasons", []), list) else [],
        "TeacherStop_t": bool(probe.get("TeacherStop_t", False)),
    }


def resolve_terminal_trace(full_rec: dict, field: str) -> dict:
    terminal_t = full_rec.get("terminal_t")
    steps = full_rec.get("steps", [])
    if isinstance(terminal_t, int) and isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            if int(step.get("t", -1)) == terminal_t and isinstance(step.get(field), dict):
                return step.get(field, {})
    value = full_rec.get(f"terminal_{field}")
    if isinstance(value, dict):
        return value
    return {}


def build_chunks_block(
    history: List[dict],
    unit_registry: Dict[str, dict],
    chunk_registry: Dict[str, dict],
) -> Tuple[List[str], List[dict]]:
    selected_chunk_ids = merge_unique_in_order(
        [item["chunk_id"] for item in history if isinstance(item.get("chunk_id"), str) and item["chunk_id"]]
    )
    chunks: List[dict] = []
    for chunk_id in selected_chunk_ids:
        chunk_info = chunk_registry.get(chunk_id, {})
        selected_unit_ids = [
            item["unit_id"]
            for item in history
            if item.get("chunk_id") == chunk_id and looks_like_raw_unit_id(item.get("unit_id"))
        ]
        selected_unit_texts = []
        for unit_id in selected_unit_ids:
            raw = unit_registry.get(unit_id)
            if raw is None:
                continue
            selected_unit_texts.append({"unit_id": unit_id, "text": raw["text"]})
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": chunk_info.get("doc_id"),
                "parent_chunk_id": chunk_info.get("parent_chunk_id", chunk_id),
                "full_chunk_text": chunk_info.get("full_chunk_text"),
                "selected_unit_ids_in_this_chunk": selected_unit_ids,
                "selected_unit_texts": selected_unit_texts,
            }
        )
    return selected_chunk_ids, chunks


def build_debug_record(
    *,
    qid: str,
    query_info: dict,
    full_rec: dict,
    terminal_state: dict,
    unit_registry: Dict[str, dict],
    chunk_registry: Dict[str, dict],
    derived_payloads_by_qid: Dict[str, Dict[str, dict]],
    run_id: str,
) -> dict:
    terminal_t = int(full_rec["terminal_t"])
    derived_payloads = derived_payloads_by_qid.get(qid, {})
    history = [build_history_entry(item, unit_registry) for item in terminal_state.get("H_t", [])]
    k_t = render_k_t_with_derived_payloads(str(terminal_state.get("K_t", "")), derived_payloads)
    selected_chunk_ids, chunks = build_chunks_block(history, unit_registry, chunk_registry)
    selected_unit_ids = [item["unit_id"] for item in history]
    answer_probe = resolve_answer_probe(full_rec)
    failure_signals = resolve_terminal_trace(full_rec, "failure_signals")
    gate_trace = resolve_terminal_trace(full_rec, "gate_trace")
    proposer_trace = resolve_terminal_trace(full_rec, "proposer_trace")
    gold_answer = str(query_info.get("answer", ""))
    all_chunk_text = "\n".join(
        str(chunk.get("full_chunk_text", "")) for chunk in chunks if isinstance(chunk.get("full_chunk_text"), str)
    )
    has_gold_answer_in_k_t = context_contains_answer(gold_answer, k_t)
    has_gold_answer_in_chunks = context_contains_answer(gold_answer, all_chunk_text)
    supporting_evidence_found = bool(answer_probe["SupportSufficient_t"])
    pred_gold_conflict = answer_conflicts(answer_probe["pred_answer"], gold_answer)

    warnings: List[str] = []
    if not answer_probe["pred_answer"]:
        warnings.append("missing_pred_answer")
    if not gold_answer:
        warnings.append("missing_gold_answer")
    if not answer_probe["answer_match_rule"]:
        warnings.append("missing_answer_match_rule")
    if any(not isinstance(chunk.get("full_chunk_text"), str) or not chunk["full_chunk_text"].strip() for chunk in chunks):
        warnings.append("missing_full_chunk_text")
    if "[missing derived payload]" in k_t:
        warnings.append("missing_derived_payload_in_K_t")
    if gold_answer and not is_yes_no_question(query_info["question"]) and not supporting_evidence_found and not has_gold_answer_in_k_t:
        warnings.append("gold_answer_not_found_in_K_t")
    if gold_answer and not is_yes_no_question(query_info["question"]) and not supporting_evidence_found and not has_gold_answer_in_chunks:
        warnings.append("gold_answer_not_found_in_chunks")
    if not supporting_evidence_found:
        warnings.append("support_insufficient")
    warnings.extend(answer_probe.get("missing_support_reasons", []))
    if answer_probe["TeacherStop_t"] and not answer_probe["pred_answer"]:
        warnings.append("teacher_stop_without_pred_answer")
        warnings.append("teacher_stop_inconsistent")
    if pred_gold_conflict and answer_probe["AnswerCorrect_t"]:
        warnings.append("answer_conflict_but_marked_correct")
    if answer_probe["TeacherStop_t"] and not answer_probe["AnswerCorrect_t"]:
        warnings.append("teacher_stop_inconsistent")
    if answer_probe["TeacherStop_t"] and not answer_probe["SupportSufficient_t"]:
        warnings.append("support_insufficient_but_stopped")

    notes_parts = [
        f"selected_chunks={len(selected_chunk_ids)}",
        f"selected_units={len(selected_unit_ids)}",
        f"answer_match_rule={answer_probe['answer_match_rule']}",
    ]
    if has_gold_answer_in_k_t:
        notes_parts.append("gold_in_K_t")
    if has_gold_answer_in_chunks:
        notes_parts.append("gold_in_chunks")
    if supporting_evidence_found:
        notes_parts.append("support_sufficient")

    return {
        "qid": qid,
        "build_meta": {
            "run_id": run_id,
            "source": "build_hotpotqa_success_debug_v2.py",
        },
        "question": query_info["question"],
        "gold_answer": gold_answer,
        "trajectory_status": {
            "status": "success",
            "terminal_step": terminal_t,
            "abort_reason": None,
        },
        "terminal_state": {
            "t": terminal_t,
            "source_state_t": terminal_t,
            "H_t": history,
            "K_t": k_t,
            "selected_unit_ids": selected_unit_ids,
            "selected_chunk_ids": selected_chunk_ids,
        },
        "chunks": chunks,
        "answer_probe": answer_probe,
        "failure_signals": failure_signals,
        "gate_trace": gate_trace,
        "proposer_trace": proposer_trace,
        "semantic_check": {
            "has_gold_answer_in_chunks": has_gold_answer_in_chunks,
            "has_gold_answer_in_K_t": has_gold_answer_in_k_t,
            "supporting_evidence_found": supporting_evidence_found,
            "support_rule": answer_probe.get("support_rule", ""),
            "support_evidence_summary": answer_probe.get("support_evidence_summary", ""),
            "missing_support_reasons": answer_probe.get("missing_support_reasons", []),
            "notes": "; ".join(notes_parts),
        },
        "debug_warnings": merge_unique_in_order(warnings),
    }


def build_split(base_dir: Path, split: str) -> Tuple[List[dict], dict, str]:
    trajectories_dir = base_dir / "trajectories"
    samples_dir = base_dir / "samples"
    queries_dir = base_dir / "queries"
    unit_registry_dir = base_dir / "unit_registry"
    index_store_dir = base_dir / "index_store"

    queries = load_queries(queries_dir / f"{split}.jsonl")
    full_by_qid = load_full(trajectories_dir / f"full_{split}.jsonl")
    run_id = extract_run_id_from_full_records(full_by_qid, split=split)
    states_by_key = load_states(trajectories_dir / f"states_{split}.jsonl")
    unit_registry, chunk_registry = load_raw_chunk_registry(
        unit_registry_dir / f"raw_units_{split}.jsonl",
        index_store_dir / f"chunks_{split}.jsonl",
    )

    derived_payloads_by_qid: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for qid, payloads in load_sample_derived_payloads(samples_dir / f"{split}.jsonl").items():
        derived_payloads_by_qid[qid].update(payloads)
    for qid, payloads in load_derived_harvest(trajectories_dir / f"derived_harvest_{split}.jsonl").items():
        derived_payloads_by_qid[qid].update(payloads)

    records: List[dict] = []
    stats = {
        "trajectories": len(full_by_qid),
        "success": 0,
        "with_pred_answer": 0,
        "with_full_chunk_text": 0,
        "gold_answer_in_K_t": 0,
        "gold_answer_in_chunks": 0,
        "warnings": 0,
        "warning_counts": Counter(),
    }

    for qid, full_rec in full_by_qid.items():
        if full_rec.get("terminal_status") != "terminal":
            continue
        terminal_t = int(full_rec["terminal_t"])
        terminal_probe = full_rec.get("terminal_probe", {})
        terminal_state = terminal_probe.get("state_snapshot") if isinstance(terminal_probe, dict) else None
        if not isinstance(terminal_state, dict):
            terminal_state = states_by_key.get((qid, terminal_t))
        if not isinstance(terminal_state, dict):
            raise ValueError(f"缺少真实 terminal state: split={split}, qid={qid}, t={terminal_t}")
        query_info = queries.get(qid)
        if query_info is None:
            raise ValueError(f"queries 中找不到 qid: split={split}, qid={qid}")
        record = build_debug_record(
            qid=qid,
            query_info=query_info,
            full_rec=full_rec,
            terminal_state=terminal_state,
            unit_registry=unit_registry,
            chunk_registry=chunk_registry,
            derived_payloads_by_qid=derived_payloads_by_qid,
            run_id=run_id,
        )
        records.append(record)

        stats["success"] += 1
        if record["answer_probe"]["pred_answer"]:
            stats["with_pred_answer"] += 1
        if all(isinstance(chunk.get("full_chunk_text"), str) and chunk["full_chunk_text"].strip() for chunk in record["chunks"]):
            stats["with_full_chunk_text"] += 1
        if record["semantic_check"]["has_gold_answer_in_K_t"]:
            stats["gold_answer_in_K_t"] += 1
        if record["semantic_check"]["has_gold_answer_in_chunks"]:
            stats["gold_answer_in_chunks"] += 1
        if record["debug_warnings"]:
            stats["warnings"] += 1
            stats["warning_counts"].update(record["debug_warnings"])

    return records, stats, run_id


def print_stats(stats_by_split: Dict[str, dict]) -> None:
    print("success semantic debug built:")
    total_warning_counts = Counter()
    for split in SPLITS:
        stats = stats_by_split[split]
        print(f"  {split}:")
        print(f"    trajectories: {stats['trajectories']}")
        print(f"    success: {stats['success']}")
        print(f"    with_pred_answer: {stats['with_pred_answer']}")
        print(f"    with_full_chunk_text: {stats['with_full_chunk_text']}")
        print(f"    gold_answer_in_K_t: {stats['gold_answer_in_K_t']}")
        print(f"    gold_answer_in_chunks: {stats['gold_answer_in_chunks']}")
        print(f"    warnings: {stats['warnings']}")
        total_warning_counts.update(stats["warning_counts"])
        print()
    print("warning counts:")
    if not total_warning_counts:
        print("  (none)")
        return
    for key, value in sorted(total_warning_counts.items()):
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build success semantic debug v2")
    parser.add_argument("--force", action="store_true", help="Accepted for compatibility; overwrites fixed output files in place")
    parser.parse_args()

    base_dir = DEFAULT_BASE
    debug_dir = base_dir / "debug"
    stats_by_split: Dict[str, dict] = {}
    run_ids_by_split: Dict[str, str] = {}
    for split in SPLITS:
        records, stats, run_id = build_split(base_dir, split)
        out_path = debug_dir / f"success_semantic_debug_{split}.jsonl"
        write_jsonl(records, out_path)
        stats_by_split[split] = stats
        run_ids_by_split[split] = run_id
    write_debug_manifest(base_dir, debug_type="success", run_ids_by_split=run_ids_by_split)
    print_stats(stats_by_split)


if __name__ == "__main__":
    main()
