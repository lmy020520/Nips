import copy
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v2"

ALPHA = 1.0
ETA_BR = 1.0
ETA_DIS = 1.0
ETA_SUP = 1.0
ETA_CTX = 1.5

KAPPA_RAW_SENTENCE = 1.00
KAPPA_RAW_CHUNK = 1.10
KAPPA_BRIDGE_NOTE = 1.15
KAPPA_VERIFICATION_NOTE = 1.20

MAX_RENDER_RAW = 4
MAX_RENDER_NOTES = 2
MAX_CHARS_PER_ITEM = 160


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


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'“”‘’`.,;:!?()\[\]{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shorten(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def load_queries(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "question", "answer"]
        for field in required:
            if field not in record:
                raise ValueError(f"queries 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"queries 中重复 qid: file={path}, qid={qid}")

        out[qid] = {
            "qid": qid,
            "question": str(record["question"]).strip(),
            "answer": str(record["answer"]).strip(),
        }
    return out


def load_targets(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "T_q_raw"]
        for field in required:
            if field not in record:
                raise ValueError(f"targets 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"targets 中重复 qid: file={path}, qid={qid}")

        t_q_raw = record["T_q_raw"]
        if not isinstance(t_q_raw, list) or len(t_q_raw) == 0:
            raise ValueError(f"T_q_raw 必须是非空 list: qid={qid}")

        target_map = {}
        role_counts = {"bridge": 0.0, "distinguish": 0.0, "support": 0.0}

        for i, item in enumerate(t_q_raw):
            required_item = ["text", "primary_role", "weight"]
            for field in required_item:
                if field not in item:
                    raise ValueError(f"T_q_raw[{i}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item.get("chunk_id", item.get("unit_id", ""))).strip()
            if not unit_id:
                raise ValueError(f"T_q_raw[{i}] 缺少 chunk_id/unit_id: qid={qid}")
            role = str(item["primary_role"]).strip()
            weight = float(item["weight"])

            if role not in {"bridge", "distinguish", "support"}:
                raise ValueError(f"非法 primary_role: qid={qid}, unit_id={unit_id}, role={role}")
            if unit_id in target_map:
                raise ValueError(f"T_q_raw 中重复 unit_id: qid={qid}, unit_id={unit_id}")

            target_map[unit_id] = {
                "unit_id": unit_id,
                "chunk_id": unit_id,
                "text": str(item["text"]).strip(),
                "primary_role": role,
                "weight": weight,
            }
            role_counts[role] += weight

        out[qid] = {
            "qid": qid,
            "target_map": target_map,
            "role_counts": role_counts,
        }
    return out


def load_raw_unit_map(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = [
            "unit_id",
            "text",
            "doc_id",
            "parent_chunk_id",
            "provenance",
            "candidate_granularity",
        ]
        for field in required:
            if field not in record:
                raise ValueError(f"raw_units 缺少字段: file={path}, row={row_idx}, field={field}")

        unit_id = str(record["unit_id"])
        if unit_id in out:
            raise ValueError(f"raw_units 中重复 unit_id: file={path}, unit_id={unit_id}")

        out[unit_id] = {
            "unit_id": unit_id,
            "text": str(record["text"]).strip(),
            "doc_id": str(record["doc_id"]),
            "parent_chunk_id": str(record["parent_chunk_id"]),
            "span_start": record.get("span_start"),
            "span_end": record.get("span_end"),
            "provenance": str(record["provenance"]),
            "candidate_granularity": str(record["candidate_granularity"]),
        }
    return out


def load_init_states(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "H_t", "A_t", "S_t", "K_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"init_state 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"init_state 中重复 qid: file={path}, qid={qid}")

        h_t = record["H_t"]
        if not isinstance(h_t, list):
            raise ValueError(f"H_t 必须是 list: qid={qid}")

        s_t = record["S_t"]
        a_t = record["A_t"]
        if not isinstance(s_t, dict):
            raise ValueError(f"S_t 必须是 dict: qid={qid}")
        if not isinstance(a_t, dict):
            raise ValueError(f"A_t 必须是 dict: qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "H_t": h_t,
            "A_t": a_t,
            "S_t": s_t,
            "K_t": str(record["K_t"]),
        }
    return out


def load_candidates(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "R_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"candidates 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"candidates 中重复 qid: file={path}, qid={qid}")

        r_t = record["R_t"]
        if not isinstance(r_t, list):
            raise ValueError(f"R_t 必须是 list: qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "R_t": [str(x) for x in r_t],
        }
    return out


def load_derived_filter(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "G_t_final", "G_t_aux", "G_t_illegal"]
        for field in required:
            if field not in record:
                raise ValueError(f"derived_filter 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"derived_filter 中重复 qid: file={path}, qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "G_t_final": [str(x) for x in record["G_t_final"]],
            "G_t_aux": [str(x) for x in record["G_t_aux"]],
            "G_t_illegal": [str(x) for x in record["G_t_illegal"]],
        }
    return out


def load_derived_harvest(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "proposal_run", "G_t_harvest"]
        for field in required:
            if field not in record:
                raise ValueError(f"derived_harvest 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"derived_harvest 中重复 qid: file={path}, qid={qid}")

        g_h = record["G_t_harvest"]
        if not isinstance(g_h, list):
            raise ValueError(f"G_t_harvest 必须是 list: qid={qid}")

        derived_map = {}
        ordered_ids = []
        for idx, item in enumerate(g_h):
            required_item = ["unit_id", "text", "provenance", "candidate_granularity", "type", "source_unit_ids"]
            for field in required_item:
                if field not in item:
                    raise ValueError(f"G_t_harvest[{idx}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item["unit_id"])
            if unit_id in derived_map:
                raise ValueError(f"G_t_harvest 中重复 unit_id: qid={qid}, unit_id={unit_id}")

            derived_map[unit_id] = {
                "unit_id": unit_id,
                "text": str(item["text"]).strip(),
                "provenance": str(item["provenance"]),
                "candidate_granularity": str(item["candidate_granularity"]),
                "type": str(item["type"]),
                "source_unit_ids": [str(x) for x in item["source_unit_ids"]],
                "coarse_priority": idx,
            }
            ordered_ids.append(unit_id)

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "proposal_run": bool(record["proposal_run"]),
            "G_t_harvest_ids": ordered_ids,
            "derived_map": derived_map,
        }
    return out


def build_unit_registry(raw_unit_map: Dict[str, dict], derived_map: Dict[str, dict]) -> Dict[str, dict]:
    registry = dict(raw_unit_map)
    for unit_id, item in derived_map.items():
        if unit_id in registry:
            raise ValueError(f"UnitRegistry 中 unit_id 冲突: {unit_id}")
        registry[unit_id] = item
    return registry


def build_candidate_pool(r_t: List[str], g_t_final: List[str]) -> List[str]:
    seen = set()
    out = []

    for unit_id in r_t + g_t_final:
        if unit_id in seen:
            continue
        seen.add(unit_id)
        out.append(unit_id)

    return out


def normalize_state_ref_list(refs: list) -> List[dict]:
    out = []
    seen = set()
    for item in refs:
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit_id = str(item["unit_id"])
        if unit_id in seen:
            continue
        seen.add(unit_id)
        out.append(
            {
                "unit_id": unit_id,
                "added_step": int(item.get("added_step", 0)),
                "used_in_summary_count": int(item.get("used_in_summary_count", 0)),
                "selected_count": int(item.get("selected_count", 1)),
            }
        )
    return out


def simulate_update(s_t: dict, u_next: dict, step_id: int) -> dict:
    s_next = copy.deepcopy(s_t)
    if "raw_refs" not in s_next or not isinstance(s_next["raw_refs"], list):
        s_next["raw_refs"] = []
    if "derived_refs" not in s_next or not isinstance(s_next["derived_refs"], list):
        s_next["derived_refs"] = []

    s_next["raw_refs"] = normalize_state_ref_list(s_next["raw_refs"])
    s_next["derived_refs"] = normalize_state_ref_list(s_next["derived_refs"])

    ref_key = "raw_refs" if u_next["provenance"] == "raw" else "derived_refs"
    refs = s_next[ref_key]

    found = False
    for ref in refs:
        if ref["unit_id"] == u_next["unit_id"]:
            ref["selected_count"] += 1
            found = True
            break

    if not found:
        refs.append(
            {
                "unit_id": u_next["unit_id"],
                "added_step": step_id,
                "used_in_summary_count": 0,
                "selected_count": 1,
            }
        )

    s_next["last_added_unit_id"] = u_next["unit_id"]
    s_next["last_updated_step"] = step_id
    return s_next


def get_role_key(role: str) -> Tuple[str, str]:
    if role == "bridge":
        return "k_bridge", "k_br"
    if role == "distinguish":
        return "k_distinguish", "k_dis"
    if role == "support":
        return "k_support", "k_sup"
    raise ValueError(f"非法 role: {role}")


def get_k_r(a_t: dict, role: str) -> float:
    key_new, key_old = get_role_key(role)
    if key_new in a_t:
        return float(a_t.get(key_new, 0.0))
    return float(a_t.get(key_old, 0.0))


def set_k_r(a_t: dict, role: str, value: float):
    key_new, key_old = get_role_key(role)
    if key_new in a_t or key_old not in a_t:
        a_t[key_new] = float(value)
    else:
        a_t[key_old] = float(value)


def simulate_ledger(a_t: dict, u_next: dict, target_info: dict) -> dict:
    a_next = copy.deepcopy(a_t)
    covered = set(str(x) for x in a_next.get("covered_target_ids", []))
    target_map = target_info["target_map"]

    if u_next["provenance"] != "raw":
        a_next["covered_target_ids"] = sorted(covered)
        return a_next

    target_unit_id = str(u_next.get("parent_chunk_id", "")).strip()
    if not target_unit_id:
        target_unit_id = str(u_next["unit_id"]).rsplit("::", 1)[0]

    if target_unit_id not in target_map:
        a_next["covered_target_ids"] = sorted(covered)
        return a_next

    if target_unit_id in covered:
        a_next["covered_target_ids"] = sorted(covered)
        return a_next

    covered.add(target_unit_id)
    target = target_map[target_unit_id]
    role = target["primary_role"]
    weight = float(target["weight"])

    current = get_k_r(a_next, role)
    set_k_r(a_next, role, current + weight)
    a_next["covered_target_ids"] = sorted(covered)
    return a_next


def get_latest_note_ids_by_type(s_t: dict, unit_registry: Dict[str, dict]) -> List[str]:
    derived_refs = normalize_state_ref_list(s_t.get("derived_refs", []))
    latest_by_type = {}

    for ref in derived_refs:
        unit_id = ref["unit_id"]
        unit = unit_registry.get(unit_id)
        if unit is None or unit.get("provenance") != "derived":
            continue

        note_type = unit.get("type")
        if note_type not in {"bridge_note", "verification_note"}:
            continue

        prev = latest_by_type.get(note_type)
        if prev is None or ref["added_step"] > prev["added_step"]:
            latest_by_type[note_type] = ref

    ordered = []
    if "bridge_note" in latest_by_type:
        ordered.append(latest_by_type["bridge_note"]["unit_id"])
    if "verification_note" in latest_by_type:
        ordered.append(latest_by_type["verification_note"]["unit_id"])
    return ordered


def render_context(q: dict, s_t: dict, unit_registry: Dict[str, dict]) -> str:
    note_ids = get_latest_note_ids_by_type(s_t, unit_registry)

    source_raw_ids = []
    for note_id in note_ids:
        note = unit_registry[note_id]
        for uid in note.get("source_unit_ids", []):
            if uid not in source_raw_ids:
                source_raw_ids.append(uid)

    raw_ids = list(source_raw_ids)

    raw_refs = normalize_state_ref_list(s_t.get("raw_refs", []))
    for ref in sorted(raw_refs, key=lambda x: x["added_step"], reverse=True):
        if len(raw_ids) >= MAX_RENDER_RAW:
            break
        if ref["unit_id"] not in raw_ids:
            raw_ids.append(ref["unit_id"])

    parts = []

    raw_units = [unit_registry[uid] for uid in raw_ids[:MAX_RENDER_RAW] if uid in unit_registry]
    if raw_units:
        parts.append("Evidence:")
        for i, u in enumerate(raw_units, start=1):
            parts.append(f"[{i}] {shorten(u['text'], MAX_CHARS_PER_ITEM)}")

    active_notes = [unit_registry[uid] for uid in note_ids if uid in unit_registry]
    if active_notes:
        parts.append("")
        parts.append("Notes:")
        for note in active_notes[:MAX_RENDER_NOTES]:
            label = "bridge" if note["type"] == "bridge_note" else "verification"
            parts.append(f"[{label}] {shorten(note['text'], MAX_CHARS_PER_ITEM)}")

    return "\n".join(parts).strip()


def answerability_probe(gold_answer: str, k_t: str) -> int:
    norm_gold = normalize_text(gold_answer)
    norm_ctx = normalize_text(k_t)
    if not norm_gold:
        return 0
    return 1 if norm_gold in norm_ctx else 0


def support_sufficient(gold_answer: str, k_t: str, a_t: dict, target_info: dict, s_t: dict, unit_registry: Dict[str, dict]) -> int:
    target_map = target_info["target_map"]

    support_targets = {uid for uid, item in target_map.items() if item["primary_role"] == "support"}
    bridge_targets = {uid for uid, item in target_map.items() if item["primary_role"] == "bridge"}
    covered = set(str(x) for x in a_t.get("covered_target_ids", []))

    ans_present = answerability_probe(gold_answer, k_t) == 1
    support_ready = support_targets.issubset(covered) if support_targets else True

    has_bridge_note = False
    derived_refs = normalize_state_ref_list(s_t.get("derived_refs", []))
    for ref in derived_refs:
        unit = unit_registry.get(ref["unit_id"])
        if unit and unit.get("provenance") == "derived" and unit.get("type") == "bridge_note":
            has_bridge_note = True
            break

    bridge_ready = (bridge_targets.issubset(covered) if bridge_targets else True) or has_bridge_note
    return 1 if (ans_present and support_ready and bridge_ready) else 0


def distinguish_sufficient(a_t: dict, target_info: dict) -> int:
    target_map = target_info["target_map"]
    dis_targets = {uid for uid, item in target_map.items() if item["primary_role"] == "distinguish"}
    if not dis_targets:
        return 1
    covered = set(str(x) for x in a_t.get("covered_target_ids", []))
    return 1 if dis_targets.issubset(covered) else 0


def compute_a_ctx(q: dict, k_t: str, a_t: dict, s_t: dict, target_info: dict, unit_registry: Dict[str, dict]) -> float:
    a_ans = answerability_probe(q["answer"], k_t)
    a_sup = support_sufficient(q["answer"], k_t, a_t, target_info, s_t, unit_registry)
    a_dis = distinguish_sufficient(a_t, target_info)
    return (a_ans + a_sup + a_dis) / 3.0


def get_N_q_r(target_info: dict, role: str) -> float:
    return float(target_info["role_counts"].get(role, 0.0))


def compute_delta_role(a_t: dict, a_u: dict, target_info: dict, role: str, alpha: float) -> float:
    N_r = get_N_q_r(target_info, role)
    if N_r <= 0:
        return 0.0

    k_t = get_k_r(a_t, role)
    k_u = get_k_r(a_u, role)

    s_t = (k_t + alpha) / (N_r + 2 * alpha)
    s_u = (k_u + alpha) / (N_r + 2 * alpha)
    return s_u - s_t


def compute_kappa(u: dict) -> float:
    if u["provenance"] == "raw":
        granularity = u.get("candidate_granularity", "sentence")
        if granularity == "chunk":
            return KAPPA_RAW_CHUNK
        return KAPPA_RAW_SENTENCE

    if u["provenance"] == "derived":
        note_type = u.get("type")
        if note_type == "bridge_note":
            return KAPPA_BRIDGE_NOTE
        if note_type == "verification_note":
            return KAPPA_VERIFICATION_NOTE
        return KAPPA_VERIFICATION_NOTE

    raise ValueError(f"非法 provenance: {u.get('provenance')}")


def candidate_priority_key(u: dict, utility: float) -> Tuple:
    is_raw = 0 if u["provenance"] == "raw" else 1
    kappa = compute_kappa(u)

    coarse_priority = u.get("coarse_priority", 10**9)
    # 负号用于更高 coarse_priority 优先
    coarse_priority_key = -int(coarse_priority) if u["provenance"] == "derived" else 0

    return (
        -utility,
        is_raw,
        kappa,
        coarse_priority_key,
        str(u["unit_id"]),
    )


def compute_U_for_candidate(q: dict, state: dict, target_info: dict, unit_registry: Dict[str, dict], u: dict) -> float:
    step_id = len(state["H_t"])
    s_u = simulate_update(state["S_t"], u, step_id=step_id)
    a_u = simulate_ledger(state["A_t"], u, target_info)
    k_u = render_context(q, s_u, unit_registry)

    delta_br = compute_delta_role(state["A_t"], a_u, target_info, "bridge", ALPHA)
    delta_dis = compute_delta_role(state["A_t"], a_u, target_info, "distinguish", ALPHA)
    delta_sup = compute_delta_role(state["A_t"], a_u, target_info, "support", ALPHA)

    a_ctx_t = compute_a_ctx(q, state["K_t"], state["A_t"], state["S_t"], target_info, unit_registry)
    a_ctx_u = compute_a_ctx(q, k_u, a_u, s_u, target_info, unit_registry)
    delta_ctx = a_ctx_u - a_ctx_t

    kappa = compute_kappa(u)

    numerator = (
        ETA_BR * delta_br +
        ETA_DIS * delta_dis +
        ETA_SUP * delta_sup +
        ETA_CTX * delta_ctx
    )
    return numerator / kappa


def build_teacher_select_record(
    qid: str,
    query_info: dict,
    state: dict,
    target_info: dict,
    c_t: List[str],
    unit_registry: Dict[str, dict],
) -> dict:
    if not c_t:
        raise ValueError(f"C_t 为空: qid={qid}")

    candidate_units = []
    for unit_id in c_t:
        if unit_id not in unit_registry:
            raise ValueError(f"C_t 中的 unit_id 不在 UnitRegistry: qid={qid}, unit_id={unit_id}")
        u = unit_registry[unit_id]
        candidate_units.append(u)

    scored = []
    for u in candidate_units:
        utility = compute_U_for_candidate(
            q=query_info,
            state=state,
            target_info=target_info,
            unit_registry=unit_registry,
            u=u,
        )
        scored.append((u, utility))

    scored.sort(key=lambda x: candidate_priority_key(x[0], x[1]))
    positive_unit_id = scored[0][0]["unit_id"]

    if positive_unit_id not in c_t:
        raise RuntimeError(f"positive_unit_id 不在 C_t 中: qid={qid}, unit_id={positive_unit_id}")

    return {
        "qid": qid,
        "t": 0,
        "C_t": c_t,
        "positive_unit_id": positive_unit_id,
    }


def convert_split(
    queries_path: Path,
    targets_path: Path,
    init_state_path: Path,
    candidates_path: Path,
    derived_filter_path: Path,
    derived_harvest_path: Path,
    raw_units_path: Path,
    output_path: Path,
) -> int:
    queries = load_queries(queries_path)
    targets = load_targets(targets_path)
    init_states = load_init_states(init_state_path)
    candidates = load_candidates(candidates_path)
    derived_filter = load_derived_filter(derived_filter_path)
    derived_harvest = load_derived_harvest(derived_harvest_path)
    raw_unit_map = load_raw_unit_map(raw_units_path)

    def generator():
        for qid in sorted(init_states.keys()):
            if qid not in queries:
                raise ValueError(f"queries 中找不到 qid: {qid}")
            if qid not in targets:
                raise ValueError(f"targets 中找不到 qid: {qid}")
            if qid not in candidates:
                raise ValueError(f"candidates 中找不到 qid: {qid}")
            if qid not in derived_filter:
                raise ValueError(f"derived_filter 中找不到 qid: {qid}")
            if qid not in derived_harvest:
                raise ValueError(f"derived_harvest 中找不到 qid: {qid}")

            g_final = derived_filter[qid]["G_t_final"]
            c_t = build_candidate_pool(candidates[qid]["R_t"], g_final)

            # 只把该 qid 的 derived harvest 注册进 UnitRegistry
            unit_registry = build_unit_registry(
                raw_unit_map=raw_unit_map,
                derived_map=derived_harvest[qid]["derived_map"],
            )

            yield build_teacher_select_record(
                qid=qid,
                query_info=queries[qid],
                state=init_states[qid],
                target_info=targets[qid],
                c_t=c_t,
                unit_registry=unit_registry,
            )

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    trajectories_dir = base_dir / "trajectories"
    queries_dir = base_dir / "queries"
    targets_dir = base_dir / "targets"
    unit_registry_dir = base_dir / "unit_registry"

    raw_units_name_map = {
        "train": "raw_units_train.jsonl",
        "val": "raw_units_val.jsonl",
        "test": "raw_units_test.jsonl",
    }
    out_name_map = {
        "train": "teacher_select_train.jsonl",
        "val": "teacher_select_val.jsonl",
        "test": "teacher_select_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        queries_path = queries_dir / f"{split}.jsonl"
        targets_path = targets_dir / f"{split}.jsonl"
        init_state_path = trajectories_dir / f"init_state_{split}.jsonl"
        candidates_path = trajectories_dir / f"candidates_{split}.jsonl"
        derived_filter_path = trajectories_dir / f"derived_filter_{split}.jsonl"
        derived_harvest_path = trajectories_dir / f"derived_harvest_{split}.jsonl"
        raw_units_path = unit_registry_dir / raw_units_name_map[split]
        output_path = trajectories_dir / out_name_map[split]

        for path in [
            queries_path,
            targets_path,
            init_state_path,
            candidates_path,
            derived_filter_path,
            derived_harvest_path,
            raw_units_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            queries_path=queries_path,
            targets_path=targets_path,
            init_state_path=init_state_path,
            candidates_path=candidates_path,
            derived_filter_path=derived_filter_path,
            derived_harvest_path=derived_harvest_path,
            raw_units_path=raw_units_path,
            output_path=output_path,
        )

    print("teacher_select v2 构建完成：")
    print(
        f"  defaults: eta_br={ETA_BR}, eta_dis={ETA_DIS}, eta_sup={ETA_SUP}, eta_ctx={ETA_CTX}, "
        f"alpha={ALPHA}"
    )
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {trajectories_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()
