import copy
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v3"

# retrieval
CHUNK_SHORTLIST_K = 8
FINAL_KR = 10
MAX_SUMMARY_CHARS = 160

# gates
RAW_COMPLETE_THRESHOLD = 0.70
TAU_SEM = 0.50
REDUNDANT_SIM_THRESHOLD = 0.90
COMPOSABLE_SIM_THRESHOLD = 0.85

# derived proposal
TOP_RAW_J = 3
MAX_DERIVED_CANDIDATES = 4
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 2.0
ALLOWED_DERIVED_TYPES = {"bridge_note", "verification_note"}

# legality / retain
MAX_SOURCE_PER_NOTE = 3
MAX_NOTE_TOKENS = 45
DUPLICATE_SIM_THRESHOLD = 0.90
MAX_FINAL_DERIVED = 2

# render helper
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


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    pieces = [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]
    return max(1, len(pieces))


def shorten(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def parse_qid_from_chunk_id(chunk_id: str) -> str:
    parts = chunk_id.split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"chunk_id 格式错误: {chunk_id}")
    return parts[0]


def parse_sent_id_from_unit_id(unit_id: str) -> int:
    parts = unit_id.split("::")
    if len(parts) < 3:
        raise ValueError(f"unit_id 格式错误，无法解析 sent_id: {unit_id}")
    return int(parts[-1])


def canonical_role(role: str) -> str:
    role = str(role).strip().lower()
    if role == "bridge":
        return "bridge"
    if role in {"distinguish", "disambiguation"}:
        return "distinguish"
    if role == "support":
        return "support"
    raise ValueError(f"非法 role: {role}")


def get_role_key(role: str) -> str:
    role = canonical_role(role)
    if role == "bridge":
        return "k_br"
    if role == "distinguish":
        return "k_dis"
    if role == "support":
        return "k_sup"
    raise ValueError(f"非法 role: {role}")


def is_derived_unit_id(unit_id: str) -> bool:
    return "::derived::" in str(unit_id)


def build_derived_unit_id(qid: str, idx: int) -> str:
    return f"{qid}::derived::{idx}"


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


class BM25Index:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("BM25 docs 不能为空")
        self.k1 = k1
        self.b = b
        self.doc_tfs: List[Counter] = []
        self.doc_lens: List[int] = []
        self.df = Counter()

        for text in docs:
            toks = tokenize(text)
            tf = Counter(toks)
            self.doc_tfs.append(tf)
            self.doc_lens.append(len(toks))
            for term in tf.keys():
                self.df[term] += 1

        self.N = len(docs)
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> List[float]:
        q_terms = tokenize(query)
        if not q_terms:
            return [0.0] * self.N

        q_tf = Counter(q_terms)
        scores = [0.0] * self.N

        for i in range(self.N):
            tf_doc = self.doc_tfs[i]
            dl = self.doc_lens[i]
            score = 0.0
            for term in q_tf.keys():
                if term not in tf_doc:
                    continue
                tf = tf_doc[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-12))
                score += idf * (tf * (self.k1 + 1)) / max(denom, 1e-12)
            scores[i] = score
        return scores


def load_queries(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        for field in ["qid", "question", "answer"]:
            if field not in record:
                raise ValueError(f"queries 缺少字段: file={path}, row={row_idx}, field={field}")
        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"queries 中重复 qid: {qid}")
        out[qid] = {
            "qid": qid,
            "question": str(record["question"]).strip(),
            "answer": str(record["answer"]).strip(),
        }
    return out


def load_targets(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "T_q_raw" not in record:
            raise ValueError(f"targets 缺少字段: file={path}, row={row_idx}")
        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"targets 中重复 qid: {qid}")

        target_map = {}
        role_counts = {"bridge": 0.0, "distinguish": 0.0, "support": 0.0}
        for i, item in enumerate(record["T_q_raw"]):
            for field in ["text", "primary_role", "weight"]:
                if field not in item:
                    raise ValueError(f"T_q_raw[{i}] 缺少字段 {field}: qid={qid}")
            unit_id = str(item.get("chunk_id", item.get("unit_id", ""))).strip()
            if not unit_id:
                raise ValueError(f"T_q_raw[{i}] 缺少 chunk_id/unit_id: qid={qid}")
            if unit_id in target_map:
                raise ValueError(f"T_q_raw 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            role = canonical_role(item["primary_role"])
            weight = float(item["weight"])
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
            "required_roles": sorted([r for r, v in role_counts.items() if v > 0]),
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
            "span_start",
            "span_end",
            "provenance",
            "candidate_granularity",
        ]
        for field in required:
            if field not in record:
                raise ValueError(f"raw_units 缺少字段: file={path}, row={row_idx}, field={field}")
        unit_id = str(record["unit_id"])
        if unit_id in out:
            raise ValueError(f"raw_units 中重复 unit_id: {unit_id}")
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


def load_chunks_grouped(path: Path) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    seen = set()
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        for field in ["chunk_id", "doc_id", "chunk_text", "summary_text"]:
            if field not in record:
                raise ValueError(f"chunks 缺少字段: file={path}, row={row_idx}, field={field}")
        chunk_id = str(record["chunk_id"])
        if chunk_id in seen:
            raise ValueError(f"chunks 中重复 chunk_id: {chunk_id}")
        seen.add(chunk_id)
        qid = parse_qid_from_chunk_id(chunk_id)
        grouped[qid].append(
            {
                "chunk_id": chunk_id,
                "doc_id": str(record["doc_id"]),
                "chunk_text": str(record["chunk_text"]).strip(),
                "summary_text": str(record["summary_text"]).strip(),
            }
        )
    return dict(grouped)


def load_atoms_by_chunk(path: Path) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    seen = set()
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        for field in ["atom_id", "parent_chunk_id", "doc_id", "atom_text"]:
            if field not in record:
                raise ValueError(f"atoms 缺少字段: file={path}, row={row_idx}, field={field}")
        atom_id = str(record["atom_id"])
        if atom_id in seen:
            raise ValueError(f"atoms 中重复 atom_id: {atom_id}")
        seen.add(atom_id)
        grouped[str(record["parent_chunk_id"])].append(
            {
                "atom_id": atom_id,
                "parent_chunk_id": str(record["parent_chunk_id"]),
                "doc_id": str(record["doc_id"]),
                "atom_text": str(record["atom_text"]).strip(),
            }
        )
    return dict(grouped)


def load_init_states(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "H_t", "A_t", "S_t", "K_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"init_state 缺少字段: file={path}, row={row_idx}, field={field}")
        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"init_state 中重复 qid: {qid}")
        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "H_t": record["H_t"],
            "A_t": record["A_t"],
            "S_t": record["S_t"],
            "K_t": str(record["K_t"]),
        }
    return out


def load_full_trajectories(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "terminal_status", "terminal_t", "abort_reason", "steps"]
        for field in required:
            if field not in record:
                raise ValueError(f"full trajectory 缺少字段: file={path}, row={row_idx}, field={field}")
        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"full trajectory 中重复 qid: {qid}")

        steps = record["steps"]
        if not isinstance(steps, list):
            raise ValueError(f"steps 必须是 list: qid={qid}")

        normalized_steps = []
        seen_t = set()
        for i, item in enumerate(steps):
            if not isinstance(item, dict):
                raise ValueError(f"steps[{i}] 必须是 dict: qid={qid}")
            if "t" not in item or "positive_unit_id" not in item:
                raise ValueError(f"steps[{i}] 缺少字段: qid={qid}")
            t = int(item["t"])
            if t in seen_t:
                raise ValueError(f"steps 中重复 t: qid={qid}, t={t}")
            seen_t.add(t)
            normalized_steps.append(
                {
                    "t": t,
                    "positive_unit_id": str(item["positive_unit_id"]),
                }
            )
        normalized_steps.sort(key=lambda x: x["t"])

        out[qid] = {
            "qid": qid,
            "terminal_status": str(record["terminal_status"]),
            "terminal_t": record["terminal_t"],
            "abort_reason": record["abort_reason"],
            "steps": normalized_steps,
        }
    return out


def load_candidate_records(path: Path) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "t" not in record:
            raise ValueError(f"candidates 缺少 qid 或 t: file={path}, row={row_idx}")
        key = (str(record["qid"]), int(record["t"]))
        if key in out:
            raise ValueError(f"candidates 中重复 prefix key: file={path}, row={row_idx}, key={key}")
        out[key] = record
    return out


def clone_state(state: dict) -> dict:
    return {
        "qid": state["qid"],
        "t": int(state["t"]),
        "H_t": copy.deepcopy(state["H_t"]),
        "A_t": copy.deepcopy(state["A_t"]),
        "S_t": copy.deepcopy(state["S_t"]),
        "K_t": str(state["K_t"]),
    }


def build_unit_registry(raw_unit_map: Dict[str, dict], derived_registry: Dict[str, dict]) -> Dict[str, dict]:
    reg = dict(raw_unit_map)
    for unit_id, item in derived_registry.items():
        if unit_id in reg:
            raise ValueError(f"UnitRegistry 中 unit_id 冲突: {unit_id}")
        reg[unit_id] = item
    return reg


def normalize_refs_minimal(refs: list) -> List[dict]:
    out = []
    seen = set()
    if not isinstance(refs, list):
        return out
    for item in refs:
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit_id = str(item["unit_id"])
        if unit_id in seen:
            continue
        seen.add(unit_id)
        out.append({"unit_id": unit_id, "added_step": int(item.get("added_step", 0))})
    return out


def simulate_update(h_t: List[dict], s_t: dict, u_next: dict) -> Tuple[list, dict]:
    h_next = copy.deepcopy(h_t)
    s_next = copy.deepcopy(s_t)

    if "raw_refs" not in s_next or not isinstance(s_next["raw_refs"], list):
        s_next["raw_refs"] = []
    if "derived_refs" not in s_next or not isinstance(s_next["derived_refs"], list):
        s_next["derived_refs"] = []

    s_next["raw_refs"] = normalize_refs_minimal(s_next["raw_refs"])
    s_next["derived_refs"] = normalize_refs_minimal(s_next["derived_refs"])

    next_step_id = len(h_next)
    unit_id = u_next["unit_id"]

    existing_h_ids = {str(x["unit_id"]) for x in h_next if isinstance(x, dict) and "unit_id" in x}
    if unit_id not in existing_h_ids:
        h_next.append({"step_id": next_step_id, "unit_id": unit_id})

    ref_key = "raw_refs" if u_next["provenance"] == "raw" else "derived_refs"
    existing_ref_ids = {str(x["unit_id"]) for x in s_next[ref_key] if isinstance(x, dict) and "unit_id" in x}
    if unit_id not in existing_ref_ids:
        s_next[ref_key].append({"unit_id": unit_id, "added_step": next_step_id})

    s_next["last_added_unit_id"] = unit_id
    s_next["last_updated_step"] = next_step_id
    return h_next, s_next


def simulate_ledger(a_t: dict, u_next: dict, target_info: dict) -> dict:
    a_next = copy.deepcopy(a_t)
    covered = [str(x) for x in a_next.get("covered_target_ids", [])]
    covered_set = set(covered)
    target_map = target_info["target_map"]

    if u_next["provenance"] != "raw":
        a_next["covered_target_ids"] = covered
        return a_next

    target_unit_id = str(u_next.get("parent_chunk_id", "")).strip()
    if not target_unit_id:
        target_unit_id = str(u_next["unit_id"]).rsplit("::", 1)[0]

    if target_unit_id not in target_map or target_unit_id in covered_set:
        a_next["covered_target_ids"] = covered
        return a_next

    covered.append(target_unit_id)
    covered_set.add(target_unit_id)

    target = target_map[target_unit_id]
    role = canonical_role(target["primary_role"])
    weight = float(target["weight"])

    current = float(a_next.get(get_role_key(role), 0.0))
    a_next[get_role_key(role)] = current + weight
    a_next["covered_target_ids"] = covered
    return a_next


def get_last_raw_text(s_t: dict, unit_registry: Dict[str, dict]) -> Optional[str]:
    raw_refs = s_t.get("raw_refs", [])
    if not isinstance(raw_refs, list) or len(raw_refs) == 0:
        return None
    last_ref = raw_refs[-1]
    if not isinstance(last_ref, dict) or "unit_id" not in last_ref:
        return None
    unit = unit_registry.get(str(last_ref["unit_id"]))
    if unit is None:
        return None
    return unit["text"]


def get_last_derived_text(s_t: dict, unit_registry: Dict[str, dict]) -> Optional[str]:
    derived_refs = s_t.get("derived_refs", [])
    if not isinstance(derived_refs, list) or len(derived_refs) == 0:
        return None
    last_ref = derived_refs[-1]
    if not isinstance(last_ref, dict) or "unit_id" not in last_ref:
        return None
    unit = unit_registry.get(str(last_ref["unit_id"]))
    if unit is None:
        return None
    return unit["text"]


def get_latest_note_ids_by_type(s_t: dict, unit_registry: Dict[str, dict]) -> List[str]:
    derived_refs = normalize_refs_minimal(s_t.get("derived_refs", []))
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
    raw_refs = normalize_refs_minimal(s_t.get("raw_refs", []))
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


def build_slot_summary(question: str, s_t: dict, unit_registry: Dict[str, dict]) -> str:
    parts = []
    last_raw = get_last_raw_text(s_t, unit_registry)
    if last_raw is not None:
        parts.append("Evidence: " + shorten(last_raw, 80))
    last_note = get_last_derived_text(s_t, unit_registry)
    if last_note is not None:
        parts.append("Note: " + shorten(last_note, 70))
    parts.append("Need: " + extract_goal_from_question(question))
    return "\n".join(parts)[:MAX_SUMMARY_CHARS].rstrip()


def build_q_t(question: str, s_t: dict, unit_registry: Dict[str, dict]) -> str:
    h_t = build_slot_summary(question, s_t, unit_registry)
    if not h_t:
        return question
    return question + "\n" + h_t


def topk_retrieve(
    q_t: str,
    chunks: List[dict],
    atoms_by_chunk: Dict[str, List[dict]],
    used_unit_ids: set,
) -> List[str]:
    if not chunks:
        return []

    summary_index = BM25Index([c["summary_text"] for c in chunks])
    chunk_index = BM25Index([c["chunk_text"] for c in chunks])
    s_scores = summary_index.score(q_t)
    c_scores = chunk_index.score(q_t)

    scored_chunks = []
    for chunk, s_sum, s_chunk in zip(chunks, s_scores, c_scores):
        coarse_score = 0.5 * s_sum + 0.5 * s_chunk
        scored_chunks.append({"chunk_id": chunk["chunk_id"], "score": float(coarse_score)})

    scored_chunks.sort(key=lambda x: (-x["score"], x["chunk_id"]))
    shortlist = scored_chunks[:CHUNK_SHORTLIST_K]

    merged = []
    for item in shortlist:
        atoms = atoms_by_chunk.get(item["chunk_id"], [])
        if not atoms:
            continue

        atom_index = BM25Index([a["atom_text"] for a in atoms])
        atom_scores = atom_index.score(q_t)

        scored_atoms = []
        for atom, score in zip(atoms, atom_scores):
            if atom["atom_id"] in used_unit_ids:
                continue
            scored_atoms.append({"unit_id": atom["atom_id"], "score": float(score)})

        if not scored_atoms:
            continue

        scored_atoms.sort(key=lambda x: (-x["score"], x["unit_id"]))
        merged.append(scored_atoms[0])

    merged.sort(key=lambda x: (-x["score"], x["unit_id"]))
    return [x["unit_id"] for x in merged[:FINAL_KR]]


def normalized_role_scores(a_t: dict, target_info: dict) -> Dict[str, float]:
    denoms = target_info["role_counts"]
    out = {}
    for role in target_info["required_roles"]:
        denom = max(float(denoms.get(role, 0.0)), 1.0)
        out[role] = float(a_t.get(get_role_key(role), 0.0)) / denom
    return out


def get_recent_raw_texts(s_t: dict, unit_registry: Dict[str, dict], n: int = 2) -> List[str]:
    raw_refs = s_t.get("raw_refs", [])
    if not isinstance(raw_refs, list):
        return []
    recent = []
    for item in raw_refs[-n:]:
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit = unit_registry.get(str(item["unit_id"]))
        if unit is not None:
            recent.append(unit["text"])
    return recent


def has_recent_note(s_t: dict, unit_registry: Dict[str, dict], note_type: str, window: int = 2) -> bool:
    derived_refs = s_t.get("derived_refs", [])
    if not isinstance(derived_refs, list):
        return False
    recent = derived_refs[-window:]
    for ref in recent:
        if not isinstance(ref, dict) or "unit_id" not in ref:
            continue
        unit = unit_registry.get(str(ref["unit_id"]))
        if unit and unit.get("provenance") == "derived" and unit.get("type") == note_type:
            return True
    return False


def is_raw_pool_redundant(r_t: List[str], s_t: dict, unit_registry: Dict[str, dict]) -> bool:
    recent_raw_texts = get_recent_raw_texts(s_t, unit_registry, n=2)
    if not recent_raw_texts:
        return False

    top_raw = r_t[:3]
    redundant_count = 0
    for unit_id in top_raw:
        unit = unit_registry.get(unit_id)
        if unit is None:
            continue
        text = unit["text"]
        if any(jaccard_similarity(text, prev) >= REDUNDANT_SIM_THRESHOLD for prev in recent_raw_texts):
            redundant_count += 1

    return redundant_count >= 2


def has_composable_raw(r_t: List[str], unit_registry: Dict[str, dict]) -> bool:
    top_raw = r_t[:3]
    if len(top_raw) < 2:
        return False

    usable = []
    for unit_id in top_raw:
        unit = unit_registry.get(unit_id)
        if unit is None:
            continue

        keep = True
        for prev in usable:
            if jaccard_similarity(unit["text"], prev["text"]) >= COMPOSABLE_SIM_THRESHOLD:
                keep = False
                break

        if keep:
            usable.append(
                {
                    "unit_id": unit_id,
                    "text": unit["text"],
                    "parent_chunk_id": unit["parent_chunk_id"],
                }
            )

    distinct_parents = len({u["parent_chunk_id"] for u in usable})
    return len(usable) >= 2 and distinct_parents >= 2


def cheap_stop_gate(target_info: dict, a_t: dict, s_t: dict, r_t: List[str], unit_registry: Dict[str, dict]) -> Dict[str, object]:
    role_scores = normalized_role_scores(a_t, target_info)
    raw_complete = all(role_scores[r] >= RAW_COMPLETE_THRESHOLD for r in target_info["required_roles"])
    has_recent_verif = has_recent_note(s_t, unit_registry, "verification_note", window=2)
    has_recent_bridge = has_recent_note(s_t, unit_registry, "bridge_note", window=2)
    raw_redundant = is_raw_pool_redundant(r_t, s_t, unit_registry)
    no_derived_need = has_recent_verif or (raw_redundant and has_recent_bridge)
    stop_candidate = raw_complete
    return {
        "role_scores": role_scores,
        "raw_complete": raw_complete,
        "no_derived_need": no_derived_need,
        "stop_candidate": stop_candidate,
    }


def need_derived_gate(target_info: dict, a_t: dict, s_t: dict, r_t: List[str], unit_registry: Dict[str, dict]) -> Dict[str, object]:
    role_scores = normalized_role_scores(a_t, target_info)
    s_sem = sum(role_scores[r] for r in target_info["required_roles"]) / max(1, len(target_info["required_roles"]))
    composable_raw = has_composable_raw(r_t, unit_registry)
    has_recent_verification = has_recent_note(s_t, unit_registry, "verification_note", window=2)
    derived_need = composable_raw and (not has_recent_verification)
    trigger_derived = (s_sem >= TAU_SEM) and derived_need
    return {
        "s_sem": s_sem,
        "composable_raw": composable_raw,
        "has_recent_verification": has_recent_verification,
        "derived_need": derived_need,
        "trigger_derived": trigger_derived,
    }


def extract_probe_answer_from_context(gold_answer: str, k_t: str) -> Optional[str]:
    norm_gold = normalize_text(gold_answer)
    norm_ctx = normalize_text(k_t)
    if not norm_gold:
        return None
    if norm_gold in norm_ctx:
        return gold_answer
    return None


def context_contains_answer(gold_answer: str, k_t: str) -> bool:
    norm_gold = normalize_text(gold_answer)
    norm_ctx = normalize_text(k_t)
    if not norm_gold:
        return False
    return norm_gold in norm_ctx


def check_answer_correct(gold_answer: str, probe_answer: Optional[str]) -> bool:
    if probe_answer is None:
        return False
    return normalize_text(gold_answer) == normalize_text(probe_answer)


def check_support_sufficient(covered_target_ids: List[str], target_unit_id_set: set) -> bool:
    covered_set = set(str(x) for x in covered_target_ids)
    return target_unit_id_set.issubset(covered_set)


def run_stop_probe(q: dict, state: dict, target_info: dict) -> dict:
    probe_answer = answer_with_deepseek(
        api_key=q["api_key"],
        model=q["model"],
        base_url=q["base_url"],
        question=q["question"],
        k_t=state["K_t"],
    )
    gold_answer = str(q["answer"])
    gold_answer_normalized = normalize_text(gold_answer)
    probe_answer_normalized = normalize_text(probe_answer) if probe_answer is not None else None
    context_exact_match = context_contains_answer(gold_answer, state["K_t"])
    answer_correct = check_answer_correct(q["answer"], probe_answer) or context_exact_match
    support_sufficient = check_support_sufficient(
        state["A_t"].get("covered_target_ids", []),
        set(target_info["target_map"].keys()),
    )
    teacher_stop = answer_correct and support_sufficient
    false_stop = not teacher_stop
    answer_match_rule = "normalized_exact_or_context_exact"
    if check_answer_correct(q["answer"], probe_answer):
        answer_match_rule = "normalized_exact"
    elif context_exact_match:
        answer_match_rule = "context_exact"
    return {
        "probe_run": True,
        "answer_source": "llm",
        "gold_answer": gold_answer,
        "gold_answer_normalized": gold_answer_normalized,
        "probe_answer": probe_answer,
        "probe_answer_normalized": probe_answer_normalized,
        "context_contains_gold_answer": context_exact_match,
        "answer_match_rule": answer_match_rule,
        "AnswerCorrect_t": answer_correct,
        "SupportSufficient_t": support_sufficient,
        "TeacherStop_t": teacher_stop,
        "FalseStop_t": false_stop,
    }


def build_state_summary_for_proposal(question: str, s_t: dict, unit_registry: Dict[str, dict]) -> str:
    lines = []
    recent_raw = get_recent_raw_texts(s_t, unit_registry, n=2)
    for text in recent_raw[:2]:
        lines.append("Evidence: " + shorten(text, 90))
    last_note = get_last_derived_text(s_t, unit_registry)
    if last_note:
        lines.append("Note: " + shorten(last_note, 70))
    lines.append("Need: " + extract_goal_from_question(question))
    return "\n".join(lines)


def build_top_raw_candidates(r_t: List[str], unit_registry: Dict[str, dict]) -> List[dict]:
    out = []
    for unit_id in r_t[:TOP_RAW_J]:
        unit = unit_registry.get(unit_id)
        if unit is None:
            continue
        out.append({"unit_id": unit_id, "text": unit["text"], "doc_id": unit["doc_id"]})
    return out


def deepseek_chat_json(api_key: str, model: str, base_url: str, system_prompt: str, user_prompt: str) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 512,
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_SEC)

    raise RuntimeError(f"DeepSeek 请求失败: {last_err}") from last_err


def answer_with_deepseek(
    *,
    api_key: str,
    model: str,
    base_url: str,
    question: str,
    k_t: str,
) -> Optional[str]:
    system_prompt = (
        "You answer a question using only the provided evidence context.\n"
        "Return strict JSON with a single key `answer`.\n"
        "If the answer span is explicitly present in the evidence, copy the shortest exact answer span.\n"
        "Return only a short answer phrase, not an explanation.\n"
        "If the context is insufficient, return an empty string."
    )
    user_prompt = (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Evidence Context:\n"
        f"{k_t.strip()}\n\n"
        "Return JSON like {\"answer\": \"...\"}."
    )
    parsed = deepseek_chat_json(
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    answer = parsed.get("answer")
    if answer is None:
        return None
    answer = str(answer).strip()
    return answer or None


def build_propose_prompt(question: str, state_summary: str, top_raw_candidates: List[dict], gold_answer: Optional[str]) -> str:
    raw_lines = []
    for i, item in enumerate(top_raw_candidates, start=1):
        raw_lines.append(
            f"{i}. unit_id={item['unit_id']}\n"
            f"   doc={item['doc_id']}\n"
            f"   text={item['text']}"
        )

    gold_line = gold_answer if gold_answer else "null"

    return (
        f"Question:\n{question}\n\n"
        f"State summary:\n{state_summary}\n\n"
        f"Top raw candidates:\n" + "\n".join(raw_lines) + "\n\n"
        f"Gold answer (offline only, optional hint):\n{gold_line}\n\n"
        f"Please decide whether to derive short grounded notes now.\n"
        f"Return strict JSON with keys:\n"
        f"- should_derive: boolean\n"
        f"- reason: string\n"
        f"- derived_candidates: list\n\n"
        f"Each derived candidate must contain:\n"
        f"- type: one of bridge_note, verification_note\n"
        f"- text: exactly one sentence\n"
        f"- source_unit_ids: list of 1 to 3 raw unit ids from the provided top raw candidates\n"
        f"- coarse_priority: integer\n\n"
        f"Constraints:\n"
        f"- produce at most {MAX_DERIVED_CANDIDATES} candidates\n"
        f"- do not output final answer style\n"
        f"- do not invent unsupported facts\n"
        f"- do not use claimed_role\n"
        f"- source_unit_ids must come only from the provided top raw candidates\n"
    )


def validate_harvest_candidates(
    qid: str,
    parsed: dict,
    visible_source_ids: set,
    start_idx: int,
) -> Tuple[List[dict], int]:
    candidates = parsed.get("derived_candidates", [])
    if not isinstance(candidates, list):
        return [], start_idx

    validated = []
    seen = set()

    for idx, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            continue

        cand_type = str(cand.get("type", "")).strip()
        text = str(cand.get("text", "")).strip()
        source_unit_ids = cand.get("source_unit_ids", [])
        coarse_priority = int(cand.get("coarse_priority", idx))

        if cand_type not in ALLOWED_DERIVED_TYPES:
            continue
        if not text:
            continue
        if not isinstance(source_unit_ids, list):
            continue

        source_unit_ids = [str(x) for x in source_unit_ids]
        if not (1 <= len(source_unit_ids) <= 3):
            continue
        if len(source_unit_ids) != len(set(source_unit_ids)):
            continue
        if not set(source_unit_ids).issubset(visible_source_ids):
            continue

        unit_id = build_derived_unit_id(qid, start_idx)
        start_idx += 1

        if unit_id in seen:
            continue
        seen.add(unit_id)

        validated.append(
            {
                "unit_id": unit_id,
                "text": text,
                "provenance": "derived",
                "candidate_granularity": "note",
                "type": cand_type,
                "source_unit_ids": source_unit_ids,
                "coarse_priority": coarse_priority,
            }
        )

    validated.sort(key=lambda x: (x["coarse_priority"], x["unit_id"]))
    return validated[:MAX_DERIVED_CANDIDATES], start_idx


def propose_derived(
    *,
    api_key: str,
    model: str,
    base_url: str,
    qid: str,
    question: str,
    state_summary: str,
    top_raw_candidates: List[dict],
    gold_answer: Optional[str],
    next_derived_idx: int,
) -> Tuple[List[dict], int]:
    if not top_raw_candidates:
        return [], next_derived_idx

    system_prompt = (
        "You are proposing grounded derived notes for offline teacher trajectory construction.\n"
        "Only propose short, grounded notes if they help organize currently visible raw evidence.\n"
        "Allowed note types are exactly: bridge_note, verification_note.\n"
        "Do not output claimed_role.\n"
        "Return strict JSON only."
    )
    user_prompt = build_propose_prompt(question, state_summary, top_raw_candidates, gold_answer)

    parsed = deepseek_chat_json(
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    if not bool(parsed.get("should_derive", False)):
        return [], next_derived_idx

    visible_source_ids = {x["unit_id"] for x in top_raw_candidates}
    return validate_harvest_candidates(qid, parsed, visible_source_ids, next_derived_idx)


def legality_filter(
    state: dict,
    r_t: List[str],
    harvest: List[dict],
    unit_registry: Dict[str, dict],
) -> Tuple[List[dict], List[dict]]:
    visible_raw_ids = set(r_t)
    for item in state["S_t"].get("raw_refs", []):
        if isinstance(item, dict) and "unit_id" in item:
            visible_raw_ids.add(str(item["unit_id"]))

    existing_note_texts = []
    for item in state["S_t"].get("derived_refs", []):
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit = unit_registry.get(str(item["unit_id"]))
        if unit is not None and unit.get("provenance") == "derived":
            existing_note_texts.append(unit["text"])

    legal = []
    illegal = []
    accepted_norm_texts = []

    for idx, z in enumerate(harvest):
        reasons = []

        z_type = str(z.get("type", "")).strip()
        z_text = str(z.get("text", "")).strip()
        src_ids = z.get("source_unit_ids", [])
        unit_id = str(z.get("unit_id", "")).strip()

        if z_type not in ALLOWED_DERIVED_TYPES:
            reasons.append("invalid_type")
        if not isinstance(src_ids, list):
            reasons.append("invalid_source_count")
            src_ids = []
        else:
            src_ids = [str(x) for x in src_ids]

        if not (1 <= len(src_ids) <= MAX_SOURCE_PER_NOTE):
            reasons.append("invalid_source_count")
        if len(src_ids) != len(set(src_ids)):
            reasons.append("duplicated_source_ids")
        if not set(src_ids).issubset(visible_raw_ids):
            reasons.append("invisible_source_ids")

        if sentence_count(z_text) != 1:
            reasons.append("not_single_sentence")
        if token_count(z_text) > MAX_NOTE_TOKENS:
            reasons.append("too_long")

        norm_text = normalize_text(z_text)
        if not norm_text:
            reasons.append("empty_text")

        for prev_norm in accepted_norm_texts:
            if norm_text == prev_norm or jaccard_similarity(norm_text, prev_norm) > DUPLICATE_SIM_THRESHOLD:
                reasons.append("duplicate_in_harvest")
                break

        for prev_text in existing_note_texts:
            prev_norm = normalize_text(prev_text)
            if norm_text == prev_norm or jaccard_similarity(norm_text, prev_norm) > DUPLICATE_SIM_THRESHOLD:
                reasons.append("duplicate_with_state")
                break

        if reasons:
            illegal.append(
                {
                    "candidate": {
                        "unit_id": unit_id,
                        "type": z_type,
                        "text": z_text,
                        "source_unit_ids": src_ids,
                    },
                    "reasons": sorted(set(reasons)),
                }
            )
            continue

        accepted_norm_texts.append(norm_text)
        legal.append(
            {
                "unit_id": unit_id,
                "type": z_type,
                "text": z_text,
                "source_unit_ids": src_ids,
                "coarse_priority": int(z.get("coarse_priority", idx)),
                "provenance": "derived",
                "candidate_granularity": "note",
            }
        )

    legal.sort(key=lambda x: (x["coarse_priority"], x["unit_id"]))
    return legal, illegal


def final_retain_selection(g_legal: List[dict]) -> Tuple[List[str], List[str]]:
    if not g_legal:
        return [], []

    retained = []
    remaining = list(g_legal)

    first = remaining.pop(0)
    retained.append(first)

    if remaining and len(retained) < MAX_FINAL_DERIVED:
        preferred_idx = None
        first_type = first["type"]
        for i, item in enumerate(remaining):
            if item["type"] != first_type:
                preferred_idx = i
                break
        if preferred_idx is None:
            second = remaining.pop(0)
        else:
            second = remaining.pop(preferred_idx)
        retained.append(second)

    final_ids = [x["unit_id"] for x in retained]
    aux_ids = [x["unit_id"] for x in remaining]
    return final_ids, aux_ids


def advance_state_with_positive(
    state: dict,
    positive_unit_id: str,
    unit_registry: Dict[str, dict],
    target_info: dict,
    query_info: dict,
) -> dict:
    if positive_unit_id not in unit_registry:
        raise ValueError(f"positive_unit_id 不在 UnitRegistry: qid={state['qid']}, unit_id={positive_unit_id}")

    u_next = unit_registry[positive_unit_id]
    h_next, s_next = simulate_update(state["H_t"], state["S_t"], u_next)
    a_next = simulate_ledger(state["A_t"], u_next, target_info)
    k_next = render_context(query_info, s_next, unit_registry)

    return {
        "qid": state["qid"],
        "t": int(state["t"]) + 1,
        "H_t": h_next,
        "A_t": a_next,
        "S_t": s_next,
        "K_t": k_next,
    }


def build_ranking_records_for_qid(
    qid: str,
    full_traj: dict,
    candidate_map: Dict[Tuple[str, int], dict],
) -> List[dict]:
    steps = full_traj["steps"]
    if not steps:
        return []

    output_records = []

    for step in steps:
        t = int(step["t"])
        positive_unit_id = str(step["positive_unit_id"])
        candidate_rec = candidate_map.get((qid, t))
        if candidate_rec is None:
            print(
                f"[WARN] candidates missing, skip prefix: "
                f"qid={qid}, t={t}, positive_unit_id={positive_unit_id}",
                flush=True,
            )
            continue

        g_aux = [str(x) for x in candidate_rec.get("G_t_aux", [])]
        g_illegal = [str(x) for x in candidate_rec.get("G_t_illegal", [])]
        c_t = [str(x) for x in candidate_rec.get("C_t", [])]

        if positive_unit_id not in c_t:
            print(
                f"[WARN] replay mismatch, skip prefix: "
                f"qid={qid}, t={t}, positive_unit_id={positive_unit_id}",
                flush=True,
            )
            continue

        negative_unit_ids = []
        neg_seen = set()

        for unit_id in c_t:
            if unit_id == positive_unit_id:
                continue
            if unit_id in neg_seen:
                continue
            neg_seen.add(unit_id)
            negative_unit_ids.append(unit_id)

        for unit_id in g_aux + g_illegal:
            if unit_id == positive_unit_id:
                continue
            if unit_id in neg_seen:
                continue
            neg_seen.add(unit_id)
            negative_unit_ids.append(unit_id)

        output_records.append(
            {
                "qid": qid,
                "t": t,
                "positive_unit_id": positive_unit_id,
                "negative_unit_ids": negative_unit_ids,
            }
        )

    return output_records


def convert_split(
    split: str,
    queries_path: Path,
    full_path: Path,
    candidates_path: Path,
    output_path: Path,
    api_key: str,
    base_url: str,
    model: str,
) -> int:
    queries = load_queries(queries_path)
    full_map = load_full_trajectories(full_path)
    candidate_map = load_candidate_records(candidates_path)

    def generator():
        for qid in sorted(full_map.keys()):
            if qid not in queries:
                raise ValueError(f"queries 中找不到 qid: {qid}")

            records = build_ranking_records_for_qid(
                qid=qid,
                full_traj=full_map[qid],
                candidate_map=candidate_map,
            )
            for rec in records:
                yield rec

    return write_jsonl(generator(), output_path)


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"

    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    queries_dir = base_dir / "queries"
    trajectories_dir = base_dir / "trajectories"
    labels_dir = base_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    out_name_map = {
        "train": "ranking_train.jsonl",
        "val": "ranking_val.jsonl",
        "test": "ranking_test.jsonl",
    }
    stats = {}

    for split in SPLITS:
        queries_path = queries_dir / f"{split}.jsonl"
        full_path = trajectories_dir / f"full_{split}.jsonl"
        candidates_path = trajectories_dir / f"candidates_{split}.jsonl"
        output_path = labels_dir / out_name_map[split]

        for path in [queries_path, full_path, candidates_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            split=split,
            queries_path=queries_path,
            full_path=full_path,
            candidates_path=candidates_path,
            output_path=output_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    print("ranking labels v2 构建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {labels_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()
