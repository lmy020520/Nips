import os
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v4")

CHUNK_SHORTLIST_K = 8
FINAL_KR = 10
MAX_SUMMARY_CHARS = 160


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


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


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
            tokens = tokenize(text)
            tf = Counter(tokens)
            self.doc_tfs.append(tf)
            self.doc_lens.append(len(tokens))
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


def parse_qid_from_chunk_id(chunk_id: str) -> str:
    parts = chunk_id.split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"chunk_id 格式错误: {chunk_id}")
    return parts[0]


def load_queries(path: Path) -> Dict[str, dict]:
    queries = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        for field in ["qid", "question"]:
            if field not in record:
                raise ValueError(f"queries 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in queries:
            raise ValueError(f"queries 中重复 qid: file={path}, qid={qid}")

        question = str(record["question"]).strip()
        if not question:
            raise ValueError(f"question 为空: qid={qid}")

        queries[qid] = {
            "qid": qid,
            "question": question,
        }

    return queries


def load_raw_units(path: Path) -> Dict[str, dict]:
    unit_map = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["unit_id", "text", "doc_id", "parent_chunk_id"]
        for field in required:
            if field not in record:
                raise ValueError(f"raw_units 缺少字段: file={path}, row={row_idx}, field={field}")

        unit_id = str(record["unit_id"])
        if unit_id in unit_map:
            raise ValueError(f"raw_units 中重复 unit_id: file={path}, unit_id={unit_id}")

        text = str(record["text"]).strip()
        if not text:
            raise ValueError(f"raw unit text 为空: unit_id={unit_id}")

        unit_map[unit_id] = {
            "unit_id": unit_id,
            "text": text,
            "doc_id": str(record["doc_id"]),
            "parent_chunk_id": str(record["parent_chunk_id"]),
        }

    return unit_map


def load_chunks_grouped(path: Path) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    seen_chunk_ids = set()

    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["chunk_id", "doc_id", "chunk_text", "summary_text"]
        for field in required:
            if field not in record:
                raise ValueError(f"chunks 缺少字段: file={path}, row={row_idx}, field={field}")

        chunk_id = str(record["chunk_id"])
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"chunks 中重复 chunk_id: file={path}, chunk_id={chunk_id}")
        seen_chunk_ids.add(chunk_id)

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


def load_atoms_grouped_by_chunk(path: Path) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    seen_atom_ids = set()

    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["atom_id", "parent_chunk_id", "doc_id", "atom_text"]
        for field in required:
            if field not in record:
                raise ValueError(f"atoms 缺少字段: file={path}, row={row_idx}, field={field}")

        atom_id = str(record["atom_id"])
        if atom_id in seen_atom_ids:
            raise ValueError(f"atoms 中重复 atom_id: file={path}, atom_id={atom_id}")
        seen_atom_ids.add(atom_id)

        atom_text = str(record["atom_text"]).strip()
        if not atom_text:
            raise ValueError(f"atom_text 为空: atom_id={atom_id}")

        grouped[str(record["parent_chunk_id"])].append(
            {
                "atom_id": atom_id,
                "parent_chunk_id": str(record["parent_chunk_id"]),
                "doc_id": str(record["doc_id"]),
                "atom_text": atom_text,
            }
        )

    return dict(grouped)


def load_states(path: Path) -> Dict[Tuple[str, int], dict]:
    states = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "H_t", "S_t", "K_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"states 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        t = int(record["t"])
        key = (qid, t)
        if key in states:
            raise ValueError(f"states 中重复 prefix key: file={path}, key={key}")

        h_t = record["H_t"]
        if not isinstance(h_t, list):
            raise ValueError(f"H_t 必须是 list: qid={qid}")

        normalized_h = []
        seen = set()
        for i, item in enumerate(h_t):
            if not isinstance(item, dict):
                raise ValueError(f"H_t[{i}] 必须是 dict: qid={qid}")
            if "step_id" not in item or "unit_id" not in item:
                raise ValueError(f"H_t[{i}] 缺少 step_id 或 unit_id: qid={qid}")

            unit_id = str(item["unit_id"])
            if unit_id in seen:
                raise ValueError(f"H_t 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen.add(unit_id)

            normalized_h.append(
                {
                    "step_id": int(item["step_id"]),
                    "unit_id": unit_id,
                }
            )

        s_t = record["S_t"]
        if not isinstance(s_t, dict):
            raise ValueError(f"S_t 必须是 dict: qid={qid}")

        states[key] = {
            "qid": qid,
            "t": t,
            "H_t": normalized_h,
            "S_t": s_t,
            "K_t": str(record["K_t"]),
        }

    return states


def load_full_trajectories(path: Path) -> Dict[str, dict]:
    full_map = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "steps" not in record:
            raise ValueError(f"full trajectories 缺少字段: file={path}, row={row_idx}")
        qid = str(record["qid"])
        if qid in full_map:
            raise ValueError(f"full trajectories 中重复 qid: file={path}, qid={qid}")
        full_map[qid] = record
    return full_map


def load_ranking_records(path: Path) -> Dict[Tuple[str, int], dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "positive_unit_id", "negative_unit_ids"]
        for field in required:
            if field not in record:
                raise ValueError(f"ranking 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        t = int(record["t"])
        key = (qid, t)
        if key in out:
            raise ValueError(f"ranking 中重复 prefix key: file={path}, key={key}")

        negative_unit_ids = record["negative_unit_ids"]
        if not isinstance(negative_unit_ids, list):
            raise ValueError(f"negative_unit_ids 必须是 list: qid={qid}, t={t}")

        out[key] = {
            "positive_unit_id": str(record["positive_unit_id"]),
            "negative_unit_ids": [str(x) for x in negative_unit_ids],
        }
    return out


def get_last_raw_text(s_t: dict, raw_unit_map: Dict[str, dict]) -> Optional[str]:
    raw_refs = s_t.get("raw_refs", [])
    if not isinstance(raw_refs, list) or len(raw_refs) == 0:
        return None

    last_ref = raw_refs[-1]
    if not isinstance(last_ref, dict) or "unit_id" not in last_ref:
        return None

    unit_id = str(last_ref["unit_id"])
    unit = raw_unit_map.get(unit_id)
    if unit is None:
        return None
    return unit["text"]


def get_last_derived_text(s_t: dict) -> Optional[str]:
    # 当前初始化状态 derived_refs 为空；这里保留接口，后续可扩展
    derived_refs = s_t.get("derived_refs", [])
    if not isinstance(derived_refs, list) or len(derived_refs) == 0:
        return None
    return None


def build_slot_summary(question: str, s_t: dict, raw_unit_map: Dict[str, dict], max_chars: int = MAX_SUMMARY_CHARS) -> str:
    parts = []

    last_raw = get_last_raw_text(s_t, raw_unit_map)
    if last_raw is not None:
        parts.append("Evidence: " + shorten(last_raw, 80))

    last_note = get_last_derived_text(s_t)
    if last_note is not None:
        parts.append("Note: " + shorten(last_note, 50))

    goal = extract_goal_from_question(question)
    if not goal:
        goal = "more evidence for the question"
    parts.append("Need: " + goal)

    text = "\n".join(parts)
    return text[:max_chars].rstrip()


def build_q_t(question: str, s_t: dict, raw_unit_map: Dict[str, dict]) -> str:
    h_t = build_slot_summary(question, s_t, raw_unit_map)
    if not h_t:
        return question
    return question + "\n" + h_t


def topk_retrieve(
    q_t: str,
    chunks: List[dict],
    atoms_by_chunk: Dict[str, List[dict]],
    used_unit_ids: set,
    shortlist_k: int,
    final_kr: int,
) -> List[str]:
    if not chunks:
        return []

    summary_index = BM25Index([c["summary_text"] for c in chunks])
    chunk_index = BM25Index([c["chunk_text"] for c in chunks])

    summary_scores = summary_index.score(q_t)
    chunk_scores = chunk_index.score(q_t)

    scored_chunks = []
    for chunk, s_sum, s_chunk in zip(chunks, summary_scores, chunk_scores):
        coarse_score = 0.5 * s_sum + 0.5 * s_chunk
        scored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "score": float(coarse_score),
            }
        )

    scored_chunks.sort(key=lambda x: (-x["score"], x["chunk_id"]))
    shortlist = scored_chunks[:shortlist_k]

    merged = []

    # 只暴露 atom / sentence 级候选，不放 chunk_id
    for item in shortlist:
        chunk_id = item["chunk_id"]
        atoms = atoms_by_chunk.get(chunk_id, [])
        if not atoms:
            continue

        atom_index = BM25Index([a["atom_text"] for a in atoms])
        atom_scores = atom_index.score(q_t)

        scored_atoms = []
        for atom, score in zip(atoms, atom_scores):
            if atom["atom_id"] in used_unit_ids:
                continue
            scored_atoms.append(
                {
                    "unit_id": atom["atom_id"],
                    "score": float(score),
                }
            )

        if not scored_atoms:
            continue

        scored_atoms.sort(key=lambda x: (-x["score"], x["unit_id"]))
        merged.append(scored_atoms[0])

    ranked = merged
    ranked.sort(key=lambda x: (-x["score"], x["unit_id"]))

    return [item["unit_id"] for item in ranked[:final_kr]]


def is_derived_unit_id(unit_id: str) -> bool:
    return "::derived::" in str(unit_id)


def merge_unique_in_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_candidates_split(
    queries_path: Path,
    states_path: Path,
    raw_units_path: Path,
    full_path: Path,
    output_path: Path,
) -> int:
    queries = load_queries(queries_path)
    states = load_states(states_path)
    raw_unit_map = load_raw_units(raw_units_path)
    full_map = load_full_trajectories(full_path)
    step_map = {
        qid: {int(step["t"]): step for step in record.get("steps", [])}
        for qid, record in full_map.items()
    }

    def generator():
        for key in sorted(states.keys(), key=lambda x: (x[0], x[1])):
            qid, t = key
            if qid not in queries:
                raise ValueError(f"queries 中找不到 qid: {qid}")
            if qid not in step_map:
                raise ValueError(f"full trajectories 中找不到 qid: {qid}")

            question = queries[qid]["question"]
            state = states[key]
            step = step_map[qid].get(t)
            if step is None:
                continue

            q_t = build_q_t(question, state["S_t"], raw_unit_map)
            candidate_debug = step.get("candidate_debug", {})

            r_t = merge_unique_in_order([str(x) for x in candidate_debug.get("R_t", step.get("R_t", []))])
            g_t_final = merge_unique_in_order([str(x) for x in candidate_debug.get("G_t_final", step.get("G_t_final", []))])
            g_t_aux = merge_unique_in_order([str(x) for x in candidate_debug.get("G_t_aux", [])])
            g_t_illegal = merge_unique_in_order([str(x) for x in candidate_debug.get("G_t_illegal", [])])
            c_t = merge_unique_in_order([str(x) for x in candidate_debug.get("C_t", step.get("C_t", r_t + g_t_final))])

            yield {
                "qid": qid,
                "t": t,
                "q_t": q_t,
                "R_t": r_t,
                "G_t_final": g_t_final,
                "G_t_aux": g_t_aux,
                "G_t_illegal": g_t_illegal,
                "C_t": c_t,
            }

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    trajectories_dir = base_dir / "trajectories"
    queries_dir = base_dir / "queries"
    unit_registry_dir = base_dir / "unit_registry"

    raw_units_name_map = {
        "train": "raw_units_train.jsonl",
        "val": "raw_units_val.jsonl",
        "test": "raw_units_test.jsonl",
    }
    output_name_map = {
        "train": "candidates_train.jsonl",
        "val": "candidates_val.jsonl",
        "test": "candidates_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        queries_path = queries_dir / f"{split}.jsonl"
        states_path = trajectories_dir / f"states_{split}.jsonl"
        raw_units_path = unit_registry_dir / raw_units_name_map[split]
        full_path = trajectories_dir / f"full_{split}.jsonl"
        output_path = trajectories_dir / output_name_map[split]

        for path in [queries_path, states_path, raw_units_path, full_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = build_candidates_split(
            queries_path=queries_path,
            states_path=states_path,
            raw_units_path=raw_units_path,
            full_path=full_path,
            output_path=output_path,
        )

    print("candidates v3 构建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {trajectories_dir / output_name_map[split]}")


if __name__ == "__main__":
    main()
