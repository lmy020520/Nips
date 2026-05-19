import os
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


SPLITS = ["train", "val", "test"]

DEFAULT_BASE = os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v2")
K_SEED = 8
K0 = 2


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


def parse_qid_from_atom_id(atom_id: str) -> str:
    parts = atom_id.split("::", 2)
    if len(parts) != 3:
        raise ValueError(f"atom_id 格式错误: {atom_id}")
    return parts[0]


def load_queries(path: Path) -> Dict[str, str]:
    queries = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        for field in ["qid", "question"]:
            if field not in record:
                raise ValueError(f"queries 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        question = str(record["question"]).strip()

        if not question:
            raise ValueError(f"question 为空: qid={qid}")
        if qid in queries:
            raise ValueError(f"queries 中重复 qid: file={path}, qid={qid}")

        queries[qid] = question

    return queries


def load_chunks_grouped(path: Path) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    seen_chunk_ids = set()

    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required_fields = ["chunk_id", "doc_id", "chunk_text", "summary_text"]
        for field in required_fields:
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
        required_fields = ["atom_id", "parent_chunk_id", "doc_id", "atom_text", "span_start", "span_end"]
        for field in required_fields:
            if field not in record:
                raise ValueError(f"atoms 缺少字段: file={path}, row={row_idx}, field={field}")

        atom_id = str(record["atom_id"])
        if atom_id in seen_atom_ids:
            raise ValueError(f"atoms 中重复 atom_id: file={path}, atom_id={atom_id}")
        seen_atom_ids.add(atom_id)

        atom_text = str(record["atom_text"]).strip()
        if not atom_text:
            raise ValueError(f"atom_text 为空: file={path}, atom_id={atom_id}")

        grouped[str(record["parent_chunk_id"])].append(
            {
                "atom_id": atom_id,
                "parent_chunk_id": str(record["parent_chunk_id"]),
                "doc_id": str(record["doc_id"]),
                "atom_text": atom_text,
                "span_start": record["span_start"],
                "span_end": record["span_end"],
            }
        )

    return dict(grouped)


def select_seed_units_for_q(
    question: str,
    chunks: List[dict],
    atoms_by_chunk: Dict[str, List[dict]],
    k_seed: int,
    k0: int,
) -> List[dict]:
    if not chunks:
        return []

    # coarse retrieval on summary + chunk text
    summary_index = BM25Index([c["summary_text"] for c in chunks])
    chunk_index = BM25Index([c["chunk_text"] for c in chunks])

    summary_scores = summary_index.score(question)
    chunk_scores = chunk_index.score(question)

    scored_chunks = []
    for chunk, s_sum, s_chunk in zip(chunks, summary_scores, chunk_scores):
        coarse_score = 0.5 * s_sum + 0.5 * s_chunk
        scored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "coarse_score": float(coarse_score),
            }
        )

    scored_chunks.sort(key=lambda x: (-x["coarse_score"], x["chunk_id"]))
    shortlist = scored_chunks[:k_seed]

    # expose top-1 atom for each shortlisted chunk
    candidates = []
    for item in shortlist:
        chunk_id = item["chunk_id"]
        atoms = atoms_by_chunk.get(chunk_id, [])
        if not atoms:
            continue

        atom_index = BM25Index([a["atom_text"] for a in atoms])
        atom_scores = atom_index.score(question)

        scored_atoms = []
        for atom, atom_score in zip(atoms, atom_scores):
            # merge score: keep atom relevance primary, use coarse score as tie-strengthening
            merged_score = float(atom_score) + float(item["coarse_score"])
            scored_atoms.append(
                {
                    "unit_id": atom["atom_id"],
                    "parent_chunk_id": chunk_id,
                    "merged_score": merged_score,
                }
            )

        scored_atoms.sort(key=lambda x: (-x["merged_score"], x["unit_id"]))
        candidates.append(scored_atoms[0])

    # light diversity: since each shortlisted chunk only contributes top-1 atom,
    # parent_chunk is already diversified; just pick top-k0
    candidates.sort(key=lambda x: (-x["merged_score"], x["unit_id"]))
    selected = candidates[:k0]

    p0 = []
    for rank, item in enumerate(selected, start=1):
        p0.append(
            {
                "unit_id": item["unit_id"],
                "rank": rank,
                "source": "seed_retrieve",
            }
        )
    return p0


def build_seed_split(
    queries_path: Path,
    chunks_path: Path,
    atoms_path: Path,
    output_path: Path,
    k_seed: int,
    k0: int,
) -> int:
    queries = load_queries(queries_path)
    chunks_grouped = load_chunks_grouped(chunks_path)
    atoms_by_chunk = load_atoms_grouped_by_chunk(atoms_path)

    def generator():
        for qid in sorted(queries.keys()):
            if qid not in chunks_grouped:
                raise ValueError(f"该 qid 没有 chunks: qid={qid}")

            p0 = select_seed_units_for_q(
                question=queries[qid],
                chunks=chunks_grouped[qid],
                atoms_by_chunk=atoms_by_chunk,
                k_seed=k_seed,
                k0=k0,
            )

            if not p0:
                raise ValueError(f"无法为该问题生成非空 P_0: qid={qid}")

            yield {
                "qid": qid,
                "P_0": p0,
            }

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    queries_dir = base_dir / "queries"
    index_store_dir = base_dir / "index_store"
    seed_dir = base_dir / "seed"
    seed_dir.mkdir(parents=True, exist_ok=True)

    stats = {}

    for split in SPLITS:
        queries_path = queries_dir / f"{split}.jsonl"
        chunks_path = index_store_dir / f"chunks_{split}.jsonl"
        atoms_path = index_store_dir / f"atoms_{split}.jsonl"
        output_path = seed_dir / f"{split}.jsonl"

        for path in [queries_path, chunks_path, atoms_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        count = build_seed_split(
            queries_path=queries_path,
            chunks_path=chunks_path,
            atoms_path=atoms_path,
            output_path=output_path,
            k_seed=K_SEED,
            k0=K0,
        )
        stats[split] = count

    print("SeedRetrieve v2 构建完成：")
    print(f"  defaults: k_0={K0}, K_seed={K_SEED}")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {seed_dir / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()