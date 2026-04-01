import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Iterable, List


TOP_K = 10


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


class BM25Retriever:
    def __init__(self, docs: List[dict], text_key: str = "retrieval_text", k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("BM25 docs 不能为空")

        self.docs = docs
        self.text_key = text_key
        self.k1 = k1
        self.b = b

        self.doc_tokens: List[List[str]] = []
        self.doc_tf: List[Counter] = []
        self.doc_lens: List[int] = []
        self.df = Counter()

        for doc in docs:
            if text_key not in doc:
                raise ValueError(f"文档缺少字段: {text_key}")
            tokens = tokenize(str(doc[text_key]))
            self.doc_tokens.append(tokens)
            tf = Counter(tokens)
            self.doc_tf.append(tf)
            self.doc_lens.append(len(tokens))

            for term in tf.keys():
                self.df[term] += 1

        self.N = len(docs)
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)

        self.idf = {}
        for term, df in self.df.items():
            # BM25 常用平滑 idf
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> List[float]:
        query_terms = tokenize(query)
        if not query_terms:
            return [0.0] * self.N

        query_tf = Counter(query_terms)
        scores = [0.0] * self.N

        for i in range(self.N):
            tf_doc = self.doc_tf[i]
            dl = self.doc_lens[i]
            score = 0.0

            for term in query_tf.keys():
                if term not in tf_doc:
                    continue

                tf = tf_doc[term]
                idf = self.idf.get(term, 0.0)

                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-12))
                term_score = idf * (tf * (self.k1 + 1)) / max(denom, 1e-12)
                score += term_score

            scores[i] = score

        return scores


def load_questions(processed_path: Path) -> List[dict]:
    questions = []
    seen_qids = set()

    for row_idx, record in enumerate(read_jsonl(processed_path), start=1):
        for field in ["qid", "question"]:
            if field not in record:
                raise ValueError(
                    f"processed 记录缺少字段: file={processed_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        question = str(record["question"]).strip()

        if not question:
            raise ValueError(f"空 question: file={processed_path}, row={row_idx}, qid={qid}")
        if qid in seen_qids:
            raise ValueError(f"processed 中发现重复 qid: file={processed_path}, qid={qid}")

        seen_qids.add(qid)
        questions.append({"qid": qid, "question": question})

    return questions


def load_index_grouped(index_path: Path) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    seen_unit_ids = set()

    for row_idx, record in enumerate(read_jsonl(index_path), start=1):
        required_fields = ["qid", "unit_id", "retrieval_text", "title", "sent_id", "text"]
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"index 记录缺少字段: file={index_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        unit_id = str(record["unit_id"])

        if unit_id in seen_unit_ids:
            raise ValueError(f"index 中发现重复 unit_id: file={index_path}, unit_id={unit_id}")
        seen_unit_ids.add(unit_id)

        grouped[qid].append(
            {
                "qid": qid,
                "unit_id": unit_id,
                "retrieval_text": str(record["retrieval_text"]),
                "title": str(record["title"]),
                "sent_id": int(record["sent_id"]),
                "text": str(record["text"]).strip(),
            }
        )

    return dict(grouped)


def build_seed_units(question: str, docs: List[dict], top_k: int) -> List[dict]:
    retriever = BM25Retriever(docs, text_key="retrieval_text")
    scores = retriever.score(question)

    ranked = []
    for doc, score in zip(docs, scores):
        ranked.append(
            {
                "unit_id": doc["unit_id"],
                "score": float(score),
                "title": doc["title"],
                "sent_id": int(doc["sent_id"]),
                "text": doc["text"],
            }
        )

    # 先按 score 降序，再按 unit_id 升序，保证结果稳定
    ranked.sort(key=lambda x: (-x["score"], x["unit_id"]))

    top_units = []
    for rank, item in enumerate(ranked[:top_k], start=1):
        top_units.append(
            {
                "unit_id": item["unit_id"],
                "rank": rank,
                "score": round(item["score"], 6),
                "title": item["title"],
                "sent_id": item["sent_id"],
                "text": item["text"],
            }
        )

    return top_units


def build_retrieval_split(processed_path: Path, index_path: Path, retrieval_path: Path, top_k: int) -> Dict[str, int]:
    questions = load_questions(processed_path)
    index_grouped = load_index_grouped(index_path)

    total_questions = 0
    total_seed_units = 0

    def record_generator():
        nonlocal total_questions, total_seed_units

        for q in questions:
            qid = q["qid"]
            question = q["question"]

            if qid not in index_grouped:
                raise ValueError(f"index 中找不到该 qid 的候选文档: qid={qid}")

            docs = index_grouped[qid]
            if not docs:
                raise ValueError(f"该 qid 的 index 候选为空: qid={qid}")

            seed_units = build_seed_units(question, docs, top_k=top_k)

            total_questions += 1
            total_seed_units += len(seed_units)

            yield {
                "qid": qid,
                "question": question,
                "seed_units": seed_units,
            }

    written = write_jsonl(record_generator(), retrieval_path)
    if written != total_questions:
        raise RuntimeError(
            f"写入条数异常: file={retrieval_path}, written={written}, expected={total_questions}"
        )

    return {
        "questions": total_questions,
        "seed_units": total_seed_units,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    processed_dir = base_dir / "processed"
    index_dir = base_dir / "index"
    retrieval_dir = base_dir / "retrieval"

    processed_files = {
        "train": processed_dir / "train.jsonl",
        "val": processed_dir / "val.jsonl",
        "test": processed_dir / "test.jsonl",
    }
    index_files = {
        "train": index_dir / "train.jsonl",
        "val": index_dir / "val.jsonl",
        "test": index_dir / "test.jsonl",
    }
    retrieval_files = {
        "train": retrieval_dir / "train.jsonl",
        "val": retrieval_dir / "val.jsonl",
        "test": retrieval_dir / "test.jsonl",
    }

    for split in ["train", "val", "test"]:
        if not processed_files[split].exists():
            raise FileNotFoundError(f"找不到 processed 文件: {processed_files[split]}")
        if not index_files[split].exists():
            raise FileNotFoundError(f"找不到 index 文件: {index_files[split]}")

    retrieval_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = build_retrieval_split(
            processed_path=processed_files[split],
            index_path=index_files[split],
            retrieval_path=retrieval_files[split],
            top_k=TOP_K,
        )
        all_stats[split] = stats

    print("HotpotQA seed retrieval 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"questions={stats['questions']}, "
            f"seed_units={stats['seed_units']}, "
            f"output={retrieval_files[split]}"
        )


if __name__ == "__main__":
    main()