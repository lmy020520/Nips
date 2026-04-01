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

        self.doc_tf: List[Counter] = []
        self.doc_lens: List[int] = []
        self.df = Counter()

        for doc in docs:
            if text_key not in doc:
                raise ValueError(f"文档缺少字段: {text_key}")
            tokens = tokenize(str(doc[text_key]))
            tf = Counter(tokens)
            self.doc_tf.append(tf)
            self.doc_lens.append(len(tokens))
            for term in tf.keys():
                self.df[term] += 1

        self.N = len(docs)
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)

        self.idf = {}
        for term, df in self.df.items():
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
                score += idf * (tf * (self.k1 + 1)) / max(denom, 1e-12)

            scores[i] = score

        return scores


def load_questions(processed_path: Path) -> Dict[str, str]:
    questions = {}
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
        if qid in questions:
            raise ValueError(f"processed 中发现重复 qid: file={processed_path}, qid={qid}")

        questions[qid] = question
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


def load_init_state(init_state_path: Path) -> Dict[str, dict]:
    states = {}
    for row_idx, record in enumerate(read_jsonl(init_state_path), start=1):
        required_fields = ["qid", "H0", "S0", "K0"]
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"init_state 记录缺少字段: file={init_state_path}, row={row_idx}, field={field}"
                )

        qid = str(record["qid"])
        if qid in states:
            raise ValueError(f"init_state 中发现重复 qid: file={init_state_path}, qid={qid}")

        h0 = record["H0"]
        if not isinstance(h0, list):
            raise ValueError(f"H0 必须是 list: qid={qid}")

        normalized_h0 = []
        seen = set()
        for i, unit_id in enumerate(h0):
            unit_id = str(unit_id)
            if unit_id in seen:
                raise ValueError(f"H0 中发现重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen.add(unit_id)
            normalized_h0.append(unit_id)

        states[qid] = {
            "qid": qid,
            "H0": normalized_h0,
            "S0": record["S0"],
            "K0": str(record["K0"]),
        }

    return states


def build_candidate_list(question: str, docs: List[dict], excluded_unit_ids: set, top_k: int) -> List[dict]:
    retriever = BM25Retriever(docs, text_key="retrieval_text")
    scores = retriever.score(question)

    ranked = []
    for doc, score in zip(docs, scores):
        if doc["unit_id"] in excluded_unit_ids:
            continue
        ranked.append(
            {
                "unit_id": doc["unit_id"],
                "score": float(score),
                "title": doc["title"],
                "sent_id": int(doc["sent_id"]),
                "text": doc["text"],
            }
        )

    ranked.sort(key=lambda x: (-x["score"], x["unit_id"]))

    results = []
    for rank, item in enumerate(ranked[:top_k], start=1):
        results.append(
            {
                "unit_id": item["unit_id"],
                "rank": rank,
                "score": round(item["score"], 6),
                "title": item["title"],
                "sent_id": item["sent_id"],
                "text": item["text"],
            }
        )
    return results


def build_rollout_step0(
    processed_path: Path,
    index_path: Path,
    init_state_path: Path,
    rollout_path: Path,
    top_k: int,
) -> Dict[str, int]:
    questions = load_questions(processed_path)
    index_grouped = load_index_grouped(index_path)
    init_states = load_init_state(init_state_path)

    total_questions = 0
    total_candidates = 0

    def record_generator():
        nonlocal total_questions, total_candidates

        qids = list(questions.keys())
        for qid in qids:
            if qid not in index_grouped:
                raise ValueError(f"index 中找不到 qid: {qid}")
            if qid not in init_states:
                raise ValueError(f"init_state 中找不到 qid: {qid}")

            question = questions[qid]
            docs = index_grouped[qid]
            h0 = init_states[qid]["H0"]

            h0_set = set(h0)
            doc_unit_ids = {doc["unit_id"] for doc in docs}

            missing = sorted(list(h0_set - doc_unit_ids))
            if missing:
                raise ValueError(f"H0 中存在不在 index 里的 unit_id: qid={qid}, missing={missing}")

            r0 = build_candidate_list(
                question=question,
                docs=docs,
                excluded_unit_ids=h0_set,
                top_k=top_k,
            )

            total_questions += 1
            total_candidates += len(r0)

            yield {
                "qid": qid,
                "t": 0,
                "query": question,
                "H_t": h0,
                "R_t": r0,
            }

    written = write_jsonl(record_generator(), rollout_path)
    if written != total_questions:
        raise RuntimeError(
            f"写入条数异常: file={rollout_path}, written={written}, expected={total_questions}"
        )

    return {
        "questions": total_questions,
        "candidates": total_candidates,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent

    base_dir = project_root / "data" / "hotpotqa_distractor"
    processed_dir = base_dir / "processed"
    index_dir = base_dir / "index"
    init_state_dir = base_dir / "init_state"
    rollout_dir = base_dir / "rollout"

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
    init_state_files = {
        "train": init_state_dir / "train.jsonl",
        "val": init_state_dir / "val.jsonl",
        "test": init_state_dir / "test.jsonl",
    }
    rollout_files = {
        "train": rollout_dir / "step0_train.jsonl",
        "val": rollout_dir / "step0_val.jsonl",
        "test": rollout_dir / "step0_test.jsonl",
    }

    for split in ["train", "val", "test"]:
        if not processed_files[split].exists():
            raise FileNotFoundError(f"找不到 processed 文件: {processed_files[split]}")
        if not index_files[split].exists():
            raise FileNotFoundError(f"找不到 index 文件: {index_files[split]}")
        if not init_state_files[split].exists():
            raise FileNotFoundError(f"找不到 init_state 文件: {init_state_files[split]}")

    rollout_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = build_rollout_step0(
            processed_path=processed_files[split],
            index_path=index_files[split],
            init_state_path=init_state_files[split],
            rollout_path=rollout_files[split],
            top_k=TOP_K,
        )
        all_stats[split] = stats

    print("HotpotQA rollout step0 构建完成：")
    for split in ["train", "val", "test"]:
        stats = all_stats[split]
        print(
            f"[{split}] "
            f"questions={stats['questions']}, "
            f"candidates={stats['candidates']}, "
            f"output={rollout_files[split]}"
        )


if __name__ == "__main__":
    main()