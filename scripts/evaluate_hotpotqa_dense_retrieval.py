#!/usr/bin/env python3
"""Evaluate dense retrieval and optional trained reranking on HotpotQA candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.ranker import CrossEncoderRanker


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_ks(value: str) -> list[int]:
    return sorted({int(x) for x in value.split(",") if x.strip()})


def load_queries(path: Path) -> dict[str, dict]:
    return {str(row["qid"]): row for row in read_jsonl(path)}


def load_targets(path: Path) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        qid = str(row["qid"])
        doc_ids = {
            str(unit.get("doc_id") or "").strip()
            for unit in row.get("T_q_raw") or []
            if unit.get("doc_id")
        }
        if doc_ids:
            targets[qid] = doc_ids
    return targets


def load_candidates(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        unit_id = str(row.get("unit_id") or "")
        qid = unit_id.split("::", 1)[0]
        text = str(row.get("text") or "").strip()
        doc_id = str(row.get("doc_id") or "").strip()
        if qid and text and doc_id:
            grouped[qid].append(
                {
                    "unit_id": unit_id,
                    "doc_id": doc_id,
                    "text": text,
                }
            )
    return grouped


def update_metrics(metrics: dict, ranked: list[dict], gold_docs: set[str], ks: list[int]) -> None:
    for k in ks:
        top_docs = {row["doc_id"] for row in ranked[:k]}
        hit_count = len(top_docs & gold_docs)
        metrics[f"doc_recall@{k}"] += hit_count / max(len(gold_docs), 1)
        metrics[f"any_support@{k}"] += int(hit_count > 0)
        metrics[f"full_support@{k}"] += int(gold_docs.issubset(top_docs))


class DenseRetriever:
    def __init__(self, model_name_or_path: str, device: str, batch_size: int):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class HotpotRanker:
    def __init__(self, model_dir: Path, checkpoint: Path, device: str, max_length: int, batch_size: int):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = CrossEncoderRanker(pretrained_name=str(model_dir), dropout=0.1).to(self.device)
        checkpoint_obj = torch.load(str(checkpoint), map_location=self.device)
        state_dict = checkpoint_obj.get("model_state_dict", checkpoint_obj)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        context = f"Question: {question}\nNotebook:\n"
        rows = []
        with torch.no_grad():
            for start in range(0, len(candidates), self.batch_size):
                batch = candidates[start : start + self.batch_size]
                tokens = self.tokenizer(
                    [context] * len(batch),
                    [row["text"] for row in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                scores, _ = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    token_type_ids=tokens.get("token_type_ids"),
                )
                for row, score in zip(batch, scores.detach().cpu().tolist()):
                    item = dict(row)
                    item["ranker_score"] = float(score)
                    rows.append(item)
        rows.sort(key=lambda row: row["ranker_score"], reverse=True)
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval on HotpotQA distractor candidates.")
    parser.add_argument("--data-root", default="data/hotpotqa_distractor_v7_10k_llm_prestep")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--dense-model", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--dense-top-k", type=int, default=80)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--ks", default="1,2,5,8,10,20,50,80")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--ranker-batch-size", type=int, default=16)
    parser.add_argument("--ranker-max-length", type=int, default=320)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    queries = load_queries(data_root / "queries" / f"{args.split}.jsonl")
    targets = load_targets(data_root / "targets" / f"{args.split}.jsonl")
    candidates_by_qid = load_candidates(data_root / "unit_registry" / f"raw_units_{args.split}.jsonl")
    ks = parse_ks(args.ks)

    qids = [qid for qid in queries if qid in targets and qid in candidates_by_qid]
    if args.max_items > 0:
        qids = qids[: args.max_items]
    if not qids:
        raise RuntimeError("no evaluable HotpotQA qids found")

    retriever = DenseRetriever(args.dense_model, device=args.device, batch_size=args.dense_batch_size)
    ranker = None
    if args.checkpoint:
        ranker = HotpotRanker(
            model_dir=Path(args.model_dir),
            checkpoint=Path(args.checkpoint),
            device=args.device,
            max_length=args.ranker_max_length,
            batch_size=args.ranker_batch_size,
        )

    dense_metrics = defaultdict(float)
    rerank_metrics = defaultdict(float)
    examples = []

    for qid in tqdm(qids, desc=f"hotpotqa-{args.split}"):
        question = str(queries[qid]["question"])
        gold_docs = targets[qid]
        candidates = candidates_by_qid[qid]
        query_embedding = retriever.encode([question])[0]
        candidate_embeddings = retriever.encode([row["text"] for row in candidates])
        sims = np.dot(candidate_embeddings, query_embedding)
        dense_ranked = []
        for index in np.argsort(sims)[::-1].tolist():
            row = dict(candidates[index])
            row["dense_score"] = float(sims[index])
            dense_ranked.append(row)

        dense_top = dense_ranked[: args.dense_top_k]
        update_metrics(dense_metrics, dense_top, gold_docs, ks)

        reranked = []
        if ranker is not None:
            reranked = ranker.rerank(question, dense_top)[: args.rerank_top_k]
            update_metrics(rerank_metrics, reranked, gold_docs, ks)

        if len(examples) < 10:
            examples.append(
                {
                    "qid": qid,
                    "question": question,
                    "gold_docs": sorted(gold_docs),
                    "dense_top5": [
                        {
                            "doc_id": row["doc_id"],
                            "unit_id": row["unit_id"],
                            "score": round(float(row["dense_score"]), 6),
                            "text": row["text"][:240],
                        }
                        for row in dense_ranked[:5]
                    ],
                    "rerank_top5": [
                        {
                            "doc_id": row["doc_id"],
                            "unit_id": row["unit_id"],
                            "score": round(float(row.get("ranker_score", 0.0)), 6),
                            "text": row["text"][:240],
                        }
                        for row in reranked[:5]
                    ],
                }
            )

    total = len(qids)
    dense_summary = {key: round(value / total, 6) for key, value in sorted(dense_metrics.items())}
    rerank_summary = {key: round(value / total, 6) for key, value in sorted(rerank_metrics.items())}
    report = {
        "data_root": str(data_root),
        "split": args.split,
        "total": total,
        "dense_model": args.dense_model,
        "checkpoint": args.checkpoint,
        "dense_top_k": args.dense_top_k,
        "rerank_top_k": args.rerank_top_k if ranker is not None else 0,
        "dense": dense_summary,
        "reranked": rerank_summary if ranker is not None else {},
        "examples": examples,
    }

    output_path = Path(args.output or f"outputs/retrieval/hotpotqa_{args.split}_dense_retrieval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, ensure_ascii=False, indent=2))
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
