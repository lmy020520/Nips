#!/usr/bin/env python3
"""Standalone RAG pipeline with the trained Hotpot trajectory reranker."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.ranker import CrossEncoderRanker


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def doc_text(item) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("content", "text", "body", "paragraph", "chunk"):
            value = item.get(key)
            if value:
                return str(value).strip()
    return ""


def load_docs(path: Path) -> list[str]:
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get("documents") or data.get("docs") or data.get("data") or []
    docs = []
    for item in data:
        text = doc_text(item)
        if text:
            docs.append(text)
    if not docs:
        raise RuntimeError(f"no documents loaded from {path}")
    return docs


def load_questions(path: Path, max_items: int = 0) -> list[dict]:
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or data.get("items") or []
    rows = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = item.get("question") or item.get("query")
        if not question:
            continue
        answer = item.get("ground_truth") or item.get("answer") or item.get("answers") or ""
        if isinstance(answer, list):
            answer = "; ".join(str(x) for x in answer)
        rows.append(
            {
                "id": item.get("id") or item.get("qid") or idx,
                "question": str(question),
                "ground_truth": str(answer),
                "question_type": str(item.get("question_type") or item.get("type") or "unknown"),
            }
        )
    if max_items > 0:
        rows = rows[:max_items]
    if not rows:
        raise RuntimeError(f"no questions loaded from {path}")
    return rows


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[\w]+|[\u4e00-\u9fff]", text.lower())
    return [t for t in tokens if t.strip()]


class HybridRetriever:
    def __init__(self, docs: list[str], dense_model: str = "", device: str = "cpu"):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(doc) for doc in docs])
        self.dense_model = None
        self.doc_embeddings = None
        if dense_model:
            from sentence_transformers import SentenceTransformer

            self.dense_model = SentenceTransformer(dense_model, device=device)
            self.doc_embeddings = self.dense_model.encode(
                docs,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

    def retrieve(self, query: str, top_k: int = 30) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        doc_scores = Counter()
        for rank, idx in enumerate(np.argsort(scores)[::-1][: min(top_k * 4, len(self.docs))]):
            doc_scores[int(idx)] += 1.0 / (60 + rank + 1)

        if self.dense_model is not None and self.doc_embeddings is not None:
            q = self.dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            sims = np.dot(self.doc_embeddings, q)
            for rank, idx in enumerate(np.argsort(sims)[::-1][: min(top_k * 4, len(self.docs))]):
                doc_scores[int(idx)] += 1.0 / (60 + rank + 1)

        ranked = [idx for idx, _ in doc_scores.most_common(top_k)]
        return [self.docs[idx] for idx in ranked]


class HotpotReranker:
    def __init__(
        self,
        model_dir: Path,
        checkpoint: Path,
        device: str,
        max_length: int = 320,
        batch_size: int = 16,
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = CrossEncoderRanker(pretrained_name=str(model_dir), dropout=0.1).to(self.device)
        ckpt = torch.load(str(checkpoint), map_location=self.device)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        self.model.eval()

    def rerank(self, query: str, candidates: list[str], top_k: int = 5) -> list[dict]:
        context = f"Question: {query}\nNotebook:\n"
        rows = []
        with torch.no_grad():
            for start in range(0, len(candidates), self.batch_size):
                docs = candidates[start : start + self.batch_size]
                tokens = self.tokenizer(
                    [context] * len(docs),
                    docs,
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
                for doc, score in zip(docs, scores.detach().cpu().tolist()):
                    rows.append({"text": doc, "score": float(score)})
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[:top_k]


def deepseek_chat(api_key: str, base_url: str, model: str, messages: list[dict], temperature: float = 0.0) -> tuple[str, int]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"], int(data.get("usage", {}).get("total_tokens") or 0)


def answer_question(api_key: str, base_url: str, model: str, question: str, contexts: list[str]) -> tuple[str, int]:
    context_text = "\n".join(f"资料片段{i+1}: {text}" for i, text in enumerate(contexts))
    messages = [
        {"role": "system", "content": "你是一个严谨的RAG问答助手。请只根据参考资料回答；资料不足就回答不知道。"},
        {"role": "user", "content": f"【参考资料】\n{context_text}\n\n【问题】\n{question}"},
    ]
    return deepseek_chat(api_key, base_url, model, messages, temperature=0.1)


def judge_answer(
    api_key: str,
    base_url: str,
    model: str,
    question: str,
    gold: str,
    prediction: str,
    contexts: list[str],
) -> dict:
    prompt = f"""
你是一名严格的RAG评测专家。请只输出JSON。

【问题】{question}
【标准答案】{gold}
【模型回答】{prediction}
【检索上下文】
{chr(10).join(contexts)[:3000]}

请输出：
{{"correctness":0或1,"context_recall":0或1,"faithfulness":0或1}}
"""
    content, _ = deepseek_chat(
        api_key,
        base_url,
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"correctness": 0, "context_recall": 0, "faithfulness": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", default="outputs/rag/hotpot_rag_report.json")
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--checkpoint", default="outputs/ranker/frozen_deberta_v3_large_v7_main_val08252/best_model.pt")
    parser.add_argument("--dense-model", default="")
    parser.add_argument("--retrieve-top-k", type=int, default=30)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    llm_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    docs = load_docs(Path(args.docs))
    questions = load_questions(Path(args.benchmark), max_items=args.max_items)
    retriever = HybridRetriever(docs, dense_model=args.dense_model, device=args.device)
    reranker = HotpotReranker(
        model_dir=Path(args.model_dir),
        checkpoint=Path(args.checkpoint),
        device=args.device,
    )

    results = []
    totals = Counter()
    for item in tqdm(questions, desc="rag"):
        question = item["question"]
        candidates = retriever.retrieve(question, top_k=args.retrieve_top_k)
        ranked = reranker.rerank(question, candidates, top_k=args.rerank_top_k)
        contexts = [row["text"] for row in ranked]
        started = time.time()
        prediction, token_cost = answer_question(api_key, base_url, llm_model, question, contexts)
        latency = time.time() - started
        scores = {"correctness": None, "context_recall": None, "faithfulness": None}
        if not args.skip_judge:
            scores = judge_answer(api_key, base_url, llm_model, question, item["ground_truth"], prediction, contexts)
            for key in ("correctness", "context_recall", "faithfulness"):
                totals[key] += int(scores.get(key) or 0)
            totals["judged"] += 1
        totals["total"] += 1
        totals["tokens"] += token_cost
        totals["latency"] += latency
        results.append(
            {
                **item,
                "prediction": prediction,
                "scores": scores,
                "contexts": contexts,
                "rerank_scores": [row["score"] for row in ranked],
                "token_cost": token_cost,
                "latency": round(latency, 3),
            }
        )

    judged = max(1, totals["judged"])
    total = max(1, totals["total"])
    summary = {
        "total": totals["total"],
        "judged": totals["judged"],
        "accuracy": round(totals["correctness"] / judged, 6) if totals["judged"] else None,
        "context_recall": round(totals["context_recall"] / judged, 6) if totals["judged"] else None,
        "faithfulness": round(totals["faithfulness"] / judged, 6) if totals["judged"] else None,
        "avg_tokens": round(totals["tokens"] / total, 2),
        "avg_generation_latency": round(totals["latency"] / total, 3),
        "docs": args.docs,
        "benchmark": args.benchmark,
        "checkpoint": args.checkpoint,
        "dense_model": args.dense_model,
        "retrieve_top_k": args.retrieve_top_k,
        "rerank_top_k": args.rerank_top_k,
    }
    report = {"summary": summary, "results": results}
    save_json(report, Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
