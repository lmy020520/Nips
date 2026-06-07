#!/usr/bin/env python3
"""Evaluate dense retrieval and trained reranking on HotpotQA sample candidate sets."""

from __future__ import annotations

import argparse
import json
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


def load_memory(path: Path) -> dict[str, dict]:
    memory = {}
    for row in read_jsonl(path):
        unit_id = str(row.get("unit_id") or "")
        if unit_id:
            title = str(row.get("title") or row.get("doc_id") or "")
            if not title:
                parts = unit_id.split("::")
                title = parts[-2] if len(parts) >= 2 else unit_id
            sent_id = row.get("sent_id")
            if sent_id is None:
                try:
                    sent_id = int(unit_id.rsplit("::", 1)[-1])
                except ValueError:
                    sent_id = 0
            memory[unit_id] = {
                "unit_id": unit_id,
                "title": title,
                "sent_id": int(sent_id),
                "text": str(row.get("text") or "").strip(),
            }
    return memory


def format_candidate_text(memory_item: dict) -> str:
    return f"{memory_item['title']} [{memory_item['sent_id']}] {memory_item['text']}"


def sample_candidate_ids(row: dict) -> list[str]:
    candidates = row.get("candidates") or {}
    for key in ("C_t", "R_t"):
        value = candidates.get(key)
        if isinstance(value, list) and value:
            return [str(unit_id) for unit_id in value]
    return []


def sample_positive_id(row: dict) -> str:
    labels = row.get("labels") or {}
    ranking = labels.get("ranking_label") or {}
    return str(ranking.get("positive_unit_id") or "")


def sample_k_t(row: dict) -> str:
    if "K_t" in row:
        return str(row["K_t"])
    state = row.get("state") or {}
    return str(state.get("K_t") or "")


class DenseScorer:
    def __init__(self, model_name_or_path: str, device: str, batch_size: int):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        q_emb = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        d_emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.dot(d_emb, q_emb)


class HotpotRanker:
    def __init__(self, model_dir: Path, checkpoint: Path, device: str, max_length: int, batch_size: int):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = CrossEncoderRanker(pretrained_name=str(model_dir), dropout=0.1).to(self.device)
        checkpoint_obj = torch.load(str(checkpoint), map_location=self.device)
        self.model.load_state_dict(checkpoint_obj.get("model_state_dict", checkpoint_obj), strict=False)
        self.model.eval()

    def score(self, context: str, texts: list[str]) -> np.ndarray:
        scores = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                tokens = self.tokenizer(
                    [context] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                output, _ = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    token_type_ids=tokens.get("token_type_ids"),
                )
                scores.extend(output.detach().cpu().tolist())
        return np.array(scores)


def accuracy_at(order: list[int], label: int, k: int) -> int:
    return int(label in order[:k])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA sample candidate retrieval.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--dense-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--ks", default="1,2,3,5,8,10")
    parser.add_argument("--dense-query-mode", choices=["question", "state"], default="question")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--ranker-batch-size", type=int, default=16)
    parser.add_argument("--ranker-max-length", type=int, default=320)
    parser.add_argument("--output", default="outputs/retrieval/hotpotqa_sample_retrieval.json")
    args = parser.parse_args()

    samples = list(read_jsonl(Path(args.samples)))
    if args.max_items > 0:
        samples = samples[: args.max_items]
    memory = load_memory(Path(args.memory))
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})

    dense = DenseScorer(args.dense_model, device=args.device, batch_size=args.dense_batch_size)
    ranker = HotpotRanker(
        model_dir=Path(args.model_dir),
        checkpoint=Path(args.checkpoint),
        device=args.device,
        max_length=args.ranker_max_length,
        batch_size=args.ranker_batch_size,
    )

    totals = {
        "total": 0,
        "skipped": 0,
        **{f"dense_acc@{k}": 0 for k in ks},
        **{f"ranker_acc@{k}": 0 for k in ks},
    }
    examples = []

    for row in tqdm(samples, desc="hotpotqa-sample-retrieval"):
        question = str(row.get("question") or "")
        k_t = sample_k_t(row)
        context = f"Question: {question}\nNotebook:\n{k_t}"
        candidate_ids = sample_candidate_ids(row)
        positive_id = sample_positive_id(row)
        if not question or not candidate_ids or positive_id not in candidate_ids:
            totals["skipped"] += 1
            continue

        texts = []
        missing = False
        for unit_id in candidate_ids:
            memory_item = memory.get(unit_id) or {}
            text = format_candidate_text(memory_item) if memory_item.get("text") else ""
            if not text:
                missing = True
                break
            texts.append(text)
        if missing:
            totals["skipped"] += 1
            continue

        label = candidate_ids.index(positive_id)
        dense_query = context if args.dense_query_mode == "state" else question
        dense_scores = dense.score(dense_query, texts)
        ranker_scores = ranker.score(context, texts)
        dense_order = np.argsort(dense_scores)[::-1].tolist()
        ranker_order = np.argsort(ranker_scores)[::-1].tolist()

        totals["total"] += 1
        for k in ks:
            totals[f"dense_acc@{k}"] += accuracy_at(dense_order, label, k)
            totals[f"ranker_acc@{k}"] += accuracy_at(ranker_order, label, k)

        if len(examples) < 10:
            examples.append(
                {
                    "qid": row.get("qid"),
                    "t": row.get("t"),
                    "question": question,
                    "positive_unit_id": positive_id,
                    "dense_pred_unit_id": candidate_ids[dense_order[0]],
                    "ranker_pred_unit_id": candidate_ids[ranker_order[0]],
                    "dense_top3": [candidate_ids[index] for index in dense_order[:3]],
                    "ranker_top3": [candidate_ids[index] for index in ranker_order[:3]],
                }
            )

    total = max(totals["total"], 1)
    summary = {
        "samples": args.samples,
        "memory": args.memory,
        "dense_model": args.dense_model,
        "dense_query_mode": args.dense_query_mode,
        "checkpoint": args.checkpoint,
        "total": totals["total"],
        "skipped": totals["skipped"],
    }
    for k in ks:
        summary[f"dense_acc@{k}"] = round(totals[f"dense_acc@{k}"] / total, 6)
        summary[f"ranker_acc@{k}"] = round(totals[f"ranker_acc@{k}"] / total, 6)

    report = {"summary": summary, "examples": examples}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
