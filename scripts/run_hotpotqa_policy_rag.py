#!/usr/bin/env python3
"""Trajectory-aware policy RAG evaluation for HotpotQA samples.

This script treats the trained model as an evidence-selection policy, not as a
conventional one-shot reranker. For each HotpotQA trajectory state, it scores the
candidate evidence units conditioned on Question + Notebook(K_t), selects the
next evidence unit, and aggregates step-level and trajectory-level metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
import urllib.request
from collections import Counter, defaultdict
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


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_memory(path: Path) -> dict[str, dict]:
    memory = {}
    for row in read_jsonl(path):
        unit_id = str(row.get("unit_id") or "")
        if not unit_id:
            continue
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
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        memory[unit_id] = {
            "unit_id": unit_id,
            "title": title,
            "sent_id": int(sent_id),
            "text": text,
            "doc_id": str(row.get("doc_id") or title),
        }
    return memory


def format_candidate_text(memory_item: dict) -> str:
    return f"{memory_item['title']} [{memory_item['sent_id']}] {memory_item['text']}"


def format_notebook_evidence(memory_item: dict, index: int) -> str:
    return f"[{index}] {memory_item['title']}: {memory_item['text']}"


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


def load_queries(path: str) -> dict[str, dict]:
    if not path:
        return {}
    query_path = Path(path)
    if not query_path.exists():
        return {}
    return {str(row.get("qid")): row for row in read_jsonl(query_path) if row.get("qid")}


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def exact_match_score(prediction: str, gold_answer: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold_answer))


def answer_contains_score(prediction: str, gold_answer: str) -> int:
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(gold_answer)
    return int(bool(norm_gold) and norm_gold in norm_pred)


def f1_score(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


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


def extract_answer_from_json(raw_answer: str) -> str:
    text = str(raw_answer or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("answer") is not None:
            return str(data["answer"]).strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and data.get("answer") is not None:
                return str(data["answer"]).strip()
        except json.JSONDecodeError:
            pass
    return text.splitlines()[0].strip() if text else ""


def answer_with_llm(question: str, evidence: list[dict], answer_mode: str) -> tuple[str, str, int, float]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return "", "", 0, 0.0
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    context = "\n".join(format_notebook_evidence(item, idx + 1) for idx, item in enumerate(evidence))
    if answer_mode == "json":
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an extractive HotpotQA answer module. "
                    "Use only the evidence. Output valid JSON only. "
                    "The answer must be the shortest exact answer phrase, usually an entity, date, number, yes, or no. "
                    "Do not include explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    'Return exactly: {"answer":"..."}'
                ),
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided evidence. "
                    "Return only the short answer, not an explanation. "
                    "If the evidence is insufficient, say unknown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Short answer:"
                ),
            },
        ]
    started = time.time()
    raw_answer, tokens = deepseek_chat(api_key, base_url, model, messages, temperature=0.0)
    answer = extract_answer_from_json(raw_answer) if answer_mode == "json" else raw_answer.strip()
    return answer, raw_answer, tokens, time.time() - started


class PolicyModel:
    def __init__(self, model_dir: Path, checkpoint: Path, device: str, max_length: int, batch_size: int):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = CrossEncoderRanker(pretrained_name=str(model_dir), dropout=0.1).to(self.device)
        checkpoint_obj = torch.load(str(checkpoint), map_location=self.device)
        self.model.load_state_dict(checkpoint_obj.get("model_state_dict", checkpoint_obj), strict=False)
        self.model.eval()

    def score(self, context: str, candidate_texts: list[str]) -> np.ndarray:
        scores = []
        with torch.no_grad():
            for start in range(0, len(candidate_texts), self.batch_size):
                batch_texts = candidate_texts[start : start + self.batch_size]
                tokens = self.tokenizer(
                    [context] * len(batch_texts),
                    batch_texts,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trajectory-aware policy RAG over HotpotQA sample states.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", default="")
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ks", default="1,2,3,5")
    parser.add_argument("--max-items", type=int, default=0, help="Limit number of sample states, not qids.")
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--max-policy-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--state-mode", choices=["dataset", "policy"], default="dataset")
    parser.add_argument("--select-top-k", type=int, default=1)
    parser.add_argument("--answer-mode", choices=["short", "json"], default="json")
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--output", default="outputs/rag/hotpotqa_policy_rag_report.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = list(read_jsonl(Path(args.samples)))
    if args.max_items > 0:
        samples = samples[: args.max_items]
    memory = load_memory(Path(args.memory))
    queries = load_queries(args.queries)
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        qid = str(row.get("qid") or "")
        if qid:
            grouped[qid].append(row)
    qids = sorted(grouped)
    if args.max_qids > 0:
        qids = qids[: args.max_qids]

    policy = PolicyModel(
        model_dir=Path(args.model_dir),
        checkpoint=Path(args.checkpoint),
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    totals = Counter()
    qid_success = Counter()
    answer_metrics = Counter()
    records = []

    for qid in tqdm(qids, desc="policy-rag"):
        rows = sorted(grouped[qid], key=lambda item: int(item.get("t", 0)))
        if args.max_policy_steps > 0:
            rows = rows[: args.max_policy_steps]
        selected_units: list[str] = []
        selected_evidence: list[dict] = []
        selected_doc_ids: set[str] = set()
        gold_units: list[str] = []
        gold_doc_ids: set[str] = set()
        step_records = []
        all_steps_correct = True

        for row in rows:
            question = str(row.get("question") or "")
            candidate_ids = sample_candidate_ids(row)
            positive_id = sample_positive_id(row)
            if not question or not candidate_ids or positive_id not in candidate_ids:
                totals["skipped"] += 1
                all_steps_correct = False
                continue

            candidate_texts = []
            usable_candidate_ids = []
            for unit_id in candidate_ids:
                item = memory.get(unit_id)
                if not item:
                    continue
                usable_candidate_ids.append(unit_id)
                candidate_texts.append(format_candidate_text(item))
            if positive_id not in usable_candidate_ids or not candidate_texts:
                totals["skipped"] += 1
                all_steps_correct = False
                continue

            if args.state_mode == "policy":
                notebook = "\n".join(
                    format_notebook_evidence(item, index + 1)
                    for index, item in enumerate(selected_evidence)
                )
                context = f"Question: {question}\nNotebook:\n{notebook}"
            else:
                context = f"Question: {question}\nNotebook:\n{sample_k_t(row)}"

            scores = policy.score(context, candidate_texts)
            order = np.argsort(scores)[::-1].tolist()
            label = usable_candidate_ids.index(positive_id)
            pred_index = order[0]
            pred_id = usable_candidate_ids[pred_index]
            positive_memory = memory[positive_id]
            selected_indices = order[: max(1, args.select_top_k)]
            selected_step_ids = [usable_candidate_ids[index] for index in selected_indices]

            totals["steps"] += 1
            step_correct = pred_id == positive_id
            all_steps_correct = all_steps_correct and step_correct
            totals["step_selected_contains_gold"] += int(positive_id in selected_step_ids)
            for k in ks:
                totals[f"step_acc@{k}"] += int(label in order[:k])

            for selected_id in selected_step_ids:
                if selected_id in selected_units:
                    continue
                selected_memory = memory[selected_id]
                selected_units.append(selected_id)
                selected_evidence.append(selected_memory)
                selected_doc_ids.add(selected_memory["doc_id"])
            gold_units.append(positive_id)
            gold_doc_ids.add(positive_memory["doc_id"])
            step_records.append(
                {
                    "t": int(row.get("t", 0)),
                    "question": question,
                    "positive_unit_id": positive_id,
                    "predicted_unit_id": pred_id,
                    "selected_unit_ids": selected_step_ids,
                    "correct": step_correct,
                    "selected_contains_gold": positive_id in selected_step_ids,
                    "positive_rank": order.index(label) + 1,
                    "top5": [
                        {
                            "unit_id": usable_candidate_ids[index],
                            "doc_id": memory[usable_candidate_ids[index]]["doc_id"],
                            "score": round(float(scores[index]), 6),
                        }
                        for index in order[:5]
                    ],
                }
            )

        if not step_records:
            continue

        qid_success["total"] += 1
        qid_success["all_steps_correct"] += int(all_steps_correct)
        qid_success["any_gold_doc_selected"] += int(bool(selected_doc_ids & gold_doc_ids))
        qid_success["full_gold_doc_coverage"] += int(gold_doc_ids.issubset(selected_doc_ids))
        qid_success["full_gold_unit_coverage"] += int(set(gold_units).issubset(set(selected_units)))

        answer = ""
        raw_answer = ""
        answer_tokens = 0
        answer_latency = 0.0
        question = str(rows[0].get("question") or "")
        if args.generate_answers:
            answer, raw_answer, answer_tokens, answer_latency = answer_with_llm(
                question,
                selected_evidence,
                answer_mode=args.answer_mode,
            )

        gold_answer = str((queries.get(qid) or {}).get("answer") or "")
        if args.generate_answers and gold_answer:
            answer_metrics["answer_judged"] += 1
            answer_metrics["answer_em"] += exact_match_score(answer, gold_answer)
            answer_metrics["answer_contains"] += answer_contains_score(answer, gold_answer)
            answer_metrics["answer_f1"] += f1_score(answer, gold_answer)
            answer_metrics["answer_tokens"] += answer_tokens
            answer_metrics["answer_latency"] += answer_latency

        records.append(
            {
                "qid": qid,
                "question": question,
                "answer": answer,
                "raw_answer": raw_answer,
                "answer_tokens": answer_tokens,
                "answer_latency": round(answer_latency, 3),
                "gold_answer": gold_answer,
                "answer_em": exact_match_score(answer, gold_answer) if gold_answer and answer else None,
                "answer_contains": answer_contains_score(answer, gold_answer) if gold_answer and answer else None,
                "answer_f1": round(f1_score(answer, gold_answer), 6) if gold_answer and answer else None,
                "selected_unit_ids": selected_units,
                "gold_unit_ids": gold_units,
                "selected_doc_ids": sorted(selected_doc_ids),
                "gold_doc_ids": sorted(gold_doc_ids),
                "all_steps_correct": all_steps_correct,
                "steps": step_records,
            }
        )

    step_total = max(totals["steps"], 1)
    qid_total = max(qid_success["total"], 1)
    summary = {
        "samples": args.samples,
        "memory": args.memory,
        "queries": args.queries,
        "checkpoint": args.checkpoint,
        "state_mode": args.state_mode,
        "select_top_k": args.select_top_k,
        "answer_mode": args.answer_mode,
        "sample_states": len(samples),
        "qids": qid_success["total"],
        "steps": totals["steps"],
        "skipped": totals["skipped"],
        "trajectory_all_steps_correct": round(qid_success["all_steps_correct"] / qid_total, 6),
        "any_gold_doc_selected": round(qid_success["any_gold_doc_selected"] / qid_total, 6),
        "full_gold_doc_coverage": round(qid_success["full_gold_doc_coverage"] / qid_total, 6),
        "full_gold_unit_coverage": round(qid_success["full_gold_unit_coverage"] / qid_total, 6),
        "step_selected_contains_gold": round(totals["step_selected_contains_gold"] / step_total, 6),
    }
    for k in ks:
        summary[f"step_acc@{k}"] = round(totals[f"step_acc@{k}"] / step_total, 6)

    judged_answers = max(answer_metrics["answer_judged"], 1)
    summary.update(
        {
            "answer_judged": answer_metrics["answer_judged"],
            "answer_em": round(answer_metrics["answer_em"] / judged_answers, 6)
            if answer_metrics["answer_judged"]
            else None,
            "answer_contains": round(answer_metrics["answer_contains"] / judged_answers, 6)
            if answer_metrics["answer_judged"]
            else None,
            "answer_f1": round(answer_metrics["answer_f1"] / judged_answers, 6)
            if answer_metrics["answer_judged"]
            else None,
            "avg_answer_tokens": round(answer_metrics["answer_tokens"] / judged_answers, 2)
            if answer_metrics["answer_judged"]
            else None,
            "avg_answer_latency": round(answer_metrics["answer_latency"] / judged_answers, 3)
            if answer_metrics["answer_judged"]
            else None,
        }
    )

    report = {"summary": summary, "results": records}
    write_json(report, Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
