#!/usr/bin/env python3
"""Evaluate DeepSeek with the question only and no retrieved evidence."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from tqdm import tqdm


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def deepseek_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> tuple[str, int]:
    body = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
    last_error = None
    for attempt in range(1, max(1, max_retries) + 1):
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            tokens = int(data.get("usage", {}).get("total_tokens") or 0)
            return content, tokens
        except (
            TimeoutError,
            ConnectionError,
            ConnectionResetError,
            http.client.RemoteDisconnected,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"DeepSeek request failed after {max_retries} retries: {last_error}") from last_error


def extract_answer_from_json(raw_answer: str) -> str:
    text = str(raw_answer or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and payload.get("answer") is not None:
            return str(payload["answer"]).strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group())
            if isinstance(payload, dict) and payload.get("answer") is not None:
                return str(payload["answer"]).strip()
        except json.JSONDecodeError:
            pass
    return text.splitlines()[0].strip() if text else ""


def cache_file_for_qid(cache_dir: Path, qid: str) -> Path:
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
    return cache_dir / f"{safe_qid}.json"


def answer_question_only(
    question: str,
    *,
    max_retries: int,
    retry_sleep: float,
) -> tuple[str, str, int, float]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a short-answer multi-hop QA module. "
                "Answer using your own knowledge because no retrieved evidence is provided. "
                "Output valid JSON only. The answer must be the shortest exact answer phrase, "
                "usually an entity, date, number, yes, or no. Do not include explanations."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nReturn exactly: {{\"answer\":\"...\"}}",
        },
    ]
    started = time.time()
    raw_answer, tokens = deepseek_chat(
        api_key,
        base_url,
        model,
        messages,
        temperature=0.0,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    return extract_answer_from_json(raw_answer), raw_answer, tokens, time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--answer-cache-dir", required=True)
    parser.add_argument("--refresh-answer-cache", action="store_true")
    parser.add_argument("--llm-max-retries", type=int, default=8)
    parser.add_argument("--llm-retry-sleep", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.queries)))
    by_qid = {}
    for row in rows:
        qid = str(row.get("qid") or row.get("_id") or row.get("id") or "")
        if qid:
            by_qid[qid] = row
    qids = sorted(by_qid)
    if args.max_qids > 0:
        qids = qids[: args.max_qids]

    cache_dir = Path(args.answer_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    records = []
    for qid in tqdm(qids, desc="question-only"):
        row = by_qid[qid]
        question = str(row.get("question") or "")
        gold_answer = str(row.get("answer") or "")
        cache_path = cache_file_for_qid(cache_dir, qid)
        answer = ""
        raw_answer = ""
        tokens = 0
        latency = 0.0
        error = ""
        if cache_path.exists() and not args.refresh_answer_cache:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            answer = str(cached.get("answer") or "")
            raw_answer = str(cached.get("raw_answer") or "")
            tokens = int(cached.get("answer_tokens") or 0)
            latency = float(cached.get("answer_latency") or 0.0)
            error = str(cached.get("error") or "")
        else:
            try:
                answer, raw_answer, tokens, latency = answer_question_only(
                    question,
                    max_retries=args.llm_max_retries,
                    retry_sleep=args.llm_retry_sleep,
                )
            except Exception as exc:
                error = str(exc)
            cache_path.write_text(
                json.dumps(
                    {
                        "qid": qid,
                        "answer": answer,
                        "raw_answer": raw_answer,
                        "answer_tokens": tokens,
                        "answer_latency": latency,
                        "error": error,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if error:
            totals["answer_errors"] += 1
        judged = bool(gold_answer) and not error
        em = exact_match_score(answer, gold_answer) if judged else None
        contains = answer_contains_score(answer, gold_answer) if judged else None
        answer_f1 = f1_score(answer, gold_answer) if judged else None
        if judged:
            totals["answer_judged"] += 1
            totals["answer_em"] += int(em)
            totals["answer_contains"] += int(contains)
            totals["answer_f1"] += float(answer_f1)
            totals["answer_tokens"] += tokens
            totals["answer_latency"] += latency
        records.append(
            {
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "answer": answer,
                "raw_answer": raw_answer,
                "answer_em": em,
                "answer_contains": contains,
                "answer_f1": round(float(answer_f1), 6) if answer_f1 is not None else None,
                "answer_tokens": tokens,
                "answer_latency": round(latency, 3),
                "error": error,
            }
        )

    judged = totals["answer_judged"]
    summary = {
        "queries": args.queries,
        "selector": "question_only_llm",
        "evidence_provided": False,
        "deepseek_model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "temperature": 0.0,
        "seed": args.seed,
        "qids": len(qids),
        "answer_judged": judged,
        "answer_errors": totals["answer_errors"],
        "answer_em": round(totals["answer_em"] / judged, 6) if judged else None,
        "answer_contains": round(totals["answer_contains"] / judged, 6) if judged else None,
        "answer_f1": round(totals["answer_f1"] / judged, 6) if judged else None,
        "avg_answer_tokens": round(totals["answer_tokens"] / judged, 2) if judged else None,
        "avg_answer_latency": round(totals["answer_latency"] / judged, 3) if judged else None,
    }
    write_json({"summary": summary, "results": records}, Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
