#!/usr/bin/env python3
"""Use DeepSeek to rerank top-2 recoverable ranker errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unit_title(unit_id: str) -> str:
    parts = unit_id.split("::")
    return parts[-2] if len(parts) >= 3 else unit_id


def unit_sent_id(unit_id: str) -> str:
    parts = unit_id.split("::")
    return parts[-1] if len(parts) >= 3 else "?"


def load_sample_map(samples_path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for record in read_jsonl(samples_path):
        rows[(str(record["qid"]), int(record["t"]))] = record
    return rows


def load_query_map(query_path: Path | None) -> dict[str, dict]:
    if query_path is None or not query_path.exists():
        return {}
    return {str(record["qid"]): record for record in read_jsonl(query_path)}


def load_doc_role_map(targets_path: Path | None) -> dict[str, dict[str, str]]:
    if targets_path is None or not targets_path.exists():
        return {}
    role_map: dict[str, dict[str, str]] = {}
    for record in read_jsonl(targets_path):
        qid = str(record["qid"])
        qid_roles = role_map.setdefault(qid, {})
        for unit in record.get("T_q_raw") or []:
            if not isinstance(unit, dict):
                continue
            doc_id = str(unit.get("doc_id") or "").strip()
            role = str(unit.get("primary_role") or "").strip()
            if doc_id and role:
                qid_roles[doc_id] = role
    return role_map


def load_memory_map(memory_path: Path) -> dict[str, dict]:
    rows = {}
    for record in read_jsonl(memory_path):
        rows[str(record["unit_id"])] = record
    return rows


def candidate_text(unit_id: str, memory_map: dict[str, dict], sample: dict) -> str:
    if unit_id in memory_map:
        item = memory_map[unit_id]
        title = str(item.get("doc_id") or item.get("title") or unit_title(unit_id))
        sent_id = unit_sent_id(unit_id)
        text = str(item.get("text") or "").strip()
        return f"{title} [{sent_id}] {text}"
    derived = sample.get("derived_payloads") or {}
    if isinstance(derived, dict) and unit_id in derived:
        payload = derived[unit_id]
        return str(payload.get("text") or payload.get("unit_text") or "").strip()
    return unit_id


def candidate_doc_id(unit_id: str, memory_map: dict[str, dict]) -> str:
    if unit_id in memory_map:
        item = memory_map[unit_id]
        return str(item.get("doc_id") or item.get("title") or unit_title(unit_id))
    return unit_title(unit_id)


def history_text(sample: dict, memory_map: dict[str, dict]) -> str:
    state = sample.get("state") or {}
    h_t = state.get("H_t") or []
    lines = []
    for item in h_t:
        if not isinstance(item, dict):
            continue
        unit_id = str(item.get("unit_id") or "")
        if not unit_id:
            continue
        lines.append(candidate_text(unit_id, memory_map, sample))
    return "\n".join(f"- {line}" for line in lines) or "(empty)"


def cache_key(qid: str, t: int, top2_ids: list[str]) -> str:
    payload = json.dumps({"qid": qid, "t": t, "top2": top2_ids}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def deepseek_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def build_prompt(
    error: dict,
    sample: dict,
    memory_map: dict[str, dict],
    query_map: dict[str, dict],
    doc_role_map: dict[str, dict[str, str]],
    style: str,
) -> tuple[list[str], str]:
    top2 = error["top_candidates"][:2]
    top2_ids = [str(item["unit_id"]) for item in top2]
    qid = str(error["qid"])
    question = str(sample.get("question") or query_map.get(qid, {}).get("question") or "")
    answer = str(query_map.get(qid, {}).get("answer") or "")
    history = history_text(sample, memory_map)
    doc_a = candidate_doc_id(top2_ids[0], memory_map)
    doc_b = candidate_doc_id(top2_ids[1], memory_map)
    role_a = doc_role_map.get(qid, {}).get(doc_a, top2[0].get("role", "unlabeled"))
    role_b = doc_role_map.get(qid, {}).get(doc_b, top2[1].get("role", "unlabeled"))
    cand_a = candidate_text(top2_ids[0], memory_map, sample)
    cand_b = candidate_text(top2_ids[1], memory_map, sample)
    if style == "basic":
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Already selected evidence before this step:\n{history}\n\n"
            "Choose which candidate should be selected as the NEXT evidence. "
            "The next evidence should best advance the multi-hop reasoning toward answering the question, "
            "not merely repeat already selected information.\n\n"
            f"Candidate A:\n{cand_a}\n\n"
            f"Candidate B:\n{cand_b}\n\n"
            'Return strict JSON: {"choice":"A"} or {"choice":"B"} with an optional short "reason".'
        )
        return top2_ids, user_prompt

    if style == "targeted":
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Gold answer: {answer or '(not provided)'}\n\n"
            f"Already selected evidence before this step:\n{history}\n\n"
            "The original ranker chose Candidate A, but this is a known hard near-miss case. "
            "Do a careful counterfactual check: would Candidate B be a better next evidence sentence?\n\n"
            "Checklist:\n"
            "- Does Candidate A merely identify an entity/background while Candidate B supplies the missing relation?\n"
            "- Does Candidate B connect the current evidence to the answer or next hop?\n"
            "- Is Candidate A a lexical distractor that matches question words but does not advance reasoning?\n"
            "- For comparison questions, does the chosen sentence support the requested comparison, not just one entity?\n"
            "- For bridge questions, does the chosen sentence establish the bridge needed to reach the answer?\n\n"
            "Choose A only if A is clearly more useful as the next evidence. "
            "Choose B if B better completes the multi-hop reasoning, even if B has lower ranker score.\n\n"
            f"Candidate A metadata: doc_id={doc_a}, role={role_a}, ranker_score={top2[0].get('score')}\n"
            f"Candidate A text:\n{cand_a}\n\n"
            f"Candidate B metadata: doc_id={doc_b}, role={role_b}, ranker_score={top2[1].get('score')}\n"
            f"Candidate B text:\n{cand_b}\n\n"
            'Return strict JSON: {"choice":"A"} or {"choice":"B"} with an optional short "reason".'
        )
        return top2_ids, user_prompt

    answer_line = f"Gold answer: {answer}\n\n" if answer else ""
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"{answer_line}"
        f"Already selected evidence before this step:\n{history}\n\n"
        "Task: choose the better NEXT evidence sentence for a trajectory-aware multi-hop retriever.\n\n"
        "Decision rules:\n"
        "1. Prefer the candidate that fills the missing hop needed to answer the question.\n"
        "2. If a candidate only repeats already selected evidence, choose the other candidate.\n"
        "3. If t=1, prefer evidence that connects the previous hop to the final answer or missing entity.\n"
        "4. Use the gold answer only as a judging aid; do not select a sentence solely because it mentions many question words.\n"
        "5. For bridge evidence, choose the sentence that creates the useful link to the next document/entity.\n"
        "6. For support evidence, choose the sentence that directly supports the final answer.\n\n"
        f"Candidate A metadata: doc_id={doc_a}, role={role_a}, ranker_score={top2[0].get('score')}\n"
        f"Candidate A text:\n{cand_a}\n\n"
        f"Candidate B metadata: doc_id={doc_b}, role={role_b}, ranker_score={top2[1].get('score')}\n"
        f"Candidate B text:\n{cand_b}\n\n"
        'Return strict JSON: {"choice":"A"} or {"choice":"B"} with an optional short "reason".'
    )
    return top2_ids, user_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", default="")
    parser.add_argument("--targets", default="")
    parser.add_argument("--prompt-style", choices=("basic", "enhanced", "targeted"), default="enhanced")
    parser.add_argument(
        "--rerun-unfixed-from",
        default="",
        help="Optional previous verifier report; only rerun cases that were not fixed.",
    )
    parser.add_argument("--output", default="outputs/analysis/top2_deepseek_verifier_results.json")
    parser.add_argument("--cache-dir", default="outputs/analysis/top2_verifier_cache")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"

    analysis = load_json(Path(args.analysis))
    sample_map = load_sample_map(Path(args.samples))
    memory_map = load_memory_map(Path(args.memory))
    query_map = load_query_map(Path(args.queries)) if args.queries else {}
    doc_role_map = load_doc_role_map(Path(args.targets)) if args.targets else {}
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    errors = [
        item
        for item in analysis.get("top_errors", [])
        if item.get("positive_rank") is not None and int(item["positive_rank"]) <= 2
    ]
    if args.rerun_unfixed_from:
        previous = load_json(Path(args.rerun_unfixed_from))
        unfixed_keys = {
            (str(item["qid"]), int(item["t"]))
            for item in previous.get("results", [])
            if not item.get("fixed")
        }
        errors = [item for item in errors if (str(item["qid"]), int(item["t"])) in unfixed_keys]
    if args.limit > 0:
        errors = errors[: args.limit]

    system_prompt = (
        "You are a careful evidence verifier for HotpotQA multi-hop retrieval. "
        "Given a question, already selected evidence, and two candidate evidence sentences, "
        "choose the better NEXT evidence. Return JSON only."
    )

    results = []
    fixed = 0
    harmful = 0
    unchanged_wrong = 0
    for idx, error in enumerate(errors, start=1):
        qid = str(error["qid"])
        t = int(error["t"])
        sample = sample_map[(qid, t)]
        top2_ids, user_prompt = build_prompt(
            error,
            sample,
            memory_map,
            query_map,
            doc_role_map,
            args.prompt_style,
        )
        key = cache_key(f"{args.prompt_style}:{qid}", t, top2_ids)
        cache_path = cache_dir / f"{key}.json"
        if cache_path.exists():
            verifier = load_json(cache_path)
        else:
            last_error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    verifier = deepseek_json(
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        timeout=args.timeout,
                    )
                    break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                    last_error = str(exc)
                    time.sleep(min(20, attempt * 2))
            else:
                raise RuntimeError(f"DeepSeek verifier failed for qid={qid}, t={t}: {last_error}")
            cache_path.write_text(json.dumps(verifier, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(max(0.0, args.sleep))

        choice = str(verifier.get("choice", "")).strip().upper()
        chosen_unit_id = top2_ids[0] if choice == "A" else top2_ids[1] if choice == "B" else ""
        positive_unit_id = str(error["positive_unit_id"])
        predicted_unit_id = str(error["predicted_unit_id"])
        is_fixed = chosen_unit_id == positive_unit_id
        is_harmful = chosen_unit_id and chosen_unit_id != predicted_unit_id and chosen_unit_id != positive_unit_id
        fixed += int(is_fixed)
        harmful += int(is_harmful)
        unchanged_wrong += int(chosen_unit_id == predicted_unit_id)
        result = {
            "qid": qid,
            "t": t,
            "positive_unit_id": positive_unit_id,
            "predicted_unit_id": predicted_unit_id,
            "top2_unit_ids": top2_ids,
            "choice": choice,
            "chosen_unit_id": chosen_unit_id,
            "fixed": is_fixed,
            "harmful": is_harmful,
            "reason": verifier.get("reason"),
        }
        results.append(result)
        print(
            f"[{idx}/{len(errors)}] qid={qid} t={t} choice={choice} fixed={is_fixed}",
            flush=True,
        )

    summary = dict(analysis["summary"])
    base_correct = int(summary["correct"])
    total = int(summary["total"])
    new_correct = base_correct + fixed
    report = {
        "analysis": args.analysis,
        "samples": args.samples,
        "memory": args.memory,
        "model": model,
        "prompt_style": args.prompt_style,
        "rerun_unfixed_from": args.rerun_unfixed_from,
        "base_summary": summary,
        "verifier_cases": len(errors),
        "fixed": fixed,
        "unchanged_wrong": unchanged_wrong,
        "harmful_within_error_set": harmful,
        "reranked_correct": new_correct,
        "reranked_accuracy": round(new_correct / max(total, 1), 6),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
