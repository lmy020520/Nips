#!/usr/bin/env python3
"""Build a deterministic hard-mining plan from ranker error reports.

The script does not run a model. It consumes JSON reports produced by
scripts/analyze_hotpotqa_ranker_errors.py and emits a JSONL plan that can be
used to choose qids/steps for the next dataset expansion round.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(data_root: Path, splits: Iterable[str]) -> Dict[Tuple[str, str, int], dict]:
    rows: Dict[Tuple[str, str, int], dict] = {}
    for split in splits:
        path = data_root / "samples" / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            qid = str(row.get("qid", ""))
            t = int(row.get("t", -1))
            rows[(split, qid, t)] = row
    return rows


def load_unit_texts(data_root: Path, splits: Iterable[str]) -> Dict[str, dict]:
    units: Dict[str, dict] = {}
    for split in splits:
        path = data_root / "unit_registry" / f"raw_units_{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            unit_id = str(row.get("unit_id", ""))
            if unit_id:
                units[unit_id] = row
    return units


def report_split(report: dict, fallback: str) -> str:
    split = str(report.get("split") or fallback).strip()
    return split if split in SPLITS else fallback


def infer_split_from_path(path: Path) -> str:
    name = path.name.lower()
    for split in SPLITS:
        if split in name:
            return split
    return "test"


def hard_score(error: dict) -> int:
    score = 1
    t = int(error.get("t", -1))
    count = int(error.get("candidate_count", 0))
    positive_role = str(error.get("positive_role", "unlabeled"))
    predicted_role = str(error.get("predicted_role", "unlabeled"))
    positive_rank = error.get("positive_rank")

    if t == 1:
        score += 4
    if positive_role == "bridge":
        score += 3
    elif positive_role == "support":
        score += 1
    elif positive_role == "distinguish":
        score += 2
    if count >= 10:
        score += 4
    elif count == 9:
        score += 3
    elif count == 8:
        score += 1
    if positive_role == predicted_role:
        score += 3
    if predicted_role == "unlabeled":
        score += 2
    if isinstance(positive_rank, int):
        if positive_rank <= 3:
            score += 3
        elif positive_rank <= 5:
            score += 1
    score_gap = error.get("score_gap")
    if isinstance(score_gap, (int, float)) and score_gap >= 0.5:
        score += 1
    return score


def priority(error: dict, score: int) -> str:
    if score >= 11:
        return "high"
    if score >= 7:
        return "medium"
    return "low"


def recommended_actions(error: dict) -> List[str]:
    actions: List[str] = []
    t = int(error.get("t", -1))
    count = int(error.get("candidate_count", 0))
    positive_role = str(error.get("positive_role", "unlabeled"))
    predicted_role = str(error.get("predicted_role", "unlabeled"))
    positive_rank = error.get("positive_rank")

    if t == 1:
        actions.append("expand_second_step_trajectory_cases")
    if positive_role == "bridge":
        actions.append("add_bridge_disambiguation_hard_negatives")
    if count >= 9:
        actions.append("add_large_candidate_set_hard_negatives")
    if positive_role == predicted_role:
        actions.append("mine_same_role_near_miss_negatives")
    if predicted_role == "unlabeled":
        actions.append("mine_unlabeled_distractor_contrast")
    if isinstance(positive_rank, int) and positive_rank <= 3:
        actions.append("prioritize_top3_near_miss")
    return actions or ["review_error_case"]


def unit_doc(sample: Optional[dict], unit_id: str, units: Dict[str, dict]) -> str:
    if unit_id in units:
        return str(units[unit_id].get("doc_id") or units[unit_id].get("title") or "")
    if sample:
        provenance = ((sample.get("candidates") or {}).get("candidate_provenance") or {}).get(unit_id) or {}
        return str(provenance.get("doc_id") or "")
    return ""


def unit_text(unit_id: str, units: Dict[str, dict], max_chars: int) -> str:
    text = str((units.get(unit_id) or {}).get("text") or "")
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def build_record(
    *,
    split: str,
    error: dict,
    sample: Optional[dict],
    units: Dict[str, dict],
    max_text_chars: int,
) -> dict:
    score = hard_score(error)
    qid = str(error.get("qid", ""))
    t = int(error.get("t", -1))
    positive_id = str(error.get("positive_unit_id", ""))
    predicted_id = str(error.get("predicted_unit_id", ""))
    question = str((sample or {}).get("question", ""))
    c_t = list(((sample or {}).get("candidates") or {}).get("C_t") or [])
    top_candidates = error.get("top_candidates") if isinstance(error.get("top_candidates"), list) else []

    return {
        "split": split,
        "qid": qid,
        "t": t,
        "priority": priority(error, score),
        "hard_score": score,
        "candidate_count": int(error.get("candidate_count", len(c_t))),
        "positive_role": str(error.get("positive_role", "unlabeled")),
        "predicted_role": str(error.get("predicted_role", "unlabeled")),
        "positive_rank": error.get("positive_rank"),
        "score_gap": error.get("score_gap"),
        "question": question,
        "positive_unit_id": positive_id,
        "positive_doc_id": unit_doc(sample, positive_id, units),
        "positive_text": unit_text(positive_id, units, max_text_chars),
        "predicted_unit_id": predicted_id,
        "predicted_doc_id": unit_doc(sample, predicted_id, units),
        "predicted_text": unit_text(predicted_id, units, max_text_chars),
        "top_candidates": top_candidates,
        "recommended_actions": recommended_actions(error),
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a HotpotQA v8 hard-mining plan from ranker error reports.")
    parser.add_argument("--data-root", default="data/hotpotqa_distractor_v7_10k_llm_prestep")
    parser.add_argument("--reports", nargs="+", required=True, help="Error analysis JSON files.")
    parser.add_argument("--output", default="outputs/analysis/v8_hard_mining_plan.jsonl")
    parser.add_argument("--summary-output", default="outputs/analysis/v8_hard_mining_summary.json")
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--min-priority", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--max-text-chars", type=int, default=240)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    reports = [Path(p) for p in args.reports]
    samples = load_samples(data_root, SPLITS)
    units = load_unit_texts(data_root, SPLITS)
    min_rank = {"low": 0, "medium": 1, "high": 2}[args.min_priority]
    priority_rank = {"low": 0, "medium": 1, "high": 2}

    records: List[dict] = []
    report_summaries = []
    for path in reports:
        report = load_json(path)
        split = report_split(report, infer_split_from_path(path))
        errors = report.get("top_errors") or []
        report_summaries.append(
            {
                "path": str(path),
                "split": split,
                "reported_total": (report.get("summary") or {}).get("total"),
                "reported_errors": (report.get("summary") or {}).get("errors"),
                "loaded_top_errors": len(errors),
            }
        )
        for error in errors:
            qid = str(error.get("qid", ""))
            t = int(error.get("t", -1))
            sample = samples.get((split, qid, t))
            record = build_record(
                split=split,
                error=error,
                sample=sample,
                units=units,
                max_text_chars=args.max_text_chars,
            )
            if priority_rank[record["priority"]] >= min_rank:
                records.append(record)

    records.sort(
        key=lambda item: (
            -int(item["hard_score"]),
            item["split"],
            item["qid"],
            int(item["t"]),
        )
    )
    if args.max_items > 0:
        records = records[: args.max_items]

    written = write_jsonl(Path(args.output), records)
    by_priority = Counter(record["priority"] for record in records)
    by_split = Counter(record["split"] for record in records)
    by_role = Counter(record["positive_role"] for record in records)
    by_t = Counter(str(record["t"]) for record in records)
    by_count = Counter(str(record["candidate_count"]) for record in records)
    actions = Counter(action for record in records for action in record["recommended_actions"])

    summary = {
        "data_root": str(data_root),
        "reports": report_summaries,
        "output": str(args.output),
        "records": written,
        "filters": {
            "max_items": args.max_items,
            "min_priority": args.min_priority,
        },
        "distribution": {
            "priority": dict(by_priority),
            "split": dict(by_split),
            "positive_role": dict(by_role),
            "t": dict(by_t),
            "candidate_count": dict(by_count),
            "recommended_actions": dict(actions.most_common()),
        },
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("HOTPOTQA V8 HARD-MINING PLAN")
    print(f"data_root: {data_root}")
    print(f"records: {written}")
    print(f"output: {args.output}")
    print(f"summary: {args.summary_output}")
    print(f"priority: {dict(by_priority)}")
    print(f"split: {dict(by_split)}")
    print(f"positive_role: {dict(by_role)}")
    print(f"t: {dict(by_t)}")
    print(f"candidate_count: {dict(by_count)}")
    print(f"actions: {dict(actions.most_common(8))}")


if __name__ == "__main__":
    main()
