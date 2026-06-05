#!/usr/bin/env python3
"""Build hard-preference samples from ranker near-miss errors."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sample_key(row: dict) -> tuple[str, int]:
    return str(row["qid"]), int(row["t"])


def load_samples(path: Path) -> dict[tuple[str, int], dict]:
    return {sample_key(row): row for row in read_jsonl(path)}


def deep_copy(row: dict) -> dict:
    return json.loads(json.dumps(row, ensure_ascii=False))


def get_candidates(row: dict) -> list[str]:
    candidates = row.get("candidates")
    if isinstance(candidates, dict):
        return [str(x) for x in candidates.get("C_t") or []]
    if isinstance(candidates, list):
        return [str(x) for x in candidates]
    raise ValueError(f"unknown candidates schema: qid={row.get('qid')}")


def positive_unit_id(row: dict) -> str:
    labels = row.get("labels") or {}
    ranking = labels.get("ranking_label") or {}
    if ranking.get("positive_unit_id"):
        return str(ranking["positive_unit_id"])
    if row.get("positive_unit_id"):
        return str(row["positive_unit_id"])
    raise ValueError(f"missing positive unit id: qid={row.get('qid')}")


def filter_map(mapping: dict, keep: set[str]) -> dict:
    if not isinstance(mapping, dict):
        return {}
    return {str(k): v for k, v in mapping.items() if str(k) in keep}


def rewrite_candidates(row: dict, selected: list[str]) -> None:
    candidates = row.get("candidates")
    keep = set(selected)
    if isinstance(candidates, dict):
        for key in ("R_t", "C_t"):
            if isinstance(candidates.get(key), list):
                candidates[key] = [unit_id for unit_id in selected if unit_id in set(map(str, candidates[key]))]
            else:
                candidates[key] = list(selected)
        for key in ("G_t_final", "G_t_aux", "G_t_illegal"):
            if isinstance(candidates.get(key), list):
                candidates[key] = [str(x) for x in candidates[key] if str(x) in keep]
        candidates["candidate_provenance"] = filter_map(candidates.get("candidate_provenance") or {}, keep)
        candidates["aux_candidate_provenance"] = filter_map(candidates.get("aux_candidate_provenance") or {}, keep)
    elif isinstance(candidates, list):
        row["candidates"] = list(selected)
    else:
        raise ValueError(f"unknown candidates schema: qid={row.get('qid')}")


def rewrite_labels(row: dict, selected: list[str], positive: str) -> None:
    labels = row.get("labels") or {}
    ranking = labels.get("ranking_label") or {}
    keep_negative = [unit_id for unit_id in selected if unit_id != positive]
    ranking["positive_unit_id"] = positive
    ranking["negative_unit_ids"] = keep_negative
    ranking["negative_provenance"] = filter_map(ranking.get("negative_provenance") or {}, set(keep_negative))
    labels["ranking_label"] = ranking
    row["labels"] = labels


def build_hard_row(source: dict, error: dict, *, max_rank: int, duplicate_index: int) -> dict | None:
    positive = str(error["positive_unit_id"])
    predicted = str(error["predicted_unit_id"])
    if int(error.get("positive_rank") or 999) > max_rank:
        return None
    top_ids = [str(item["unit_id"]) for item in error.get("top_candidates") or []]
    if positive not in top_ids or predicted not in top_ids:
        return None
    selected = []
    for unit_id in top_ids[:max_rank]:
        if unit_id not in selected:
            selected.append(unit_id)
    if positive not in selected:
        selected.append(positive)
    if predicted not in selected:
        selected.insert(0, predicted)
    selected = selected[:max_rank]
    if positive not in selected:
        selected[-1] = positive
    if len(selected) < 2:
        return None

    row = deep_copy(source)
    rewrite_candidates(row, selected)
    rewrite_labels(row, selected, positive)
    meta = row.setdefault("meta", {})
    meta["hard_preference"] = {
        "source": "build_hotpotqa_hard_preference_samples.py",
        "duplicate_index": duplicate_index,
        "analysis_qid": str(error["qid"]),
        "analysis_t": int(error["t"]),
        "predicted_unit_id": predicted,
        "positive_unit_id": positive,
        "positive_rank": int(error["positive_rank"]),
        "score_gap": error.get("score_gap"),
        "candidate_count_original": error.get("candidate_count"),
        "candidate_count_hard": len(selected),
        "top_candidate_ids": selected,
    }
    return row


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/hotpotqa_distractor_v7_10k_llm_prestep")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output-root", default="data/hotpotqa_distractor_v9_hard_preference")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rank", type=int, default=2)
    parser.add_argument("--max-hard", type=int, default=3000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--include-original", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    samples_out = output_root / "samples"
    samples_out.mkdir(parents=True, exist_ok=True)

    for subdir in ("unit_registry", "targets"):
        src = source_root / subdir
        if src.exists():
            copy_tree(src, output_root / subdir)

    source_samples_path = source_root / "samples" / f"{args.split}.jsonl"
    source_samples = load_samples(source_samples_path)
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))

    stats = Counter()
    hard_rows = []
    for error in analysis.get("top_errors") or []:
        if args.max_hard > 0 and stats["base_hard_cases"] >= args.max_hard:
            break
        if error.get("positive_rank") is None:
            continue
        if int(error["positive_rank"]) > args.max_rank:
            continue
        key = (str(error["qid"]), int(error["t"]))
        source = source_samples.get(key)
        if source is None:
            stats["missing_source"] += 1
            continue
        for duplicate_index in range(1, max(1, args.repeat) + 1):
            row = build_hard_row(source, error, max_rank=args.max_rank, duplicate_index=duplicate_index)
            if row is not None:
                hard_rows.append(row)
        stats[f"rank::{error['positive_rank']}"] += 1
        stats[f"t::{error['t']}"] += 1
        stats["base_hard_cases"] += 1

    train_rows = []
    if args.include_original:
        train_rows.extend(read_jsonl(source_samples_path))
    train_rows.extend(hard_rows)
    train_count = write_jsonl(samples_out / "train.jsonl", train_rows)
    hard_count = write_jsonl(output_root / "debug" / "hard_preference_rows.jsonl", hard_rows)

    split_counts = {"train": train_count}
    for split in ("val", "test"):
        src = source_root / "samples" / f"{split}.jsonl"
        dst = samples_out / f"{split}.jsonl"
        shutil.copy2(src, dst)
        split_counts[split] = sum(1 for _ in read_jsonl(dst))

    summary = {
        "source_root": str(source_root),
        "analysis": args.analysis,
        "output_root": str(output_root),
        "params": {
            "split": args.split,
            "max_rank": args.max_rank,
            "max_hard": args.max_hard,
            "repeat": args.repeat,
            "include_original": args.include_original,
        },
        "samples": split_counts,
        "hard_rows": hard_count,
        "stats": dict(stats),
    }
    summary_path = output_root / "debug" / "hard_preference_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
