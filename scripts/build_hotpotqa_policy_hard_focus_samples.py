#!/usr/bin/env python3
"""Build a hard-focused training set from front-end/policy rank records.

This is for cases where the fixed front-end already keeps the positive evidence
inside MMR-compressed cand10, but the policy ranks it poorly. The script copies a
source dataset and appends repeated copies of those hard rows to the train split.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def copy_tree_or_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def sample_key(row: dict) -> tuple[str, int]:
    return str(row.get("qid") or ""), int(row.get("t", 0))


def deep_copy(row: dict) -> dict:
    return json.loads(json.dumps(row, ensure_ascii=False))


def hard_record(record: dict, *, min_policy_rank: int, max_mmr_rank: int) -> bool:
    ranks = record.get("ranks") or {}
    policy_rank = ranks.get("policy")
    mmr_rank = ranks.get("mmr_compressed")
    if policy_rank is None or mmr_rank is None:
        return False
    return int(policy_rank) >= min_policy_rank and int(mmr_rank) <= max_mmr_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/hotpotqa_distractor_v11_hybrid_natural")
    parser.add_argument("--rank-records", required=True)
    parser.add_argument("--output-root", default="data/hotpotqa_distractor_v13_hard_focus")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-policy-rank", type=int, default=3)
    parser.add_argument("--max-mmr-rank", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-hard", type=int, default=5000)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for subdir in ("unit_registry", "targets", "queries"):
        copy_tree_or_file(source_root / subdir, output_root / subdir)

    source_samples_path = source_root / "samples" / f"{args.split}.jsonl"
    source_samples = {sample_key(row): row for row in read_jsonl(source_samples_path)}

    stats = Counter()
    hard_rows = []
    for record in read_jsonl(Path(args.rank_records)):
        if args.max_hard > 0 and stats["hard_base_cases"] >= args.max_hard:
            break
        if not hard_record(record, min_policy_rank=args.min_policy_rank, max_mmr_rank=args.max_mmr_rank):
            continue
        key = sample_key(record)
        source = source_samples.get(key)
        if source is None:
            stats["missing_source"] += 1
            continue
        for duplicate_index in range(1, max(1, args.repeat) + 1):
            row = deep_copy(source)
            meta = row.setdefault("meta", {})
            meta["policy_hard_focus"] = {
                "source": "build_hotpotqa_policy_hard_focus_samples.py",
                "duplicate_index": duplicate_index,
                "rank_record": {
                    "qid": key[0],
                    "t": key[1],
                    "ranks": record.get("ranks") or {},
                    "positive_unit_id": record.get("positive_unit_id"),
                    "predicted_unit_id": record.get("predicted_unit_id"),
                    "selected_contains_gold": record.get("selected_contains_gold"),
                },
            }
            hard_rows.append(row)
        stats["hard_base_cases"] += 1
        ranks = record.get("ranks") or {}
        stats[f"policy_rank::{ranks.get('policy')}"] += 1
        stats[f"mmr_rank::{ranks.get('mmr_compressed')}"] += 1

    train_rows = []
    if args.include_original:
        train_rows.extend(read_jsonl(source_samples_path))
    train_rows.extend(hard_rows)
    train_count = write_jsonl(train_rows, output_root / "samples" / "train.jsonl")
    hard_count = write_jsonl(hard_rows, output_root / "debug" / "hard_focus_rows.jsonl")

    split_counts = {"train": train_count}
    for split in ("val", "test"):
        src = source_root / "samples" / f"{split}.jsonl"
        dst = output_root / "samples" / f"{split}.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        split_counts[split] = sum(1 for _ in read_jsonl(dst))

    summary = {
        "source_root": str(source_root),
        "rank_records": args.rank_records,
        "output_root": str(output_root),
        "params": {
            "split": args.split,
            "min_policy_rank": args.min_policy_rank,
            "max_mmr_rank": args.max_mmr_rank,
            "repeat": args.repeat,
            "max_hard": args.max_hard,
            "include_original": args.include_original,
        },
        "samples": split_counts,
        "hard_rows": hard_count,
        "stats": dict(stats),
    }
    write_path = output_root / "debug" / "hard_focus_summary.json"
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
