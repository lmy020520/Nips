#!/usr/bin/env python3
"""Build an expanded HotpotQA source split without changing val/test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from build_hotpotqa_processed import convert_file
from prepare_hotpotqa_split import project_fields
from rebuild_hotpotqa_queries_v2 import convert_split as convert_queries


SPLITS = ("train", "val", "test")
TARGET_MIX = {
    ("bridge", "hard"): 0.20,
    ("bridge", "medium"): 0.45,
    ("bridge", "easy"): 0.10,
    ("comparison", "hard"): 0.08,
    ("comparison", "medium"): 0.14,
    ("comparison", "easy"): 0.03,
}


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON list: {path}")
    return rows


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def stable_key(qid: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{qid}".encode()).hexdigest()


def qid_of(row: dict) -> str:
    for key in ("qid", "_id", "id"):
        if key in row:
            return str(row[key])
    raise ValueError("sample has no qid/_id/id")


def allocate_quotas(total: int) -> dict[tuple[str, str], int]:
    quotas = {key: int(total * weight) for key, weight in TARGET_MIX.items()}
    remainder = total - sum(quotas.values())
    order = sorted(TARGET_MIX, key=lambda key: TARGET_MIX[key], reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def select_new_rows(pool: list[dict], count: int, seed: int) -> tuple[list[dict], dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pool:
        buckets[(str(row["type"]), str(row["level"]))].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: stable_key(qid_of(row), seed))

    quotas = allocate_quotas(count)
    selected: list[dict] = []
    selected_qids: set[str] = set()
    bucket_stats = {}
    for key, quota in quotas.items():
        chosen = buckets.get(key, [])[:quota]
        selected.extend(chosen)
        selected_qids.update(qid_of(row) for row in chosen)
        bucket_stats[f"{key[0]}/{key[1]}"] = {
            "requested": quota,
            "available": len(buckets.get(key, [])),
            "selected": len(chosen),
        }

    if len(selected) < count:
        remaining = [row for row in pool if qid_of(row) not in selected_qids]
        remaining.sort(key=lambda row: stable_key(qid_of(row), seed + 1))
        selected.extend(remaining[: count - len(selected)])

    if len(selected) != count:
        raise RuntimeError(f"not enough unused train qids: requested={count}, selected={len(selected)}")
    selected.sort(key=lambda row: stable_key(qid_of(row), seed + 2))
    return selected, bucket_stats


def distribution(rows: list[dict]) -> dict:
    return {
        "type": dict(Counter(str(row["type"]) for row in rows)),
        "level": dict(Counter(str(row["level"]) for row in rows)),
        "type_level": {
            f"{key[0]}/{key[1]}": value
            for key, value in Counter((str(row["type"]), str(row["level"])) for row in rows).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-source", type=Path, default=Path("data/hotpotqa_distractor_v6_10k_source"))
    parser.add_argument("--output-source", type=Path, default=Path("data/hotpotqa_distractor_v8_15k_source"))
    parser.add_argument("--new-train-qids", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output_source
    manifest_path = output / "splits" / "incremental_split_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"output already exists; pass --force to rebuild: {output}")

    existing = {split: load_json(args.existing_source / "raw" / f"{split}.json") for split in SPLITS}
    existing_qids = {qid_of(row) for split in SPLITS for row in existing[split]}

    print("Loading full HotpotQA distractor train split...", flush=True)
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
        cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
    )
    pool = [dict(row) for row in dataset if qid_of(row) not in existing_qids]
    new_raw, bucket_stats = select_new_rows(pool, args.new_train_qids, args.seed)
    new_rows = [project_fields(row) for row in new_raw]

    combined = {
        "train": existing["train"] + new_rows,
        "val": existing["val"],
        "test": existing["test"],
    }
    all_qids = [qid_of(row) for split in SPLITS for row in combined[split]]
    if len(all_qids) != len(set(all_qids)):
        raise RuntimeError("expanded split contains duplicate or cross-split qids")

    for split in SPLITS:
        raw_path = output / "raw" / f"{split}.json"
        processed_path = output / "processed" / f"{split}.jsonl"
        query_path = output / "queries" / f"{split}.jsonl"
        write_json(combined[split], raw_path)
        convert_file(raw_path, processed_path)
        convert_queries(processed_path, query_path, split)

    manifest = {
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "existing_source": str(args.existing_source),
        "output_source": str(output),
        "new_train_qids": len(new_rows),
        "split_sizes": {split: len(combined[split]) for split in SPLITS},
        "val_test_unchanged": True,
        "excluded_existing_qids": len(existing_qids),
        "unused_train_pool": len(pool),
        "target_mix": {f"{key[0]}/{key[1]}": value for key, value in TARGET_MIX.items()},
        "bucket_selection": bucket_stats,
        "new_distribution": distribution(new_rows),
        "combined_train_distribution": distribution(combined["train"]),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
