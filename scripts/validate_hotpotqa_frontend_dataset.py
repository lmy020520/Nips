#!/usr/bin/env python3
"""Validate rebuilt HotpotQA front-end training/evaluation data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def positive_id(row: dict) -> str:
    return str((((row.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id")) or "")


def candidate_ids(row: dict) -> list[str]:
    candidates = row.get("candidates") or {}
    value = candidates.get("C_t") or candidates.get("R_t") or []
    return [str(unit_id) for unit_id in value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/hotpotqa_distractor_v10_hybrid_frontend"))
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--splits", default="train,val,test")
    args = parser.parse_args()

    root = args.data_root
    summary = {"data_root": str(root), "splits": {}}
    issues = []

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        samples_path = root / "samples" / f"{split}.jsonl"
        memory_path = root / "unit_registry" / f"raw_units_{split}.jsonl"
        if not samples_path.exists():
            issues.append({"split": split, "issue": "missing_samples", "path": str(samples_path)})
            continue
        if not memory_path.exists():
            issues.append({"split": split, "issue": "missing_memory", "path": str(memory_path)})
            continue

        memory_ids = {str(row.get("unit_id")) for row in read_jsonl(memory_path)}
        counters = Counter()
        by_qid = defaultdict(list)
        candidate_count = Counter()
        rank_buckets = Counter()

        for row_idx, row in enumerate(read_jsonl(samples_path), start=1):
            qid = str(row.get("qid") or "")
            by_qid[qid].append(row)
            candidates = candidate_ids(row)
            positive = positive_id(row)
            meta = row.get("frontend_meta") or {}

            counters["samples"] += 1
            candidate_count[len(candidates)] += 1
            counters["natural_positive"] += int(bool(meta.get("natural_positive_in_candidates")))
            counters["forced_positive"] += int(bool(meta.get("forced_positive")))

            rank = int(meta.get("rrf_rank") or 0)
            if rank:
                bucket = ((min(rank, 100) - 1) // 10 + 1) * 10
                rank_buckets[f"rrf_rank<={bucket}"] += 1

            if not positive:
                issues.append({"split": split, "row": row_idx, "issue": "missing_positive", "qid": qid})
            elif positive not in candidates:
                issues.append({"split": split, "row": row_idx, "issue": "positive_not_in_candidates", "qid": qid})
            missing = [unit_id for unit_id in candidates if unit_id not in memory_ids]
            if missing:
                issues.append(
                    {
                        "split": split,
                        "row": row_idx,
                        "issue": "candidate_missing_memory",
                        "qid": qid,
                        "examples": missing[:3],
                    }
                )
            if args.expected_candidates > 0 and len(candidates) != args.expected_candidates:
                issues.append(
                    {
                        "split": split,
                        "row": row_idx,
                        "issue": "unexpected_candidate_count",
                        "qid": qid,
                        "count": len(candidates),
                    }
                )

        summary["splits"][split] = {
            "qids": len(by_qid),
            "samples": counters["samples"],
            "candidate_count": dict(candidate_count),
            "natural_positive_rate": round(counters["natural_positive"] / max(1, counters["samples"]), 6),
            "forced_positive_rate": round(counters["forced_positive"] / max(1, counters["samples"]), 6),
            "rank_buckets": dict(rank_buckets),
        }

    summary["status"] = "OK" if not issues else "FAILED"
    summary["issue_count"] = len(issues)
    summary["issues"] = issues[:20]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
