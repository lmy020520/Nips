#!/usr/bin/env python3
"""Check the contribution-supervision scope in KBS samples.

Current official v1 supports positive-only c_t* supervision. Full
candidate-wise counterfactual c_t*(u) should only be claimed when samples carry
per-candidate contribution labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CONTRIBUTION_KEYS = ("c_br", "c_dis", "c_sup", "c_der")
CANDIDATE_WISE_KEYS = (
    "candidate_c_t_star",
    "candidate_contribution_labels",
    "candidate_contributions",
    "c_t_star_by_candidate",
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_nested(obj: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def has_positive_contribution(row: dict[str, Any]) -> bool:
    c_t_star = get_nested(row, ["labels", "c_t_star"], {})
    if not isinstance(c_t_star, dict):
        return False
    return all(key in c_t_star and isinstance(c_t_star.get(key), (int, float)) for key in CONTRIBUTION_KEYS)


def candidate_wise_payload(row: dict[str, Any]) -> Any:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    for key in CANDIDATE_WISE_KEYS:
        if key in labels:
            return labels[key]
        if key in row:
            return row[key]
    return None


def has_candidate_wise_contribution(row: dict[str, Any]) -> bool:
    payload = candidate_wise_payload(row)
    if not isinstance(payload, dict) or not payload:
        return False
    for value in payload.values():
        if isinstance(value, dict) and all(key in value for key in CONTRIBUTION_KEYS):
            return True
    return False


def inspect_split(data_root: Path, split: str) -> dict[str, Any]:
    path = data_root / "samples" / f"{split}.jsonl"
    counter = Counter()
    examples = []
    if not path.exists():
        return {
            "split": split,
            "path": str(path),
            "exists": False,
            "total": 0,
            "status": "missing",
        }

    for row in read_jsonl(path):
        counter["total"] += 1
        positive = has_positive_contribution(row)
        candidate_wise = has_candidate_wise_contribution(row)
        counter["positive_contribution"] += int(positive)
        counter["candidate_wise_contribution"] += int(candidate_wise)
        if len(examples) < 5 and (positive or candidate_wise):
            examples.append(
                {
                    "qid": str(row.get("qid") or ""),
                    "t": row.get("t"),
                    "positive_contribution": positive,
                    "candidate_wise_contribution": candidate_wise,
                    "candidate_wise_keys_present": [
                        key
                        for key in CANDIDATE_WISE_KEYS
                        if key in row or key in (row.get("labels") if isinstance(row.get("labels"), dict) else {})
                    ],
                }
            )

    total = max(1, counter["total"])
    return {
        "split": split,
        "path": str(path),
        "exists": True,
        "total": counter["total"],
        "positive_contribution_count": counter["positive_contribution"],
        "positive_contribution_rate": round(counter["positive_contribution"] / total, 6),
        "candidate_wise_contribution_count": counter["candidate_wise_contribution"],
        "candidate_wise_contribution_rate": round(counter["candidate_wise_contribution"] / total, 6),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    split_reports = [inspect_split(data_root, split) for split in splits]
    total = sum(report.get("total", 0) for report in split_reports)
    positive_count = sum(report.get("positive_contribution_count", 0) for report in split_reports)
    candidate_wise_count = sum(report.get("candidate_wise_contribution_count", 0) for report in split_reports)

    summary = {
        "data_root": str(data_root),
        "splits": splits,
        "total": total,
        "positive_contribution_count": positive_count,
        "positive_contribution_rate": round(positive_count / max(1, total), 6),
        "candidate_wise_contribution_count": candidate_wise_count,
        "candidate_wise_contribution_rate": round(candidate_wise_count / max(1, total), 6),
        "official_v1_scope": (
            "positive_only_contribution"
            if positive_count > 0 and candidate_wise_count == 0
            else "candidate_wise_contribution_available"
            if candidate_wise_count > 0
            else "no_contribution_labels"
        ),
        "claim_guardrail": (
            "Do not claim full candidate-wise counterfactual contribution unless "
            "candidate_wise_contribution_count > 0."
        ),
        "splits_report": split_reports,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        write_json(summary, Path(args.output))
    print(text)


if __name__ == "__main__":
    main()
