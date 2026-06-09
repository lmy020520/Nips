#!/usr/bin/env python3
"""Validate a lightweight HotpotQA Policy-RAG evaluation dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON decode failed: file={path} line={line_idx}") from exc


def positive_id(row: dict) -> str:
    return str(((row.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id") or "")


def candidate_ids(row: dict) -> list[str]:
    candidates = row.get("candidates") or {}
    value = candidates.get("C_t") or candidates.get("R_t") or []
    return [str(item) for item in value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/hotpotqa_distractor_eval_3000"))
    args = parser.parse_args()

    root = args.data_root
    paths = {
        "queries": root / "queries" / "test.jsonl",
        "samples": root / "samples" / "test.jsonl",
        "raw_units": root / "unit_registry" / "raw_units_test.jsonl",
        "targets": root / "targets" / "test.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required files: {missing}")

    queries = {str(row.get("qid")): row for row in read_jsonl(paths["queries"])}
    raw_units = {str(row.get("unit_id")): row for row in read_jsonl(paths["raw_units"])}
    targets = {str(row.get("qid")): row for row in read_jsonl(paths["targets"])}
    samples = list(read_jsonl(paths["samples"]))

    issues = []
    by_qid = defaultdict(list)
    candidate_sizes = []
    for row in samples:
        qid = str(row.get("qid") or "")
        by_qid[qid].append(row)
        if qid not in queries:
            issues.append(("missing_query", qid, row.get("t")))
        if qid not in targets:
            issues.append(("missing_target", qid, row.get("t")))
        pos = positive_id(row)
        cands = candidate_ids(row)
        candidate_sizes.append(len(cands))
        if not pos:
            issues.append(("missing_positive", qid, row.get("t")))
        if pos and pos not in cands:
            issues.append(("positive_not_in_candidates", qid, row.get("t")))
        missing_units = [unit_id for unit_id in cands if unit_id not in raw_units]
        if missing_units:
            issues.append(("candidate_missing_raw_unit", qid, row.get("t"), missing_units[:3]))
        if pos and pos not in raw_units:
            issues.append(("positive_missing_raw_unit", qid, row.get("t")))

    for qid, rows in by_qid.items():
        ts = sorted(int(row.get("t", -1)) for row in rows)
        if ts != list(range(len(ts))):
            issues.append(("non_contiguous_steps", qid, ts))

    answer_missing = [qid for qid, row in queries.items() if not str(row.get("answer") or "").strip()]
    if answer_missing:
        issues.append(("missing_answers", answer_missing[:10]))

    status = "OK" if not issues else "FAILED"
    summary = {
        "status": status,
        "data_root": str(root),
        "qids": len(queries),
        "samples": len(samples),
        "raw_units": len(raw_units),
        "targets": len(targets),
        "avg_steps_per_qid": round(len(samples) / max(1, len(queries)), 4),
        "candidate_count": {
            "min": min(candidate_sizes) if candidate_sizes else 0,
            "max": max(candidate_sizes) if candidate_sizes else 0,
            "avg": round(sum(candidate_sizes) / max(1, len(candidate_sizes)), 4),
        },
        "query_type": dict(Counter(str((row.get("metadata") or {}).get("type") or "") for row in queries.values())),
        "query_level": dict(Counter(str((row.get("metadata") or {}).get("level") or "") for row in queries.values())),
        "issues": issues[:20],
        "issue_count": len(issues),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
