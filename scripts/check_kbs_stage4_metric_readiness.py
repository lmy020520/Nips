#!/usr/bin/env python3
"""Check that Stage 4 reports contain enough per-qid data for offline scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_REPORTS = {
    "KSG-EA-Compact": "outputs/rag/kbs_v22_stage2_closure_balanced/full_compact.json",
    "KSG-EA-Recall": "outputs/rag/kbs_v22_stage2_hotpot/full_recall.json",
    "Anchor": "outputs/rag/kbs_stage3_acra_anchor_hotpot/previous_only_compact.json",
    "Direct-Indirect": "outputs/rag/kbs_stage3_ecdr_direct_indirect_hotpot/direct_only_compact.json",
    "Gold-Oracle": "outputs/rag/full3000_gold_oracle.json",
}
REQUIRED_RESULT_KEYS = {
    "qid", "answer", "gold_answer", "selected_unit_ids", "gold_unit_ids", "steps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", default=[])
    parser.add_argument("--expected-qids", type=int, default=3000)
    parser.add_argument("--require-paths", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_reports(values: list[str]) -> dict[str, Path]:
    reports = {name: Path(path) for name, path in DEFAULT_REPORTS.items()}
    for value in values:
        if "=" not in value:
            raise ValueError("--report must use NAME=PATH")
        name, path = value.split("=", 1)
        reports[name] = Path(path)
    return reports


def main() -> None:
    args = parse_args()
    reports = parse_reports(args.report)
    failures = []
    details = {}
    for name, path in reports.items():
        detail = {"path": str(path), "exists": path.is_file()}
        details[name] = detail
        if not path.is_file():
            if args.require_paths:
                failures.append(f"missing report: {name}={path}")
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            summary = obj.get("summary")
            records = obj.get("results")
            if not isinstance(summary, dict) or not isinstance(records, list):
                raise ValueError("missing summary/results")
            qids = [str(record.get("qid") or "") for record in records]
            missing_keys = sorted(
                REQUIRED_RESULT_KEYS - set(records[0]) if records else REQUIRED_RESULT_KEYS
            )
            detail.update(
                qids=len(set(qids)),
                records=len(records),
                duplicate_or_empty_qids=len(qids) - len({qid for qid in qids if qid}),
                missing_result_keys=missing_keys,
                answer_judged=summary.get("answer_judged"),
            )
            if len(records) != args.expected_qids:
                failures.append(f"{name}: expected {args.expected_qids} records")
            if detail["duplicate_or_empty_qids"]:
                failures.append(f"{name}: duplicate or empty qids")
            if missing_keys:
                failures.append(f"{name}: missing result keys {missing_keys}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            detail["error"] = str(exc)
            failures.append(f"{name}: {exc}")

    result = {
        "status": "OK" if not failures else "MISSING",
        "expected_qids": args.expected_qids,
        "reports": details,
        "failures": failures,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
