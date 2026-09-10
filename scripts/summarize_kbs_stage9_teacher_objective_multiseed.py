#!/usr/bin/env python3
"""Summarize three-seed Closure-versus-Coverage selection reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
METHODS = ("closure", "coverage")
METRICS = (
    "step_at_1",
    "step_at_5",
    "mrr",
    "full_unit_coverage",
    "full_doc_coverage",
    "top1_acquired_reselection_rate",
    "top5_acquired_slot_rate",
)


def parse_named_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("summary must use SEED=PATH")
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(statistics.mean(values), 6),
        "sample_std": round(statistics.stdev(values), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=parse_named_path, required=True)
    parser.add_argument("--expected-qids", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = dict(args.summary)
    if set(paths) != set(SEEDS):
        raise ValueError(f"summaries must contain exactly seeds {SEEDS}")

    failures: list[str] = []
    reports: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        report = json.loads(paths[seed].read_text(encoding="utf-8"))
        reports[seed] = report
        if report.get("status") != "OK":
            failures.append(f"seed {seed}: paired report status is not OK")
        if int(report.get("qids") or 0) != args.expected_qids:
            failures.append(f"seed {seed}: qids != {args.expected_qids}")
        if int(report.get("api_calls") or 0) != 0:
            failures.append(f"seed {seed}: unexpected API calls")
        if report.get("failures"):
            failures.append(f"seed {seed}: paired report contains failures")

    qid_hashes = {
        str(seed): reports[seed].get("ordered_qids_sha256") for seed in SEEDS
    }
    target_hashes = {
        str(seed): reports[seed].get("paired_step_targets_sha256") for seed in SEEDS
    }
    if len(set(qid_hashes.values())) != 1 or None in qid_hashes.values():
        failures.append("ordered qid hashes differ across seeds")
    if len(set(target_hashes.values())) != 1 or None in target_hashes.values():
        failures.append("paired step-target hashes differ across seeds")

    method_metrics = {
        method: {
            metric: stats(
                [
                    float(reports[seed]["method_metrics"][method][metric])
                    for seed in SEEDS
                ]
            )
            for metric in METRICS
        }
        for method in METHODS
    }
    paired_deltas = {}
    for metric in METRICS:
        entries = [reports[seed]["paired_deltas"][metric] for seed in SEEDS]
        values = [float(entry["coverage_minus_closure"]) for entry in entries]
        paired_deltas[metric] = {
            **stats(values),
            "ci95_by_seed": {
                str(seed): [
                    float(entry["ci95_low"]),
                    float(entry["ci95_high"]),
                ]
                for seed, entry in zip(SEEDS, entries)
            },
            "coverage_higher_for_every_seed": all(value > 0 for value in values),
            "closure_higher_for_every_seed": all(value < 0 for value in values),
        }

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.1",
        "mode": "teacher_objective_multiseed_selection",
        "api_calls": 0,
        "seeds": list(SEEDS),
        "qids_per_seed": args.expected_qids,
        "ordered_qids_sha256": qid_hashes,
        "paired_step_targets_sha256": target_hashes,
        "method_metrics": method_metrics,
        "coverage_minus_closure": paired_deltas,
        "interpretation": (
            "Selection evidence is a prerequisite diagnostic; the registered "
            "teacher-objective claim requires downstream answer/support/joint metrics."
        ),
        "next_gate": (
            "Review selection evidence before exact-context cache preparation."
            if not failures
            else "Resolve failures; do not call an API."
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
