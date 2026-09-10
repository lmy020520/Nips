#!/usr/bin/env python3
"""Summarize matched Closure-versus-Coverage downstream results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
METRICS = (
    "answer_em",
    "answer_f1",
    "supporting_fact_f1",
    "supporting_fact_em",
    "joint_f1",
    "joint_em",
    "full_support_coverage",
    "closure_success_at_10",
)
GATE_METRICS = (
    "answer_f1",
    "supporting_fact_f1",
    "joint_f1",
    "full_support_coverage",
    "closure_success_at_10",
)


def parse_seed_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("bootstrap must use SEED=PATH")
    raw_seed, raw_path = value.split("=", 1)
    seed = int(raw_seed)
    if seed not in SEEDS or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"seed must be one of {SEEDS}")
    return seed, Path(raw_path)


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(statistics.mean(values), 6),
        "sample_std": round(statistics.stdev(values), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-summary", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument(
        "--bootstrap", action="append", type=parse_seed_path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bootstrap_paths = dict(args.bootstrap)
    if set(bootstrap_paths) != set(SEEDS):
        raise ValueError(f"bootstraps must contain exactly seeds {SEEDS}")

    failures: list[str] = []
    standard = load_json(args.standard_summary)
    selection = load_json(args.selection_summary)
    if standard.get("status") != "OK" or standard.get("failures"):
        failures.append("standard metric summary is not a clean OK")
    if selection.get("status") != "OK" or selection.get("failures"):
        failures.append("selection summary is not a clean OK")

    methods = standard.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("standard summary lacks methods")

    method_metrics: dict[str, dict[str, Any]] = {
        "closure": {},
        "coverage": {},
    }
    paired: dict[str, Any] = {}
    bootstrap_reports: dict[int, dict[str, Any]] = {}

    for seed in SEEDS:
        closure_name = f"Closure-s{seed}"
        coverage_name = f"Coverage-s{seed}"
        for name in (closure_name, coverage_name):
            method = methods.get(name)
            if not isinstance(method, dict):
                failures.append(f"missing standard method: {name}")
                continue
            if int(method.get("qids") or 0) != 3000:
                failures.append(f"{name}: qids != 3000")

        bootstrap = load_json(bootstrap_paths[seed])
        bootstrap_reports[seed] = bootstrap
        comparison_name = f"{coverage_name}-minus-{closure_name}"
        if bootstrap.get("status") != "OK":
            failures.append(f"seed {seed}: bootstrap status is not OK")
        comparison = (bootstrap.get("comparisons") or {}).get(comparison_name)
        if not isinstance(comparison, dict):
            failures.append(f"seed {seed}: missing comparison {comparison_name}")

    if failures:
        result = {
            "status": "FAIL",
            "stage": 9,
            "step": "9.1",
            "mode": "teacher_objective_multiseed_downstream",
            "api_calls": 0,
            "failures": failures,
        }
    else:
        for method, prefix in (("closure", "Closure"), ("coverage", "Coverage")):
            for metric in METRICS:
                values = [
                    float(methods[f"{prefix}-s{seed}"]["metrics"][metric])
                    for seed in SEEDS
                ]
                method_metrics[method][metric] = stats(values)

        for metric in METRICS:
            entries = []
            for seed in SEEDS:
                comparison_name = f"Coverage-s{seed}-minus-Closure-s{seed}"
                entries.append(
                    bootstrap_reports[seed]["comparisons"][comparison_name][metric]
                )
            values = [float(entry["observed_delta"]) for entry in entries]
            paired[metric] = {
                **stats(values),
                "delta_definition": "Coverage minus Closure",
                "ci95_by_seed": {
                    str(seed): [
                        float(entry["ci95_low"]),
                        float(entry["ci95_high"]),
                    ]
                    for seed, entry in zip(SEEDS, entries)
                },
                "closure_higher_for_every_seed": all(value < 0 for value in values),
                "coverage_higher_for_every_seed": all(value > 0 for value in values),
                "closure_direction_ci_excludes_zero_by_seed": {
                    str(seed): float(entry["ci95_high"]) < 0
                    for seed, entry in zip(SEEDS, entries)
                },
            }

        consistent = all(
            paired[metric]["closure_higher_for_every_seed"]
            for metric in GATE_METRICS
        )
        reselection_metrics = (
            "top1_acquired_reselection_rate",
            "top5_acquired_slot_rate",
        )
        result = {
            "status": "OK",
            "stage": 9,
            "step": "9.1",
            "mode": "teacher_objective_multiseed_downstream",
            "api_calls": 0,
            "seeds": list(SEEDS),
            "qids_per_seed": 3000,
            "n_bootstrap_per_seed": 10000,
            "metric_semantics": standard.get("metric_semantics"),
            "method_metrics": method_metrics,
            "coverage_minus_closure": paired,
            "interpretation_gate": {
                "registered_metrics": list(GATE_METRICS),
                "closure_higher_for_every_seed_and_metric": consistent,
                "decision": "CLOSURE_SUPERIOR" if consistent else "MIXED",
                "note": (
                    "The decision uses matched downstream directions across all "
                    "three seeds. Per-seed qid bootstrap intervals are reported "
                    "separately and are not pooled across training seeds."
                ),
            },
            "selection_diagnostics": {
                "source": str(args.selection_summary),
                "method_metrics": {
                    method: {
                        metric: selection["method_metrics"][method][metric]
                        for metric in reselection_metrics
                    }
                    for method in ("closure", "coverage")
                },
                "coverage_minus_closure": {
                    metric: selection["coverage_minus_closure"][metric]
                    for metric in reselection_metrics
                },
            },
            "standard_metrics": str(args.standard_summary),
            "paired_bootstraps": {
                str(seed): str(bootstrap_paths[seed]) for seed in SEEDS
            },
            "failures": [],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if result["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
