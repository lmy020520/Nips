#!/usr/bin/env python3
"""Summarize the Stage 9.2 Full-versus-CE+Margin downstream comparison."""

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
DOWNSTREAM_GATE_METRICS = (
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
    if (
        selection.get("status") != "OK"
        or selection.get("mode") != "acquired_loss_multiseed_selection"
        or selection.get("failures")
    ):
        failures.append("selection summary is not a clean Stage 9.2 OK")

    methods = standard.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("standard summary lacks methods")

    bootstraps: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        full_name = f"Full-s{seed}"
        baseline_name = f"CE-Margin-s{seed}"
        for name in (full_name, baseline_name):
            method = methods.get(name)
            if not isinstance(method, dict):
                failures.append(f"missing standard method: {name}")
            elif int(method.get("qids") or 0) != 3000:
                failures.append(f"{name}: qids != 3000")

        bootstrap = load_json(bootstrap_paths[seed])
        bootstraps[seed] = bootstrap
        comparison_name = f"{full_name}-minus-{baseline_name}"
        if bootstrap.get("status") != "OK":
            failures.append(f"seed {seed}: bootstrap status is not OK")
        if not isinstance((bootstrap.get("comparisons") or {}).get(comparison_name), dict):
            failures.append(f"seed {seed}: missing comparison {comparison_name}")

    if failures:
        result: dict[str, Any] = {
            "status": "FAIL",
            "stage": 9,
            "step": "9.2",
            "mode": "acquired_loss_multiseed_downstream",
            "api_calls": 0,
            "failures": failures,
        }
    else:
        method_metrics: dict[str, dict[str, Any]] = {"full": {}, "ce_margin": {}}
        for family, prefix in (("full", "Full"), ("ce_margin", "CE-Margin")):
            for metric in METRICS:
                method_metrics[family][metric] = stats(
                    [
                        float(methods[f"{prefix}-s{seed}"]["metrics"][metric])
                        for seed in SEEDS
                    ]
                )

        paired: dict[str, Any] = {}
        for metric in METRICS:
            entries = [
                bootstraps[seed]["comparisons"][
                    f"Full-s{seed}-minus-CE-Margin-s{seed}"
                ][metric]
                for seed in SEEDS
            ]
            values = [float(entry["observed_delta"]) for entry in entries]
            paired[metric] = {
                **stats(values),
                "delta_definition": "Full minus CE+Margin",
                "ci95_by_seed": {
                    str(seed): [
                        float(entry["ci95_low"]),
                        float(entry["ci95_high"]),
                    ]
                    for seed, entry in zip(SEEDS, entries)
                },
                "full_higher_for_every_seed": all(value > 0 for value in values),
                "full_lower_for_every_seed": all(value < 0 for value in values),
            }

        selection_contrast = selection["paired_seed_contrasts"][
            "full_minus_ce_margin"
        ]
        top1 = selection_contrast["top1_acquired_reselection_rate"]
        targeted_effect = (
            float(top1["mean"]) < 0
            and all(float(high) < 0 for _, high in top1["ci95_by_seed"].values())
        )
        downstream_consistent = all(
            paired[metric]["full_higher_for_every_seed"]
            for metric in DOWNSTREAM_GATE_METRICS
        )
        decision = "DOWNSTREAM_SUPPORTED" if downstream_consistent else (
            "TARGETED_ANTI_RESELECTION_ONLY" if targeted_effect else "MIXED_OR_NULL"
        )
        result = {
            "status": "OK",
            "stage": 9,
            "step": "9.2",
            "mode": "acquired_loss_multiseed_downstream",
            "api_calls": 0,
            "seeds": list(SEEDS),
            "qids_per_seed": 3000,
            "n_bootstrap_per_seed": 10000,
            "primary_contrast": "full_minus_ce_margin",
            "metric_semantics": standard.get("metric_semantics"),
            "method_metrics": method_metrics,
            "full_minus_ce_margin": paired,
            "interpretation_gate": {
                "downstream_metrics": list(DOWNSTREAM_GATE_METRICS),
                "full_higher_for_every_seed_and_downstream_metric": (
                    downstream_consistent
                ),
                "top1_anti_reselection_replicated": targeted_effect,
                "decision": decision,
                "note": (
                    "Per-seed qid bootstrap intervals are reported separately; "
                    "training seeds are not pooled as qid observations."
                ),
            },
            "selection_diagnostics": {
                "source": str(args.selection_summary),
                "full_minus_ce_margin": {
                    key: selection_contrast[key]
                    for key in (
                        "step_at_1",
                        "step_at_5",
                        "mrr",
                        "full_unit_coverage",
                        "top1_acquired_reselection_rate",
                        "top5_acquired_slot_rate",
                    )
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
