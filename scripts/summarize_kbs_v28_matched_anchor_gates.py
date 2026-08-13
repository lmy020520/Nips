#!/usr/bin/env python3
"""Summarize v27 Full versus v28 matched-Anchor gates across seeds."""

import argparse
import json
import statistics
from pathlib import Path


EXPECTED_SEEDS = ("42", "43", "44")
SLICES = ("all_states", "hop2_plus")
METRICS = ("step_at_1", "step_at_5", "mrr", "full_unit_coverage")
METHODS = ("v27_full", "v28_anchor")


def parse_named_path(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    seed, path = value.split("=", 1)
    return str(seed), Path(path)


def mean_std(values):
    return {
        "values": [round(float(value), 6) for value in values],
        "mean": round(float(statistics.mean(values)), 6),
        "sample_std": round(float(statistics.stdev(values)), 6) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gates = {}
    audit_failures = []
    scientific_failures = []
    for seed, path in args.gate:
        if seed in gates:
            raise ValueError(f"duplicate seed: {seed}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        gates[seed] = obj
        if obj.get("mode") != "validation_gate":
            audit_failures.append(f"seed {seed} mode={obj.get('mode')!r}")
        if int(obj.get("qids", 0)) != 1000:
            audit_failures.append(f"seed {seed} qids={obj.get('qids')} != 1000")
        if obj.get("protocol_failures"):
            audit_failures.append(f"seed {seed} has protocol failures")
        if obj.get("status") not in ("PASS", "FAIL"):
            audit_failures.append(f"seed {seed} invalid status={obj.get('status')!r}")
        if not bool((obj.get("acceptance_gate") or {}).get("evaluated")):
            audit_failures.append(f"seed {seed} acceptance gate was not evaluated")
        if not bool((obj.get("acceptance_gate") or {}).get("passes")):
            scientific_failures.append(f"seed {seed} does not pass the matched-Anchor gate")

    if set(gates) != set(EXPECTED_SEEDS):
        audit_failures.append(
            f"expected seeds {list(EXPECTED_SEEDS)}, found {sorted(gates, key=int)}"
        )
    ordered_seeds = [seed for seed in EXPECTED_SEEDS if seed in gates]

    method_metrics = {
        method: {
            slice_name: {
                metric: mean_std(
                    [
                        gates[seed]["method_metrics"][method][slice_name][metric]
                        for seed in ordered_seeds
                    ]
                )
                for metric in METRICS
            }
            for slice_name in SLICES
        }
        for method in METHODS
    } if ordered_seeds else {}
    full_minus_anchor = {
        slice_name: {
            metric: mean_std(
                [
                    gates[seed]["comparisons"][slice_name][metric]["full_minus_anchor"]
                    for seed in ordered_seeds
                ]
            )
            for metric in METRICS
        }
        for slice_name in SLICES
    } if ordered_seeds else {}
    direction_consistency = {
        slice_name: {
            metric: {
                "positive_seeds": sum(
                    gates[seed]["comparisons"][slice_name][metric]["full_minus_anchor"] > 0
                    for seed in ordered_seeds
                ),
                "negative_seeds": sum(
                    gates[seed]["comparisons"][slice_name][metric]["full_minus_anchor"] < 0
                    for seed in ordered_seeds
                ),
                "zero_seeds": sum(
                    gates[seed]["comparisons"][slice_name][metric]["full_minus_anchor"] == 0
                    for seed in ordered_seeds
                ),
            }
            for metric in METRICS
        }
        for slice_name in SLICES
    } if ordered_seeds else {}

    experiment_complete = not audit_failures
    scientific_gate_passes = experiment_complete and not scientific_failures
    result = {
        "status": "PASS" if scientific_gate_passes else "FAIL",
        "stage": 5,
        "mode": "matched_anchor_multiseed_summary",
        "experiment_complete": experiment_complete,
        "scientific_gate_passes": scientific_gate_passes,
        "seeds": ordered_seeds,
        "qids_per_seed": 1000,
        "per_seed_status": {
            seed: {
                "status": gates[seed].get("status"),
                "acceptance_gate_passes": bool(
                    (gates[seed].get("acceptance_gate") or {}).get("passes")
                ),
            }
            for seed in ordered_seeds
        },
        "method_metrics": method_metrics,
        "full_minus_anchor": full_minus_anchor,
        "direction_consistency": direction_consistency,
        "audit_failures": audit_failures,
        "scientific_failures": scientific_failures,
        "interpretation_rule": (
            "A negative scientific result remains a completed experiment when "
            "experiment_complete=true; it must be retained rather than rerun or hidden."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if audit_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
