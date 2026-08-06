#!/usr/bin/env python3
"""Summarize matched v27 validation gates across random seeds."""

import argparse
import json
import statistics
from pathlib import Path


SLICES = ("all_states", "hop2_plus")
METRICS = ("step_at_1", "step_at_5", "mrr", "full_unit_coverage")
CHAIN_BASELINES = ("v23_anchor", "v24_direct_indirect")


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
    failures = []
    for seed, path in args.gate:
        if seed in gates:
            raise ValueError(f"duplicate seed: {seed}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        gates[seed] = obj
        if obj.get("status") != "PASS":
            failures.append(f"seed {seed} gate status is {obj.get('status')!r}")
        if obj.get("protocol_failures"):
            failures.append(f"seed {seed} has protocol failures")
        if int(obj.get("qids", 0)) != 1000:
            failures.append(f"seed {seed} qids={obj.get('qids')} != 1000")
        passed = set((obj.get("acceptance_gate") or {}).get("passed_chain_baselines") or [])
        if passed != set(CHAIN_BASELINES):
            failures.append(f"seed {seed} did not pass both chain baselines: {sorted(passed)}")

    ordered_seeds = sorted(gates, key=int)
    method_metrics = {
        slice_name: {
            metric: mean_std(
                [
                    gates[seed]["method_metrics"]["v27_dual"][slice_name][metric]
                    for seed in ordered_seeds
                ]
            )
            for metric in METRICS
        }
        for slice_name in SLICES
    }
    chain_deltas = {}
    for baseline in CHAIN_BASELINES:
        comparison_key = f"v27_dual_minus_{baseline}"
        chain_deltas[baseline] = {
            slice_name: {
                metric: mean_std(
                    [
                        gates[seed]["comparisons"][comparison_key]["metrics"][slice_name][metric]["v27_minus_reference"]
                        for seed in ordered_seeds
                    ]
                )
                for metric in METRICS
            }
            for slice_name in SLICES
        }

    result = {
        "status": "PASS" if not failures else "FAIL",
        "seeds": ordered_seeds,
        "qids_per_seed": 1000,
        "all_seed_gates_pass": not failures,
        "v27_metrics": method_metrics,
        "v27_minus_chain_baselines": chain_deltas,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
