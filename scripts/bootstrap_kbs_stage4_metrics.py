#!/usr/bin/env python3
"""Paired qid bootstrap for Stage-4 standard and closure-aware metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_METRICS = (
    "supporting_fact_f1",
    "supporting_fact_em",
    "joint_f1",
    "joint_em",
    "full_support_coverage",
    "closure_success_at_10",
)


def load_records(path: Path) -> dict[str, dict[str, dict]]:
    methods: dict[str, dict[str, dict]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            method = str(row.get("method") or "")
            qid = str(row.get("qid") or "")
            if not method or not qid:
                raise ValueError(f"line {line_number}: missing method or qid")
            method_rows = methods.setdefault(method, {})
            if qid in method_rows:
                raise ValueError(f"line {line_number}: duplicate {method}/{qid}")
            method_rows[qid] = row
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--primary", default="KSG-EA-Compact")
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    if not metrics or args.n_bootstrap <= 0:
        raise ValueError("metrics and n-bootstrap must be non-empty/positive")

    methods = load_records(args.records)
    if args.primary not in methods:
        raise ValueError(f"missing primary method: {args.primary}")
    primary_qids = set(methods[args.primary])
    rng = np.random.default_rng(args.seed)
    result = {
        "status": "OK",
        "primary": args.primary,
        "baselines": args.baseline,
        "metrics": list(metrics),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "comparisons": {},
    }

    for baseline in args.baseline:
        if baseline not in methods:
            raise ValueError(f"missing baseline method: {baseline}")
        if set(methods[baseline]) != primary_qids:
            raise ValueError(f"qid set differs for {baseline}")
        qids = sorted(primary_qids)
        comparison = {}
        for metric in metrics:
            try:
                differences = np.asarray(
                    [
                        float(methods[args.primary][qid][metric])
                        - float(methods[baseline][qid][metric])
                        for qid in qids
                    ],
                    dtype=np.float64,
                )
            except KeyError as exc:
                raise ValueError(f"missing metric {metric!r}") from exc

            bootstrap_means = []
            for start in range(0, args.n_bootstrap, 250):
                size = min(250, args.n_bootstrap - start)
                indices = rng.integers(0, len(qids), size=(size, len(qids)))
                bootstrap_means.append(differences[indices].mean(axis=1))
            samples = np.concatenate(bootstrap_means)
            comparison[metric] = {
                "observed_delta": round(float(differences.mean()), 6),
                "ci95_low": round(float(np.quantile(samples, 0.025)), 6),
                "ci95_high": round(float(np.quantile(samples, 0.975)), 6),
                "shared_qids": len(qids),
                "bootstrap_samples": args.n_bootstrap,
            }
        result["comparisons"][f"{args.primary}-minus-{baseline}"] = comparison

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
