#!/usr/bin/env python3
"""Paired qid bootstrap for Stage 9.6 MuSiQue downstream metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = (
    "KSG-EA-Compact",
    "KSG-EA-Balanced",
    "Hybrid-RAG",
    "BGE-Reranker-RAG",
)
METRICS = (
    "answer_em",
    "answer_f1",
    "supporting_paragraph_f1",
    "supporting_paragraph_em",
    "joint_f1",
    "joint_em",
    "full_support_paragraph_coverage",
    "full_support_title_coverage",
    "closure_success_at_10",
    "closure_success_at_15",
)
COMPARISONS = (
    ("KSG-EA-Compact", "Hybrid-RAG"),
    ("KSG-EA-Compact", "BGE-Reranker-RAG"),
    ("KSG-EA-Balanced", "Hybrid-RAG"),
    ("KSG-EA-Balanced", "BGE-Reranker-RAG"),
    ("KSG-EA-Balanced", "KSG-EA-Compact"),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_records(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    methods: dict[str, dict[str, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            method = str(row.get("method") or "")
            qid = str(row.get("qid") or "")
            if method not in METHODS or not qid:
                raise ValueError(
                    f"line {line_number}: invalid method/qid {method!r}/{qid!r}"
                )
            indexed = methods.setdefault(method, {})
            if qid in indexed:
                raise ValueError(f"line {line_number}: duplicate {method}/{qid}")
            indexed[qid] = row
    return methods


def interval(values: np.ndarray) -> tuple[float, float]:
    return (
        round(float(np.quantile(values, 0.025)), 6),
        round(float(np.quantile(values, 0.975)), 6),
    )


def bootstrap_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    samples = []
    for start in range(0, n_bootstrap, 250):
        current = min(250, n_bootstrap - start)
        indices = rng.integers(0, len(values), size=(current, len(values)))
        samples.append(values[indices].mean(axis=1))
    joined = np.concatenate(samples)
    low, high = interval(joined)
    return {
        "observed": round(float(values.mean()), 6),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_samples": n_bootstrap,
    }


def bootstrap_delta(
    differences: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    result = bootstrap_mean(differences, rng=rng, n_bootstrap=n_bootstrap)
    result["observed_delta"] = result.pop("observed")
    result["shared_qids"] = len(differences)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--standard-summary", type=Path, required=True)
    parser.add_argument("--expected-qids", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("n-bootstrap must be positive")

    standard = read_json(args.standard_summary)
    if (
        standard.get("status") != "OK"
        or standard.get("mode") != "musique_zero_shot_downstream_metrics"
        or standard.get("qids") != args.expected_qids
        or standard.get("unit_granularity") != "paragraph"
        or standard.get("failures")
    ):
        raise ValueError("standard summary is not a clean registered MuSiQue OK")

    records = load_records(args.records)
    if set(records) != set(METHODS):
        raise ValueError(f"record methods={sorted(records)} != {sorted(METHODS)}")
    reference_qids = list(records[METHODS[0]])
    if len(reference_qids) != args.expected_qids:
        raise ValueError(
            f"qids={len(reference_qids)} != expected={args.expected_qids}"
        )
    for method in METHODS[1:]:
        if list(records[method]) != reference_qids:
            raise ValueError(f"ordered qids differ for {method}")

    arrays: dict[str, dict[str, np.ndarray]] = {}
    methods: dict[str, Any] = {}
    rng = np.random.default_rng(args.seed)
    standard_methods = standard.get("methods") or {}
    for method in METHODS:
        arrays[method] = {}
        methods[method] = {}
        standard_method = standard_methods.get(method) or {}
        standard_metrics = standard_method.get("metrics") or {}
        for metric in METRICS:
            try:
                values = np.asarray(
                    [float(records[method][qid][metric]) for qid in reference_qids],
                    dtype=np.float64,
                )
            except KeyError as exc:
                raise ValueError(f"missing {method}/{metric}") from exc
            observed = float(values.mean())
            registered = standard_metrics.get(metric)
            if registered is None or abs(observed - float(registered)) > 2e-6:
                raise ValueError(
                    f"{method}/{metric}: records={observed:.6f} "
                    f"!= standard={registered!r}"
                )
            arrays[method][metric] = values
            methods[method][metric] = bootstrap_mean(
                values, rng=rng, n_bootstrap=args.n_bootstrap
            )

    comparisons = {}
    for contender, baseline in COMPARISONS:
        name = f"{contender}-minus-{baseline}"
        comparisons[name] = {}
        for metric in METRICS:
            differences = arrays[contender][metric] - arrays[baseline][metric]
            comparisons[name][metric] = bootstrap_delta(
                differences, rng=rng, n_bootstrap=args.n_bootstrap
            )

    result = {
        "status": "OK",
        "stage": 9,
        "step": "9.6",
        "mode": "musique_zero_shot_downstream_bootstrap",
        "api_calls": 0,
        "qids": args.expected_qids,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "resampling_unit": "qid",
        "unit_granularity": "paragraph",
        "metrics": list(METRICS),
        "methods": methods,
        "comparisons": comparisons,
        "metric_boundary": (
            "Evidence metrics use MuSiQue paragraphs and are not sentence-level "
            "HotpotQA/2Wiki supporting-fact metrics."
        ),
        "failures": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
