#!/usr/bin/env python3
"""Paired qid bootstrap for Stage 9.6 MuSiQue paragraph selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METHODS = (
    "compact_seed42",
    "balanced_seed42",
    "hybrid",
    "bge_reranker",
)
METRICS = (
    "paragraph_alignment_at_1",
    "paragraph_alignment_at_5",
    "full_support_paragraph_coverage",
    "full_support_title_coverage",
)
COMPARISONS = (
    ("compact_seed42", "hybrid"),
    ("compact_seed42", "bge_reranker"),
    ("balanced_seed42", "hybrid"),
    ("balanced_seed42", "bge_reranker"),
    ("balanced_seed42", "compact_seed42"),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_reports(values: list[str]) -> dict[str, Path]:
    reports = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--report must be name=path: {value}")
        name, raw_path = value.split("=", 1)
        if name not in METHODS or name in reports:
            raise ValueError(f"unknown or duplicate method: {name}")
        reports[name] = Path(raw_path)
    missing = sorted(set(METHODS) - set(reports))
    if missing:
        raise ValueError(f"missing reports: {missing}")
    return reports


def index_records(path: Path) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    report = read_json(path)
    summary = report.get("summary")
    results = report.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report must contain summary/results: {path}")
    order = []
    indexed = {}
    for row in results:
        if not isinstance(row, dict):
            raise ValueError(f"non-object result row: {path}")
        qid = str(row.get("qid") or "")
        if not qid or qid in indexed:
            raise ValueError(f"missing or duplicate qid={qid!r}: {path}")
        order.append(qid)
        indexed[qid] = row
    return summary, order, indexed


def metric_components(
    records: list[dict[str, Any]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    numerators = []
    denominators = []
    for row in records:
        if metric.startswith("paragraph_alignment_at_"):
            k = int(metric.rsplit("_", 1)[1])
            steps = row.get("steps") or []
            hits = 0
            for step in steps:
                rank = step.get("positive_rank")
                if isinstance(rank, (int, float)):
                    hits += int(int(rank) <= k)
                else:
                    positive = str(step.get("positive_unit_id") or "")
                    selected = [str(value) for value in step.get("selected_unit_ids") or []]
                    hits += int(bool(positive) and positive in selected[:k])
            numerators.append(float(hits))
            denominators.append(float(len(steps)))
        elif metric == "full_support_paragraph_coverage":
            gold = {str(value) for value in row.get("gold_unit_ids") or [] if value}
            selected = {
                str(value) for value in row.get("selected_unit_ids") or [] if value
            }
            numerators.append(float(bool(gold) and gold.issubset(selected)))
            denominators.append(float(bool(gold)))
        elif metric == "full_support_title_coverage":
            gold = {str(value) for value in row.get("gold_doc_ids") or [] if value}
            selected = {
                str(value) for value in row.get("selected_doc_ids") or [] if value
            }
            numerators.append(float(bool(gold) and gold.issubset(selected)))
            denominators.append(float(bool(gold)))
        else:
            raise ValueError(f"unsupported metric: {metric}")
    return np.asarray(numerators, dtype=np.float64), np.asarray(
        denominators, dtype=np.float64
    )


def aggregate(numerators: np.ndarray, denominators: np.ndarray) -> float:
    denominator = float(denominators.sum())
    if denominator <= 0:
        raise ValueError("metric has no valid denominator")
    return float(numerators.sum() / denominator)


def interval(values: list[float]) -> tuple[float, float]:
    return (
        round(float(np.quantile(values, 0.025)), 6),
        round(float(np.quantile(values, 0.975)), 6),
    )


def bootstrap_metric(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    values = []
    size = len(numerators)
    for start in range(0, n_bootstrap, 250):
        current = min(250, n_bootstrap - start)
        indices = rng.integers(0, size, size=(current, size))
        sampled_num = numerators[indices].sum(axis=1)
        sampled_den = denominators[indices].sum(axis=1)
        values.extend((sampled_num / sampled_den).tolist())
    low, high = interval(values)
    return {
        "observed": round(aggregate(numerators, denominators), 6),
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_samples": len(values),
    }


def bootstrap_delta(
    contender: tuple[np.ndarray, np.ndarray],
    baseline: tuple[np.ndarray, np.ndarray],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    contender_num, contender_den = contender
    baseline_num, baseline_den = baseline
    values = []
    size = len(contender_num)
    for start in range(0, n_bootstrap, 250):
        current = min(250, n_bootstrap - start)
        indices = rng.integers(0, size, size=(current, size))
        contender_values = contender_num[indices].sum(axis=1) / contender_den[
            indices
        ].sum(axis=1)
        baseline_values = baseline_num[indices].sum(axis=1) / baseline_den[
            indices
        ].sum(axis=1)
        values.extend((contender_values - baseline_values).tolist())
    low, high = interval(values)
    return {
        "observed_delta": round(
            aggregate(contender_num, contender_den)
            - aggregate(baseline_num, baseline_den),
            6,
        ),
        "ci95_low": low,
        "ci95_high": high,
        "shared_qids": size,
        "bootstrap_samples": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--expected-qids", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("n-bootstrap must be positive")

    selection_summary = read_json(args.selection_summary)
    if (
        selection_summary.get("status") != "OK"
        or selection_summary.get("mode") != "musique_zero_shot_selection"
        or selection_summary.get("qids") != args.expected_qids
        or selection_summary.get("failures")
    ):
        raise ValueError("selection summary is not a clean registered OK")

    paths = parse_reports(args.report)
    report_summaries = {}
    orders = {}
    records = {}
    for name in METHODS:
        summary, order, indexed = index_records(paths[name])
        if len(order) != args.expected_qids:
            raise ValueError(f"{name}: qids={len(order)} != {args.expected_qids}")
        if summary.get("answer_judged") != 0 or summary.get("skipped") != 0:
            raise ValueError(f"{name}: report is not complete selection-only output")
        report_summaries[name] = summary
        orders[name] = order
        records[name] = indexed
    reference_order = orders[METHODS[0]]
    for name in METHODS[1:]:
        if orders[name] != reference_order:
            raise ValueError(f"ordered qids differ for {name}")

    components: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name in METHODS:
        ordered_records = [records[name][qid] for qid in reference_order]
        components[name] = {}
        for metric in METRICS:
            value = metric_components(ordered_records, metric)
            components[name][metric] = value
            observed = aggregate(*value)
            registered = selection_summary["method_metrics"][name][metric]
            if abs(observed - float(registered)) > 2e-6:
                raise ValueError(
                    f"{name}/{metric}: per-qid={observed:.6f} "
                    f"!= selection summary={float(registered):.6f}"
                )

    rng = np.random.default_rng(args.seed)
    methods = {}
    for name in METHODS:
        methods[name] = {}
        for metric in METRICS:
            methods[name][metric] = bootstrap_metric(
                *components[name][metric],
                rng=rng,
                n_bootstrap=args.n_bootstrap,
            )

    paired_deltas = {}
    for contender, baseline in COMPARISONS:
        comparison = f"{contender}-minus-{baseline}"
        paired_deltas[comparison] = {}
        for metric in METRICS:
            paired_deltas[comparison][metric] = bootstrap_delta(
                components[contender][metric],
                components[baseline][metric],
                rng=rng,
                n_bootstrap=args.n_bootstrap,
            )

    result = {
        "status": "OK",
        "stage": 9,
        "step": "9.6",
        "mode": "musique_zero_shot_selection_bootstrap",
        "api_calls": 0,
        "qids": args.expected_qids,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "resampling_unit": "qid cluster",
        "metrics": list(METRICS),
        "methods": methods,
        "paired_deltas": paired_deltas,
        "metric_boundary": (
            "All evidence metrics are paragraph-level and are not directly "
            "identical to sentence-level HotpotQA/2Wiki Step@k metrics."
        ),
        "next_gate": "Review paired intervals before answer-cache preparation.",
        "failures": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
