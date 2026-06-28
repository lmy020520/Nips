#!/usr/bin/env python3
"""Bootstrap confidence intervals for KBS RAG reports.

The script consumes reports produced by scripts/run_hotpotqa_policy_rag.py.
It resamples qids with replacement and reports 95% bootstrap confidence
intervals for answer and evidence-selection metrics. For method comparisons,
it performs paired bootstrap over the qid intersection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_METRICS = ("answer_em", "answer_f1", "step_at_5", "full_unit_coverage")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(float(value), 6)


def parse_report_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty report name in spec: {spec}")
    return name, Path(path)


def record_step_hits(record: dict[str, Any]) -> tuple[int, int]:
    hits = 0
    total = 0
    for step in record.get("steps") or []:
        if not isinstance(step, dict):
            continue
        total += 1
        if bool(step.get("selected_contains_gold")):
            hits += 1
    return hits, total


def record_full_unit_coverage(record: dict[str, Any]) -> float | None:
    gold = set(str(item) for item in (record.get("gold_unit_ids") or []) if item)
    selected = set(str(item) for item in (record.get("selected_unit_ids") or []) if item)
    if not gold:
        return None
    return float(gold.issubset(selected))


def record_full_doc_coverage(record: dict[str, Any]) -> float | None:
    gold = set(str(item) for item in (record.get("gold_doc_ids") or []) if item)
    selected = set(str(item) for item in (record.get("selected_doc_ids") or []) if item)
    if not gold:
        return None
    return float(gold.issubset(selected))


def normalize_records(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("report does not contain a records list; rerun with full JSON output")
    by_qid = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        qid = str(record.get("qid") or "")
        if qid:
            by_qid[qid] = record
    if not by_qid:
        raise ValueError("report records list is empty or missing qid fields")
    return by_qid


def aggregate_metric(records: list[dict[str, Any]], metric: str) -> float | None:
    if not records:
        return None

    if metric in {"answer_em", "answer_f1", "answer_contains"}:
        values = []
        for record in records:
            value = record.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return float(mean(values)) if values else None

    if metric == "step_at_5":
        hits = 0
        total = 0
        for record in records:
            step_hits, step_total = record_step_hits(record)
            hits += step_hits
            total += step_total
        return hits / total if total else None

    if metric == "full_unit_coverage":
        values = [record_full_unit_coverage(record) for record in records]
        values = [value for value in values if value is not None]
        return float(mean(values)) if values else None

    if metric == "full_doc_coverage":
        values = [record_full_doc_coverage(record) for record in records]
        values = [value for value in values if value is not None]
        return float(mean(values)) if values else None

    raise ValueError(f"unsupported metric: {metric}")


def bootstrap_ci(
    by_qid: dict[str, dict[str, Any]],
    metric: str,
    *,
    rng: random.Random,
    n_bootstrap: int,
) -> dict[str, Any]:
    qids = sorted(by_qid)
    observed = aggregate_metric([by_qid[qid] for qid in qids], metric)
    samples = []
    for _ in range(n_bootstrap):
        sampled = [by_qid[rng.choice(qids)] for _ in qids]
        value = aggregate_metric(sampled, metric)
        if value is not None:
            samples.append(float(value))
    return {
        "observed": round(float(observed), 6) if observed is not None else None,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "bootstrap_samples": len(samples),
    }


def paired_delta_ci(
    baseline: dict[str, dict[str, Any]],
    contender: dict[str, dict[str, Any]],
    metric: str,
    *,
    rng: random.Random,
    n_bootstrap: int,
) -> dict[str, Any]:
    qids = sorted(set(baseline) & set(contender))
    if not qids:
        return {
            "observed_delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "shared_qids": 0,
            "bootstrap_samples": 0,
        }
    observed_base = aggregate_metric([baseline[qid] for qid in qids], metric)
    observed_contender = aggregate_metric([contender[qid] for qid in qids], metric)
    observed_delta = (
        float(observed_contender) - float(observed_base)
        if observed_base is not None and observed_contender is not None
        else None
    )

    samples = []
    for _ in range(n_bootstrap):
        sampled_qids = [rng.choice(qids) for _ in qids]
        base_value = aggregate_metric([baseline[qid] for qid in sampled_qids], metric)
        contender_value = aggregate_metric([contender[qid] for qid in sampled_qids], metric)
        if base_value is not None and contender_value is not None:
            samples.append(float(contender_value) - float(base_value))

    return {
        "observed_delta": round(observed_delta, 6) if observed_delta is not None else None,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "shared_qids": len(qids),
        "bootstrap_samples": len(samples),
    }


def write_tsv(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, metrics in summary["methods"].items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "section": "method_ci",
                    "method": method,
                    "comparison": "",
                    "metric": metric,
                    "observed": values.get("observed"),
                    "ci95_low": values.get("ci95_low"),
                    "ci95_high": values.get("ci95_high"),
                    "qids": summary["qids_by_method"].get(method),
                }
            )
    for comparison, metrics in summary["paired_deltas"].items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "section": "paired_delta",
                    "method": "",
                    "comparison": comparison,
                    "metric": metric,
                    "observed": values.get("observed_delta"),
                    "ci95_low": values.get("ci95_low"),
                    "ci95_high": values.get("ci95_high"),
                    "qids": values.get("shared_qids"),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "method", "comparison", "metric", "observed", "ci95_low", "ci95_high", "qids"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Report spec as MethodName=path/to/report.json. Repeat for multiple methods.",
    )
    parser.add_argument("--baseline", default="", help="Baseline method for paired deltas, e.g. Hybrid.")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tsv-output", default="")
    args = parser.parse_args()

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    reports = {}
    for spec in args.report:
        name, path = parse_report_spec(spec)
        reports[name] = normalize_records(read_json(path))

    rng = random.Random(args.seed)
    method_summary = {
        name: {
            metric: bootstrap_ci(records, metric, rng=rng, n_bootstrap=args.n_bootstrap)
            for metric in metrics
        }
        for name, records in reports.items()
    }

    paired = {}
    if args.baseline:
        if args.baseline not in reports:
            raise ValueError(f"--baseline {args.baseline!r} not found in reports: {sorted(reports)}")
        for name, records in reports.items():
            if name == args.baseline:
                continue
            comparison = f"{name}-minus-{args.baseline}"
            paired[comparison] = {
                metric: paired_delta_ci(
                    reports[args.baseline],
                    records,
                    metric,
                    rng=rng,
                    n_bootstrap=args.n_bootstrap,
                )
                for metric in metrics
            }

    summary = {
        "reports": list(reports),
        "baseline": args.baseline,
        "metrics": metrics,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "qids_by_method": {name: len(records) for name, records in reports.items()},
        "methods": method_summary,
        "paired_deltas": paired,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.tsv_output:
        write_tsv(summary, Path(args.tsv_output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if args.tsv_output:
        print(f"tsv: {args.tsv_output}")


if __name__ == "__main__":
    main()
