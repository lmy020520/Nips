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
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DEFAULT_METRICS = ("answer_em", "answer_f1", "step_at_5", "full_unit_coverage")

SUMMARY_KEYS = {
    "answer_em": "answer_em",
    "answer_f1": "answer_f1",
    "answer_contains": "answer_contains",
    "step_at_5": "step_acc@5",
    "full_unit_coverage": "full_gold_unit_coverage",
    "full_doc_coverage": "full_gold_doc_coverage",
}


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
    # RAG reports use ``results``; older analysis exports used ``records``.
    records = report.get("records")
    if not isinstance(records, list):
        records = report.get("results")
    if not isinstance(records, list):
        raise ValueError(
            "report does not contain a 'records' or 'results' list; "
            "rerun with full per-qid JSON output"
        )
    by_qid = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        qid = str(record.get("qid") or "")
        if qid:
            if qid in by_qid:
                raise ValueError(f"report contains duplicate qid: {qid}")
            by_qid[qid] = record
    if not by_qid:
        raise ValueError("report records list is empty or missing qid fields")
    return by_qid


def validate_report_summary(
    report: dict[str, Any],
    records: dict[str, dict[str, Any]],
    metrics: list[str],
    *,
    report_name: str,
    tolerance: float = 2e-6,
) -> None:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{report_name}: report does not contain a summary object")

    summary_qids = summary.get("qids")
    if not isinstance(summary_qids, int) or summary_qids != len(records):
        raise ValueError(
            f"{report_name}: summary qids={summary_qids!r}, "
            f"but per-qid results contain {len(records)} records"
        )

    record_list = list(records.values())
    for metric in metrics:
        summary_key = SUMMARY_KEYS.get(metric)
        if summary_key is None:
            continue
        expected = summary.get(summary_key)
        observed = aggregate_metric(record_list, metric)
        if expected is None or observed is None:
            if expected is not None or observed is not None:
                raise ValueError(
                    f"{report_name}: {metric} availability differs between "
                    f"summary ({expected!r}) and per-qid results ({observed!r})"
                )
            continue
        if abs(float(expected) - float(observed)) > tolerance:
            raise ValueError(
                f"{report_name}: {metric} mismatch: summary={float(expected):.6f}, "
                f"per-qid={float(observed):.6f}"
            )


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


def metric_components(
    records: list[dict[str, Any]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-qid numerators and denominators for exact aggregation."""
    numerators = []
    denominators = []
    for record in records:
        if metric in {"answer_em", "answer_f1", "answer_contains"}:
            value = record.get(metric)
            if isinstance(value, (int, float)):
                numerators.append(float(value))
                denominators.append(1.0)
            else:
                numerators.append(0.0)
                denominators.append(0.0)
        elif metric == "step_at_5":
            hits, total = record_step_hits(record)
            numerators.append(float(hits))
            denominators.append(float(total))
        elif metric == "full_unit_coverage":
            value = record_full_unit_coverage(record)
            numerators.append(float(value or 0.0))
            denominators.append(float(value is not None))
        elif metric == "full_doc_coverage":
            value = record_full_doc_coverage(record)
            numerators.append(float(value or 0.0))
            denominators.append(float(value is not None))
        else:
            raise ValueError(f"unsupported metric: {metric}")
    return np.asarray(numerators, dtype=np.float64), np.asarray(denominators, dtype=np.float64)


def bootstrap_values(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    batch_size: int = 256,
) -> list[float]:
    """Resample qids in bounded batches to avoid a large bootstrap matrix."""
    size = len(numerators)
    values: list[float] = []
    for start in range(0, n_bootstrap, batch_size):
        current = min(batch_size, n_bootstrap - start)
        indices = rng.integers(0, size, size=(current, size))
        sampled_num = numerators[indices].sum(axis=1)
        sampled_den = denominators[indices].sum(axis=1)
        valid = sampled_den > 0
        values.extend((sampled_num[valid] / sampled_den[valid]).tolist())
    return values


def bootstrap_ci(
    by_qid: dict[str, dict[str, Any]],
    metric: str,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, Any]:
    records = [by_qid[qid] for qid in sorted(by_qid)]
    numerators, denominators = metric_components(records, metric)
    denominator = denominators.sum()
    observed = numerators.sum() / denominator if denominator > 0 else None
    samples = bootstrap_values(
        numerators,
        denominators,
        rng=rng,
        n_bootstrap=n_bootstrap,
    )
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
    rng: np.random.Generator,
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
    base_num, base_den = metric_components([baseline[qid] for qid in qids], metric)
    contender_num, contender_den = metric_components([contender[qid] for qid in qids], metric)
    observed_base = base_num.sum() / base_den.sum() if base_den.sum() > 0 else None
    observed_contender = (
        contender_num.sum() / contender_den.sum() if contender_den.sum() > 0 else None
    )
    observed_delta = (
        float(observed_contender) - float(observed_base)
        if observed_base is not None and observed_contender is not None
        else None
    )

    samples = []
    size = len(qids)
    for start in range(0, n_bootstrap, 256):
        current = min(256, n_bootstrap - start)
        indices = rng.integers(0, size, size=(current, size))
        sampled_base_den = base_den[indices].sum(axis=1)
        sampled_contender_den = contender_den[indices].sum(axis=1)
        valid = (sampled_base_den > 0) & (sampled_contender_den > 0)
        sampled_base = base_num[indices].sum(axis=1)[valid] / sampled_base_den[valid]
        sampled_contender = (
            contender_num[indices].sum(axis=1)[valid] / sampled_contender_den[valid]
        )
        samples.extend((sampled_contender - sampled_base).tolist())

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
    parser.add_argument(
        "--require-identical-qids",
        action="store_true",
        help="Fail unless every report contains exactly the same qid set.",
    )
    parser.add_argument(
        "--require-summary-match",
        action="store_true",
        help="Recompute requested metrics from per-qid results and verify the report summary.",
    )
    parser.add_argument(
        "--expected-qids",
        type=int,
        default=0,
        help="Fail unless every report contains this many qids. Zero disables the check.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--tsv-output", default="")
    args = parser.parse_args()

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    reports = {}
    for spec in args.report:
        name, path = parse_report_spec(spec)
        if not path.is_file():
            raise FileNotFoundError(f"{name}: report not found: {path}")
        report = read_json(path)
        records = normalize_records(report)
        if args.expected_qids and len(records) != args.expected_qids:
            raise ValueError(
                f"{name}: expected {args.expected_qids} qids, found {len(records)}"
            )
        if args.require_summary_match:
            validate_report_summary(report, records, metrics, report_name=name)
        reports[name] = records

    if args.require_identical_qids and reports:
        reference_name = next(iter(reports))
        reference_qids = set(reports[reference_name])
        mismatches = []
        for name, records in reports.items():
            qids = set(records)
            missing = reference_qids - qids
            extra = qids - reference_qids
            if missing or extra:
                mismatches.append(
                    f"{name}: missing={len(missing)}, extra={len(extra)}"
                )
        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(
                f"report qid sets differ from {reference_name}: {details}"
            )

    rng = np.random.default_rng(args.seed)
    method_summary = {}
    for name, records in reports.items():
        method_summary[name] = {}
        for metric in metrics:
            print(f"[bootstrap] method={name} metric={metric}", flush=True)
            method_summary[name][metric] = bootstrap_ci(
                records, metric, rng=rng, n_bootstrap=args.n_bootstrap
            )

    paired = {}
    if args.baseline:
        if args.baseline not in reports:
            raise ValueError(f"--baseline {args.baseline!r} not found in reports: {sorted(reports)}")
        for name, records in reports.items():
            if name == args.baseline:
                continue
            comparison = f"{name}-minus-{args.baseline}"
            paired[comparison] = {}
            for metric in metrics:
                print(f"[bootstrap] comparison={comparison} metric={metric}", flush=True)
                paired[comparison][metric] = paired_delta_ci(
                    reports[args.baseline],
                    records,
                    metric,
                    rng=rng,
                    n_bootstrap=args.n_bootstrap,
                )

    summary = {
        "reports": list(reports),
        "baseline": args.baseline,
        "metrics": metrics,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "identical_qids_required": args.require_identical_qids,
        "summary_match_required": args.require_summary_match,
        "expected_qids": args.expected_qids or None,
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
