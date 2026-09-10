#!/usr/bin/env python3
"""Audit paired Closure-versus-Coverage online selection reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "step_at_1",
    "step_at_5",
    "mrr",
    "full_unit_coverage",
    "full_doc_coverage",
    "top1_acquired_reselection_rate",
    "top5_acquired_slot_rate",
)


def load_report(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    results = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report must contain summary/results: {path}")
    records: dict[str, dict[str, Any]] = {}
    for record in results:
        qid = str(record.get("qid") or "")
        if not qid or qid in records:
            raise ValueError(f"missing or duplicate qid in {path}: {qid!r}")
        records[qid] = record
    return summary, records


def qid_totals(record: dict[str, Any]) -> np.ndarray:
    steps = record.get("steps") or []
    step1 = 0
    step5 = 0
    reciprocal_rank = 0.0
    top1_reselected = 0
    top5_reselected = 0
    top5_slots = 0
    acquired: set[str] = set()

    for step in steps:
        rank = int(step.get("positive_rank") or 0)
        if rank <= 0:
            raise ValueError("positive_rank must be a positive integer")
        step1 += int(rank == 1)
        step5 += int(rank <= 5)
        reciprocal_rank += 1.0 / rank

        predicted = str(step.get("predicted_unit_id") or "")
        top1_reselected += int(bool(predicted) and predicted in acquired)
        selected = [str(value) for value in step.get("selected_unit_ids") or []]
        top5_reselected += sum(value in acquired for value in selected)
        top5_slots += len(selected)
        acquired.update(
            str(value) for value in step.get("state_update_unit_ids") or []
        )

    gold_units = {str(value) for value in record.get("gold_unit_ids") or []}
    selected_units = {str(value) for value in record.get("selected_unit_ids") or []}
    gold_docs = {str(value) for value in record.get("gold_doc_ids") or []}
    selected_docs = {str(value) for value in record.get("selected_doc_ids") or []}
    return np.asarray(
        [
            step1,
            step5,
            reciprocal_rank,
            int(bool(gold_units) and gold_units.issubset(selected_units)),
            int(bool(gold_docs) and gold_docs.issubset(selected_docs)),
            top1_reselected,
            top5_reselected,
            len(steps),
            1,
            top5_slots,
        ],
        dtype=np.float64,
    )


def metric(rows: np.ndarray, name: str) -> float:
    numerator = {
        "step_at_1": 0,
        "step_at_5": 1,
        "mrr": 2,
        "full_unit_coverage": 3,
        "full_doc_coverage": 4,
        "top1_acquired_reselection_rate": 5,
        "top5_acquired_slot_rate": 6,
    }[name]
    denominator = {
        "step_at_1": 7,
        "step_at_5": 7,
        "mrr": 7,
        "full_unit_coverage": 8,
        "full_doc_coverage": 8,
        "top1_acquired_reselection_rate": 7,
        "top5_acquired_slot_rate": 9,
    }[name]
    total = rows[:, denominator].sum()
    return float(rows[:, numerator].sum() / total) if total else 0.0


def summarize(rows: np.ndarray) -> dict[str, float]:
    return {name: round(metric(rows, name), 6) for name in METRICS}


def bootstrap(
    coverage: np.ndarray,
    closure: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, name in enumerate(METRICS):
        observed = metric(coverage, name) - metric(closure, name)
        rng = np.random.default_rng(seed + offset)
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for index in range(n_bootstrap):
            selected = rng.integers(0, len(coverage), len(coverage))
            samples[index] = metric(coverage[selected], name) - metric(
                closure[selected], name
            )
        low, high = np.percentile(samples, [2.5, 97.5])
        result[name] = {
            "coverage_minus_closure": round(observed, 6),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "bootstrap_samples": n_bootstrap,
        }
    return result


def protocol_audit(
    summary: dict[str, Any], expected_checkpoint: str, method: str
) -> list[str]:
    expected = {
        "checkpoint": expected_checkpoint,
        "policy_context_source": "online_state",
        "selector": "hybrid_policy",
        "front_pool_k": 30,
        "front_fusion": "rrf",
        "local_expansion_window": 1,
        "candidate_top_k": 10,
        "select_top_k": 5,
        "state_update_top_k": 1,
        "policy_score_mode": "front_policy_blend",
        "policy_blend_weight": 0.5,
        "generate_answers": False,
        "answer_judged": 0,
        "answer_errors": 0,
    }
    return [
        f"{method} {key}: {summary.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-report", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--closure-checkpoint", required=True)
    parser.add_argument("--coverage-checkpoint", required=True)
    parser.add_argument("--expected-qids", type=int, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    closure_summary, closure_records = load_report(args.closure_report)
    coverage_summary, coverage_records = load_report(args.coverage_report)
    failures = protocol_audit(
        closure_summary, args.closure_checkpoint, "closure"
    ) + protocol_audit(coverage_summary, args.coverage_checkpoint, "coverage")

    closure_qids = list(closure_records)
    coverage_qids = list(coverage_records)
    if closure_qids != coverage_qids:
        failures.append("ordered qids differ between Closure and Coverage")
    if len(closure_qids) != args.expected_qids:
        failures.append(
            f"expected {args.expected_qids} qids, found {len(closure_qids)}"
        )

    shared_qids = [qid for qid in closure_qids if qid in coverage_records]
    for qid in shared_qids:
        closure_steps = closure_records[qid].get("steps") or []
        coverage_steps = coverage_records[qid].get("steps") or []
        if len(closure_steps) != len(coverage_steps):
            failures.append(f"step count differs for qid={qid}")
            continue
        for left, right in zip(closure_steps, coverage_steps):
            if (
                int(left.get("t") or 0) != int(right.get("t") or 0)
                or left.get("positive_unit_id") != right.get("positive_unit_id")
            ):
                failures.append(f"paired step target differs for qid={qid}")
                break

    if len(shared_qids) != args.expected_qids:
        failures.append(
            f"expected {args.expected_qids} shared qids, found {len(shared_qids)}"
        )

    method_metrics: dict[str, Any] = {}
    paired_deltas: dict[str, Any] = {}
    if shared_qids:
        closure_rows = np.asarray(
            [qid_totals(closure_records[qid]) for qid in shared_qids]
        )
        coverage_rows = np.asarray(
            [qid_totals(coverage_records[qid]) for qid in shared_qids]
        )
        method_metrics = {
            "closure": summarize(closure_rows),
            "coverage": summarize(coverage_rows),
        }
        paired_deltas = bootstrap(
            coverage_rows,
            closure_rows,
            args.n_bootstrap,
            args.seed,
        )

    result = {
        "status": "SMOKE_OK" if args.smoke and not failures else (
            "OK" if not failures else "FAIL"
        ),
        "stage": 9,
        "step": "9.1",
        "mode": "paired_selection_smoke" if args.smoke else "paired_selection",
        "api_calls": 0,
        "qids": len(shared_qids),
        "n_bootstrap": args.n_bootstrap,
        "protocol": {
            "operating_point": "Compact",
            "candidate_top_k": 10,
            "select_top_k": 5,
            "state_update_top_k": 1,
            "policy_blend_weight": 0.5,
            "answers_generated": False,
            "closure_checkpoint": args.closure_checkpoint,
            "coverage_checkpoint": args.coverage_checkpoint,
        },
        "method_metrics": method_metrics,
        "paired_deltas": paired_deltas,
        "interpretation": (
            "Smoke metrics validate execution only and are not a scientific gate."
            if args.smoke
            else "Coverage-minus-Closure deltas must be interpreted downstream."
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
