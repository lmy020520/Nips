#!/usr/bin/env python3
"""Audit matched online selection reports for the Stage 9.2 loss ablation."""

from __future__ import annotations

import argparse
import hashlib
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


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("value must use non-empty NAME=PATH")
    return name, Path(path)


def parse_named_text(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use NAME=VALUE")
    name, text = value.split("=", 1)
    if not name or not text:
        raise argparse.ArgumentTypeError("value must use non-empty NAME=VALUE")
    return name, text


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
    step1 = 0
    step5 = 0
    reciprocal_rank = 0.0
    top1_reselected = 0
    top5_reselected = 0
    top5_slots = 0
    acquired: set[str] = set()
    steps = record.get("steps") or []
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


def paired_bootstrap(
    reference: np.ndarray,
    comparison: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for offset, name in enumerate(METRICS):
        observed = metric(reference, name) - metric(comparison, name)
        rng = np.random.default_rng(seed + offset)
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for index in range(n_bootstrap):
            selected = rng.integers(0, len(reference), len(reference))
            samples[index] = metric(reference[selected], name) - metric(
                comparison[selected], name
            )
        low, high = np.percentile(samples, [2.5, 97.5])
        output[name] = {
            "reference_minus_comparison": round(observed, 6),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "bootstrap_samples": n_bootstrap,
        }
    return output


def protocol_failures(
    method: str, summary: dict[str, Any], checkpoint: str
) -> list[str]:
    expected = {
        "checkpoint": checkpoint,
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
        f"{method} {key}: {summary.get(key)!r} != {expected_value!r}"
        for key, expected_value in expected.items()
        if summary.get(key) != expected_value
    ]


def sequence_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=parse_named_path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=parse_named_text, required=True
    )
    parser.add_argument("--reference", default="full")
    parser.add_argument("--expected-qids", type=int, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report_paths = dict(args.report)
    checkpoints = dict(args.checkpoint)
    failures: list[str] = []
    if set(report_paths) != set(checkpoints):
        failures.append("report and checkpoint method sets differ")
    if args.reference not in report_paths:
        failures.append(f"reference method is missing: {args.reference}")

    summaries: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for method, path in report_paths.items():
        try:
            summaries[method], records[method] = load_report(path)
            failures.extend(
                protocol_failures(method, summaries[method], checkpoints.get(method, ""))
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{method}: {exc}")

    ordered_qids: dict[str, list[str]] = {
        method: list(method_records) for method, method_records in records.items()
    }
    reference_qids = ordered_qids.get(args.reference, [])
    if len(reference_qids) != args.expected_qids:
        failures.append(
            f"expected {args.expected_qids} reference qids, found {len(reference_qids)}"
        )

    reference_targets: list[str] = []
    if args.reference in records:
        for qid in reference_qids:
            for step in records[args.reference][qid].get("steps") or []:
                reference_targets.append(
                    f"{qid}\t{int(step.get('t') or 0)}\t"
                    f"{str(step.get('positive_unit_id') or '')}"
                )
    for method, qids in ordered_qids.items():
        if qids != reference_qids:
            failures.append(f"ordered qids differ: {method} versus {args.reference}")
            continue
        targets: list[str] = []
        for qid in qids:
            for step in records[method][qid].get("steps") or []:
                targets.append(
                    f"{qid}\t{int(step.get('t') or 0)}\t"
                    f"{str(step.get('positive_unit_id') or '')}"
                )
        if targets != reference_targets:
            failures.append(f"step targets differ: {method} versus {args.reference}")

    rows: dict[str, np.ndarray] = {}
    if reference_qids:
        for method, method_records in records.items():
            if list(method_records) == reference_qids:
                rows[method] = np.asarray(
                    [qid_totals(method_records[qid]) for qid in reference_qids]
                )

    method_metrics = {method: summarize(values) for method, values in rows.items()}
    paired_deltas = {}
    if args.reference in rows:
        paired_deltas = {
            f"{args.reference}_minus_{method}": paired_bootstrap(
                rows[args.reference],
                values,
                args.n_bootstrap,
                args.seed + index * 100,
            )
            for index, (method, values) in enumerate(rows.items())
            if method != args.reference
        }

    result = {
        "status": (
            "SMOKE_OK" if args.smoke and not failures else "OK" if not failures else "FAIL"
        ),
        "stage": 9,
        "step": "9.2",
        "mode": "acquired_loss_selection_smoke" if args.smoke else "acquired_loss_selection",
        "api_calls": 0,
        "qids": len(reference_qids),
        "n_bootstrap": args.n_bootstrap,
        "reference": args.reference,
        "ordered_qids_sha256": sequence_hash(reference_qids),
        "step_targets_sha256": sequence_hash(reference_targets),
        "protocol": {
            "operating_point": "Compact",
            "candidate_top_k": 10,
            "select_top_k": 5,
            "state_update_top_k": 1,
            "policy_blend_weight": 0.5,
            "answers_generated": False,
            "checkpoints": checkpoints,
        },
        "method_metrics": method_metrics,
        "paired_deltas": paired_deltas,
        "interpretation": (
            "Smoke metrics validate execution only and are not a scientific gate."
            if args.smoke
            else "Selection diagnostics precede any downstream answer evaluation."
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
