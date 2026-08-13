#!/usr/bin/env python3
"""Paired no-answer validation analysis for v27 Full versus v28 Anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("step_at_1", "step_at_5", "mrr", "full_unit_coverage")


def load_report(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary, results = obj.get("summary"), obj.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report must contain summary/results: {path}")
    records = {}
    order = []
    for row in results:
        qid = str(row.get("qid") or "")
        if not qid or qid in records:
            raise ValueError(f"missing or duplicate qid in {path}: {qid!r}")
        order.append(qid)
        records[qid] = row
    return summary, records, order


def full_unit(record: dict) -> float:
    gold = {str(value) for value in record.get("gold_unit_ids") or []}
    selected = {str(value) for value in record.get("selected_unit_ids") or []}
    return float(bool(gold) and gold.issubset(selected))


def qid_totals(record: dict, later_only: bool) -> np.ndarray:
    steps = [
        step
        for step in record.get("steps") or []
        if not later_only or int(step.get("t") or 0) >= 1
    ]
    ranks = [int(step.get("positive_rank") or 0) for step in steps]
    if any(rank <= 0 for rank in ranks):
        raise ValueError("positive_rank must be a positive integer")
    return np.asarray(
        [
            sum(rank == 1 for rank in ranks),
            sum(rank <= 5 for rank in ranks),
            sum(1.0 / rank for rank in ranks),
            full_unit(record),
            len(ranks),
            1.0,
        ],
        dtype=np.float64,
    )


def metric(rows: np.ndarray, name: str) -> float:
    numerator = {"step_at_1": 0, "step_at_5": 1, "mrr": 2, "full_unit_coverage": 3}
    denominator = {"step_at_1": 4, "step_at_5": 4, "mrr": 4, "full_unit_coverage": 5}
    return float(rows[:, numerator[name]].sum() / rows[:, denominator[name]].sum())


def summarize(rows: np.ndarray):
    return {name: round(metric(rows, name), 6) for name in METRICS}


def bootstrap(full_rows, anchor_rows, n_bootstrap, seed):
    output = {}
    for index, name in enumerate(METRICS):
        observed = metric(full_rows, name) - metric(anchor_rows, name)
        rng = np.random.default_rng(seed + index)
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for sample_index in range(n_bootstrap):
            selected = rng.integers(0, len(full_rows), len(full_rows))
            samples[sample_index] = metric(full_rows[selected], name) - metric(
                anchor_rows[selected], name
            )
        low, high = np.percentile(samples, [2.5, 97.5])
        output[name] = {
            "full_minus_anchor": round(observed, 6),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "bootstrap_samples": n_bootstrap,
        }
    return output


def validate_pairing(full_records, anchor_records, qids):
    failures = []
    for qid in qids:
        full_steps = full_records[qid].get("steps") or []
        anchor_steps = anchor_records[qid].get("steps") or []
        if len(full_steps) != len(anchor_steps):
            failures.append(f"step count mismatch qid={qid}")
            continue
        for full_step, anchor_step in zip(full_steps, anchor_steps):
            if (
                int(full_step.get("t") or 0) != int(anchor_step.get("t") or 0)
                or full_step.get("positive_unit_id") != anchor_step.get("positive_unit_id")
            ):
                failures.append(f"paired step mismatch qid={qid}")
                break
    return failures


def protocol_failures(summary, expected_checkpoint, expected_context, name):
    expected = {
        "checkpoint": expected_checkpoint,
        "policy_context_source": expected_context,
        "selector": "hybrid_policy",
        "candidate_top_k": 10,
        "select_top_k": 5,
        "state_update_top_k": 1,
        "policy_blend_weight": 0.5,
        "answer_judged": 0,
    }
    return [
        f"{name} {key}: {summary.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--anchor-report", type=Path, required=True)
    parser.add_argument("--full-checkpoint", required=True)
    parser.add_argument("--anchor-checkpoint", required=True)
    parser.add_argument("--expected-qids", type=int, required=True)
    parser.add_argument("--n-bootstrap", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    full_summary, full_records, full_order = load_report(args.full_report)
    anchor_summary, anchor_records, anchor_order = load_report(args.anchor_report)
    failures = []
    if full_order != anchor_order:
        failures.append("ordered qid lists differ")
    if set(full_records) != set(anchor_records):
        failures.append(
            f"qid sets differ: full={len(full_records)}, anchor={len(anchor_records)}"
        )
    qids = full_order
    if len(qids) != args.expected_qids:
        failures.append(f"expected {args.expected_qids} qids, found {len(qids)}")
    if not failures:
        failures.extend(validate_pairing(full_records, anchor_records, qids))
    failures.extend(
        protocol_failures(full_summary, args.full_checkpoint, "online_state", "full")
    )
    failures.extend(
        protocol_failures(
            anchor_summary,
            args.anchor_checkpoint,
            "previous_evidence_only",
            "anchor",
        )
    )

    method_metrics = {}
    comparisons = {}
    gate = {"passes": False, "evaluated": False}
    if not failures:
        rows = {
            name: {
                slice_name: np.asarray(
                    [qid_totals(records[qid], later_only) for qid in qids]
                )
                for slice_name, later_only in (("all_states", False), ("hop2_plus", True))
            }
            for name, records in (("v27_full", full_records), ("v28_anchor", anchor_records))
        }
        method_metrics = {
            name: {slice_name: summarize(values) for slice_name, values in slices.items()}
            for name, slices in rows.items()
        }
        comparisons = {
            slice_name: bootstrap(
                rows["v27_full"][slice_name],
                rows["v28_anchor"][slice_name],
                args.n_bootstrap,
                args.seed + slice_index * 100,
            )
            for slice_index, slice_name in enumerate(("all_states", "hop2_plus"))
        }
        if not args.smoke:
            hop2 = comparisons["hop2_plus"]
            all_states = comparisons["all_states"]
            rank_improves = (
                hop2["step_at_1"]["ci95_low"] > 0
                or hop2["mrr"]["ci95_low"] > 0
            )
            coverage_non_regression = all_states["full_unit_coverage"]["ci95_high"] >= 0
            gate = {
                "evaluated": True,
                "significant_hop2_step1_or_mrr_improvement": bool(rank_improves),
                "no_significant_full_unit_regression": bool(coverage_non_regression),
                "passes": bool(rank_improves and coverage_non_regression),
            }

    status = "FAIL" if failures else ("SMOKE_OK" if args.smoke else ("PASS" if gate["passes"] else "FAIL"))
    result = {
        "status": status,
        "stage": 5,
        "mode": "runtime_smoke" if args.smoke else "validation_gate",
        "qids": len(qids),
        "n_bootstrap": args.n_bootstrap,
        "protocol": {
            "full": {
                "checkpoint": full_summary.get("checkpoint"),
                "policy_context_source": full_summary.get("policy_context_source"),
            },
            "anchor": {
                "checkpoint": anchor_summary.get("checkpoint"),
                "policy_context_source": anchor_summary.get("policy_context_source"),
            },
        },
        "protocol_failures": failures,
        "method_metrics": method_metrics,
        "comparisons": comparisons,
        "acceptance_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
