#!/usr/bin/env python3
"""Evaluate v27 against matched Full-state and controlled chain baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("v27_dual", "v22_full", "v23_anchor", "v24_direct_indirect")
REFERENCES = METHODS[1:]
CHAIN_BASELINES = ("v23_anchor", "v24_direct_indirect")
METRICS = ("step_at_1", "step_at_5", "mrr", "full_unit_coverage")


def load_report(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary, results = obj.get("summary"), obj.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report must contain summary/results: {path}")
    by_qid = {}
    for row in results:
        qid = str(row.get("qid") or "")
        if not qid or qid in by_qid:
            raise ValueError(f"missing or duplicate qid in {path}: {qid!r}")
        by_qid[qid] = row
    return summary, by_qid


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
    denominators = {"step_at_1": 4, "step_at_5": 4, "mrr": 4, "full_unit_coverage": 5}
    numerators = {"step_at_1": 0, "step_at_5": 1, "mrr": 2, "full_unit_coverage": 3}
    return float(rows[:, numerators[name]].sum() / rows[:, denominators[name]].sum())


def summarize(rows: np.ndarray):
    return {name: round(metric(rows, name), 6) for name in METRICS}


def validate_pairing(records, qids):
    reference = records["v27_dual"]
    for method in REFERENCES:
        for qid in qids:
            left_steps = reference[qid].get("steps") or []
            right_steps = records[method][qid].get("steps") or []
            if len(left_steps) != len(right_steps):
                raise ValueError(f"step count mismatch: method={method}, qid={qid}")
            for left, right in zip(left_steps, right_steps):
                if (
                    int(left.get("t") or 0) != int(right.get("t") or 0)
                    or left.get("positive_unit_id") != right.get("positive_unit_id")
                ):
                    raise ValueError(f"paired step mismatch: method={method}, qid={qid}")


def bootstrap(candidate, baseline, n_bootstrap, seed):
    output = {}
    for metric_index, name in enumerate(METRICS):
        rng = np.random.default_rng(seed + metric_index)
        observed = metric(candidate, name) - metric(baseline, name)
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for index in range(n_bootstrap):
            selected = rng.integers(0, len(candidate), len(candidate))
            samples[index] = metric(candidate[selected], name) - metric(
                baseline[selected], name
            )
        low, high = np.percentile(samples, [2.5, 97.5])
        output[name] = {
            "v27_minus_reference": round(observed, 6),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "bootstrap_samples": n_bootstrap,
        }
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v27-report", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--anchor-report", type=Path, required=True)
    parser.add_argument("--direct-report", type=Path, required=True)
    parser.add_argument("--expected-qids", type=int, required=True)
    parser.add_argument("--n-bootstrap", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = {
        "v27_dual": args.v27_report,
        "v22_full": args.full_report,
        "v23_anchor": args.anchor_report,
        "v24_direct_indirect": args.direct_report,
    }
    summaries, records = {}, {}
    for method, path in paths.items():
        summaries[method], records[method] = load_report(path)

    qid_sets = {method: set(rows) for method, rows in records.items()}
    if any(qids != qid_sets["v27_dual"] for qids in qid_sets.values()):
        raise ValueError(
            "qid sets differ: "
            + ", ".join(f"{method}={len(qids)}" for method, qids in qid_sets.items())
        )
    qids = sorted(qid_sets["v27_dual"])
    if len(qids) != args.expected_qids:
        raise ValueError(f"expected {args.expected_qids} qids, found {len(qids)}")
    validate_pairing(records, qids)

    rows = {
        method: {
            "all_states": np.asarray(
                [qid_totals(method_rows[qid], False) for qid in qids]
            ),
            "hop2_plus": np.asarray(
                [qid_totals(method_rows[qid], True) for qid in qids]
            ),
        }
        for method, method_rows in records.items()
    }
    method_metrics = {
        method: {slice_name: summarize(values) for slice_name, values in slices.items()}
        for method, slices in rows.items()
    }

    comparisons = {}
    passed_chain_baselines = []
    for reference_index, reference in enumerate(REFERENCES):
        slices = {
            slice_name: bootstrap(
                rows["v27_dual"][slice_name],
                rows[reference][slice_name],
                args.n_bootstrap,
                args.seed + reference_index * 100 + slice_index * 10,
            )
            for slice_index, slice_name in enumerate(("all_states", "hop2_plus"))
        }
        hop2 = slices["hop2_plus"]
        all_states = slices["all_states"]
        rank_improves = (
            hop2["step_at_1"]["ci95_low"] > 0
            or hop2["mrr"]["ci95_low"] > 0
        )
        coverage_non_regression = all_states["full_unit_coverage"]["ci95_high"] >= 0
        passes = bool(rank_improves and coverage_non_regression)
        if reference in CHAIN_BASELINES and passes:
            passed_chain_baselines.append(reference)
        comparisons[f"v27_dual_minus_{reference}"] = {
            "metrics": slices,
            "gate": {
                "significant_hop2_step1_or_mrr_improvement": bool(rank_improves),
                "no_significant_full_unit_regression": bool(coverage_non_regression),
                "passes": passes,
            },
        }

    expected_protocol = {
        "v27_dual": (
            "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
            "online_state",
        ),
        "v22_full": (
            "outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt",
            "online_state",
        ),
        "v23_anchor": (
            "outputs/ranker/deberta_v3_large_v23_acra_anchor/best_model.pt",
            "previous_evidence_only",
        ),
        "v24_direct_indirect": (
            "outputs/ranker/deberta_v3_large_v24_ecdr_direct_indirect/best_model.pt",
            "direct_evidence_only",
        ),
    }
    protocol, protocol_failures = {}, []
    for method, summary in summaries.items():
        protocol[method] = {
            key: summary.get(key)
            for key in (
                "checkpoint", "policy_context_source", "selector",
                "candidate_top_k", "select_top_k", "state_update_top_k",
                "policy_blend_weight", "answer_judged",
            )
        }
        checkpoint, context = expected_protocol[method]
        expected = {
            "checkpoint": checkpoint,
            "policy_context_source": context,
            "selector": "hybrid_policy",
            "candidate_top_k": 10,
            "select_top_k": 5,
            "state_update_top_k": 1,
            "policy_blend_weight": 0.5,
            "answer_judged": 0,
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                protocol_failures.append(
                    f"{method} {key}: {summary.get(key)!r} != {value!r}"
                )

    strict_pass = set(passed_chain_baselines) == set(CHAIN_BASELINES)
    if args.smoke:
        status = "SMOKE_OK" if not protocol_failures else "FAIL"
    else:
        status = "PASS" if strict_pass and not protocol_failures else "FAIL"
    result = {
        "status": status,
        "mode": "runtime_smoke" if args.smoke else "validation_gate",
        "qids": len(qids),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "protocol": protocol,
        "protocol_failures": protocol_failures,
        "method_metrics": method_metrics,
        "comparisons": comparisons,
        "acceptance_gate": {
            "passes": bool(strict_pass and not protocol_failures),
            "passed_chain_baselines": passed_chain_baselines,
            "required_chain_baselines": list(CHAIN_BASELINES),
            "rule": (
                "v27 must significantly improve hop2+ Step@1 or MRR over both "
                "v23 anchor and v24 direct-indirect while avoiding significant "
                "full-unit coverage regression against each."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if protocol_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
