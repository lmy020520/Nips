#!/usr/bin/env python3
"""Evaluate the pre-registered v25 validation gate against chain baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("v25_full", "v23_anchor", "v24_direct_indirect")
BASELINES = ("v23_anchor", "v24_direct_indirect")
METRICS = ("step_at_1", "step_at_5", "mrr", "full_unit_coverage")


def load_report(path: Path) -> tuple[dict, dict[str, dict]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    results = obj.get("results")
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


def qid_totals(record: dict, *, later_only: bool) -> np.ndarray:
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
    if name == "step_at_1":
        return float(rows[:, 0].sum() / rows[:, 4].sum())
    if name == "step_at_5":
        return float(rows[:, 1].sum() / rows[:, 4].sum())
    if name == "mrr":
        return float(rows[:, 2].sum() / rows[:, 4].sum())
    if name == "full_unit_coverage":
        return float(rows[:, 3].sum() / rows[:, 5].sum())
    raise KeyError(name)


def validate_pairing(records: dict[str, dict[str, dict]], qids: list[str]) -> None:
    reference = records["v25_full"]
    for method in BASELINES:
        for qid in qids:
            reference_steps = reference[qid].get("steps") or []
            candidate_steps = records[method][qid].get("steps") or []
            if len(reference_steps) != len(candidate_steps):
                raise ValueError(f"step count mismatch: method={method}, qid={qid}")
            for left, right in zip(reference_steps, candidate_steps):
                if (
                    int(left.get("t") or 0) != int(right.get("t") or 0)
                    or left.get("positive_unit_id") != right.get("positive_unit_id")
                ):
                    raise ValueError(f"paired step mismatch: method={method}, qid={qid}")


def summarize(rows: np.ndarray) -> dict:
    return {name: round(metric(rows, name), 6) for name in METRICS}


def bootstrap_comparison(
    candidate_rows: np.ndarray,
    baseline_rows: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    for metric_index, name in enumerate(METRICS):
        observed = metric(candidate_rows, name) - metric(baseline_rows, name)
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for index in range(n_bootstrap):
            selected = rng.integers(0, len(candidate_rows), len(candidate_rows))
            samples[index] = metric(candidate_rows[selected], name) - metric(
                baseline_rows[selected],
                name,
            )
        low, high = np.percentile(samples, [2.5, 97.5])
        result[name] = {
            "v25_minus_baseline": round(observed, 6),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "bootstrap_samples": n_bootstrap,
        }
        # Keep each metric's random stream deterministic but independent.
        rng = np.random.default_rng(seed + metric_index + 1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v25-report", type=Path, required=True)
    parser.add_argument("--anchor-report", type=Path, required=True)
    parser.add_argument("--direct-report", type=Path, required=True)
    parser.add_argument("--expected-qids", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "v25_full": args.v25_report,
        "v23_anchor": args.anchor_report,
        "v24_direct_indirect": args.direct_report,
    }
    summaries = {}
    records = {}
    for method, path in paths.items():
        summaries[method], records[method] = load_report(path)

    qid_sets = {method: set(values) for method, values in records.items()}
    if any(qid_set != qid_sets["v25_full"] for qid_set in qid_sets.values()):
        raise ValueError(
            "qid sets differ: "
            + ", ".join(f"{method}={len(qids)}" for method, qids in qid_sets.items())
        )
    qids = sorted(qid_sets["v25_full"])
    if len(qids) != args.expected_qids:
        raise ValueError(f"expected {args.expected_qids} qids, found {len(qids)}")
    validate_pairing(records, qids)

    rows = {
        method: {
            "all_states": np.asarray(
                [qid_totals(method_records[qid], later_only=False) for qid in qids]
            ),
            "hop2_plus": np.asarray(
                [qid_totals(method_records[qid], later_only=True) for qid in qids]
            ),
        }
        for method, method_records in records.items()
    }
    method_metrics = {
        method: {
            slice_name: summarize(slice_rows)
            for slice_name, slice_rows in method_rows.items()
        }
        for method, method_rows in rows.items()
    }

    comparisons = {}
    passed_against = []
    for baseline_index, baseline in enumerate(BASELINES):
        slice_comparisons = {}
        for slice_index, slice_name in enumerate(("all_states", "hop2_plus")):
            slice_comparisons[slice_name] = bootstrap_comparison(
                rows["v25_full"][slice_name],
                rows[baseline][slice_name],
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + baseline_index * 100 + slice_index * 10,
            )
        hop2 = slice_comparisons["hop2_plus"]
        all_states = slice_comparisons["all_states"]
        rank_improves = (
            hop2["step_at_1"]["ci95_low"] > 0
            or hop2["mrr"]["ci95_low"] > 0
        )
        coverage_non_regression = (
            all_states["full_unit_coverage"]["ci95_high"] >= 0
        )
        gate = {
            "significant_hop2_step1_or_mrr_improvement": bool(rank_improves),
            "no_clear_full_unit_regression": bool(coverage_non_regression),
            "passes": bool(rank_improves and coverage_non_regression),
        }
        if gate["passes"]:
            passed_against.append(baseline)
        comparisons[f"v25_minus_{baseline}"] = {
            "metrics": slice_comparisons,
            "gate": gate,
        }

    expected_protocol = {
        "v25_full": {
            "checkpoint": "outputs/ranker/deberta_v3_large_v25_rollout_aligned/best_model.pt",
            "policy_context_source": "online_state",
        },
        "v23_anchor": {
            "checkpoint": "outputs/ranker/deberta_v3_large_v23_acra_anchor/best_model.pt",
            "policy_context_source": "previous_evidence_only",
        },
        "v24_direct_indirect": {
            "checkpoint": "outputs/ranker/deberta_v3_large_v24_ecdr_direct_indirect/best_model.pt",
            "policy_context_source": "direct_evidence_only",
        },
    }
    protocol_failures = []
    protocol = {}
    for method, summary in summaries.items():
        protocol[method] = {
            "checkpoint": summary.get("checkpoint"),
            "policy_context_source": summary.get("policy_context_source"),
            "selector": summary.get("selector"),
            "candidate_top_k": summary.get("candidate_top_k"),
            "select_top_k": summary.get("select_top_k"),
            "state_update_top_k": summary.get("state_update_top_k"),
            "policy_blend_weight": summary.get("policy_blend_weight"),
            "answer_judged": summary.get("answer_judged"),
        }
        for key, expected in expected_protocol[method].items():
            if summary.get(key) != expected:
                protocol_failures.append(
                    f"{method} {key}: {summary.get(key)!r} != {expected!r}"
                )
        common_expected = {
            "selector": "hybrid_policy",
            "candidate_top_k": 10,
            "select_top_k": 5,
            "state_update_top_k": 1,
            "policy_blend_weight": 0.5,
            "answer_judged": 0,
        }
        for key, expected in common_expected.items():
            if summary.get(key) != expected:
                protocol_failures.append(
                    f"{method} {key}: {summary.get(key)!r} != {expected!r}"
                )

    overall_pass = bool(passed_against) and not protocol_failures
    result = {
        "status": "PASS" if overall_pass else "FAIL",
        "qids": len(qids),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "protocol": protocol,
        "protocol_failures": protocol_failures,
        "method_metrics": method_metrics,
        "comparisons": comparisons,
        "acceptance_gate": {
            "passes": overall_pass,
            "passed_against": passed_against,
            "rule": (
                "v25 must significantly improve hop2+ Step@1 or MRR over at "
                "least one chain baseline and must not significantly regress "
                "full-unit coverage against that baseline"
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
