#!/usr/bin/env python3
"""Compare the v26 clue-state policy with matched state/chain baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("v26_clue", "v22_full", "v23_anchor", "v24_direct_indirect")
BASELINES = METHODS[1:]
METRICS = ("step_at_1", "step_at_5", "mrr", "full_unit_coverage")


def load_report(path: Path) -> tuple[dict, dict[str, dict]]:
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


def qid_totals(record: dict, *, later_only: bool) -> np.ndarray:
    steps = [
        step for step in record.get("steps") or []
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


def summarize(rows: np.ndarray) -> dict:
    return {name: round(metric(rows, name), 6) for name in METRICS}


def validate_pairing(records: dict[str, dict[str, dict]], qids: list[str]) -> None:
    reference = records["v26_clue"]
    for method in BASELINES:
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


def audit_clue_transitions(records: dict[str, dict]) -> dict:
    failures = []
    steps = 0
    monotonic_pairs = 0
    linked_pairs = 0
    for qid, record in records.items():
        previous_after = None
        question_hash = None
        for step in record.get("steps") or []:
            steps += 1
            before = step.get("clue_state_before")
            after = step.get("clue_state_after")
            if not isinstance(before, dict) or not isinstance(after, dict):
                failures.append(f"missing clue state: qid={qid}, t={step.get('t')}")
                continue
            before_vector = list(before.get("coverage_vector") or [])
            after_vector = list(after.get("coverage_vector") or [])
            if len(before_vector) != len(after_vector) or any(
                right < left for left, right in zip(before_vector, after_vector)
            ):
                failures.append(f"non-monotonic clue state: qid={qid}, t={step.get('t')}")
            else:
                monotonic_pairs += 1
            if previous_after is not None:
                if before != previous_after:
                    failures.append(f"unlinked clue transition: qid={qid}, t={step.get('t')}")
                else:
                    linked_pairs += 1
            current_hash = before.get("question_sha256")
            if question_hash is None:
                question_hash = current_hash
                if any(before_vector):
                    failures.append(f"non-empty initial clue coverage: qid={qid}")
            elif current_hash != question_hash:
                failures.append(f"question hash changed: qid={qid}, t={step.get('t')}")
            previous_after = after
        if record.get("final_clue_state") != previous_after:
            failures.append(f"final clue state mismatch: qid={qid}")
    return {
        "steps": steps,
        "monotonic_pairs": monotonic_pairs,
        "linked_pairs": linked_pairs,
        "failure_count": len(failures),
        "failures": failures[:30],
    }


def bootstrap_comparison(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    result = {}
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
        result[name] = {
            "clue_minus_baseline": round(observed, 6),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "bootstrap_samples": n_bootstrap,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clue-report", type=Path, required=True)
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--anchor-report", type=Path, required=True)
    parser.add_argument("--direct-report", type=Path, required=True)
    parser.add_argument("--expected-qids", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "v26_clue": args.clue_report,
        "v22_full": args.full_report,
        "v23_anchor": args.anchor_report,
        "v24_direct_indirect": args.direct_report,
    }
    summaries, records = {}, {}
    for method, path in paths.items():
        summaries[method], records[method] = load_report(path)

    qid_sets = {method: set(rows) for method, rows in records.items()}
    if any(qids != qid_sets["v26_clue"] for qids in qid_sets.values()):
        raise ValueError(
            "qid sets differ: "
            + ", ".join(f"{method}={len(qids)}" for method, qids in qid_sets.items())
        )
    qids = sorted(qid_sets["v26_clue"])
    if len(qids) != args.expected_qids:
        raise ValueError(f"expected {args.expected_qids} qids, found {len(qids)}")
    validate_pairing(records, qids)

    rows = {
        method: {
            "all_states": np.asarray(
                [qid_totals(method_rows[qid], later_only=False) for qid in qids]
            ),
            "hop2_plus": np.asarray(
                [qid_totals(method_rows[qid], later_only=True) for qid in qids]
            ),
        }
        for method, method_rows in records.items()
    }
    method_metrics = {
        method: {slice_name: summarize(values) for slice_name, values in slices.items()}
        for method, slices in rows.items()
    }

    comparisons = {}
    competitive_against = []
    for baseline_index, baseline in enumerate(BASELINES):
        slices = {
            slice_name: bootstrap_comparison(
                rows["v26_clue"][slice_name],
                rows[baseline][slice_name],
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + baseline_index * 100 + slice_index * 10,
            )
            for slice_index, slice_name in enumerate(("all_states", "hop2_plus"))
        }
        hop2 = slices["hop2_plus"]
        all_states = slices["all_states"]
        rank_non_regression = (
            hop2["step_at_1"]["ci95_high"] >= 0
            or hop2["mrr"]["ci95_high"] >= 0
        )
        coverage_non_regression = (
            all_states["full_unit_coverage"]["ci95_high"] >= 0
        )
        competitive = bool(rank_non_regression and coverage_non_regression)
        if competitive:
            competitive_against.append(baseline)
        comparisons[f"v26_clue_minus_{baseline}"] = {
            "metrics": slices,
            "gate": {
                "hop2_rank_non_regression": bool(rank_non_regression),
                "full_unit_non_regression": bool(coverage_non_regression),
                "competitive": competitive,
            },
        }

    expected_protocol = {
        "v26_clue": (
            "outputs/ranker/deberta_v3_large_v26_fiske_clue_state/best_model.pt",
            "clue_state",
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
    protocol_failures = []
    protocol = {}
    for method, summary in summaries.items():
        protocol[method] = {
            key: summary.get(key)
            for key in (
                "checkpoint", "policy_context_source", "clue_state_version",
                "selector", "candidate_top_k", "select_top_k",
                "state_update_top_k", "policy_blend_weight", "answer_judged",
                "save_online_states",
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
        if method == "v26_clue":
            expected.update(
                clue_state_version="fiske_inspired_textual_clues_v1",
                save_online_states=True,
            )
        for key, value in expected.items():
            if summary.get(key) != value:
                protocol_failures.append(
                    f"{method} {key}: {summary.get(key)!r} != {value!r}"
                )

    clue_audit = audit_clue_transitions(records["v26_clue"])
    if clue_audit["failure_count"]:
        protocol_failures.append(
            f"clue transition audit: {clue_audit['failure_count']} failures"
        )

    result = {
        "status": "OK" if not protocol_failures else "FAIL",
        "qids": len(qids),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "protocol": protocol,
        "protocol_failures": protocol_failures,
        "clue_transition_audit": clue_audit,
        "method_metrics": method_metrics,
        "comparisons": comparisons,
        "validation_decision": {
            "competitive_against": competitive_against,
            "answer_api_eligible": bool(competitive_against) and not protocol_failures,
            "rule": (
                "The clue baseline must avoid significant hop2 rank and full-unit "
                "regression against at least one controlled baseline before any "
                "answer-API evaluation."
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
