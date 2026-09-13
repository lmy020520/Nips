#!/usr/bin/env python3
"""Audit Stage 9.5 matched downstream state-rollout selection reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CONDITIONS = {
    "online_state": "online_state",
    "query_only": "query_only",
    "frozen_initial_state": "frozen_initial_state",
    "other_question_state": "other_question_state",
    "previous_evidence_only": "previous_evidence_only",
}
METRICS = (
    "step_at_1",
    "step_at_5",
    "mrr",
    "full_unit_coverage",
    "full_doc_coverage",
    "top1_acquired_reselection_rate",
    "top5_acquired_slot_rate",
)


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--report must use NAME=PATH")
    name, path = value.split("=", 1)
    if name not in CONDITIONS:
        raise argparse.ArgumentTypeError(f"unknown condition: {name}")
    return name, Path(path)


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    results = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report must contain summary/results: {path}")
    return summary, results


def target_sequence(results: list[dict[str, Any]]) -> list[str]:
    return [
        f"{record.get('qid')}\t{int(step.get('t') or 0)}\t{step.get('positive_unit_id')}"
        for record in results
        for step in record.get("steps") or []
    ]


def qid_totals(record: dict[str, Any]) -> np.ndarray:
    step1 = step5 = top1_reselected = top5_reselected = top5_slots = 0
    reciprocal_rank = 0.0
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
        acquired.update(str(value) for value in step.get("state_update_unit_ids") or [])
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


def selection_metrics(rows: np.ndarray) -> dict[str, float | int]:
    return {
        "qids": len(rows),
        "steps": int(rows[:, 7].sum()),
        **{name: round(metric(rows, name), 6) for name in METRICS},
    }


def paired_bootstrap(
    reference: np.ndarray,
    comparison: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    output = {}
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
            "online_state_minus_comparison": round(observed, 6),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "bootstrap_samples": n_bootstrap,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=named_path, required=True)
    parser.add_argument("--expected-qids", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = dict(args.report)
    failures = []
    if set(paths) != set(CONDITIONS):
        failures.append(f"reports must contain exactly {sorted(CONDITIONS)}")

    method_metrics = {}
    qid_hashes = {}
    target_hashes = {}
    method_rows = {}
    common_expected = {
        "samples": "data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl",
        "memory": "data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl",
        "queries": "data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl",
        "checkpoint": "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
        "state_mode": "policy",
        "selector": "hybrid_policy",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "hybrid_alpha": 0.5,
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
        "save_online_states": True,
        "qids": args.expected_qids,
        "skipped": 0,
    }

    for name in CONDITIONS:
        path = paths.get(name)
        if path is None:
            continue
        try:
            summary, results = read_report(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{name}: {exc}")
            continue
        expected = {**common_expected, "policy_context_source": CONDITIONS[name]}
        if name == "other_question_state":
            expected.update(
                {
                    "external_policy_state_pairing": "cyclic_next_qid",
                    "external_policy_state_report": str(paths["online_state"]),
                }
            )
        for key, value in expected.items():
            if summary.get(key) != value:
                failures.append(f"{name} {key}: {summary.get(key)!r} != {value!r}")

        qids = [str(record.get("qid") or "") for record in results]
        if len(qids) != args.expected_qids or len(set(qids)) != len(qids):
            failures.append(
                f"{name}: expected {args.expected_qids} unique qids, "
                f"found {len(qids)} rows/{len(set(qids))} unique"
            )
        for record in results:
            qid = str(record.get("qid") or "")
            for step in record.get("steps") or []:
                if not isinstance(step.get("online_state_before"), dict):
                    failures.append(f"{name}: missing online_state_before for qid={qid}")
                    break
                metadata = step.get("context_state_metadata") or {}
                if name == "frozen_initial_state" and (
                    metadata.get("source_qid") != qid
                    or metadata.get("source_t") != 0
                    or metadata.get("frozen") is not True
                ):
                    failures.append(f"{name}: invalid frozen metadata for qid={qid}")
                    break
                if name == "other_question_state" and (
                    not metadata.get("source_qid")
                    or metadata.get("source_qid") == qid
                    or int(metadata.get("source_t", -1)) > int(step.get("t") or 0)
                    or metadata.get("pairing") != "cyclic_next_qid"
                ):
                    failures.append(f"{name}: invalid other-question metadata for qid={qid}")
                    break
        qid_hashes[name] = digest(qids)
        target_hashes[name] = digest(target_sequence(results))
        try:
            rows = np.stack([qid_totals(record) for record in results])
            method_rows[name] = rows
            method_metrics[name] = selection_metrics(rows)
        except ValueError as exc:
            failures.append(f"{name}: {exc}")

    if qid_hashes and len(set(qid_hashes.values())) != 1:
        failures.append("ordered qid hashes differ across conditions")
    if target_hashes and len(set(target_hashes.values())) != 1:
        failures.append("ordered teacher-target hashes differ across conditions")
    step_counts = {metrics["steps"] for metrics in method_metrics.values()}
    if len(step_counts) > 1:
        failures.append("evaluated step counts differ across conditions")

    paired_deltas = {}
    if not failures and "online_state" in method_rows:
        for offset, name in enumerate(CONDITIONS):
            if name == "online_state":
                continue
            paired_deltas[f"online_state-minus-{name}"] = paired_bootstrap(
                method_rows["online_state"],
                method_rows[name],
                args.n_bootstrap,
                args.seed + 100 * offset,
            )

    output = {
        "status": "SMOKE_OK" if args.smoke and not failures else (
            "OK" if not failures else "FAIL"
        ),
        "stage": 9,
        "step": "9.5",
        "mode": (
            "downstream_state_rollout_selection_smoke"
            if args.smoke
            else "downstream_state_rollout_selection"
        ),
        "api_calls": 0,
        "qids": args.expected_qids,
        "protocol": {
            "operating_point": "Compact",
            "candidate_top_k": 10,
            "front_pool_k": 30,
            "select_top_k": 5,
            "state_update_top_k": 1,
            "policy_blend_weight": 0.5,
            "answers_generated": False,
            "other_question_pairing": "cyclic_next_qid",
        },
        "ordered_qids_sha256": qid_hashes,
        "ordered_teacher_targets_sha256": target_hashes,
        "method_metrics": method_metrics,
        "paired_deltas": paired_deltas,
        "interpretation": (
            "Smoke metrics validate execution only and are not scientific results."
            if args.smoke
            else "Selection results precede the exact-context answer-cache gate."
        ),
        "next_gate": (
            "Run five complete 3,000-qid selection-only rollouts."
            if args.smoke and not failures
            else "Prepare exact-context answer caches without API calls."
            if not failures
            else "Resolve failures; do not start full rollouts or answer generation."
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
