#!/usr/bin/env python3
"""Audit Stage 9.4 final-policy 2Wiki selection reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METHODS = {
    "compact_seed42": {
        "reported_name": "KSG-EA-Compact",
        "selector": "hybrid_policy",
        "checkpoint": "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
        "candidate_top_k": 10,
        "front_pool_k": 30,
        "state_update_top_k": 1,
    },
    "balanced_knee_seed42": {
        "reported_name": "KSG-EA-Balanced",
        "selector": "hybrid_policy",
        "checkpoint": "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
        "candidate_top_k": 15,
        "front_pool_k": 30,
        "state_update_top_k": 1,
    },
    "recall_seed42": {
        "reported_name": "KSG-EA-Recall",
        "selector": "hybrid_policy",
        "checkpoint": "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
        "candidate_top_k": 50,
        "front_pool_k": 50,
        "state_update_top_k": 1,
    },
    "hybrid": {
        "reported_name": "Hybrid-RAG",
        "selector": "hybrid",
        "checkpoint": "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
        "candidate_top_k": 8,
        "front_pool_k": 30,
        "state_update_top_k": 5,
    },
    "bge_reranker": {
        "reported_name": "BGE-Reranker-RAG",
        "selector": "generic_reranker",
        "checkpoint": "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
        "candidate_top_k": 8,
        "front_pool_k": 30,
        "state_update_top_k": 5,
    },
}


def read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    results = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report must contain summary/results: {path}")
    return summary, results


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def target_sequence(results: list[dict[str, Any]]) -> list[str]:
    values = []
    for row in results:
        qid = str(row.get("qid") or "")
        for step in row.get("steps") or []:
            values.append(
                f"{qid}\t{int(step.get('t') or 0)}\t"
                f"{str(step.get('positive_unit_id') or '')}"
            )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--expected-qids", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {}
    for item in args.report:
        if "=" not in item:
            raise ValueError(f"--report must be name=path: {item}")
        name, raw_path = item.split("=", 1)
        if name not in METHODS or name in paths:
            raise ValueError(f"unknown or duplicate method: {name}")
        paths[name] = Path(raw_path)

    failures = []
    missing = sorted(set(METHODS) - set(paths))
    if missing:
        failures.append(f"missing method reports: {missing}")
    metrics: dict[str, Any] = {}
    qid_hashes = {}
    target_hashes = {}
    data_root = "data/2wiki_multihopqa_eval_1000_cand50"

    for name, spec in METHODS.items():
        path = paths.get(name)
        if path is None:
            continue
        try:
            summary, results = read_report(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{name}: {exc}")
            continue

        expected = {
            "samples": f"{data_root}/samples/test.jsonl",
            "memory": f"{data_root}/unit_registry/raw_units_test.jsonl",
            "queries": f"{data_root}/queries/test.jsonl",
            "checkpoint": spec["checkpoint"],
            "state_mode": "policy",
            "policy_context_source": "online_state",
            "selector": spec["selector"],
            "hybrid_alpha": 0.5,
            "front_pool_k": spec["front_pool_k"],
            "candidate_top_k": spec["candidate_top_k"],
            "select_top_k": 5,
            "state_update_top_k": spec["state_update_top_k"],
            "generate_answers": False,
            "answer_judged": 0,
            "answer_errors": 0,
            "qids": args.expected_qids,
            "skipped": 0,
        }
        if spec["selector"] == "hybrid_policy":
            expected.update(
                {
                    "dense_model": "models/bge-large-en-v1.5",
                    "dense_query_mode": "state",
                    "front_fusion": "rrf",
                    "local_expansion_window": 1,
                    "policy_score_mode": "front_policy_blend",
                    "policy_blend_weight": 0.5,
                }
            )
        elif spec["selector"] == "hybrid":
            expected.update(
                {
                    "dense_model": "models/bge-large-en-v1.5",
                    "dense_query_mode": "state",
                    "reranker_model": "",
                }
            )
        else:
            expected.update(
                {
                    "dense_model": "",
                    "dense_query_mode": "question",
                    "reranker_model": "models/bge-reranker-large",
                }
            )
        for key, value in expected.items():
            if summary.get(key) != value:
                failures.append(f"{name} {key}: {summary.get(key)!r} != {value!r}")

        qids = [str(row.get("qid") or "") for row in results]
        if len(qids) != args.expected_qids or len(set(qids)) != len(qids):
            failures.append(
                f"{name}: expected {args.expected_qids} unique qids, found "
                f"{len(qids)} rows/{len(set(qids))} unique"
            )
        qid_hashes[name] = digest(qids)
        target_hashes[name] = digest(target_sequence(results))
        metrics[name] = {
            "reported_name": spec["reported_name"],
            "qids": summary.get("qids"),
            "steps": summary.get("steps"),
            "step_acc@1": summary.get("step_acc@1"),
            "step_acc@5": summary.get("step_acc@5"),
            "full_gold_doc_coverage": summary.get("full_gold_doc_coverage"),
            "full_gold_unit_coverage": summary.get("full_gold_unit_coverage"),
        }

    if len(set(qid_hashes.values())) > 1:
        failures.append("ordered qid hashes differ across methods")
    if len(set(target_hashes.values())) > 1:
        failures.append("ordered teacher-target hashes differ across methods")

    is_smoke = args.smoke
    result = {
        "status": "SMOKE_OK" if is_smoke and not failures else (
            "OK" if not failures else "FAIL"
        ),
        "stage": 9,
        "step": "9.4",
        "mode": "final_policy_2wiki_selection_smoke" if is_smoke else (
            "final_policy_2wiki_selection"
        ),
        "api_calls": 0,
        "qids": args.expected_qids,
        "ordered_qids_sha256": qid_hashes,
        "ordered_teacher_targets_sha256": target_hashes,
        "method_metrics": metrics,
        "interpretation": (
            "Smoke metrics validate execution only and are not scientific results."
            if is_smoke
            else "Full selection results precede answer-cache authorization."
        ),
        "next_gate": (
            "Resolve failures before proceeding."
            if failures
            else (
                "Run all registered 1,000-qid selection-only reports; do not call the answer API."
                if is_smoke
                else "Review multiseed robustness before preparing exact-context answer caches."
            )
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
