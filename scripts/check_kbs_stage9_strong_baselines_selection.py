#!/usr/bin/env python3
"""Audit Stage 9.3 same-protocol strong-baseline selection reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BASELINES = {
    "bm25": {
        "reported_name": "BM25-RAG",
        "selector": "bm25",
        "dense_model": "",
        "dense_query_mode": "question",
        "reranker_model": "",
    },
    "dense": {
        "reported_name": "Dense-RAG",
        "selector": "dense",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "reranker_model": "",
    },
    "hybrid": {
        "reported_name": "Hybrid-RAG",
        "selector": "hybrid",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "reranker_model": "",
    },
    "iterative_hybrid": {
        "reported_name": "Iterative-Hybrid-RAG",
        "selector": "iterative_hybrid",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "reranker_model": "",
    },
    "bge_reranker": {
        "reported_name": "BGE-Reranker-RAG",
        "selector": "generic_reranker",
        "dense_model": "",
        "dense_query_mode": "question",
        "reranker_model": "models/bge-reranker-large",
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
    values: list[str] = []
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-qids", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths: dict[str, Path] = {}
    for item in args.report:
        if "=" not in item:
            raise ValueError(f"--report must be name=path: {item}")
        name, raw_path = item.split("=", 1)
        if name not in BASELINES or name in paths:
            raise ValueError(f"unknown or duplicate baseline: {name}")
        paths[name] = Path(raw_path)

    failures: list[str] = []
    missing = sorted(set(BASELINES) - set(paths))
    if missing:
        failures.append(f"missing baseline reports: {missing}")

    metrics: dict[str, Any] = {}
    qid_hashes: dict[str, str] = {}
    target_hashes: dict[str, str] = {}
    expected_common = {
        "samples": "data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl",
        "memory": "data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl",
        "queries": "data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl",
        "checkpoint": args.checkpoint,
        "state_mode": "policy",
        "policy_context_source": "online_state",
        "hybrid_alpha": 0.5,
        "candidate_top_k": 8,
        "select_top_k": 5,
        "state_update_top_k": 5,
        "generate_answers": False,
        "answer_judged": 0,
        "answer_errors": 0,
        "qids": args.expected_qids,
        "skipped": 0,
    }

    for name, spec in BASELINES.items():
        path = paths.get(name)
        if path is None:
            continue
        try:
            summary, results = read_report(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{name}: {exc}")
            continue

        expected = {
            **expected_common,
            "selector": spec["selector"],
            "dense_model": spec["dense_model"],
            "dense_query_mode": spec["dense_query_mode"],
            "reranker_model": spec["reranker_model"],
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                failures.append(
                    f"{name} {key}: {summary.get(key)!r} != {value!r}"
                )

        qids = [str(row.get("qid") or "") for row in results]
        if len(qids) != args.expected_qids or len(set(qids)) != len(qids):
            failures.append(
                f"{name}: expected {args.expected_qids} unique qids, found "
                f"{len(qids)} rows/{len(set(qids))} unique"
            )
        qid_hashes[name] = digest(qids)
        target_hashes[name] = digest(target_sequence(results))
        runtime = summary.get("runtime_profile")
        if not isinstance(runtime, dict):
            runtime = {}
        metrics[name] = {
            "reported_name": spec["reported_name"],
            "qids": summary.get("qids"),
            "steps": summary.get("steps"),
            "step_acc@1": summary.get("step_acc@1"),
            "step_acc@5": summary.get("step_acc@5"),
            "full_gold_doc_coverage": summary.get("full_gold_doc_coverage"),
            "full_gold_unit_coverage": summary.get("full_gold_unit_coverage"),
            "selection_ms_per_qid": runtime.get("selection_avg_ms_per_qid"),
            "selection_throughput_qids_per_second": runtime.get(
                "selection_throughput_qids_per_second"
            ),
            "peak_gpu_allocated_mb": runtime.get("peak_gpu_allocated_mb"),
            "peak_gpu_reserved_mb": runtime.get("peak_gpu_reserved_mb"),
        }

    if len(set(qid_hashes.values())) > 1:
        failures.append("ordered qid hashes differ across baselines")
    if len(set(target_hashes.values())) > 1:
        failures.append("ordered teacher-target hashes differ across baselines")

    result = {
        "status": "SMOKE_OK" if args.smoke and not failures else (
            "OK" if not failures else "FAIL"
        ),
        "stage": 9,
        "step": "9.3",
        "mode": "strong_baseline_selection_smoke" if args.smoke else (
            "strong_baseline_selection"
        ),
        "api_calls": 0,
        "qids": args.expected_qids,
        "protocol": {
            "candidate_top_k": 8,
            "select_top_k": 5,
            "state_update_top_k": 5,
            "hybrid_alpha": 0.5,
            "answers_generated": False,
            "checkpoint_argument": args.checkpoint,
        },
        "ordered_qids_sha256": qid_hashes,
        "ordered_teacher_targets_sha256": target_hashes,
        "method_metrics": metrics,
        "interpretation": (
            "Smoke metrics validate execution only and are not scientific results."
            if args.smoke
            else "Full-run selection metrics precede the answer-cache gate."
        ),
        "next_gate": (
            "Resolve failures before proceeding."
            if failures
            else (
                "Run five 3,000-qid selection-only reports; do not call the answer API."
                if args.smoke
                else "Prepare and review exact-context answer caches; do not call the API."
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
