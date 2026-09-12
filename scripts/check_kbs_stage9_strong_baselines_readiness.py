#!/usr/bin/env python3
"""Audit prerequisites for Stage 9.3 same-generator strong baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ANSWER_PROTOCOL = {
    "answer_model": "deepseek-v4-flash",
    "answer_thinking_mode": "disabled",
    "answer_mode": "json",
    "answer_temperature": 0.0,
    "answer_prompt_version": "kbs_extractive_answer_json_v1",
}
BASELINES = {
    "bm25": {
        "reported_name": "BM25-RAG",
        "selector": "bm25",
        "dense_model": "",
        "dense_query_mode": "question",
        "reranker_model": "",
        "candidate_top_k": 8,
        "historical_report": "outputs/rag/full3000_bm25.json",
    },
    "dense": {
        "reported_name": "Dense-RAG",
        "selector": "dense",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "reranker_model": "",
        "candidate_top_k": 8,
        "historical_report": "outputs/rag/full3000_dense.json",
    },
    "hybrid": {
        "reported_name": "Hybrid-RAG",
        "selector": "hybrid",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "reranker_model": "",
        "candidate_top_k": 8,
        "historical_report": "outputs/rag/full3000_hybrid.json",
    },
    "iterative_hybrid": {
        "reported_name": "Iterative-Hybrid-RAG",
        "selector": "iterative_hybrid",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "reranker_model": "",
        "candidate_top_k": 8,
        "historical_report": (
            "outputs/rag/missing_baselines/hotpot_iterative_hybrid.json"
        ),
    },
    "bge_reranker": {
        "reported_name": "BGE-Reranker-RAG",
        "selector": "generic_reranker",
        "dense_model": "",
        "dense_query_mode": "question",
        "reranker_model": "models/bge-reranker-large",
        "candidate_top_k": 8,
        "historical_report": "outputs/rag/bge_reranker_large_eval3000.json",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def protocol_differences(summary: dict[str, Any]) -> dict[str, Any]:
    expected = {
        **ANSWER_PROTOCOL,
        "qids": 3000,
        "answer_judged": 3000,
        "answer_errors": 0,
        "generate_answers": True,
    }
    return {
        key: {"observed": summary.get(key), "expected": value}
        for key, value in expected.items()
        if summary.get(key) != value
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_strong_baselines"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.output_root / "readiness.json"

    data_root = Path("data/hotpotqa_distractor_eval_3000_cand50")
    samples = data_root / "samples/test.jsonl"
    queries = data_root / "queries/test.jsonl"
    memory = data_root / "unit_registry/raw_units_test.jsonl"
    required = [
        Path("md/kbs_three_review_execution_plan.md"),
        Path("md/kbs_review_75_85_execution_plan.md"),
        Path("scripts/run_hotpotqa_policy_rag.py"),
        Path("scripts/evaluate_kbs_standard_metrics.py"),
        Path("scripts/bootstrap_kbs_stage4_metrics.py"),
        samples,
        queries,
        memory,
        Path("models/deberta-v3-large"),
        Path("models/bge-large-en-v1.5"),
        Path("models/bge-reranker-large"),
        Path("outputs/rag/kbs_v27_final_hotpot/full_compact.json"),
        Path("outputs/rag/kbs_v27_stage5_multiseed/seed42/full_recall.json"),
        Path("outputs/rag/full3000_gold_oracle.json"),
    ]
    failures = [f"missing required path: {path}" for path in required if not path.exists()]

    counts: dict[str, int | None] = {"queries": None, "samples": None, "memory": None}
    if not failures:
        try:
            counts = {
                "queries": count_jsonl(queries),
                "samples": count_jsonl(samples),
                "memory": count_jsonl(memory),
            }
            if counts["queries"] != 3000:
                failures.append(f"query rows={counts['queries']} != 3000")
            if counts["samples"] != 7296:
                failures.append(f"sample rows={counts['samples']} != 7296")
        except OSError as exc:
            failures.append(str(exc))

    runtime_text = ""
    runtime_path = Path("scripts/run_hotpotqa_policy_rag.py")
    if runtime_path.is_file():
        runtime_text = runtime_path.read_text(encoding="utf-8")
    implementation_audit: dict[str, Any] = {}
    historical_audit: dict[str, Any] = {}
    proposed_outputs: dict[str, str] = {}
    for key, spec in BASELINES.items():
        selector = str(spec["selector"])
        implementation_audit[key] = {
            "selector": selector,
            "implemented": f'args.selector == "{selector}"' in runtime_text,
        }
        if not implementation_audit[key]["implemented"]:
            failures.append(f"runtime does not implement selector={selector}")

        historical_path = Path(str(spec["historical_report"]))
        historical: dict[str, Any] = {
            "path": str(historical_path),
            "exists": historical_path.is_file(),
            "eligible_for_final_protocol_reuse": False,
            "protocol_differences": {},
        }
        if historical_path.is_file():
            try:
                obj = read_json(historical_path)
                summary = obj.get("summary")
                if not isinstance(summary, dict):
                    raise ValueError("missing summary")
                differences = protocol_differences(summary)
                historical["protocol_differences"] = differences
                historical["eligible_for_final_protocol_reuse"] = not differences
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                historical["read_error"] = str(exc)
        historical_audit[key] = historical

        proposed = args.output_root / "selection3000" / f"{key}.json"
        proposed_outputs[key] = str(proposed)
        if proposed.exists():
            failures.append(f"proposed selection output already exists: {proposed}")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.3",
        "mode": "same_generator_strong_baseline_readiness",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "data": {
            "root": str(data_root),
            "counts": counts,
        },
        "frozen_answer_protocol": ANSWER_PROTOCOL,
        "baseline_registry": BASELINES,
        "implementation_audit": implementation_audit,
        "historical_report_audit": historical_audit,
        "historical_policy": (
            "Historical reports are descriptive only. A report can seed an exact-"
            "context answer cache only when its full answer protocol matches."
        ),
        "proposed_selection_outputs": proposed_outputs,
        "next_gate": (
            "Run a five-baseline 20-qid selection-only smoke; do not call the API."
            if not failures
            else "Resolve readiness failures before GPU inference or API calls."
        ),
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
