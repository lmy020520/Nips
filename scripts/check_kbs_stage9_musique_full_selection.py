#!/usr/bin/env python3
"""Audit complete Stage 9.6 MuSiQue paragraph-level selection reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CHECKPOINT = "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
METHODS = {
    "compact_seed42": {
        "reported_name": "KSG-EA-Compact",
        "selector": "hybrid_policy",
        "candidate_top_k": 10,
        "state_update_top_k": 1,
    },
    "balanced_seed42": {
        "reported_name": "KSG-EA-Balanced",
        "selector": "hybrid_policy",
        "candidate_top_k": 15,
        "state_update_top_k": 1,
    },
    "hybrid": {
        "reported_name": "Hybrid-RAG",
        "selector": "hybrid",
        "candidate_top_k": 8,
        "state_update_top_k": 5,
    },
    "bge_reranker": {
        "reported_name": "BGE-Reranker-RAG",
        "selector": "generic_reranker",
        "candidate_top_k": 8,
        "state_update_top_k": 5,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def target_sequence(results: list[dict[str, Any]]) -> list[str]:
    values = []
    for row in results:
        qid = str(row.get("qid") or "")
        for step in row.get("steps") or []:
            values.append(
                f"{qid}\t{int(step.get('t', -1))}\t"
                f"{str(step.get('positive_unit_id') or '')}"
            )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/musique_ans_eval_1000_paragraph20"),
    )
    parser.add_argument("--expected-qids", type=int, default=1000)
    parser.add_argument(
        "--adapter-audit",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_breadth/adapter_readiness.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths: dict[str, Path] = {}
    for item in args.report:
        if "=" not in item:
            raise ValueError(f"--report must be name=path: {item}")
        name, raw_path = item.split("=", 1)
        if name not in METHODS or name in paths:
            raise ValueError(f"unknown or duplicate method: {name}")
        paths[name] = Path(raw_path)

    failures = []
    missing_methods = sorted(set(METHODS) - set(paths))
    if missing_methods:
        failures.append(f"missing method reports: {missing_methods}")

    samples_path = args.data_root / "samples/test.jsonl"
    queries_path = args.data_root / "queries/test.jsonl"
    memory_path = args.data_root / "unit_registry/raw_units_test.jsonl"
    for path in (samples_path, queries_path, memory_path, args.adapter_audit):
        if not path.is_file():
            failures.append(f"missing prerequisite: {path}")

    expected_qids: list[str] = []
    expected_targets: list[str] = []
    expected_steps = 0
    non_paragraph_memory = 0
    if not failures:
        adapter_audit = read_json(args.adapter_audit)
        if adapter_audit.get("status") != "OK" or adapter_audit.get("failures"):
            failures.append("adapter readiness is not a clean OK")

        queries = read_jsonl(queries_path)
        samples = read_jsonl(samples_path)
        memory = read_jsonl(memory_path)
        query_qids = {str(row.get("qid") or "") for row in queries}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in samples:
            qid = str(row.get("qid") or "")
            if qid:
                grouped[qid].append(row)
        expected_qids = sorted(grouped)[: args.expected_qids]
        if len(expected_qids) != args.expected_qids:
            failures.append(
                f"adapter qids: {len(expected_qids)} != {args.expected_qids}"
            )
        if not set(expected_qids).issubset(query_qids):
            failures.append("runtime qids are missing from the query file")
        for qid in expected_qids:
            rows = sorted(grouped[qid], key=lambda row: int(row.get("t", -1)))
            expected_steps += len(rows)
            for row in rows:
                positive = (
                    ((row.get("labels") or {}).get("ranking_label") or {}).get(
                        "positive_unit_id"
                    )
                    or ""
                )
                expected_targets.append(
                    f"{qid}\t{int(row.get('t', -1))}\t{str(positive)}"
                )
        non_paragraph_memory = sum(
            1 for row in memory if row.get("candidate_granularity") != "paragraph"
        )
        if non_paragraph_memory:
            failures.append(f"non-paragraph memory rows: {non_paragraph_memory}")

    method_metrics: dict[str, Any] = {}
    qid_hashes: dict[str, str] = {}
    target_hashes: dict[str, str] = {}
    for name, spec in METHODS.items():
        path = paths.get(name)
        if path is None:
            continue
        try:
            report = read_json(path)
            summary = report.get("summary")
            results = report.get("results")
            if not isinstance(summary, dict) or not isinstance(results, list):
                raise ValueError("report must contain summary and results")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{name}: {exc}")
            continue

        expected = {
            "samples": str(samples_path),
            "memory": str(memory_path),
            "queries": str(queries_path),
            "checkpoint": CHECKPOINT,
            "state_mode": "policy",
            "policy_context_source": "online_state",
            "selector": spec["selector"],
            "hybrid_alpha": 0.5,
            "front_pool_k": 30,
            "candidate_top_k": spec["candidate_top_k"],
            "select_top_k": 5,
            "state_update_top_k": spec["state_update_top_k"],
            "answer_mode": "json",
            "generate_answers": False,
            "qids": args.expected_qids,
            "steps": expected_steps,
            "skipped": 0,
            "answer_judged": 0,
            "answer_errors": 0,
        }
        if spec["selector"] == "hybrid_policy":
            expected.update(
                {
                    "dense_model": "models/bge-large-en-v1.5",
                    "dense_query_mode": "state",
                    "front_fusion": "rrf",
                    "local_expansion_window": 1,
                    "mmr_lambda": 0.7,
                    "mmr_same_doc_similarity": 0.35,
                    "policy_score_mode": "front_policy_blend",
                    "policy_blend_weight": 0.5,
                    "save_online_states": True,
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

        observed_qids = [str(row.get("qid") or "") for row in results]
        observed_targets = target_sequence(results)
        answer_output_violations = sum(
            1
            for row in results
            if row.get("answer") or row.get("raw_answer") or row.get("answer_tokens")
        )
        missing_online_states = 0
        if spec["selector"] == "hybrid_policy":
            missing_online_states = sum(
                1
                for row in results
                for step in row.get("steps") or []
                if not isinstance(step.get("online_state_before"), dict)
                or not isinstance(step.get("online_state_after"), dict)
            )
        if observed_qids != expected_qids:
            failures.append(f"{name}: ordered qids differ from the adapter")
        if observed_targets != expected_targets:
            failures.append(f"{name}: ordered paragraph targets differ from the adapter")
        if len(set(observed_qids)) != len(observed_qids):
            failures.append(f"{name}: duplicate result qids")
        if answer_output_violations:
            failures.append(
                f"{name}: selection-only rows contain answer output: "
                f"{answer_output_violations}"
            )
        if missing_online_states:
            failures.append(
                f"{name}: steps missing saved online states: {missing_online_states}"
            )
        qid_hashes[name] = digest(observed_qids)
        target_hashes[name] = digest(observed_targets)

        runtime = summary.get("runtime_profile")
        if not isinstance(runtime, dict):
            failures.append(f"{name}: missing runtime profile")
            runtime = {}
        for field in (
            "selection_avg_ms_per_qid",
            "selection_throughput_qids_per_second",
            "peak_gpu_allocated_mb",
            "peak_gpu_reserved_mb",
        ):
            if not isinstance(runtime.get(field), (int, float)):
                failures.append(f"{name}: missing runtime metric {field}")

        method_metrics[name] = {
            "reported_name": spec["reported_name"],
            "qids": summary.get("qids"),
            "steps": summary.get("steps"),
            "paragraph_alignment_at_1": summary.get("step_acc@1"),
            "paragraph_alignment_at_5": summary.get("step_acc@5"),
            "full_support_title_coverage": summary.get("full_gold_doc_coverage"),
            "full_support_paragraph_coverage": summary.get("full_gold_unit_coverage"),
            "selection_ms_per_qid": runtime.get("selection_avg_ms_per_qid"),
            "selection_throughput_qids_per_second": runtime.get(
                "selection_throughput_qids_per_second"
            ),
            "peak_gpu_allocated_mb": runtime.get("peak_gpu_allocated_mb"),
            "peak_gpu_reserved_mb": runtime.get("peak_gpu_reserved_mb"),
            "answer_output_violations": answer_output_violations,
            "missing_online_state_steps": missing_online_states,
        }

    if len(set(qid_hashes.values())) > 1:
        failures.append("ordered qid hashes differ across methods")
    if len(set(target_hashes.values())) > 1:
        failures.append("ordered paragraph-target hashes differ across methods")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.6",
        "mode": "musique_zero_shot_selection",
        "api_calls": 0,
        "qids": args.expected_qids,
        "steps": expected_steps,
        "protocol": {
            "dataset": "fixed MuSiQue-Ans dev subset",
            "student_training": "HotpotQA only; no MuSiQue fine-tuning",
            "unit_granularity": "paragraph",
            "primary_operating_point": "Compact-10",
            "secondary_operating_point": "Balanced-15",
            "baselines": ["Hybrid-RAG", "BGE-Reranker-RAG"],
            "recall_50_supported": False,
            "answers_generated": False,
        },
        "ordered_qids_sha256": qid_hashes,
        "ordered_paragraph_targets_sha256": target_hashes,
        "method_metrics": method_metrics,
        "audit": {
            "expected_steps": expected_steps,
            "non_paragraph_memory_rows": non_paragraph_memory,
        },
        "metric_boundary": (
            "All evidence metrics are paragraph-level and are not directly "
            "identical to sentence-level HotpotQA/2Wiki Step@k metrics."
        ),
        "next_gate": (
            "Review complete selection results before any answer-cache preparation."
            if not failures
            else "Resolve full-selection failures before any answer generation."
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
