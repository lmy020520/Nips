#!/usr/bin/env python3
"""Audit the bounded Stage 9.6 MuSiQue Compact selection smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CHECKPOINT = "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/musique_ans_eval_1000_paragraph20"),
    )
    parser.add_argument("--expected-qids", type=int, default=20)
    parser.add_argument(
        "--adapter-audit",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_breadth/adapter_readiness.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    report = read_json(args.report)
    summary = report.get("summary")
    results = report.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError("selection report must contain summary and results")

    samples_path = args.data_root / "samples/test.jsonl"
    queries_path = args.data_root / "queries/test.jsonl"
    memory_path = args.data_root / "unit_registry/raw_units_test.jsonl"
    adapter_audit_path = args.adapter_audit
    for path in (samples_path, queries_path, memory_path, adapter_audit_path):
        if not path.is_file():
            failures.append(f"missing smoke prerequisite: {path}")

    expected_qids: list[str] = []
    expected_targets: list[str] = []
    expected_steps = 0
    non_paragraph_memory = 0
    if not failures:
        adapter_audit = read_json(adapter_audit_path)
        if adapter_audit.get("status") != "OK" or adapter_audit.get("failures"):
            failures.append("adapter readiness is not a clean OK")

        queries = read_jsonl(queries_path)
        samples = read_jsonl(samples_path)
        memory = read_jsonl(memory_path)
        query_qids = {str(row.get("qid") or "") for row in queries}
        # Match the runtime exactly: it groups sample rows, sorts qids, and only
        # then applies --max-qids. Query-file order is not used for selection.
        runtime_qids = sorted(
            {str(row.get("qid") or "") for row in samples if row.get("qid")}
        )
        expected_qids = runtime_qids[: args.expected_qids]
        qid_set = set(expected_qids)
        if not qid_set.issubset(query_qids):
            failures.append("runtime smoke qids are missing from the query file")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in samples:
            qid = str(row.get("qid") or "")
            if qid in qid_set:
                grouped[qid].append(row)
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

    expected_summary = {
        "samples": str(samples_path),
        "memory": str(memory_path),
        "queries": str(queries_path),
        "checkpoint": CHECKPOINT,
        "state_mode": "policy",
        "policy_context_source": "online_state",
        "selector": "hybrid_policy",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "hybrid_alpha": 0.5,
        "front_pool_k": 30,
        "front_fusion": "rrf",
        "local_expansion_window": 1,
        "mmr_lambda": 0.7,
        "mmr_same_doc_similarity": 0.35,
        "reranker_model": "",
        "candidate_top_k": 10,
        "select_top_k": 5,
        "state_update_top_k": 1,
        "policy_score_mode": "front_policy_blend",
        "policy_blend_weight": 0.5,
        "answer_mode": "json",
        "generate_answers": False,
        "save_online_states": True,
        "seed": 20260608,
        "qids": args.expected_qids,
        "steps": expected_steps,
        "skipped": 0,
        "answer_judged": 0,
        "answer_errors": 0,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            failures.append(f"summary {key}: {summary.get(key)!r} != {value!r}")

    observed_qids = [str(row.get("qid") or "") for row in results]
    if observed_qids != expected_qids:
        failures.append("ordered result qids differ from the fixed adapter prefix")
    if len(set(observed_qids)) != len(observed_qids):
        failures.append("duplicate result qids")

    observed_targets = []
    observed_steps = 0
    empty_answer_violations = 0
    missing_online_states = 0
    for row in results:
        qid = str(row.get("qid") or "")
        if row.get("answer") or row.get("raw_answer") or row.get("answer_tokens"):
            empty_answer_violations += 1
        for step in row.get("steps") or []:
            observed_steps += 1
            observed_targets.append(
                f"{qid}\t{int(step.get('t', -1))}\t"
                f"{str(step.get('positive_unit_id') or '')}"
            )
            if not isinstance(step.get("online_state_before"), dict) or not isinstance(
                step.get("online_state_after"), dict
            ):
                missing_online_states += 1
    if observed_targets != expected_targets:
        failures.append("ordered teacher targets differ from the fixed adapter prefix")
    if observed_steps != expected_steps:
        failures.append(f"observed steps: {observed_steps} != expected {expected_steps}")
    if empty_answer_violations:
        failures.append(f"selection-only rows contain answer output: {empty_answer_violations}")
    if missing_online_states:
        failures.append(f"steps missing saved online states: {missing_online_states}")

    result = {
        "status": "SMOKE_OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.6",
        "mode": "musique_compact_selection_smoke",
        "api_calls": 0,
        "qids": args.expected_qids,
        "steps": observed_steps,
        "protocol": {
            "dataset": "fixed MuSiQue-Ans dev subset",
            "unit_granularity": "paragraph",
            "checkpoint": CHECKPOINT,
            "operating_point": "Compact",
            "candidate_top_k": 10,
            "front_pool_k": 30,
            "select_top_k": 5,
            "state_update_top_k": 1,
            "policy_blend_weight": 0.5,
            "answers_generated": False,
        },
        "ordered_qids_sha256": digest(observed_qids),
        "ordered_paragraph_targets_sha256": digest(observed_targets),
        "metrics": {
            "paragraph_alignment_at_1": summary.get("step_acc@1"),
            "paragraph_alignment_at_5": summary.get("step_acc@5"),
            "full_support_paragraph_coverage": summary.get("full_gold_unit_coverage"),
            "full_support_title_coverage": summary.get("full_gold_doc_coverage"),
        },
        "metric_boundary": (
            "Paragraph-level evidence metrics; not directly identical to the "
            "sentence-level HotpotQA/2Wiki Step@k and unit coverage metrics."
        ),
        "audit": {
            "expected_steps": expected_steps,
            "non_paragraph_memory_rows": non_paragraph_memory,
            "answer_output_violations": empty_answer_violations,
            "missing_online_state_steps": missing_online_states,
        },
        "interpretation": "Smoke metrics validate execution only and are not scientific results.",
        "next_gate": (
            "Review the smoke before authorizing complete selection-only MuSiQue runs."
            if not failures
            else "Resolve smoke failures before any complete run or answer generation."
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
