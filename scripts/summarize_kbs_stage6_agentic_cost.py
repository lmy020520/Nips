#!/usr/bin/env python3
"""Recover auditable agentic-selection cost fields from the stored report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=Path,
        default=Path("outputs/rag/agentic_llm_eval3000.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    args = parser.parse_args()

    obj = json.loads(args.report.read_text(encoding="utf-8"))
    summary = obj.get("summary") if isinstance(obj, dict) else None
    records = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError("agentic report lacks summary/results")

    failures: list[str] = []
    expected = {
        "selector": "agentic_llm",
        "qids": 3000,
        "steps": 7296,
        "answer_judged": 3000,
        "answer_errors": 0,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            failures.append(f"{key}={summary.get(key)!r} != {value!r}")
    qids: set[str] = set()
    selection_calls = 0
    selection_tokens = 0
    selection_latency = 0.0
    selection_empty_outputs = 0
    selection_empty_indices = 0
    answer_tokens = 0
    answer_latency = 0.0
    answer_empty_outputs = 0
    invalid_steps = 0
    for record in records:
        if not isinstance(record, dict):
            failures.append("report contains a non-object result")
            continue
        qid = str(record.get("qid") or "")
        if not qid or qid in qids:
            failures.append(f"empty or duplicate qid: {qid!r}")
            continue
        qids.add(qid)
        answer_tokens += int(record.get("answer_tokens") or 0)
        answer_latency += float(record.get("answer_latency") or 0.0)
        answer_empty_outputs += int(not str(record.get("raw_answer") or "").strip())
        steps = record.get("steps")
        if not isinstance(steps, list):
            failures.append(f"qid={qid} lacks steps")
            continue
        for step in steps:
            decision = step.get("agentic_decision") if isinstance(step, dict) else None
            if not isinstance(decision, dict):
                invalid_steps += 1
                continue
            selection_calls += 1
            selection_tokens += int(decision.get("tokens") or 0)
            selection_latency += float(decision.get("latency") or 0.0)
            selection_empty_outputs += int(not str(decision.get("raw_answer") or "").strip())
            indices = decision.get("selected_indices")
            selection_empty_indices += int(not isinstance(indices, list) or not indices)

    qid_count = len(qids)
    if len(records) != 3000 or qid_count != 3000:
        failures.append(f"expected 3,000 unique qids, found {len(records)}/{qid_count}")
    if selection_calls != int(summary.get("steps") or 0):
        failures.append(
            f"agentic decisions={selection_calls} != summary steps={summary.get('steps')}"
        )
    if invalid_steps or selection_empty_outputs or selection_empty_indices:
        failures.append(
            "invalid selection records: "
            f"missing_decision={invalid_steps}, empty_output={selection_empty_outputs}, "
            f"empty_indices={selection_empty_indices}"
        )
    if answer_empty_outputs:
        failures.append(f"empty final-answer outputs={answer_empty_outputs}")

    final_answer_calls = int(summary.get("answer_judged") or 0)
    total_calls = selection_calls + final_answer_calls
    total_tokens = selection_tokens + answer_tokens
    total_latency = selection_latency + answer_latency
    row: dict[str, Any] = {
        "method": "LLM-guided iterative selection",
        "qids": qid_count,
        "selection_llm_calls_total": selection_calls,
        "selection_llm_calls_per_qid": round(selection_calls / qid_count, 6),
        "selection_api_tokens_total": selection_tokens,
        "selection_api_tokens_per_qid": round(selection_tokens / qid_count, 6),
        "selection_api_tokens_per_call": round(selection_tokens / selection_calls, 6),
        "selection_api_latency_seconds_total": round(selection_latency, 6),
        "selection_api_latency_seconds_per_qid": round(selection_latency / qid_count, 6),
        "selection_api_latency_seconds_per_call": round(selection_latency / selection_calls, 6),
        "final_answer_calls_total": final_answer_calls,
        "final_answer_api_tokens_total": answer_tokens,
        "final_answer_api_tokens_per_qid": round(answer_tokens / qid_count, 6),
        "final_answer_api_latency_seconds_total": round(answer_latency, 6),
        "final_answer_api_latency_seconds_per_qid": round(answer_latency / qid_count, 6),
        "total_llm_calls": total_calls,
        "total_llm_calls_per_qid": round(total_calls / qid_count, 6),
        "total_api_tokens": total_tokens,
        "total_api_tokens_per_qid": round(total_tokens / qid_count, 6),
        "measured_sequential_api_latency_seconds_total": round(total_latency, 6),
        "measured_sequential_api_latency_seconds_per_qid": round(total_latency / qid_count, 6),
        "answer_em": summary.get("answer_em"),
        "answer_f1": summary.get("answer_f1"),
        "teacher_alignment_at_5": summary.get("step_acc@5"),
        "full_unit_coverage": summary.get("full_gold_unit_coverage"),
        "terminal_selection_failures": invalid_steps + selection_empty_outputs + selection_empty_indices,
        "terminal_answer_failures": int(summary.get("answer_errors") or 0) + answer_empty_outputs,
    }
    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 6,
        "step": "6.4",
        "mode": "stored_agentic_cost_accounting",
        "api_calls": 0,
        "source_report": str(args.report),
        "reported_method_name": "LLM-guided iterative selection",
        "source_protocol": {
            "selector": summary.get("selector"),
            "requested_answer_model": summary.get("answer_model") or "not recorded",
            "answer_prompt_version": summary.get("answer_prompt_version") or "not recorded",
            "refresh_answer_cache": summary.get("refresh_answer_cache"),
            "agentic_max_candidates": summary.get("agentic_max_candidates"),
        },
        "metrics": row,
        "recoverability": {
            "selection_prompt_tokens": None,
            "selection_completion_tokens": None,
            "answer_prompt_tokens": None,
            "answer_completion_tokens": None,
            "retry_attempts": None,
            "estimated_api_cost_usd": None,
            "reason": (
                "The historical runtime stored only API total_tokens and terminal latency. "
                "It did not persist prompt/completion token splits or retry counts, so an "
                "exact price-weighted cost and retry audit cannot be reconstructed."
            ),
        },
        "latency_note": (
            "Latency is the sum of stored selection and final-answer API waiting times; "
            "it is not a newly profiled full-process wall-clock measurement."
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tsv_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.tsv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    print(f"tsv: {args.tsv_output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
