#!/usr/bin/env python3
"""Merge quality, closure, context, and unified cost metrics for Stage 6.3."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


BUDGETS = (10, 15, 20, 50)
FRONT_POOLS = {10: 30, 15: 30, 20: 30, 50: 50}
METHODS = {budget: f"cand{budget}" for budget in BUDGETS}
FINAL_CHECKPOINT = "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"


def parse_budget_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must use BUDGET=PATH")
    budget_text, path_text = value.split("=", 1)
    try:
        budget = int(budget_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid budget: {budget_text}") from exc
    if budget not in BUDGETS or not path_text.strip():
        raise argparse.ArgumentTypeError(f"budget must be one of {BUDGETS}")
    return budget, Path(path_text)


def load_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary") if isinstance(obj, dict) else None
    records = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError(f"report lacks summary/results: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"report contains a non-object record: {path}")
    return summary, records


def index_records(records: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(record.get("qid") or "")
        if not qid or qid in indexed:
            raise ValueError(f"empty or duplicate qid in {path}")
        indexed[qid] = record
    return indexed


def load_memory(path: Path) -> dict[str, str]:
    memory: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            unit_id = str(row.get("unit_id") or "")
            text = str(row.get("text") or "")
            if not unit_id or not text:
                raise ValueError(f"invalid memory row at line {line_number}: {path}")
            memory[unit_id] = text
    return memory


def context_metrics(
    records: dict[str, dict[str, Any]], memory: dict[str, str], path: Path
) -> dict[str, float]:
    unit_counts: list[int] = []
    char_counts: list[int] = []
    token_counts: list[int] = []
    missing: set[str] = set()
    for record in records.values():
        unit_ids = record.get("selected_unit_ids")
        if not isinstance(unit_ids, list):
            raise ValueError(f"missing selected_unit_ids in {path}")
        texts = []
        for raw_unit_id in unit_ids:
            unit_id = str(raw_unit_id)
            text = memory.get(unit_id)
            if text is None:
                missing.add(unit_id)
            else:
                texts.append(text)
        context = "\n".join(texts)
        unit_counts.append(len(texts))
        char_counts.append(len(context))
        token_counts.append(len(re.findall(r"\S+", context)))
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} selected units missing from memory; "
            f"examples={sorted(missing)[:5]}"
        )
    count = len(records)
    return {
        "avg_selected_context_units": round(sum(unit_counts) / count, 6),
        "avg_selected_context_chars": round(sum(char_counts) / count, 6),
        "avg_selected_context_lexical_tokens": round(sum(token_counts) / count, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=parse_budget_path, required=True)
    parser.add_argument("--profile-report", action="append", type=parse_budget_path, required=True)
    parser.add_argument("--standard-summary", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    args = parser.parse_args()

    report_paths = dict(args.report)
    profile_paths = dict(args.profile_report)
    if set(report_paths) != set(BUDGETS) or len(args.report) != len(BUDGETS):
        raise ValueError(f"answer reports must contain each budget exactly once: {BUDGETS}")
    if set(profile_paths) != set(BUDGETS) or len(args.profile_report) != len(BUDGETS):
        raise ValueError(f"profile reports must contain each budget exactly once: {BUDGETS}")

    standard = json.loads(args.standard_summary.read_text(encoding="utf-8"))
    if standard.get("status") != "OK":
        raise ValueError("standard-metric summary did not pass")
    standard_methods = standard.get("methods")
    if not isinstance(standard_methods, dict):
        raise ValueError("standard-metric summary lacks methods")
    memory = load_memory(args.memory)

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    answer_qids: dict[int, set[str]] = {}
    profile_qids: dict[int, set[str]] = {}
    for budget in BUDGETS:
        answer_summary, answer_rows = load_report(report_paths[budget])
        profile_summary, profile_rows = load_report(profile_paths[budget])
        answers = index_records(answer_rows, report_paths[budget])
        profiles = index_records(profile_rows, profile_paths[budget])
        answer_qids[budget] = set(answers)
        profile_qids[budget] = set(profiles)
        expected_answer = {
            "checkpoint": FINAL_CHECKPOINT,
            "qids": 3000,
            "steps": 7296,
            "candidate_top_k": budget,
            "front_pool_k": FRONT_POOLS[budget],
            "answer_judged": 3000,
            "answer_errors": 0,
            "answer_model": "deepseek-v4-flash",
            "answer_thinking_mode": "disabled",
            "answer_prompt_version": "kbs_extractive_answer_json_v1",
        }
        for key, expected in expected_answer.items():
            if answer_summary.get(key) != expected:
                failures.append(
                    f"cand{budget} answer {key}={answer_summary.get(key)!r} != {expected!r}"
                )
        expected_profile = {
            "checkpoint": FINAL_CHECKPOINT,
            "qids": 500,
            "candidate_top_k": budget,
            "front_pool_k": FRONT_POOLS[budget],
            "answer_judged": 0,
            "generate_answers": False,
        }
        for key, expected in expected_profile.items():
            if profile_summary.get(key) != expected:
                failures.append(
                    f"cand{budget} profile {key}={profile_summary.get(key)!r} != {expected!r}"
                )
        runtime = profile_summary.get("runtime_profile")
        if not isinstance(runtime, dict):
            failures.append(f"cand{budget} profile lacks runtime_profile")
            runtime = {}
        if runtime.get("includes_answer_api") is not False:
            failures.append(f"cand{budget} profile selection runtime includes answer API")

        method = standard_methods.get(METHODS[budget])
        if not isinstance(method, dict):
            failures.append(f"standard metrics missing {METHODS[budget]}")
            method = {}
        metrics = method.get("metrics") if isinstance(method.get("metrics"), dict) else {}
        closure_key = f"closure_success_at_{budget}"
        selection_ms = runtime.get("selection_avg_ms_per_qid")
        answer_ms = (
            1000.0 * float(answer_summary["avg_answer_latency"])
            if answer_summary.get("avg_answer_latency") is not None else None
        )
        e2e_ms = (
            float(selection_ms) + answer_ms
            if selection_ms is not None and answer_ms is not None else None
        )
        row = {
            "budget": budget,
            "front_pool_k": FRONT_POOLS[budget],
            "qids": len(answers),
            "answer_em": answer_summary.get("answer_em"),
            "answer_f1": answer_summary.get("answer_f1"),
            "joint_f1": metrics.get("joint_f1"),
            "teacher_alignment_at_5": answer_summary.get("step_acc@5"),
            "full_unit_coverage": answer_summary.get("full_gold_unit_coverage"),
            "closure_success_at_budget": metrics.get(closure_key),
            **context_metrics(answers, memory, report_paths[budget]),
            "avg_answer_api_tokens": answer_summary.get("avg_answer_tokens"),
            "selection_ms_per_qid": selection_ms,
            "selection_throughput_qids_per_second": runtime.get(
                "selection_throughput_qids_per_second"
            ),
            "answer_api_ms_per_qid": round(answer_ms, 3) if answer_ms is not None else None,
            "estimated_e2e_ms_per_qid": round(e2e_ms, 3) if e2e_ms is not None else None,
            "peak_gpu_allocated_mb": runtime.get("peak_gpu_allocated_mb"),
            "peak_gpu_reserved_mb": runtime.get("peak_gpu_reserved_mb"),
        }
        rows.append(row)

    if len({frozenset(qids) for qids in answer_qids.values()}) != 1:
        failures.append("four answer reports do not use identical qid sets")
    if len({frozenset(qids) for qids in profile_qids.values()}) != 1:
        failures.append("four profile reports do not use identical 500-qid sets")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 6,
        "step": "6.3",
        "mode": "final_cost_closure_frontier",
        "api_calls": 0,
        "protocol": {
            "checkpoint": FINAL_CHECKPOINT,
            "quality_qids": 3000,
            "unified_runtime_qids": 500,
            "budgets": list(BUDGETS),
            "front_pool_schedule": {str(key): value for key, value in FRONT_POOLS.items()},
            "selected_context_tokens": "whitespace-delimited tokens in unique selected evidence text",
            "selection_latency": "same-session 500-qid profile after 20-qid warmup; excludes answer API",
            "estimated_e2e_latency": "unified selection latency plus measured average answer API latency",
            "cost_curve_note": (
                "Measured operating points are not assumed monotonic because cand20 runs "
                "iterative MMR-20 while cand50 often uses retained-pool passthrough."
            ),
        },
        "answer_reports": {str(key): str(value) for key, value in report_paths.items()},
        "profile_reports": {str(key): str(value) for key, value in profile_paths.items()},
        "standard_metrics": str(args.standard_summary),
        "rows": rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tsv_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.tsv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    print(f"tsv: {args.tsv_output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
