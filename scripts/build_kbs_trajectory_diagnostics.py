#!/usr/bin/env python3
"""Build readable KBS trajectory diagnostics from a policy-RAG report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def short_text(text: Any, limit: int = 160) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return str(value)


def selected_contains_all_gold(result: dict) -> bool:
    selected = set(result.get("selected_unit_ids") or [])
    gold = set(result.get("gold_unit_ids") or [])
    return bool(gold) and gold.issubset(selected)


def extract_last_deficit(steps: list[dict]) -> dict | None:
    for step in reversed(steps):
        deficit = step.get("deficit_estimate")
        if isinstance(deficit, dict):
            return deficit
    return None


def build_case_record(result: dict) -> dict:
    steps = result.get("steps") if isinstance(result.get("steps"), list) else []
    first_error_step = None
    for step in steps:
        if not step.get("selected_contains_gold"):
            first_error_step = int(step.get("t", len(steps)))
            break

    last_deficit = extract_last_deficit(steps)
    stop_record = result.get("stop_record") if isinstance(result.get("stop_record"), dict) else None
    final_state = result.get("final_online_state") if isinstance(result.get("final_online_state"), dict) else {}
    final_k_t = short_text(final_state.get("K_t"), limit=360) if final_state else ""

    step_summaries = []
    for step in steps:
        top5 = step.get("top5") if isinstance(step.get("top5"), list) else []
        step_summaries.append(
            {
                "t": step.get("t"),
                "positive_unit_id": step.get("positive_unit_id"),
                "predicted_unit_id": step.get("predicted_unit_id"),
                "selected_unit_ids": step.get("selected_unit_ids") or [],
                "selected_contains_gold": bool(step.get("selected_contains_gold")),
                "positive_rank": step.get("positive_rank"),
                "deficit_mean": (step.get("deficit_estimate") or {}).get("mean")
                if isinstance(step.get("deficit_estimate"), dict)
                else None,
                "stop_reason": (step.get("stop_decision") or {}).get("reason")
                if isinstance(step.get("stop_decision"), dict)
                else "",
                "top5": [
                    {
                        "rank": idx + 1,
                        "unit_id": item.get("unit_id"),
                        "doc_id": item.get("doc_id"),
                        "score": item.get("score"),
                    }
                    for idx, item in enumerate(top5)
                    if isinstance(item, dict)
                ],
            }
        )

    return {
        "qid": result.get("qid"),
        "question": result.get("question"),
        "answer": result.get("answer"),
        "gold_answer": result.get("gold_answer"),
        "answer_em": result.get("answer_em"),
        "answer_f1": result.get("answer_f1"),
        "all_steps_correct": bool(result.get("all_steps_correct")),
        "full_gold_unit_coverage": selected_contains_all_gold(result),
        "selected_units": len(result.get("selected_unit_ids") or []),
        "gold_units": len(result.get("gold_unit_ids") or []),
        "stopped_early": bool(result.get("stopped_early")),
        "stop_reason": stop_record.get("reason") if stop_record else "",
        "first_error_step": first_error_step,
        "last_deficit_mean": last_deficit.get("mean") if last_deficit else None,
        "last_deficit_max": last_deficit.get("max") if last_deficit else None,
        "final_K_t_preview": final_k_t,
        "steps": step_summaries,
    }


def build_markdown(summary: dict, cases: list[dict], *, max_cases: int) -> str:
    lines = [
        "# KBS Trajectory Diagnostics",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "qids",
        "cases_all_steps_correct",
        "cases_full_gold_unit_coverage",
        "cases_stopped_early",
        "avg_selected_units",
        "avg_gold_units",
        "avg_last_deficit_mean",
    ]:
        lines.append(f"| `{key}` | {format_float(summary.get(key))} |")

    lines.extend(
        [
            "",
            "## Case Table",
            "",
            "| # | qid | all_steps | full_gold | stopped | first_error_t | selected/gold | last_def_mean | question |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx, case in enumerate(cases[:max_cases], start=1):
        selected_gold = f"{case['selected_units']}/{case['gold_units']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(case.get("qid") or ""),
                    "1" if case.get("all_steps_correct") else "0",
                    "1" if case.get("full_gold_unit_coverage") else "0",
                    "1" if case.get("stopped_early") else "0",
                    str(case.get("first_error_step") if case.get("first_error_step") is not None else "-"),
                    selected_gold,
                    format_float(case.get("last_deficit_mean")),
                    short_text(case.get("question"), 90).replace("|", " "),
                ]
            )
            + " |"
        )

    lines.extend(["", "## First Error Examples", ""])
    error_cases = [case for case in cases if case.get("first_error_step") is not None]
    if not error_cases:
        lines.append("No step-level selected-gold failures found in this report.")
    for case in error_cases[: max_cases]:
        lines.extend(
            [
                f"### {case.get('qid')}",
                "",
                f"- Question: {case.get('question')}",
                f"- First error step: {case.get('first_error_step')}",
                f"- Selected/gold units: {case.get('selected_units')}/{case.get('gold_units')}",
                f"- Last deficit mean: {format_float(case.get('last_deficit_mean'))}",
                "",
            ]
        )
        for step in case.get("steps", []):
            if step.get("t") == case.get("first_error_step"):
                lines.extend(
                    [
                        f"- Positive: `{step.get('positive_unit_id')}`",
                        f"- Predicted: `{step.get('predicted_unit_id')}`",
                        f"- Selected contains gold: `{step.get('selected_contains_gold')}`",
                        f"- Positive rank: `{step.get('positive_rank')}`",
                        "",
                    ]
                )
                break
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--max-md-cases", type=int, default=30)
    args = parser.parse_args()

    report_path = Path(args.report)
    output_prefix = Path(args.output_prefix)
    report = load_json(report_path)
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("report.results must be a list")

    cases = [build_case_record(result) for result in results if isinstance(result, dict)]
    counters = Counter()
    for case in cases:
        counters["qids"] += 1
        counters["cases_all_steps_correct"] += int(case["all_steps_correct"])
        counters["cases_full_gold_unit_coverage"] += int(case["full_gold_unit_coverage"])
        counters["cases_stopped_early"] += int(case["stopped_early"])

    qids = max(counters["qids"], 1)
    last_deficits = [case["last_deficit_mean"] for case in cases if case["last_deficit_mean"] is not None]
    summary = {
        "report": str(report_path),
        "qids": counters["qids"],
        "cases_all_steps_correct": round(counters["cases_all_steps_correct"] / qids, 6),
        "cases_full_gold_unit_coverage": round(counters["cases_full_gold_unit_coverage"] / qids, 6),
        "cases_stopped_early": round(counters["cases_stopped_early"] / qids, 6),
        "avg_selected_units": round(sum(case["selected_units"] for case in cases) / qids, 4),
        "avg_gold_units": round(sum(case["gold_units"] for case in cases) / qids, 4),
        "avg_last_deficit_mean": round(sum(last_deficits) / len(last_deficits), 6) if last_deficits else None,
    }

    summary_path = output_prefix.with_suffix(".summary.json")
    cases_path = output_prefix.with_suffix(".cases.jsonl")
    md_path = output_prefix.with_suffix(".md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(cases, cases_path)
    md_path.write_text(build_markdown(summary, cases, max_cases=args.max_md_cases), encoding="utf-8")

    print(json.dumps({"summary": str(summary_path), "cases": str(cases_path), "markdown": str(md_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
