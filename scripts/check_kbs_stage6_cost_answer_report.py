#!/usr/bin/env python3
"""Validate one Stage 6.3 middle-budget answer report and its cache."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FINAL_CHECKPOINT = "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
ANSWER_PROTOCOL = {
    "answer_model": "deepseek-v4-flash",
    "answer_thinking_mode": "disabled",
    "answer_mode": "json",
    "answer_prompt_version": "kbs_extractive_answer_json_v1",
}
EXPECTED_REUSE = {15: 1259, 20: 1126}


def read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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


def cache_file(cache_dir: Path, qid: str) -> Path:
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
    return cache_dir / f"{safe_qid}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, choices=sorted(EXPECTED_REUSE), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    metrics: dict[str, Any] = {}
    cache_stats = {
        "files": 0,
        "reused_exact_context": 0,
        "freshly_generated": 0,
        "invalid": 0,
    }
    required = [args.report, args.selection_report, args.cache_dir]
    failures.extend(f"missing required path: {path}" for path in required if not path.exists())

    if not failures:
        try:
            summary, result_rows = read_report(args.report)
            selection_summary, selection_rows = read_report(args.selection_report)
            results = index_records(result_rows, args.report)
            selections = index_records(selection_rows, args.selection_report)
            expected_summary = {
                "checkpoint": FINAL_CHECKPOINT,
                "qids": 3000,
                "steps": 7296,
                "candidate_top_k": args.budget,
                "front_pool_k": 30,
                "select_top_k": 5,
                "state_update_top_k": 1,
                "policy_blend_weight": 0.5,
                "answer_judged": 3000,
                "answer_errors": 0,
                "generate_answers": True,
                "refresh_answer_cache": False,
                **ANSWER_PROTOCOL,
            }
            for key, value in expected_summary.items():
                if summary.get(key) != value:
                    failures.append(f"summary {key}={summary.get(key)!r} != {value!r}")
            if len(results) != 3000 or len(selections) != 3000:
                failures.append(
                    f"expected 3,000 report/selection qids, found {len(results)}/{len(selections)}"
                )
            if set(results) != set(selections):
                failures.append("answer and selection reports have different qid sets")
            selection_expected = {
                "checkpoint": FINAL_CHECKPOINT,
                "candidate_top_k": args.budget,
                "front_pool_k": 30,
                "answer_judged": 0,
                "generate_answers": False,
            }
            for key, value in selection_expected.items():
                if selection_summary.get(key) != value:
                    failures.append(
                        f"selection summary {key}={selection_summary.get(key)!r} != {value!r}"
                    )

            context_mismatches = 0
            empty_answers = 0
            error_answers = 0
            cache_mismatches = 0
            for qid, record in results.items():
                selection = selections.get(qid, {})
                if record.get("question") != selection.get("question"):
                    context_mismatches += 1
                if record.get("gold_answer") != selection.get("gold_answer"):
                    context_mismatches += 1
                selected_units = record.get("selected_unit_ids")
                if selected_units != selection.get("selected_unit_ids"):
                    context_mismatches += 1
                raw_answer = str(record.get("raw_answer") or "").strip()
                empty_answers += int(not raw_answer)
                error_answers += int(raw_answer.startswith("ERROR:"))

                path = cache_file(args.cache_dir, qid)
                if not path.exists():
                    cache_mismatches += 1
                    continue
                cached = json.loads(path.read_text(encoding="utf-8"))
                cache_stats["files"] += 1
                if (
                    str(cached.get("qid") or "") != qid
                    or str(cached.get("answer") or "") != str(record.get("answer") or "")
                    or str(cached.get("raw_answer") or "") != str(record.get("raw_answer") or "")
                    or int(cached.get("answer_tokens") or 0) != int(record.get("answer_tokens") or 0)
                ):
                    cache_mismatches += 1
                reuse = cached.get("cache_reuse")
                if isinstance(reuse, dict):
                    cache_stats["reused_exact_context"] += 1
                    if (
                        reuse.get("type") != "exact_ordered_selected_context"
                        or int(reuse.get("target_budget") or 0) != args.budget
                        or reuse.get("selected_unit_ids") != selected_units
                    ):
                        cache_stats["invalid"] += 1
                else:
                    cache_stats["freshly_generated"] += 1

            extra_cache_files = len(list(args.cache_dir.glob("*.json"))) - cache_stats["files"]
            expected_reuse = EXPECTED_REUSE[args.budget]
            expected_fresh = 3000 - expected_reuse
            if context_mismatches:
                failures.append(f"context/question/gold mismatches={context_mismatches}")
            if empty_answers or error_answers:
                failures.append(f"empty answers={empty_answers}, error answers={error_answers}")
            if cache_mismatches or extra_cache_files:
                failures.append(
                    f"cache mismatches={cache_mismatches}, extra cache files={extra_cache_files}"
                )
            if cache_stats["reused_exact_context"] != expected_reuse:
                failures.append(
                    f"reused caches={cache_stats['reused_exact_context']} != {expected_reuse}"
                )
            if cache_stats["freshly_generated"] != expected_fresh:
                failures.append(
                    f"fresh caches={cache_stats['freshly_generated']} != {expected_fresh}"
                )
            if cache_stats["invalid"]:
                failures.append(f"invalid reuse metadata={cache_stats['invalid']}")
            runtime = summary.get("runtime_profile")
            if not isinstance(runtime, dict):
                failures.append("runtime_profile is missing")
                runtime = {}
            elif runtime.get("includes_answer_api") is not False:
                failures.append("selection runtime must exclude answer API")

            for key in ("step_acc@1", "step_acc@5", "full_gold_unit_coverage"):
                if summary.get(key) != selection_summary.get(key):
                    failures.append(
                        f"{key} changed from selection report: "
                        f"{summary.get(key)!r} != {selection_summary.get(key)!r}"
                    )
            metrics = {
                "answer_em": summary.get("answer_em"),
                "answer_f1": summary.get("answer_f1"),
                "step_acc@1": summary.get("step_acc@1"),
                "step_acc@5": summary.get("step_acc@5"),
                "full_gold_unit_coverage": summary.get("full_gold_unit_coverage"),
                "avg_answer_tokens": summary.get("avg_answer_tokens"),
                "avg_answer_latency": summary.get("avg_answer_latency"),
                "selection_ms_per_qid_this_run": runtime.get("selection_avg_ms_per_qid"),
                "peak_gpu_allocated_mb_this_run": runtime.get("peak_gpu_allocated_mb"),
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(str(exc))

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 6,
        "step": "6.3",
        "mode": "middle_budget_answer_report_audit",
        "budget": args.budget,
        "report": str(args.report),
        "selection_report": str(args.selection_report),
        "cache_dir": str(args.cache_dir),
        "protocol": {
            "checkpoint": FINAL_CHECKPOINT,
            **ANSWER_PROTOCOL,
            "reused_contexts_expected": EXPECTED_REUSE[args.budget],
            "fresh_answer_calls_expected": 3000 - EXPECTED_REUSE[args.budget],
        },
        "metrics": metrics,
        "cache": cache_stats,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
