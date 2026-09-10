#!/usr/bin/env python3
"""Validate a Coverage-teacher answer report against frozen selection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANSWER_PROTOCOL = {
    "answer_model": "deepseek-v4-flash",
    "answer_thinking_mode": "disabled",
    "answer_mode": "json",
    "answer_temperature": 0.0,
    "answer_prompt_version": "kbs_extractive_answer_json_v1",
}
COVERAGE_CHECKPOINTS = {
    42: "outputs/ranker/deberta_v3_large_v29_coverage_greedy/best_model.pt",
    43: "outputs/ranker/deberta_v3_large_v29_coverage_greedy_seed43/best_model.pt",
    44: "outputs/ranker/deberta_v3_large_v29_coverage_greedy_seed44/best_model.pt",
}


def load_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary") if isinstance(obj, dict) else None
    results = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report lacks summary/results: {path}")
    if not all(isinstance(record, dict) for record in results):
        raise ValueError(f"report contains a non-object result: {path}")
    return summary, results


def index(records: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(record.get("qid") or "")
        if not qid or qid in output:
            raise ValueError(f"empty or duplicate qid in {path}: {qid!r}")
        output[qid] = record
    return output


def cache_file(cache_dir: Path, qid: str) -> Path:
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
    return cache_dir / f"{safe_qid}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=sorted(COVERAGE_CHECKPOINTS), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-qids", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    metrics: dict[str, Any] = {}
    cache_stats = {
        "evaluated_files": 0,
        "reused_exact_closure_context": 0,
        "coverage_fresh_answer": 0,
        "invalid": 0,
    }
    required = [args.report, args.selection_report, args.cache_dir]
    failures.extend(
        f"missing required path: {path}" for path in required if not path.exists()
    )

    if not failures:
        try:
            summary, result_rows = load_report(args.report)
            selection_summary, selection_rows = load_report(args.selection_report)
            results = index(result_rows, args.report)
            selections_all = index(selection_rows, args.selection_report)
            selections = {
                qid: selections_all[qid]
                for qid in list(selections_all)[: args.expected_qids]
            }
            expected_summary = {
                "checkpoint": COVERAGE_CHECKPOINTS[args.seed],
                "qids": args.expected_qids,
                "steps": sum(len(row.get("steps") or []) for row in selections.values()),
                "policy_context_source": "online_state",
                "selector": "hybrid_policy",
                "front_pool_k": 30,
                "front_fusion": "rrf",
                "local_expansion_window": 1,
                "candidate_top_k": 10,
                "select_top_k": 5,
                "state_update_top_k": 1,
                "policy_score_mode": "front_policy_blend",
                "policy_blend_weight": 0.5,
                "generate_answers": True,
                "answer_judged": args.expected_qids,
                "answer_errors": 0,
                "refresh_answer_cache": False,
                **ANSWER_PROTOCOL,
            }
            for key, expected in expected_summary.items():
                if summary.get(key) != expected:
                    failures.append(
                        f"summary {key}={summary.get(key)!r} != {expected!r}"
                    )
            if list(results) != list(selections):
                failures.append("answer qid order differs from the selection prefix")

            context_mismatches = 0
            empty_answers = 0
            error_answers = 0
            cache_mismatches = 0
            for qid, result in results.items():
                selection = selections.get(qid, {})
                if any(
                    result.get(key) != selection.get(key)
                    for key in ("question", "gold_answer", "selected_unit_ids")
                ):
                    context_mismatches += 1
                raw_answer = str(result.get("raw_answer") or "").strip()
                empty_answers += int(not raw_answer)
                error_answers += int(raw_answer.startswith("ERROR:"))

                path = cache_file(args.cache_dir, qid)
                if not path.is_file():
                    cache_mismatches += 1
                    continue
                cached = json.loads(path.read_text(encoding="utf-8"))
                cache_stats["evaluated_files"] += 1
                if (
                    str(cached.get("qid") or "") != qid
                    or str(cached.get("answer") or "") != str(result.get("answer") or "")
                    or str(cached.get("raw_answer") or "")
                    != str(result.get("raw_answer") or "")
                    or int(cached.get("answer_tokens") or 0)
                    != int(result.get("answer_tokens") or 0)
                ):
                    cache_mismatches += 1
                reuse = cached.get("cache_reuse")
                if isinstance(reuse, dict):
                    cache_stats["reused_exact_closure_context"] += 1
                    if (
                        reuse.get("type") != "exact_ordered_selected_context"
                        or int(reuse.get("seed") or 0) != args.seed
                        or reuse.get("selected_unit_ids") != result.get("selected_unit_ids")
                    ):
                        cache_stats["invalid"] += 1
                else:
                    cache_stats["coverage_fresh_answer"] += 1

            if context_mismatches:
                failures.append(f"selection context mismatches={context_mismatches}")
            if empty_answers or error_answers:
                failures.append(
                    f"empty answers={empty_answers}, error answers={error_answers}"
                )
            if cache_mismatches or cache_stats["invalid"]:
                failures.append(
                    f"cache mismatches={cache_mismatches}, "
                    f"invalid reuse metadata={cache_stats['invalid']}"
                )
            if cache_stats["evaluated_files"] != args.expected_qids:
                failures.append(
                    f"evaluated cache files={cache_stats['evaluated_files']} "
                    f"!= {args.expected_qids}"
                )

            for key in ("step_acc@1", "step_acc@5", "full_gold_unit_coverage"):
                if summary.get(key) != selection_summary.get(key) and args.expected_qids == 3000:
                    failures.append(f"{key} differs from full selection report")
            metrics = {
                "answer_em": summary.get("answer_em"),
                "answer_f1": summary.get("answer_f1"),
                "step_acc@1": summary.get("step_acc@1"),
                "step_acc@5": summary.get("step_acc@5"),
                "full_gold_unit_coverage": summary.get("full_gold_unit_coverage"),
                "avg_answer_tokens": summary.get("avg_answer_tokens"),
                "avg_answer_latency": summary.get("avg_answer_latency"),
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append(str(exc))

    result = {
        "status": "SMOKE_OK" if args.smoke and not failures else (
            "OK" if not failures else "FAIL"
        ),
        "stage": 9,
        "step": "9.1",
        "mode": "coverage_answer_smoke" if args.smoke else "coverage_answer_report",
        "seed": args.seed,
        "qids": args.expected_qids,
        "report": str(args.report),
        "selection_report": str(args.selection_report),
        "cache_dir": str(args.cache_dir),
        "protocol": ANSWER_PROTOCOL,
        "metrics": metrics,
        "cache": cache_stats,
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
