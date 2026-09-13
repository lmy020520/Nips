#!/usr/bin/env python3
"""Validate Stage 9.5 answers against a frozen state-rollout selection report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.prepare_kbs_stage9_strong_baseline_answer_caches import ANSWER_PROTOCOL


METHODS = (
    "online_state",
    "query_only",
    "frozen_initial_state",
    "other_question_state",
    "previous_evidence_only",
)


def read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary") if isinstance(obj, dict) else None
    results = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report lacks summary/results: {path}")
    return summary, results


def index(records: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        qid = str(record.get("qid") or "")
        if not qid or qid in output:
            raise ValueError(f"empty or duplicate qid in {path}: {qid!r}")
        output[qid] = record
    return output


def cache_file(cache_dir: Path, qid: str) -> Path:
    return cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', qid)}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--online-state-report", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-qids", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    metrics: dict[str, Any] = {}
    cache_stats = {
        "evaluated_files": 0,
        "reused_exact_context": 0,
        "fresh_answers": 0,
        "invalid": 0,
    }
    for path in (args.report, args.selection_report, args.online_state_report, args.cache_dir):
        if not path.exists():
            failures.append(f"missing required path: {path}")

    if not failures:
        try:
            summary, answer_rows = read_report(args.report)
            selection_summary, selection_rows = read_report(args.selection_report)
            answers = index(answer_rows, args.report)
            selections_all = index(selection_rows, args.selection_report)
            selections = {
                qid: selections_all[qid]
                for qid in list(selections_all)[: args.expected_qids]
            }
            expected_summary = {
                "checkpoint": "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
                "qids": args.expected_qids,
                "steps": sum(len(row.get("steps") or []) for row in selections.values()),
                "state_mode": "policy",
                "policy_context_source": args.method,
                "selector": "hybrid_policy",
                "dense_model": "models/bge-large-en-v1.5",
                "dense_query_mode": "state",
                "hybrid_alpha": 0.5,
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
                "save_online_states": True,
                **ANSWER_PROTOCOL,
            }
            if args.method == "other_question_state":
                expected_summary.update(
                    {
                        "external_policy_state_report": str(args.online_state_report),
                        "external_policy_state_pairing": "cyclic_next_qid",
                    }
                )
            for key, expected in expected_summary.items():
                if summary.get(key) != expected:
                    failures.append(f"summary {key}={summary.get(key)!r} != {expected!r}")
            if list(answers) != list(selections):
                failures.append("answer qid order differs from frozen selection prefix")

            context_mismatches = empty_answers = error_answers = cache_mismatches = 0
            for qid, answer in answers.items():
                selection = selections.get(qid, {})
                if any(
                    answer.get(key) != selection.get(key)
                    for key in ("question", "gold_answer", "selected_unit_ids")
                ):
                    context_mismatches += 1
                raw_answer = str(answer.get("raw_answer") or "").strip()
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
                    or str(cached.get("answer") or "") != str(answer.get("answer") or "")
                    or str(cached.get("raw_answer") or "") != raw_answer
                    or int(cached.get("answer_tokens") or 0) != int(answer.get("answer_tokens") or 0)
                ):
                    cache_mismatches += 1
                for key in (
                    "answer_model",
                    "answer_thinking_mode",
                    "answer_mode",
                    "answer_prompt_version",
                ):
                    if cached.get(key) != ANSWER_PROTOCOL[key]:
                        cache_stats["invalid"] += 1
                if "source_report" in cached:
                    cache_stats["reused_exact_context"] += 1
                    if any(
                        cached.get(key) != answer.get(key)
                        for key in ("question", "gold_answer", "selected_unit_ids")
                    ):
                        cache_stats["invalid"] += 1
                else:
                    cache_stats["fresh_answers"] += 1

            if context_mismatches:
                failures.append(f"selection context mismatches={context_mismatches}")
            if empty_answers or error_answers:
                failures.append(f"empty answers={empty_answers}, error answers={error_answers}")
            if cache_mismatches or cache_stats["invalid"]:
                failures.append(
                    f"cache mismatches={cache_mismatches}, invalid metadata={cache_stats['invalid']}"
                )
            if cache_stats["evaluated_files"] != args.expected_qids:
                failures.append(
                    f"evaluated cache files={cache_stats['evaluated_files']} != {args.expected_qids}"
                )
            if args.expected_qids == 3000:
                for key in (
                    "step_acc@1",
                    "step_acc@5",
                    "full_gold_doc_coverage",
                    "full_gold_unit_coverage",
                ):
                    if summary.get(key) != selection_summary.get(key):
                        failures.append(f"{key} differs from frozen selection report")
            metrics = {
                "answer_em": summary.get("answer_em"),
                "answer_f1": summary.get("answer_f1"),
                "step_acc@1": summary.get("step_acc@1"),
                "step_acc@5": summary.get("step_acc@5"),
                "full_gold_doc_coverage": summary.get("full_gold_doc_coverage"),
                "full_gold_unit_coverage": summary.get("full_gold_unit_coverage"),
                "avg_answer_tokens": summary.get("avg_answer_tokens"),
                "avg_answer_latency": summary.get("avg_answer_latency"),
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append(str(exc))

    output = {
        "status": "SMOKE_OK" if args.smoke and not failures else (
            "OK" if not failures else "FAIL"
        ),
        "stage": 9,
        "step": "9.5",
        "mode": "state_rollout_answer_smoke" if args.smoke else "state_rollout_answer_report",
        "method": args.method,
        "qids": args.expected_qids,
        "report": str(args.report),
        "selection_report": str(args.selection_report),
        "cache_dir": str(args.cache_dir),
        "protocol": ANSWER_PROTOCOL,
        "metrics": metrics,
        "cache": cache_stats,
        "next_gate": (
            "Resolve failures before proceeding."
            if failures
            else "Review the bounded smoke before authorizing the complete answer chain."
            if args.smoke
            else "Finalize standard metrics and paired confidence intervals."
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
