#!/usr/bin/env python3
"""Prepare exact-context answer caches for Stage 9.3 without API calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


METHODS = ("bm25", "dense", "hybrid", "iterative_hybrid", "bge_reranker")
ANSWER_PROTOCOL = {
    "answer_model": "deepseek-v4-flash",
    "answer_thinking_mode": "disabled",
    "answer_mode": "json",
    "answer_temperature": 0.0,
    "answer_prompt_version": "kbs_extractive_answer_json_v1",
}
SOURCE_REPORTS = (
    Path("outputs/rag/kbs_v27_final_hotpot/full_compact.json"),
    Path("outputs/rag/kbs_v27_stage5_multiseed/seed42/full_recall.json"),
    Path("outputs/rag/kbs_v27_stage5_multiseed/seed43/full_compact.json"),
    Path("outputs/rag/kbs_v27_stage5_multiseed/seed44/full_compact.json"),
    Path("outputs/rag/kbs_stage9_teacher_objective/coverage_seed42_full3000.json"),
    Path("outputs/rag/kbs_stage9_teacher_objective/coverage_seed43_full3000.json"),
    Path("outputs/rag/kbs_stage9_teacher_objective/coverage_seed44_full3000.json"),
    Path("outputs/rag/kbs_stage9_acquired_loss/ce_margin_seed42_full3000.json"),
    Path("outputs/rag/kbs_stage9_acquired_loss/ce_margin_seed43_full3000.json"),
    Path("outputs/rag/kbs_stage9_acquired_loss/ce_margin_seed44_full3000.json"),
)


def read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    results = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report must contain summary/results: {path}")
    return summary, results


def context_key(record: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    question = str(record.get("question") or "")
    gold_answer = str(record.get("gold_answer") or "")
    units = record.get("selected_unit_ids")
    if not question or not gold_answer or not isinstance(units, list):
        raise ValueError(f"invalid answer-context fields for qid={record.get('qid')}")
    return question, gold_answer, tuple(str(value) for value in units)


def source_protocol_differences(summary: dict[str, Any]) -> dict[str, Any]:
    expected = {
        **ANSWER_PROTOCOL,
        "generate_answers": True,
        "qids": 3000,
        "answer_judged": 3000,
        "answer_errors": 0,
    }
    return {
        key: {"observed": summary.get(key), "expected": value}
        for key, value in expected.items()
        if summary.get(key) != value
    }


def cache_payload(
    target_method: str,
    target_qid: str,
    source_path: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qid": target_qid,
        "answer": str(record.get("answer") or ""),
        "raw_answer": str(record.get("raw_answer") or ""),
        "answer_tokens": int(record.get("answer_tokens") or 0),
        "answer_latency": float(record.get("answer_latency") or 0.0),
        **ANSWER_PROTOCOL,
        "target_method": target_method,
        "source_report": str(source_path),
        "question": str(record.get("question") or ""),
        "gold_answer": str(record.get("gold_answer") or ""),
        "selected_unit_ids": [str(value) for value in record.get("selected_unit_ids") or []],
        "reuse_rule": "exact_ordered_selected_context_and_frozen_answer_protocol",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_strong_baselines/selection3000"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/rag/cache_kbs_stage9_strong_baselines"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/kbs_stage9_strong_baselines/answer_cache_readiness.json"
        ),
    )
    args = parser.parse_args()

    failures: list[str] = []
    selection_summary_path = args.selection_root / "summary.json"
    if not selection_summary_path.is_file():
        failures.append(f"missing selection summary: {selection_summary_path}")
    else:
        selection_summary = json.loads(
            selection_summary_path.read_text(encoding="utf-8")
        )
        if selection_summary.get("status") != "OK" or selection_summary.get("failures"):
            failures.append("full selection summary is not a clean OK")

    source_audit: dict[str, Any] = {}
    source_by_context: dict[
        tuple[str, str, tuple[str, ...]], tuple[Path, dict[str, Any]]
    ] = {}
    answer_disagreements = 0
    for path in SOURCE_REPORTS:
        audit: dict[str, Any] = {"exists": path.is_file(), "eligible": False}
        if not path.is_file():
            source_audit[str(path)] = audit
            continue
        try:
            summary, records = read_report(path)
            differences = source_protocol_differences(summary)
            audit["protocol_differences"] = differences
            audit["eligible"] = not differences
            audit["qids"] = len(records)
            if not differences:
                accepted = 0
                for record in records:
                    raw_answer = str(record.get("raw_answer") or "").strip()
                    answer = str(record.get("answer") or "").strip()
                    if not raw_answer or raw_answer.startswith("ERROR:") or not answer:
                        raise ValueError(f"invalid answer for qid={record.get('qid')}")
                    key = context_key(record)
                    previous = source_by_context.get(key)
                    if previous is not None:
                        previous_raw = str(previous[1].get("raw_answer") or "")
                        if previous_raw != raw_answer:
                            answer_disagreements += 1
                        continue
                    source_by_context[key] = (path, record)
                    accepted += 1
                audit["new_unique_contexts"] = accepted
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            audit["eligible"] = False
            audit["error"] = str(exc)
        source_audit[str(path)] = audit

    targets_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    unresolved_contexts: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
    per_method: dict[str, Any] = {}
    intended: dict[str, dict[Path, dict[str, Any]]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        path = args.selection_root / f"{method}.json"
        if not path.is_file():
            failures.append(f"missing selection report: {path}")
            continue
        try:
            summary, records = read_report(path)
            if (
                summary.get("qids") != 3000
                or summary.get("answer_judged") != 0
                or len(records) != 3000
            ):
                raise ValueError("selection report is not a complete no-answer 3000-qid run")
            indexed: dict[str, dict[str, Any]] = {}
            reused = 0
            for record in records:
                qid = str(record.get("qid") or "")
                if not qid or qid in indexed:
                    raise ValueError(f"missing or duplicate qid={qid!r}")
                indexed[qid] = record
                key = context_key(record)
                source = source_by_context.get(key)
                if source is None:
                    unresolved_contexts[key] += 1
                    continue
                source_path, source_record = source
                intended[method][args.cache_root / method / f"{qid}.json"] = (
                    cache_payload(method, qid, source_path, source_record)
                )
                reused += 1
            targets_by_method[method] = indexed
            per_method[method] = {
                "selection_report": str(path),
                "qids": len(records),
                "reused_exact_context": reused,
                "fresh_target_answers_without_cross_baseline_propagation": (
                    len(records) - reused
                ),
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{method}: {exc}")

    if targets_by_method:
        qid_orders = [list(records) for records in targets_by_method.values()]
        if any(order != qid_orders[0] for order in qid_orders[1:]):
            failures.append("ordered qids differ across target baseline reports")

    existing_identical = {method: 0 for method in METHODS}
    if not failures:
        for method in METHODS:
            cache_dir = args.cache_root / method
            expected = intended[method]
            if not cache_dir.exists():
                continue
            for path in cache_dir.glob("*.json"):
                if path not in expected:
                    failures.append(f"unexpected pre-existing cache: {path}")
                    continue
                try:
                    current = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(f"invalid pre-existing cache {path}: {exc}")
                    continue
                if current != expected[path]:
                    failures.append(f"pre-existing cache payload differs: {path}")
                else:
                    existing_identical[method] += 1

    written = {method: 0 for method in METHODS}
    if not failures:
        for method in METHODS:
            cache_dir = args.cache_root / method
            cache_dir.mkdir(parents=True, exist_ok=True)
            for path, payload in intended[method].items():
                if path.exists():
                    continue
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                written[method] += 1

    unresolved_target_files = sum(unresolved_contexts.values())
    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.3",
        "mode": "strong_baseline_exact_context_answer_cache_preparation",
        "api_calls": 0,
        "protocol": {
            **ANSWER_PROTOCOL,
            "reuse_rule": (
                "question, gold answer, full ordered selected-unit sequence, "
                "and complete answer protocol must match exactly"
            ),
            "source_priority": [str(path) for path in SOURCE_REPORTS],
            "cache_root": str(args.cache_root),
        },
        "eligible_source_contexts": len(source_by_context),
        "source_report_audit": source_audit,
        "duplicate_source_raw_answer_disagreements": answer_disagreements,
        "per_method": per_method,
        "existing_identical_cache_files": existing_identical,
        "written_cache_files": written,
        "total_target_contexts": sum(
            int(row.get("qids") or 0) for row in per_method.values()
        ),
        "fresh_target_answers_without_cross_baseline_propagation": (
            unresolved_target_files
        ),
        "unique_fresh_contexts_with_cross_baseline_deduplication": len(
            unresolved_contexts
        ),
        "cross_baseline_duplicate_target_files": (
            unresolved_target_files - len(unresolved_contexts)
        ),
        "next_gate": (
            "Review cache reuse and unique fresh-call counts before one bounded answer smoke."
            if not failures
            else "Resolve failures; do not call the answer API."
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
