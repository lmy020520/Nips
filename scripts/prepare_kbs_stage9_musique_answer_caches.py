#!/usr/bin/env python3
"""Audit and prepare exact-context MuSiQue answer caches without API calls."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


METHODS = (
    "compact_seed42",
    "balanced_seed42",
    "hybrid",
    "bge_reranker",
)
ANSWER_PROTOCOL = {
    "answer_model": "deepseek-v4-flash",
    "answer_thinking_mode": "disabled",
    "answer_mode": "json",
    "answer_temperature": 0.0,
    "answer_prompt_version": "kbs_extractive_answer_json_v1",
}
ANSWER_REFERENCE_MODE = "max_over_canonical_and_aliases"
SOURCE_REPORTS = tuple(
    Path(f"outputs/rag/kbs_stage9_musique/{method}_full1000.json")
    for method in METHODS
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary") if isinstance(obj, dict) else None
    results = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report must contain summary/results: {path}")
    return summary, results


def answer_aliases(query: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(value).strip()
        for value in query.get("answer_aliases") or []
        if str(value).strip()
    )


def context_key(
    record: dict[str, Any],
    queries: dict[str, dict[str, Any]],
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    qid = str(record.get("qid") or "")
    query = queries.get(qid)
    if query is None:
        raise ValueError(f"missing query metadata for qid={qid!r}")
    question = str(query.get("question") or "")
    gold_answer = str(query.get("answer") or "")
    units = record.get("selected_unit_ids")
    if not question or not gold_answer or not isinstance(units, list):
        raise ValueError(f"invalid answer-context fields for qid={qid}")
    if str(record.get("question") or "") != question:
        raise ValueError(f"question differs from adapter query for qid={qid}")
    record_gold = str(record.get("gold_answer") or "")
    if record_gold and record_gold != gold_answer:
        raise ValueError(f"gold answer differs from adapter query for qid={qid}")
    return (
        question,
        gold_answer,
        answer_aliases(query),
        tuple(str(value) for value in units),
    )


def protocol_differences(summary: dict[str, Any]) -> dict[str, Any]:
    expected = {
        **ANSWER_PROTOCOL,
        "answer_reference_mode": ANSWER_REFERENCE_MODE,
        "generate_answers": True,
        "qids": 1000,
        "answer_judged": 1000,
        "answer_errors": 0,
    }
    return {
        key: {"observed": summary.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if summary.get(key) != expected_value
    }


def safe_cache_path(cache_dir: Path, qid: str) -> Path:
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
    return cache_dir / f"{safe_qid}.json"


def cache_payload(
    method: str,
    qid: str,
    source_path: Path,
    source: dict[str, Any],
    query: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qid": qid,
        "answer": str(source.get("answer") or ""),
        "raw_answer": str(source.get("raw_answer") or ""),
        "answer_tokens": int(source.get("answer_tokens") or 0),
        "answer_latency": float(source.get("answer_latency") or 0.0),
        **ANSWER_PROTOCOL,
        "answer_reference_mode": ANSWER_REFERENCE_MODE,
        "target_method": method,
        "source_report": str(source_path),
        "question": str(query.get("question") or ""),
        "gold_answer": str(query.get("answer") or ""),
        "gold_answer_aliases": list(answer_aliases(query)),
        "selected_unit_ids": [
            str(value) for value in source.get("selected_unit_ids") or []
        ],
        "reuse_rule": "exact_context_with_ordered_aliases_and_frozen_protocol",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_breadth/selection1000"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/musique_ans_eval_1000_paragraph20/queries/test.jsonl"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/rag/cache_kbs_stage9_musique"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/kbs_stage9_breadth/answer_cache_readiness.json"
        ),
    )
    args = parser.parse_args()

    failures: list[str] = []
    for name, expected_mode in (
        ("summary.json", "musique_zero_shot_selection"),
        ("paired_bootstrap.json", "musique_zero_shot_selection_bootstrap"),
    ):
        path = args.selection_root / name
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if (
                report.get("status") != "OK"
                or report.get("mode") != expected_mode
                or report.get("qids") != 1000
                or report.get("failures")
            ):
                failures.append(f"{path} is not a clean registered OK")
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"cannot read {path}: {exc}")

    queries: dict[str, dict[str, Any]] = {}
    try:
        for row in read_jsonl(args.queries):
            qid = str(row.get("qid") or "")
            if not qid or qid in queries:
                raise ValueError(f"empty or duplicate query qid={qid!r}")
            queries[qid] = row
        if len(queries) != 1000:
            failures.append(f"query count={len(queries)} != 1000")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"cannot load adapter queries: {exc}")

    source_audit: dict[str, Any] = {}
    source_by_context: dict[
        tuple[str, str, tuple[str, ...], tuple[str, ...]],
        tuple[Path, dict[str, Any]],
    ] = {}
    answer_disagreements = 0
    for path in SOURCE_REPORTS:
        audit: dict[str, Any] = {"exists": path.is_file(), "eligible": False}
        if path.is_file() and queries:
            try:
                summary, records = read_report(path)
                differences = protocol_differences(summary)
                audit["protocol_differences"] = differences
                audit["qids"] = len(records)
                audit["eligible"] = not differences and len(records) == 1000
                if audit["eligible"]:
                    accepted = 0
                    for record in records:
                        answer = str(record.get("answer") or "").strip()
                        raw = str(record.get("raw_answer") or "").strip()
                        if not answer or not raw or raw.startswith("ERROR:"):
                            raise ValueError(
                                f"invalid source answer for qid={record.get('qid')}"
                            )
                        key = context_key(record, queries)
                        previous = source_by_context.get(key)
                        if previous is not None:
                            if str(previous[1].get("raw_answer") or "") != raw:
                                answer_disagreements += 1
                            continue
                        source_by_context[key] = (path, record)
                        accepted += 1
                    audit["new_unique_contexts"] = accepted
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                audit["eligible"] = False
                audit["error"] = str(exc)
        source_audit[str(path)] = audit

    target_rows: dict[str, dict[str, dict[str, Any]]] = {}
    intended: dict[str, dict[Path, dict[str, Any]]] = {
        method: {} for method in METHODS
    }
    unresolved: Counter[
        tuple[str, str, tuple[str, ...], tuple[str, ...]]
    ] = Counter()
    per_method: dict[str, Any] = {}
    for method in METHODS:
        path = args.selection_root / f"{method}.json"
        try:
            summary, records = read_report(path)
            if (
                summary.get("qids") != 1000
                or summary.get("answer_judged") != 0
                or summary.get("skipped") != 0
                or len(records) != 1000
            ):
                raise ValueError("selection report is not a complete no-answer run")
            indexed: dict[str, dict[str, Any]] = {}
            reused = 0
            for record in records:
                qid = str(record.get("qid") or "")
                if not qid or qid in indexed:
                    raise ValueError(f"empty or duplicate qid={qid!r}")
                indexed[qid] = record
                key = context_key(record, queries)
                source = source_by_context.get(key)
                if source is None:
                    unresolved[key] += 1
                    continue
                source_path, source_record = source
                cache_path = safe_cache_path(args.cache_root / method, qid)
                intended[method][cache_path] = cache_payload(
                    method, qid, source_path, source_record, queries[qid]
                )
                reused += 1
            target_rows[method] = indexed
            per_method[method] = {
                "selection_report": str(path),
                "qids": len(records),
                "reused_exact_context": reused,
                "fresh_target_answers_without_cross_method_propagation": (
                    len(records) - reused
                ),
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{method}: {exc}")

    if len(target_rows) == len(METHODS):
        orders = [list(rows) for rows in target_rows.values()]
        if any(order != orders[0] for order in orders[1:]):
            failures.append("ordered qids differ across MuSiQue target reports")

    existing_identical = {method: 0 for method in METHODS}
    existing_valid_fresh = {method: 0 for method in METHODS}
    if not failures:
        for method in METHODS:
            cache_dir = args.cache_root / method
            for path in cache_dir.glob("*.json") if cache_dir.exists() else []:
                try:
                    current = json.loads(path.read_text(encoding="utf-8"))
                    qid = str(current.get("qid") or "")
                    target = target_rows[method].get(qid)
                    if target is None or path != safe_cache_path(cache_dir, qid):
                        raise ValueError("cache does not map to a target qid")
                    if not str(current.get("answer") or "").strip() or not str(
                        current.get("raw_answer") or ""
                    ).strip():
                        raise ValueError("cache has an empty answer")
                    if any(
                        current.get(key) != ANSWER_PROTOCOL[key]
                        for key in (
                            "answer_model",
                            "answer_thinking_mode",
                            "answer_mode",
                            "answer_prompt_version",
                        )
                    ):
                        raise ValueError("cache answer protocol differs")
                    if path in intended[method] and current == intended[method][path]:
                        existing_identical[method] += 1
                    elif "source_report" not in current:
                        existing_valid_fresh[method] += 1
                    else:
                        raise ValueError("propagated cache differs from intended source")
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    failures.append(f"invalid pre-existing cache {path}: {exc}")

    written = {method: 0 for method in METHODS}
    if not failures:
        for method in METHODS:
            cache_dir = args.cache_root / method
            cache_dir.mkdir(parents=True, exist_ok=True)
            for path, payload in intended[method].items():
                if not path.exists():
                    path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    written[method] += 1

    unresolved_files = sum(unresolved.values())
    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.6",
        "mode": "musique_exact_context_answer_cache_preparation",
        "api_calls": 0,
        "protocol": {
            **ANSWER_PROTOCOL,
            "answer_reference_mode": ANSWER_REFERENCE_MODE,
            "reuse_rule": (
                "question, canonical answer, ordered aliases, and full ordered "
                "selected-unit sequence must match exactly"
            ),
            "cache_root": str(args.cache_root),
            "source_priority": [str(path) for path in SOURCE_REPORTS],
            "answer_methods": list(METHODS),
        },
        "query_audit": {
            "qids": len(queries),
            "qids_with_aliases": sum(
                bool(answer_aliases(query)) for query in queries.values()
            ),
        },
        "eligible_source_contexts": len(source_by_context),
        "source_report_audit": source_audit,
        "duplicate_source_raw_answer_disagreements": answer_disagreements,
        "per_method": per_method,
        "existing_identical_cache_files": existing_identical,
        "existing_valid_fresh_cache_files": existing_valid_fresh,
        "written_cache_files": written,
        "total_target_contexts": sum(row.get("qids", 0) for row in per_method.values()),
        "fresh_target_answers_without_cross_method_propagation": unresolved_files,
        "unique_fresh_contexts_with_cross_method_deduplication": len(unresolved),
        "cross_method_duplicate_target_files": unresolved_files - len(unresolved),
        "next_gate": (
            "Review unique fresh-call counts before one bounded Compact answer smoke."
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
