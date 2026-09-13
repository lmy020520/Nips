#!/usr/bin/env python3
"""Prepare exact-context Stage 9.5 answer caches without API calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.prepare_kbs_stage9_strong_baseline_answer_caches import (
    ANSWER_PROTOCOL,
    SOURCE_REPORTS,
    context_key,
    read_report,
    source_protocol_differences,
)


METHODS = (
    "online_state",
    "query_only",
    "frozen_initial_state",
    "other_question_state",
    "previous_evidence_only",
)


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
        default=Path("outputs/analysis/kbs_stage9_state_rollout/selection3000"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/rag/cache_kbs_stage9_state_rollout"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_state_rollout/answer_cache_readiness.json"),
    )
    args = parser.parse_args()

    failures: list[str] = []
    summary_path = args.selection_root / "summary.json"
    if not summary_path.is_file():
        failures.append(f"missing selection summary: {summary_path}")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "OK" or summary.get("failures"):
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
            audit.update(
                protocol_differences=differences,
                eligible=not differences,
                qids=len(records),
            )
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
                        if str(previous[1].get("raw_answer") or "") != raw_answer:
                            answer_disagreements += 1
                        continue
                    source_by_context[key] = (path, record)
                    accepted += 1
                audit["new_unique_contexts"] = accepted
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            audit.update(eligible=False, error=str(exc))
        source_audit[str(path)] = audit

    targets: dict[str, dict[str, dict[str, Any]]] = {}
    intended: dict[str, dict[Path, dict[str, Any]]] = {name: {} for name in METHODS}
    unresolved: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
    per_method = {}
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
                or summary.get("policy_context_source") != method
                or len(records) != 3000
            ):
                raise ValueError("selection report is not a complete matched no-answer run")
            indexed = {}
            reused = 0
            for record in records:
                qid = str(record.get("qid") or "")
                if not qid or qid in indexed:
                    raise ValueError(f"missing or duplicate qid={qid!r}")
                indexed[qid] = record
                key = context_key(record)
                source = source_by_context.get(key)
                if source is None:
                    unresolved[key] += 1
                    continue
                source_path, source_record = source
                cache_path = args.cache_root / method / f"{qid}.json"
                intended[method][cache_path] = cache_payload(
                    method, qid, source_path, source_record
                )
                reused += 1
            targets[method] = indexed
            per_method[method] = {
                "selection_report": str(path),
                "qids": len(records),
                "reused_exact_context": reused,
                "fresh_target_answers_without_cross_method_propagation": len(records) - reused,
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{method}: {exc}")

    if targets:
        qid_orders = [list(rows) for rows in targets.values()]
        if any(order != qid_orders[0] for order in qid_orders[1:]):
            failures.append("ordered qids differ across state conditions")

    existing_identical = {name: 0 for name in METHODS}
    existing_valid_fresh = {name: 0 for name in METHODS}
    if not failures:
        for method in METHODS:
            cache_dir = args.cache_root / method
            if not cache_dir.exists():
                continue
            for path in cache_dir.glob("*.json"):
                try:
                    current = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(f"invalid pre-existing cache {path}: {exc}")
                    continue
                qid = str(current.get("qid") or "")
                target = targets.get(method, {}).get(qid)
                if target is None or path.name != f"{qid}.json":
                    failures.append(f"cache does not map to a target qid: {path}")
                    continue
                valid_protocol = all(
                    current.get(key) == ANSWER_PROTOCOL[key]
                    for key in (
                        "answer_model",
                        "answer_thinking_mode",
                        "answer_mode",
                        "answer_prompt_version",
                    )
                )
                raw_answer = str(current.get("raw_answer") or "").strip()
                if not valid_protocol or not raw_answer or raw_answer.startswith("ERROR:"):
                    failures.append(f"invalid pre-existing answer cache: {path}")
                    continue
                expected = intended[method].get(path)
                if expected is not None and current == expected:
                    existing_identical[method] += 1
                elif "source_report" not in current:
                    existing_valid_fresh[method] += 1
                else:
                    failures.append(f"pre-existing propagated cache differs: {path}")

    written = {name: 0 for name in METHODS}
    if not failures:
        for method in METHODS:
            for path, payload in intended[method].items():
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                written[method] += 1

    unresolved_files = sum(unresolved.values())
    output = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.5",
        "mode": "state_rollout_exact_context_answer_cache_preparation",
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
        "existing_valid_fresh_cache_files": existing_valid_fresh,
        "written_cache_files": written,
        "total_target_contexts": sum(row.get("qids", 0) for row in per_method.values()),
        "fresh_target_answers_without_cross_method_propagation": unresolved_files,
        "unique_fresh_contexts_with_cross_method_deduplication": len(unresolved),
        "cross_method_duplicate_target_files": unresolved_files - len(unresolved),
        "next_gate": (
            "Review cache reuse and unique fresh-call counts before one bounded answer smoke."
            if not failures
            else "Resolve failures; do not call the answer API."
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
