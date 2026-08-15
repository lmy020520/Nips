#!/usr/bin/env python3
"""Prepare exact-context answer caches for the Stage 6.3 middle budgets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
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


def read_report(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    summary = obj.get("summary")
    records = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError(f"report lacks summary/results: {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"report contains a non-object record: {path}")
        qid = str(record.get("qid") or "")
        if not qid or qid in indexed:
            raise ValueError(f"report contains an empty or duplicate qid: {path}")
        indexed[qid] = record
    return summary, indexed


def cache_file(cache_dir: Path, qid: str) -> Path:
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
    return cache_dir / f"{safe_qid}.json"


def normalized_units(record: dict[str, Any], path: Path) -> list[str]:
    units = record.get("selected_unit_ids")
    if not isinstance(units, list) or not units or not all(isinstance(unit, str) for unit in units):
        raise ValueError(f"missing/invalid selected_unit_ids for {record.get('qid')} in {path}")
    return units


def audit_source(
    path: Path,
    summary: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    budget: int,
    front_pool: int,
) -> list[str]:
    failures: list[str] = []
    expected = {
        "checkpoint": FINAL_CHECKPOINT,
        "qids": 3000,
        "candidate_top_k": budget,
        "front_pool_k": front_pool,
        "answer_judged": 3000,
        "answer_errors": 0,
        **ANSWER_PROTOCOL,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            failures.append(f"{path}: {key}={summary.get(key)!r} != {value!r}")
    if len(records) != 3000:
        failures.append(f"{path}: qid records={len(records)} != 3000")
    for qid, record in records.items():
        if not str(record.get("raw_answer") or "").strip():
            failures.append(f"{path}: qid={qid} has an empty raw answer")
            break
        try:
            normalized_units(record, path)
        except ValueError as exc:
            failures.append(str(exc))
            break
    return failures


def audit_target(
    path: Path,
    summary: dict[str, Any],
    records: dict[str, dict[str, Any]],
    *,
    budget: int,
) -> list[str]:
    failures: list[str] = []
    expected = {
        "checkpoint": FINAL_CHECKPOINT,
        "qids": 3000,
        "steps": 7296,
        "candidate_top_k": budget,
        "front_pool_k": 30,
        "select_top_k": 5,
        "state_update_top_k": 1,
        "policy_blend_weight": 0.5,
        "answer_judged": 0,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            failures.append(f"{path}: {key}={summary.get(key)!r} != {value!r}")
    if summary.get("generate_answers") is not False:
        failures.append(f"{path}: target report must be selection-only")
    if len(records) != 3000:
        failures.append(f"{path}: qid records={len(records)} != 3000")
    for record in records.values():
        try:
            normalized_units(record, path)
        except ValueError as exc:
            failures.append(str(exc))
            break
    return failures


def cache_payload(
    qid: str,
    source_record: dict[str, Any],
    *,
    source_path: Path,
    source_budget: int,
    target_budget: int,
) -> dict[str, Any]:
    return {
        "qid": qid,
        "answer": str(source_record.get("answer") or ""),
        "raw_answer": str(source_record.get("raw_answer") or ""),
        "answer_tokens": int(source_record.get("answer_tokens") or 0),
        "answer_latency": float(source_record.get("answer_latency") or 0.0),
        **ANSWER_PROTOCOL,
        "cache_reuse": {
            "type": "exact_ordered_selected_context",
            "source_report": str(source_path),
            "source_budget": source_budget,
            "target_budget": target_budget,
            "selected_unit_ids": normalized_units(source_record, source_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cand10-report", type=Path,
        default=Path("outputs/rag/kbs_v27_final_hotpot/full_compact.json"),
    )
    parser.add_argument(
        "--cand50-report", type=Path,
        default=Path("outputs/rag/kbs_v27_stage5_multiseed/seed42/full_recall.json"),
    )
    parser.add_argument(
        "--cand15-report", type=Path,
        default=Path("outputs/analysis/kbs_stage6_cost_frontier_selection3000/cand15_selection3000.json"),
    )
    parser.add_argument(
        "--cand20-report", type=Path,
        default=Path("outputs/analysis/kbs_stage6_cost_frontier_selection3000/cand20_selection3000.json"),
    )
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("outputs/rag/cache_kbs_stage6_cost_frontier"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/analysis/kbs_stage6_cost_frontier/cache_reuse_readiness.json"),
    )
    args = parser.parse_args()

    paths = {
        10: args.cand10_report,
        15: args.cand15_report,
        20: args.cand20_report,
        50: args.cand50_report,
    }
    failures = [f"missing report: {path}" for path in paths.values() if not path.exists()]
    summaries: dict[int, dict[str, Any]] = {}
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    intended: dict[int, dict[str, dict[str, Any]]] = {15: {}, 20: {}}
    stats: dict[str, Any] = {}

    if not failures:
        try:
            for budget, path in paths.items():
                summaries[budget], reports[budget] = read_report(path)
            failures.extend(audit_source(paths[10], summaries[10], reports[10], budget=10, front_pool=30))
            failures.extend(audit_source(paths[50], summaries[50], reports[50], budget=50, front_pool=50))
            failures.extend(audit_target(paths[15], summaries[15], reports[15], budget=15))
            failures.extend(audit_target(paths[20], summaries[20], reports[20], budget=20))

            qid_sets = {budget: set(records) for budget, records in reports.items()}
            if len({frozenset(qids) for qids in qid_sets.values()}) != 1:
                failures.append("the four reports do not contain identical qid sets")

            if not failures:
                common_qids = sorted(qid_sets[10])
                for qid in common_qids:
                    reference = reports[10][qid]
                    for budget in (15, 20, 50):
                        candidate = reports[budget][qid]
                        for key in ("question", "gold_answer"):
                            if candidate.get(key) != reference.get(key):
                                failures.append(f"qid={qid}: {key} differs between cand10 and cand{budget}")
                                break
                    if failures:
                        break

            if not failures:
                for target_budget in (15, 20):
                    source_counts: Counter[str] = Counter()
                    both_match = 0
                    both_answer_disagreements = 0
                    target_records = reports[target_budget]
                    for qid in sorted(target_records):
                        target_units = normalized_units(target_records[qid], paths[target_budget])
                        matches10 = target_units == normalized_units(reports[10][qid], paths[10])
                        matches50 = target_units == normalized_units(reports[50][qid], paths[50])
                        if matches10 and matches50:
                            both_match += 1
                            answer10 = str(reports[10][qid].get("raw_answer") or "")
                            answer50 = str(reports[50][qid].get("raw_answer") or "")
                            both_answer_disagreements += int(answer10 != answer50)
                        if matches10:
                            source_budget = 10
                        elif matches50:
                            source_budget = 50
                        else:
                            continue
                        source_counts[str(source_budget)] += 1
                        intended[target_budget][qid] = cache_payload(
                            qid,
                            reports[source_budget][qid],
                            source_path=paths[source_budget],
                            source_budget=source_budget,
                            target_budget=target_budget,
                        )
                    reused = len(intended[target_budget])
                    stats[str(target_budget)] = {
                        "qids": len(target_records),
                        "reused_exact_context": reused,
                        "new_api_answers_required": len(target_records) - reused,
                        "source_budget_counts": dict(sorted(source_counts.items())),
                        "matches_both_sources": both_match,
                        "both_source_raw_answer_disagreements": both_answer_disagreements,
                    }
                    if reused != EXPECTED_REUSE[target_budget]:
                        failures.append(
                            f"cand{target_budget}: exact reuse={reused} != {EXPECTED_REUSE[target_budget]}"
                        )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(str(exc))

    existing_identical: dict[int, int] = {15: 0, 20: 0}
    if not failures:
        for target_budget in (15, 20):
            cache_dir = args.cache_root / f"cand{target_budget}"
            expected_paths = {
                cache_file(cache_dir, qid): payload
                for qid, payload in intended[target_budget].items()
            }
            if cache_dir.exists():
                for path in cache_dir.glob("*.json"):
                    if path not in expected_paths:
                        failures.append(f"unexpected pre-existing cache file: {path}")
                        continue
                    try:
                        existing = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        failures.append(f"cannot read pre-existing cache {path}: {exc}")
                        continue
                    if existing != expected_paths[path]:
                        failures.append(f"pre-existing cache differs from audited payload: {path}")
                    else:
                        existing_identical[target_budget] += 1

    written: dict[int, int] = {15: 0, 20: 0}
    if not failures:
        for target_budget in (15, 20):
            cache_dir = args.cache_root / f"cand{target_budget}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            for qid, payload in intended[target_budget].items():
                path = cache_file(cache_dir, qid)
                if path.exists():
                    continue
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                written[target_budget] += 1

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 6,
        "step": "6.3",
        "mode": "exact_context_answer_cache_preparation",
        "api_calls": 0,
        "protocol": {
            "checkpoint": FINAL_CHECKPOINT,
            **ANSWER_PROTOCOL,
            "reuse_rule": "question, gold answer, and ordered selected_unit_ids must match exactly",
            "source_preference_if_both_match": "cand10",
            "cache_root": str(args.cache_root),
        },
        "reports": {str(budget): str(path) for budget, path in sorted(paths.items())},
        "budgets": stats,
        "existing_identical_cache_files": {str(key): value for key, value in existing_identical.items()},
        "written_cache_files": {str(key): value for key, value in written.items()},
        "total_new_api_answers_required": sum(
            int(stats.get(str(budget), {}).get("new_api_answers_required", 0))
            for budget in (15, 20)
        ),
        "next_gate": (
            "Review this audit before authorizing answer generation."
            if not failures else "Resolve all failures; do not call the answer API."
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
