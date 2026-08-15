#!/usr/bin/env python3
"""Audit artifacts and freeze the final-v27 Stage 6.3 cost-frontier protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FINAL_CHECKPOINT = "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
EXPECTED_VARIANTS = {
    "rrf_top30",
    "rrf_local_mmr10",
    "bge_top10",
    "no_compression",
    "oracle_target_preserving_mmr10",
}


def read_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def audit_answer_report(path: Path, *, budget: int, front_pool_k: int) -> tuple[dict, list[str]]:
    failures = []
    obj = read_json(path)
    summary = obj.get("summary")
    records = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError(f"report lacks summary/results: {path}")
    expected = {
        "checkpoint": FINAL_CHECKPOINT,
        "qids": 3000,
        "steps": 7296,
        "front_pool_k": front_pool_k,
        "candidate_top_k": budget,
        "select_top_k": 5,
        "state_update_top_k": 1,
        "policy_blend_weight": 0.5,
        "answer_judged": 3000,
        "answer_errors": 0,
        "answer_model": "deepseek-v4-flash",
        "answer_thinking_mode": "disabled",
        "answer_prompt_version": "kbs_extractive_answer_json_v1",
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            failures.append(f"{path}: {key}={summary.get(key)!r} != {value!r}")
    qids = [str(record.get("qid") or "") for record in records]
    empty_answers = sum(not str(record.get("raw_answer") or "").strip() for record in records)
    missing_fields = Counter()
    for record in records:
        for key in ("selected_unit_ids", "gold_unit_ids", "steps", "answer_f1"):
            missing_fields[key] += int(record.get(key) is None)
    if len(records) != 3000 or len(set(qids)) != 3000 or not all(qids):
        failures.append(f"{path}: expected 3,000 unique non-empty qids")
    if empty_answers:
        failures.append(f"{path}: {empty_answers} empty raw answers")
    if any(missing_fields.values()):
        failures.append(f"{path}: missing result fields {dict(missing_fields)}")
    runtime = summary.get("runtime_profile") or {}
    for key in ("selection_avg_ms_per_qid", "peak_gpu_allocated_mb"):
        if runtime.get(key) is None:
            failures.append(f"{path}: runtime_profile.{key} is missing")
    if runtime.get("includes_answer_api") is not False:
        failures.append(f"{path}: selection runtime must exclude answer API")
    return {
        "path": str(path),
        "budget": budget,
        "front_pool_k": front_pool_k,
        "qids": len(records),
        "answer_f1": summary.get("answer_f1"),
        "full_unit_coverage": summary.get("full_gold_unit_coverage"),
        "avg_answer_tokens": summary.get("avg_answer_tokens"),
        "answer_latency_ms_per_qid": (
            round(1000.0 * float(summary["avg_answer_latency"]), 3)
            if summary.get("avg_answer_latency") is not None else None
        ),
        "selection_ms_per_qid": runtime.get("selection_avg_ms_per_qid"),
        "peak_gpu_allocated_mb": runtime.get("peak_gpu_allocated_mb"),
        "answer_cache_dir": summary.get("answer_cache_dir"),
        "empty_answers": empty_answers,
    }, failures


def audit_compression(summary_path: Path, records_path: Path) -> tuple[dict, list[str]]:
    failures = []
    obj = read_json(summary_path)
    protocol = obj.get("protocol") or {}
    variants = obj.get("variants") or {}
    expected_protocol = {
        "checkpoint": FINAL_CHECKPOINT,
        "qids": 3000,
        "front_pool_k": 30,
        "candidate_top_k": 10,
        "select_top_k": 5,
        "state_update_top_k": 1,
        "policy_blend_weight": 0.5,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            failures.append(
                f"compression summary: {key}={protocol.get(key)!r} != {value!r}"
            )
    if set(variants) != EXPECTED_VARIANTS:
        failures.append(f"compression variants differ: {sorted(variants)}")
    for name, metrics in variants.items():
        if metrics.get("qids") != 3000 or metrics.get("steps") != 7296:
            failures.append(f"compression {name}: incomplete qids/states")
        if metrics.get("skipped") != 0:
            failures.append(f"compression {name}: skipped={metrics.get('skipped')}")

    line_count = 0
    variant_qids: dict[str, set[str]] = {name: set() for name in EXPECTED_VARIANTS}
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            line_count += 1
            row = json.loads(line)
            variant = str(row.get("variant") or "")
            qid = str(row.get("qid") or "")
            if variant in variant_qids and qid:
                variant_qids[variant].add(qid)
            else:
                failures.append(f"invalid compression record at line {line_number}")
    if line_count != 15000:
        failures.append(f"compression records={line_count} != 15000")
    for variant, qids in variant_qids.items():
        if len(qids) != 3000:
            failures.append(f"compression {variant}: record qids={len(qids)} != 3000")
    return {
        "summary": str(summary_path),
        "records": str(records_path),
        "records_count": line_count,
        "qids_per_variant": {name: len(qids) for name, qids in sorted(variant_qids.items())},
    }, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compact-report", type=Path,
        default=Path("outputs/rag/kbs_v27_final_hotpot/full_compact.json"),
    )
    parser.add_argument(
        "--recall-report", type=Path,
        default=Path("outputs/rag/kbs_v27_stage5_multiseed/seed42/full_recall.json"),
    )
    parser.add_argument(
        "--compression-summary", type=Path,
        default=Path("outputs/analysis/kbs_stage6_compression_funnel_full3000/summary.json"),
    )
    parser.add_argument(
        "--compression-records", type=Path,
        default=Path("outputs/analysis/kbs_stage6_compression_funnel_full3000/records.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/analysis/kbs_stage6_cost_frontier/readiness.json"),
    )
    args = parser.parse_args()

    required = [
        Path("md/kbs_three_review_execution_plan.md"),
        Path(FINAL_CHECKPOINT),
        Path("models/deberta-v3-large"),
        Path("models/bge-large-en-v1.5"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl"),
        args.compact_report,
        args.recall_report,
        args.compression_summary,
        args.compression_records,
    ]
    failures = [f"missing required path: {path}" for path in required if not path.exists()]
    reports = {}
    compression = {}
    if not failures:
        try:
            reports["10"], report_failures = audit_answer_report(
                args.compact_report, budget=10, front_pool_k=30
            )
            failures.extend(report_failures)
            reports["50"], report_failures = audit_answer_report(
                args.recall_report, budget=50, front_pool_k=50
            )
            failures.extend(report_failures)
            compression, compression_failures = audit_compression(
                args.compression_summary, args.compression_records
            )
            failures.extend(compression_failures)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(str(exc))

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 6,
        "step": "6.3",
        "mode": "cost_frontier_readiness",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "protocol": {
            "checkpoint": FINAL_CHECKPOINT,
            "budgets": [10, 15, 20, 50],
            "front_pool_schedule": {"10": 30, "15": 30, "20": 30, "50": 50},
            "front_pool_schedule_note": (
                "This is the established system operating-point frontier, not a "
                "single-factor candidate-budget ablation; Stage 6.2 supplies the "
                "controlled compression intervention."
            ),
            "select_top_k": 5,
            "state_update_top_k": 1,
            "policy_blend_weight": 0.5,
            "answer_model": "deepseek-v4-flash",
            "answer_thinking_mode": "disabled",
            "answer_prompt_version": "kbs_extractive_answer_json_v1",
            "fresh_budget_specific_cache_required": True,
        },
        "existing_answer_reports": reports,
        "compression_artifact_audit": compression,
        "missing_budget_reports": [15, 20],
        "next_gate": (
            "Run one no-answer 20-qid smoke for cand15 and cand20; do not call "
            "the API or start full runs until both smoke reports are reviewed."
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
