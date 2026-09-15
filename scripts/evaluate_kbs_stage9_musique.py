#!/usr/bin/env python3
"""Compute alias-aware MuSiQue paragraph and downstream metrics offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_kbs_standard_metrics import answer_metrics, set_metrics
except ModuleNotFoundError:
    from evaluate_kbs_standard_metrics import answer_metrics, set_metrics


METHODS = (
    "KSG-EA-Compact",
    "KSG-EA-Balanced",
    "Hybrid-RAG",
    "BGE-Reranker-RAG",
)
ANSWER_PROTOCOL = {
    "answer_model": "deepseek-v4-flash",
    "answer_thinking_mode": "disabled",
    "answer_mode": "json",
    "answer_temperature": 0.0,
    "answer_prompt_version": "kbs_extractive_answer_json_v1",
    "answer_reference_mode": "max_over_canonical_and_aliases",
}


def parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("report must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if name not in METHODS:
        raise argparse.ArgumentTypeError(f"unsupported method: {name}")
    return name, Path(raw_path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def title_metrics(predicted: list[Any], gold: list[Any]) -> dict[str, float]:
    predicted_set = {str(value) for value in predicted if value}
    gold_set = {str(value) for value in gold if value}
    if not gold_set:
        raise ValueError("row has no gold support titles")
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support_title_precision": precision,
        "support_title_recall": recall,
        "support_title_f1": f1,
        "support_title_em": float(predicted_set == gold_set),
        "full_support_title_coverage": float(gold_set.issubset(predicted_set)),
    }


def alignment_at_k(row: dict[str, Any], k: int) -> tuple[float, float]:
    hits = 0
    steps = row.get("steps") or []
    for step in steps:
        rank = step.get("positive_rank")
        if isinstance(rank, (int, float)):
            hits += int(int(rank) <= k)
        else:
            target = str(step.get("positive_unit_id") or "")
            selected = [str(value) for value in step.get("selected_unit_ids") or []]
            hits += int(bool(target) and target in selected[:k])
    return float(hits), float(len(steps))


def evaluate_report(
    name: str,
    path: Path,
    expected_qids: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    obj = read_json(path)
    summary = obj.get("summary")
    source_rows = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(source_rows, list):
        raise ValueError(f"report lacks summary/results: {path}")

    failures = []
    expected_summary = {
        **ANSWER_PROTOCOL,
        "generate_answers": True,
        "answer_judged": expected_qids,
        "answer_errors": 0,
        "qids": expected_qids,
        "skipped": 0,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            failures.append(f"{name}: summary {key}={summary.get(key)!r} != {expected!r}")
    if len(source_rows) != expected_qids:
        failures.append(f"{name}: records={len(source_rows)} != {expected_qids}")

    seen = set()
    records = []
    alignment_totals = {1: [0.0, 0.0], 5: [0.0, 0.0]}
    for source in source_rows:
        qid = str(source.get("qid") or "")
        if not qid or qid in seen:
            raise ValueError(f"{name}: empty or duplicate qid={qid!r}")
        seen.add(qid)
        answer = answer_metrics(
            source.get("answer", ""),
            source.get("gold_answer", ""),
            source.get("gold_answer_aliases") or [],
        )
        if (
            float(source.get("answer_em") or 0.0) != answer["answer_em"]
            or abs(float(source.get("answer_f1") or 0.0) - answer["answer_f1"])
            > 1e-5
        ):
            failures.append(f"{name}: alias-aware answer replay differs for qid={qid}")

        gold_ids = source.get("gold_unit_ids") or []
        predicted_chain = [
            step.get("predicted_unit_id")
            for step in source.get("steps") or []
            if step.get("predicted_unit_id")
        ]
        paragraph = set_metrics(predicted_chain, gold_ids, qid)
        selected = set_metrics(source.get("selected_unit_ids") or [], gold_ids, qid)
        titles = title_metrics(
            source.get("selected_doc_ids") or [],
            source.get("gold_doc_ids") or [],
        )
        joint_precision = answer["answer_precision"] * paragraph["precision"]
        joint_recall = answer["answer_recall"] * paragraph["recall"]
        joint_f1 = (
            2 * joint_precision * joint_recall / (joint_precision + joint_recall)
            if joint_precision + joint_recall
            else 0.0
        )
        record = {
            "qid": qid,
            **answer,
            "supporting_paragraph_precision": paragraph["precision"],
            "supporting_paragraph_recall": paragraph["recall"],
            "supporting_paragraph_f1": paragraph["f1"],
            "supporting_paragraph_em": paragraph["em"],
            "selected_paragraph_precision": selected["precision"],
            "selected_paragraph_recall": selected["recall"],
            "selected_paragraph_f1": selected["f1"],
            "selected_paragraph_em": selected["em"],
            "full_support_paragraph_coverage": selected["full_coverage"],
            "selected_paragraph_count": selected["predicted_count"],
            "gold_paragraph_count": selected["gold_count"],
            **titles,
            "joint_em": answer["answer_em"] * paragraph["em"],
            "joint_precision": joint_precision,
            "joint_recall": joint_recall,
            "joint_f1": joint_f1,
        }
        for budget in (10, 15):
            record[f"closure_success_at_{budget}"] = float(
                answer["answer_em"] == 1.0
                and selected["full_coverage"] == 1.0
                and selected["predicted_count"] <= budget
            )
        for k in alignment_totals:
            numerator, denominator = alignment_at_k(source, k)
            alignment_totals[k][0] += numerator
            alignment_totals[k][1] += denominator
        records.append(record)

    metric_keys = [
        "answer_em",
        "answer_precision",
        "answer_recall",
        "answer_f1",
        "supporting_paragraph_precision",
        "supporting_paragraph_recall",
        "supporting_paragraph_f1",
        "supporting_paragraph_em",
        "selected_paragraph_precision",
        "selected_paragraph_recall",
        "selected_paragraph_f1",
        "selected_paragraph_em",
        "full_support_paragraph_coverage",
        "support_title_precision",
        "support_title_recall",
        "support_title_f1",
        "support_title_em",
        "full_support_title_coverage",
        "joint_em",
        "joint_precision",
        "joint_recall",
        "joint_f1",
        "closure_success_at_10",
        "closure_success_at_15",
        "selected_paragraph_count",
        "gold_paragraph_count",
    ]
    metrics = {key: round(mean(records, key), 6) for key in metric_keys}
    for k, (numerator, denominator) in alignment_totals.items():
        metrics[f"paragraph_alignment_at_{k}"] = round(numerator / denominator, 6)

    checks = {}
    for source_key, metric_key in (
        ("answer_em", "answer_em"),
        ("answer_f1", "answer_f1"),
        ("step_acc@1", "paragraph_alignment_at_1"),
        ("step_acc@5", "paragraph_alignment_at_5"),
        ("full_gold_unit_coverage", "full_support_paragraph_coverage"),
        ("full_gold_doc_coverage", "full_support_title_coverage"),
    ):
        source_value = summary.get(source_key)
        recomputed = metrics[metric_key]
        matches = source_value is not None and abs(float(source_value) - recomputed) <= 1e-5
        checks[source_key] = {
            "source": source_value,
            "recomputed": recomputed,
            "match": matches,
        }
        if not matches:
            failures.append(f"{name}: source summary mismatch for {source_key}")

    method_summary = {
        "name": name,
        "source_report": str(path),
        "qids": len(records),
        "unit_granularity": "paragraph",
        "answer_reference_mode": ANSWER_PROTOCOL["answer_reference_mode"],
        "metrics": metrics,
        "source_summary_checks": checks,
    }
    return method_summary, records, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=parse_report, required=True)
    parser.add_argument("--answer-audit", action="append", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument("--expected-qids", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    reports = dict(args.report)
    if len(reports) != len(args.report):
        failures.append("duplicate report method")
    if set(reports) != set(METHODS):
        failures.append(f"report methods={sorted(reports)} != {sorted(METHODS)}")

    cache_summary = read_json(args.cache_summary)
    if (
        cache_summary.get("status") != "OK"
        or cache_summary.get("eligible_source_contexts") != 3758
        or cache_summary.get("total_target_contexts") != 4000
        or cache_summary.get("fresh_target_answers_without_cross_method_propagation") != 0
        or cache_summary.get("unique_fresh_contexts_with_cross_method_deduplication") != 0
        or cache_summary.get("failures")
    ):
        failures.append("final cache-propagation summary is not a clean closed ledger")

    audits = {}
    for path in args.answer_audit:
        audit = read_json(path)
        method = str(audit.get("method") or "")
        if method in audits:
            failures.append(f"duplicate answer audit method={method!r}")
        audits[method] = audit
        if (
            audit.get("status") != "OK"
            or audit.get("qids") != args.expected_qids
            or audit.get("failures")
        ):
            failures.append(f"answer audit is not a clean OK: {path}")
    expected_audits = {"compact_seed42", "balanced_seed42", "hybrid", "bge_reranker"}
    if set(audits) != expected_audits:
        failures.append(f"answer audit methods={sorted(audits)} != {sorted(expected_audits)}")

    methods = {}
    all_records = {}
    orders = {}
    for name in METHODS:
        if name not in reports:
            continue
        method, records, method_failures = evaluate_report(
            name, reports[name], args.expected_qids
        )
        methods[name] = method
        all_records[name] = records
        orders[name] = [record["qid"] for record in records]
        failures.extend(method_failures)
    if orders:
        reference = orders[METHODS[0]]
        for name, order in orders.items():
            if order != reference:
                failures.append(f"ordered qids differ for {name}")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.6",
        "mode": "musique_zero_shot_downstream_metrics",
        "api_calls": 0,
        "qids": args.expected_qids,
        "dataset": "fixed MuSiQue-Ans development subset",
        "student_training": "HotpotQA only; no MuSiQue fine-tuning",
        "unit_granularity": "paragraph",
        "metric_boundary": (
            "Paragraph evidence metrics are separate from sentence-level "
            "HotpotQA/2Wiki supporting-fact metrics."
        ),
        "answer_protocol": ANSWER_PROTOCOL,
        "cache_accounting": {
            "target_contexts": cache_summary.get("total_target_contexts"),
            "unique_generated_contexts": cache_summary.get("eligible_source_contexts"),
            "exact_reused_target_files": (
                int(cache_summary.get("total_target_contexts") or 0)
                - int(cache_summary.get("eligible_source_contexts") or 0)
            ),
        },
        "methods": methods,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.records_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.records_output.open("w", encoding="utf-8") as stream:
        for name in METHODS:
            for record in all_records.get(name, []):
                stream.write(json.dumps({"method": name, **record}, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    print(f"records: {args.records_output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
