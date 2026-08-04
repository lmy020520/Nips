#!/usr/bin/env python3
"""Compute HotpotQA evidence, answer, joint, and closure metrics offline."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


def normalize_answer(value: object) -> str:
    text = str(value).lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_metrics(prediction: object, gold: object) -> dict[str, float]:
    prediction_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    exact_match = float(prediction_norm == gold_norm)
    if (
        prediction_norm in SPECIAL_ANSWERS
        or gold_norm in SPECIAL_ANSWERS
    ) and prediction_norm != gold_norm:
        return {"answer_em": exact_match, "answer_precision": 0.0,
                "answer_recall": 0.0, "answer_f1": 0.0}

    prediction_tokens = prediction_norm.split()
    gold_tokens = gold_norm.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not prediction_tokens or not gold_tokens:
        precision = recall = f1 = float(prediction_tokens == gold_tokens)
    elif overlap == 0:
        precision = recall = f1 = 0.0
    else:
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "answer_em": exact_match,
        "answer_precision": precision,
        "answer_recall": recall,
        "answer_f1": f1,
    }


def unit_pair(unit_id: object, qid: str) -> tuple[str, int]:
    value = str(unit_id)
    prefix = f"{qid}::"
    if not value.startswith(prefix):
        raise ValueError(f"unit does not belong to qid={qid}: {value}")
    body = value[len(prefix):]
    try:
        title, sent_id = body.rsplit("::", 1)
        return title, int(sent_id)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid unit id: {value}") from exc


def set_metrics(predicted_ids: list, gold_ids: list, qid: str) -> dict[str, float]:
    predicted = {unit_pair(value, qid) for value in predicted_ids}
    gold = {unit_pair(value, qid) for value in gold_ids}
    if not gold:
        raise ValueError(f"qid={qid} has no gold supporting facts")
    true_positive = len(predicted & gold)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / (true_positive + false_positive) if predicted else 0.0
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = float(false_positive == 0 and false_negative == 0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "em": exact_match,
        "full_coverage": float(false_negative == 0),
        "predicted_count": float(len(predicted)),
        "gold_count": float(len(gold)),
    }


def joint_metrics(answer: dict[str, float], evidence: dict[str, float]) -> dict[str, float]:
    precision = answer["answer_precision"] * evidence["supporting_fact_precision"]
    recall = answer["answer_recall"] * evidence["supporting_fact_recall"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "joint_em": answer["answer_em"] * evidence["supporting_fact_em"],
        "joint_precision": precision,
        "joint_recall": recall,
        "joint_f1": f1,
    }


def parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("report must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("report must use non-empty NAME=PATH")
    return name.strip(), Path(path)


def parse_budgets(value: str) -> list[int]:
    budgets = sorted({int(item) for item in value.split(",") if item.strip()})
    if not budgets or any(item <= 0 for item in budgets):
        raise argparse.ArgumentTypeError("closure budgets must be positive integers")
    return budgets


def mean(records: list[dict], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records)


def evaluate_report(name: str, path: Path, budgets: list[int]) -> tuple[dict, list[dict]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    source_summary = obj.get("summary")
    source_records = obj.get("results")
    if not isinstance(source_summary, dict) or not isinstance(source_records, list):
        raise ValueError(f"{path} must contain summary and results")

    records = []
    seen = set()
    source_replay = {"answer_em": [], "answer_f1": []}
    for source in source_records:
        qid = str(source.get("qid") or "")
        if not qid or qid in seen:
            raise ValueError(f"missing or duplicate qid in {path}: {qid!r}")
        seen.add(qid)
        for key in source_replay:
            if source.get(key) is None:
                raise ValueError(f"qid={qid} is missing source per-qid {key}")
            source_replay[key].append(float(source[key]))
        answer = answer_metrics(source.get("answer", ""), source.get("gold_answer", ""))
        gold_ids = source.get("gold_unit_ids") or []
        predicted_fact_ids = [
            step.get("predicted_unit_id")
            for step in source.get("steps") or []
            if step.get("predicted_unit_id")
        ]
        facts = set_metrics(predicted_fact_ids, gold_ids, qid)
        selected = set_metrics(
            source.get("selected_unit_ids") or [],
            gold_ids,
            qid,
        )
        evidence = {
            "supporting_fact_precision": facts["precision"],
            "supporting_fact_recall": facts["recall"],
            "supporting_fact_f1": facts["f1"],
            "supporting_fact_em": facts["em"],
            "selected_evidence_precision": selected["precision"],
            "selected_evidence_recall": selected["recall"],
            "selected_evidence_f1": selected["f1"],
            "selected_evidence_em": selected["em"],
            "full_support_coverage": selected["full_coverage"],
            "selected_evidence_count": selected["predicted_count"],
            "gold_evidence_count": selected["gold_count"],
        }
        joint = joint_metrics(answer, evidence)
        record = {"qid": qid, **answer, **evidence, **joint}
        for budget in budgets:
            record[f"closure_success_at_{budget}"] = float(
                answer["answer_em"] == 1.0
                and evidence["full_support_coverage"] == 1.0
                and evidence["selected_evidence_count"] <= budget
            )
        records.append(record)

    metric_keys = [
        "answer_em", "answer_precision", "answer_recall", "answer_f1",
        "supporting_fact_precision", "supporting_fact_recall",
        "supporting_fact_f1", "supporting_fact_em", "full_support_coverage",
        "selected_evidence_precision", "selected_evidence_recall",
        "selected_evidence_f1", "selected_evidence_em",
        "selected_evidence_count", "gold_evidence_count",
        "joint_em", "joint_precision", "joint_recall", "joint_f1",
    ] + [f"closure_success_at_{budget}" for budget in budgets]
    metrics = {key: round(mean(records, key), 6) for key in metric_keys}
    checks = {}
    for source_key, computed_key in (
        ("answer_em", "answer_em"),
        ("answer_f1", "answer_f1"),
        ("full_gold_unit_coverage", "full_support_coverage"),
    ):
        source_value = source_summary.get(source_key)
        if source_value is None:
            continue
        if source_key in source_replay:
            recomputed = round(sum(source_replay[source_key]) / len(records), 6)
        else:
            recomputed = metrics[computed_key]
        delta = recomputed - float(source_value)
        checks[source_key] = {
            "source": round(float(source_value), 6),
            "recomputed": recomputed,
            "delta": round(delta, 9),
            "match": abs(delta) <= 1e-5,
        }
    summary = {
        "name": name,
        "source_report": str(path),
        "source_checkpoint": source_summary.get("checkpoint"),
        "qids": len(records),
        "closure_cost": "number of unique selected (title, sentence_id) evidence units",
        "closure_budgets": budgets,
        "metrics": metrics,
        "answer_metric_note": (
            "metrics use official HotpotQA special-answer handling; source-summary "
            "checks replay the historical per-qid values stored in each report"
        ),
        "source_summary_checks": checks,
    }
    return summary, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=parse_report, required=True)
    parser.add_argument("--gold-oracle-name", default="Gold-Oracle")
    parser.add_argument("--expected-qids", type=int, default=3000)
    parser.add_argument("--closure-unit-budgets", type=parse_budgets, default=[5, 10, 15, 20, 50])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = {}
    all_records = {}
    qid_sets = {}
    failures = []
    for name, path in args.report:
        if name in methods:
            raise ValueError(f"duplicate method name: {name}")
        summary, records = evaluate_report(name, path, args.closure_unit_budgets)
        methods[name] = summary
        all_records[name] = records
        qid_sets[name] = {record["qid"] for record in records}
        if summary["qids"] != args.expected_qids:
            failures.append(f"{name}: expected {args.expected_qids} qids, found {summary['qids']}")
        for metric, check in summary["source_summary_checks"].items():
            if not check["match"]:
                failures.append(f"{name}: source summary mismatch for {metric}")

    reference_name = next(iter(qid_sets))
    for name, qids in qid_sets.items():
        if qids != qid_sets[reference_name]:
            failures.append(
                f"{name}: qid set differs from {reference_name} "
                f"({len(qids)} versus {len(qid_sets[reference_name])})"
            )

    oracle = methods.get(args.gold_oracle_name)
    oracle_check = None
    if oracle is None:
        failures.append(f"missing gold oracle method: {args.gold_oracle_name}")
    else:
        metrics = oracle["metrics"]
        oracle_check = {
            "full_support_coverage": metrics["full_support_coverage"],
            "supporting_fact_recall": metrics["supporting_fact_recall"],
            "supporting_fact_em": metrics["supporting_fact_em"],
            "recall_upper_bound_pass": (
                abs(metrics["full_support_coverage"] - 1.0) <= 1e-9
                and abs(metrics["supporting_fact_recall"] - 1.0) <= 1e-9
            ),
            "em_upper_bound_pass": abs(metrics["supporting_fact_em"] - 1.0) <= 1e-9,
        }
        if not oracle_check["recall_upper_bound_pass"]:
            failures.append("gold oracle does not reach full supporting-fact recall")
        if not oracle_check["em_upper_bound_pass"]:
            failures.append("gold oracle does not reach supporting-fact EM upper bound")

    result = {
        "status": "OK" if not failures else "FAIL",
        "expected_qids": args.expected_qids,
        "identical_qids_required": True,
        "metric_semantics": {
            "supporting_fact_prediction": "unique per-step predicted_unit_id chain",
            "selected_evidence_prediction": "unique final selected_unit_ids answer context",
            "unit_mapping": "qid::title::sentence_id parsed from the right",
            "joint": "HotpotQA product precision/recall and product EM",
            "closure_success": "answer EM AND full supporting-fact coverage AND unique-unit budget",
        },
        "methods": methods,
        "gold_oracle_check": oracle_check,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.records_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.records_output.open("w", encoding="utf-8") as stream:
        for name, records in all_records.items():
            for record in records:
                stream.write(json.dumps({"method": name, **record}, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    print(f"records: {args.records_output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
