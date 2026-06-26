#!/usr/bin/env python3
"""Aggregate KBS failure, trajectory, and deficit diagnostics.

This script is intentionally offline: it consumes an existing RAG report and,
optionally, a front-end trace report produced by diagnose_frontend_trace.py.
It does not call the LLM.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


DEFICIT_KEYS = ("d_br", "d_dis", "d_sup", "d_der")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0 or y_var <= 0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return round(cov / math.sqrt(x_var * y_var), 6)


def load_teacher_deficits(samples_path: Path) -> dict[tuple[str, int], dict[str, float]]:
    deficits = {}
    for row in read_jsonl(samples_path):
        qid = str(row.get("qid") or "")
        t = int(row.get("t", 0))
        labels = row.get("labels") or {}
        d_t_star = labels.get("d_t_star") or {}
        if qid and d_t_star:
            deficits[(qid, t)] = {key: float(d_t_star.get(key, 0.0)) for key in DEFICIT_KEYS}
    return deficits


def summarize_rag_report(report: dict, teacher_deficits: dict[tuple[str, int], dict[str, float]]) -> dict:
    summary = report.get("summary") or {}
    results = report.get("results") or []

    qid_total = len(results)
    trajectory_lengths = [len(row.get("steps") or []) for row in results]
    success_count = sum(1 for row in results if row.get("all_steps_correct"))
    failed_count = qid_total - success_count

    answer_success_by_qid = {}
    for row in results:
        qid = str(row.get("qid") or "")
        f1 = row.get("answer_f1")
        if f1 is None:
            answer_success_by_qid[qid] = None
        else:
            answer_success_by_qid[qid] = 1.0 if float(f1) >= 0.5 else 0.0

    positive_rank_counter = Counter()
    selected_contains_by_t = Counter()
    step_total_by_t = Counter()
    false_stop_count = 0
    stop_reasons = Counter()

    abs_errors = defaultdict(list)
    sq_errors = defaultdict(list)
    deficit_pairs_by_qid = defaultdict(list)
    predicted_deficit_for_corr = []
    answer_success_for_corr = []

    deficit_available_steps = 0
    teacher_deficit_available_steps = 0
    for row in results:
        qid = str(row.get("qid") or "")
        if row.get("stopped_early"):
            stop_reasons[str((row.get("stop_record") or {}).get("reason") or "unknown")] += 1
            gold_units = set(row.get("gold_unit_ids") or [])
            selected_units = set(row.get("selected_unit_ids") or [])
            if not gold_units.issubset(selected_units):
                false_stop_count += 1

        qid_deficit_values = []
        for step in row.get("steps") or []:
            t = int(step.get("t", 0))
            positive_rank_counter[str(step.get("positive_rank", "missing"))] += 1
            step_total_by_t[str(t)] += 1
            selected_contains_by_t[str(t)] += int(bool(step.get("selected_contains_gold")))

            teacher = teacher_deficits.get((qid, t))
            if teacher:
                teacher_deficit_available_steps += 1
            pred = step.get("deficit_estimate") or {}
            if not pred:
                continue
            deficit_available_steps += 1
            pred_mean = float(pred.get("mean", 0.0))
            qid_deficit_values.append(pred_mean)

            if teacher:
                for key in DEFICIT_KEYS:
                    error = float(pred.get(key, 0.0)) - float(teacher.get(key, 0.0))
                    abs_errors[key].append(abs(error))
                    sq_errors[key].append(error * error)
                deficit_pairs_by_qid[qid].append((t, pred_mean))

        success = answer_success_by_qid.get(qid)
        if success is not None and qid_deficit_values:
            predicted_deficit_for_corr.append(float(qid_deficit_values[-1]))
            answer_success_for_corr.append(success)

    role_wise = {}
    for key in DEFICIT_KEYS:
        values = abs_errors[key]
        role_wise[key] = {
            "mae": round(sum(values) / len(values), 6) if values else None,
            "mse": round(sum(sq_errors[key]) / len(sq_errors[key]), 6) if sq_errors[key] else None,
            "count": len(values),
        }

    all_abs = [value for values in abs_errors.values() for value in values]
    all_sq = [value for values in sq_errors.values() for value in values]

    monotonic_total = 0
    monotonic_non_increase = 0
    for pairs in deficit_pairs_by_qid.values():
        ordered = [value for _, value in sorted(pairs)]
        for left, right in zip(ordered, ordered[1:]):
            monotonic_total += 1
            monotonic_non_increase += int(right <= left + 1e-9)

    return {
        "report_summary": {
            key: summary.get(key)
            for key in (
                "qids",
                "steps",
                "trajectory_all_steps_correct",
                "full_gold_doc_coverage",
                "full_gold_unit_coverage",
                "step_selected_contains_gold",
                "answer_em",
                "answer_f1",
                "answer_contains",
                "avg_answer_tokens",
                "avg_answer_latency",
            )
        },
        "trajectory": {
            "qids": qid_total,
            "success_count": success_count,
            "failed_count": failed_count,
            "success_ratio": round(success_count / qid_total, 6) if qid_total else None,
            "failed_ratio": round(failed_count / qid_total, 6) if qid_total else None,
            "avg_trajectory_length": round(sum(trajectory_lengths) / qid_total, 6) if qid_total else None,
            "trajectory_length_distribution": dict(Counter(map(str, trajectory_lengths))),
            "false_stop_count": false_stop_count,
            "stop_reason_distribution": dict(stop_reasons),
        },
        "step": {
            "positive_rank_distribution": dict(positive_rank_counter),
            "selected_contains_gold_by_t": {
                t: round(selected_contains_by_t[t] / max(1, step_total_by_t[t]), 6)
                for t in sorted(step_total_by_t, key=lambda item: int(item))
            },
        },
        "deficit": {
            "available": deficit_available_steps > 0,
            "predicted_deficit_steps": deficit_available_steps,
            "teacher_deficit_steps": teacher_deficit_available_steps,
            "overall_mae": round(sum(all_abs) / len(all_abs), 6) if all_abs else None,
            "overall_mse": round(sum(all_sq) / len(all_sq), 6) if all_sq else None,
            "role_wise": role_wise,
            "monotonic_non_increase_rate": round(monotonic_non_increase / monotonic_total, 6)
            if monotonic_total
            else None,
            "monotonic_pairs": monotonic_total,
            "final_predicted_deficit_vs_answer_success_pearson": pearson(
                predicted_deficit_for_corr,
                answer_success_for_corr,
            ),
            "answer_success_correlation_pairs": len(predicted_deficit_for_corr),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-report", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--trace-report", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rag_report = read_json(Path(args.rag_report))
    teacher_deficits = load_teacher_deficits(Path(args.samples))
    output = summarize_rag_report(rag_report, teacher_deficits)

    if args.trace_report:
        trace = read_json(Path(args.trace_report))
        output["frontend_trace"] = {
            "core_breakdown": trace.get("core_breakdown"),
            "stage_hit_rates": trace.get("stage_hit_rates"),
            "rank_buckets": trace.get("rank_buckets"),
            "top1_error_types": trace.get("top1_error_types"),
            "trace_output": trace.get("trace_output"),
        }

    write_json(output, Path(args.output))
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
