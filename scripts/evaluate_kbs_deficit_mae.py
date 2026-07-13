#!/usr/bin/env python3
"""Evaluate typed-deficit prediction against teacher d_t* labels.

This is a direct calibration script for the student deficit head. It reads
trajectory samples that contain labels.d_t_star, runs the student model on the
state/candidate pairs, averages the predicted typed deficit over candidates,
and reports overall and role-wise MAE/MSE.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm import tqdm


DEFICIT_KEYS = ("d_br", "d_dis", "d_sup", "d_der")


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


def teacher_deficit(row: dict) -> dict[str, float] | None:
    labels = row.get("labels") or {}
    d_t_star = labels.get("d_t_star") or row.get("d_t_star") or {}
    if not isinstance(d_t_star, dict) or not d_t_star:
        return None
    return {key: float(d_t_star.get(key, 0.0)) for key in DEFICIT_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output", default="")
    args = parser.parse_args()

    # Import lazily so static checks do not require the full training env.
    from run_hotpotqa_policy_rag import (
        PolicyModel,
        format_candidate_text,
        load_memory,
        sample_candidate_ids,
        sample_k_t,
    )

    rows = list(read_jsonl(Path(args.samples)))
    if args.max_items > 0:
        rows = rows[: args.max_items]
    memory = load_memory(Path(args.memory))
    policy = PolicyModel(
        model_dir=Path(args.model_dir),
        checkpoint=Path(args.checkpoint),
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    totals = Counter()
    abs_errors: dict[str, list[float]] = defaultdict(list)
    sq_errors: dict[str, list[float]] = defaultdict(list)
    pred_mean_by_qid: dict[str, list[tuple[int, float]]] = defaultdict(list)
    teacher_mean_by_qid: dict[str, list[tuple[int, float]]] = defaultdict(list)
    pred_teacher_means: list[float] = []
    teacher_means: list[float] = []
    records = []

    for row in tqdm(rows, desc="deficit-mae"):
        totals["rows"] += 1
        qid = str(row.get("qid") or "")
        t = int(row.get("t", 0))
        teacher = teacher_deficit(row)
        if teacher is None:
            totals["missing_teacher_deficit"] += 1
            continue

        question = str(row.get("question") or "")
        context = f"Question: {question}\nNotebook:\n{sample_k_t(row)}"
        candidate_ids = sample_candidate_ids(row)
        candidate_texts = []
        usable_candidate_ids = []
        for unit_id in candidate_ids:
            item = memory.get(unit_id)
            if item:
                usable_candidate_ids.append(unit_id)
                candidate_texts.append(format_candidate_text(item))
        if not candidate_texts:
            totals["missing_candidates"] += 1
            continue

        ranking_scores, _, deficit_preds = policy.score_with_aux(context, candidate_texts, return_aux=True)
        if deficit_preds is None or len(deficit_preds) == 0:
            totals["missing_prediction"] += 1
            continue
        pred_values = np.mean(deficit_preds, axis=0)
        pred = {key: float(pred_values[index]) for index, key in enumerate(DEFICIT_KEYS)}
        positive_id = str(((row.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id") or "")
        positive_rank = None
        if positive_id in usable_candidate_ids:
            positive_index = usable_candidate_ids.index(positive_id)
            order = np.argsort(ranking_scores)[::-1].tolist()
            positive_rank = order.index(positive_index) + 1

        totals["evaluated"] += 1
        pred_mean = float(np.mean([pred[key] for key in DEFICIT_KEYS]))
        teacher_mean = float(np.mean([teacher[key] for key in DEFICIT_KEYS]))
        pred_mean_by_qid[qid].append((t, pred_mean))
        teacher_mean_by_qid[qid].append((t, teacher_mean))
        pred_teacher_means.append(pred_mean)
        teacher_means.append(teacher_mean)

        role_errors = {}
        for key in DEFICIT_KEYS:
            error = pred[key] - teacher[key]
            abs_error = abs(error)
            sq_error = error * error
            abs_errors[key].append(abs_error)
            sq_errors[key].append(sq_error)
            role_errors[key] = {
                "pred": round(pred[key], 6),
                "teacher": round(teacher[key], 6),
                "abs_error": round(abs_error, 6),
            }

        records.append(
            {
                "qid": qid,
                "t": t,
                "pred_mean": round(pred_mean, 6),
                "teacher_mean": round(teacher_mean, 6),
                "positive_rank": positive_rank,
                "step_correct": positive_rank == 1 if positive_rank is not None else None,
                "role_errors": role_errors,
            }
        )

    role_wise = {}
    for key in DEFICIT_KEYS:
        role_wise[key] = {
            "mae": round(sum(abs_errors[key]) / len(abs_errors[key]), 6)
            if abs_errors[key]
            else None,
            "mse": round(sum(sq_errors[key]) / len(sq_errors[key]), 6)
            if sq_errors[key]
            else None,
            "count": len(abs_errors[key]),
        }

    all_abs = [value for values in abs_errors.values() for value in values]
    all_sq = [value for values in sq_errors.values() for value in values]

    pred_monotonic_total = 0
    pred_monotonic_non_increase = 0
    teacher_monotonic_total = 0
    teacher_monotonic_non_increase = 0
    for qid, pairs in pred_mean_by_qid.items():
        ordered_pred = [value for _, value in sorted(pairs)]
        for left, right in zip(ordered_pred, ordered_pred[1:]):
            pred_monotonic_total += 1
            pred_monotonic_non_increase += int(right <= left + 1e-9)
        ordered_teacher = [value for _, value in sorted(teacher_mean_by_qid.get(qid, []))]
        for left, right in zip(ordered_teacher, ordered_teacher[1:]):
            teacher_monotonic_total += 1
            teacher_monotonic_non_increase += int(right <= left + 1e-9)

    report = {
        "samples": args.samples,
        "memory": args.memory,
        "checkpoint": args.checkpoint,
        "model_dir": args.model_dir,
        "note": (
            "MAE is computed by directly matching each sample's labels.d_t_star "
            "with the student deficit head prediction averaged over that sample's candidates."
        ),
        "rows": totals["rows"],
        "evaluated": totals["evaluated"],
        "missing_teacher_deficit": totals["missing_teacher_deficit"],
        "missing_candidates": totals["missing_candidates"],
        "missing_prediction": totals["missing_prediction"],
        "overall_mae": round(sum(all_abs) / len(all_abs), 6) if all_abs else None,
        "overall_mse": round(sum(all_sq) / len(all_sq), 6) if all_sq else None,
        "role_wise": role_wise,
        "predicted_mean_vs_teacher_mean_pearson": pearson(pred_teacher_means, teacher_means),
        "predicted_deficit_monotonic_non_increase_rate": round(
            pred_monotonic_non_increase / pred_monotonic_total,
            6,
        )
        if pred_monotonic_total
        else None,
        "teacher_deficit_monotonic_non_increase_rate": round(
            teacher_monotonic_non_increase / teacher_monotonic_total,
            6,
        )
        if teacher_monotonic_total
        else None,
        "monotonic_pairs": {
            "predicted": pred_monotonic_total,
            "teacher": teacher_monotonic_total,
        },
    }
    if totals["evaluated"] == 0:
        report["status"] = "NO_EVALUATED_ROWS"
        report["diagnosis"] = (
            "No sample had both labels.d_t_star and valid candidates. "
            "Use a sample file with teacher deficit labels, for example the v7/v16 labeled samples."
        )
    else:
        report["status"] = "OK"

    write_json(report, Path(args.output))
    if args.records_output:
        Path(args.records_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.records_output).open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if args.records_output:
        print(f"records: {args.records_output}")


if __name__ == "__main__":
    main()
