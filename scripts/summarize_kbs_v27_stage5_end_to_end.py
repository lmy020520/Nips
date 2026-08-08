#!/usr/bin/env python3
"""Audit and summarize v27 Stage-5 end-to-end reports across seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_kbs_standard_metrics import evaluate_report


EXPECTED_SEEDS = ("42", "43", "44")
SUMMARY_METRICS = (
    "answer_em",
    "answer_f1",
    "step_at_1",
    "step_at_5",
    "full_unit_coverage",
    "supporting_fact_f1",
    "joint_f1",
    "hop2_plus_step_at_1",
    "hop2_plus_step_at_5",
    "hop2_plus_mrr",
    "avg_answer_tokens",
    "selection_avg_ms_per_qid",
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("report must use SEED=PATH")
    seed, path = value.split("=", 1)
    return seed.strip(), Path(path.strip())


def mean_std(values: list[float]) -> dict:
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(statistics.mean(values), 6),
        "sample_std": round(statistics.stdev(values), 6),
    }


def report_metrics(
    seed: str,
    path: Path,
    operating_point: str,
) -> tuple[dict, list[str], list[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    records = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError(f"{path} must contain summary and results")

    failures = []
    if int(summary.get("qids", 0)) != 3000:
        failures.append(f"seed {seed}: qids={summary.get('qids')} != 3000")
    if int(summary.get("answer_judged", 0)) != 3000:
        failures.append(f"seed {seed}: answer_judged != 3000")
    if int(summary.get("answer_errors", -1)) != 0:
        failures.append(f"seed {seed}: answer_errors != 0")
    expected_candidate_top_k = 10 if operating_point == "compact" else 50
    for key, expected in (
        ("policy_blend_weight", 0.5),
        ("state_update_top_k", 1),
        ("candidate_top_k", expected_candidate_top_k),
        ("select_top_k", 5),
    ):
        if summary.get(key) != expected:
            failures.append(
                f"seed {seed}: {key}={summary.get(key)!r} != {expected!r}"
            )

    qids = []
    later_steps = []
    empty_answers = 0
    for record in records:
        qid = str(record.get("qid") or "")
        qids.append(qid)
        if not str(record.get("raw_answer") or "").strip():
            empty_answers += 1
        later_steps.extend(
            step
            for step in record.get("steps") or []
            if int(step.get("t", 0)) >= 1
        )
    if len(qids) != 3000 or len(set(qids)) != 3000 or not all(qids):
        failures.append(f"seed {seed}: missing or duplicate qids")
    if empty_answers:
        failures.append(f"seed {seed}: {empty_answers} empty raw answers")
    if not later_steps:
        failures.append(f"seed {seed}: no hop-2+ states")

    standard, _ = evaluate_report(f"seed{seed}", path, [10])
    for metric, check in standard["source_summary_checks"].items():
        if not check["match"]:
            failures.append(f"seed {seed}: summary replay mismatch for {metric}")

    def rank(step: dict) -> int | None:
        value = step.get("positive_rank")
        return int(value) if value is not None else None

    hop_count = len(later_steps)
    metrics = {
        "answer_em": float(summary["answer_em"]),
        "answer_f1": float(summary["answer_f1"]),
        "step_at_1": float(summary["step_acc@1"]),
        "step_at_5": float(summary["step_acc@5"]),
        "full_unit_coverage": float(summary["full_gold_unit_coverage"]),
        "supporting_fact_f1": float(standard["metrics"]["supporting_fact_f1"]),
        "joint_f1": float(standard["metrics"]["joint_f1"]),
        "hop2_plus_step_at_1": sum(rank(step) == 1 for step in later_steps) / hop_count,
        "hop2_plus_step_at_5": sum(
            rank(step) is not None and rank(step) <= 5 for step in later_steps
        ) / hop_count,
        "hop2_plus_mrr": sum(
            1.0 / rank(step) if rank(step) else 0.0 for step in later_steps
        ) / hop_count,
        "avg_answer_tokens": float(summary["avg_answer_tokens"]),
        "selection_avg_ms_per_qid": float(
            summary["runtime_profile"]["selection_avg_ms_per_qid"]
        ),
        "hop2_plus_states": hop_count,
    }
    protocol = {
        "checkpoint": summary.get("checkpoint"),
        "qids": summary.get("qids"),
        "steps": summary.get("steps"),
        "policy_blend_weight": summary.get("policy_blend_weight"),
        "state_update_top_k": summary.get("state_update_top_k"),
        "candidate_top_k": summary.get("candidate_top_k"),
        "select_top_k": summary.get("select_top_k"),
        "answer_cache_dir": summary.get("answer_cache_dir"),
    }
    return {"source_report": str(path), "protocol": protocol, "metrics": metrics}, qids, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=parse_named_path, required=True)
    parser.add_argument(
        "--operating-point",
        choices=("compact", "recall"),
        default="compact",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = dict(args.report)
    if set(paths) != set(EXPECTED_SEEDS):
        raise ValueError(f"reports must contain exactly seeds {EXPECTED_SEEDS}")

    per_seed = {}
    qids_by_seed = {}
    failures = []
    for seed in EXPECTED_SEEDS:
        per_seed[seed], qids_by_seed[seed], seed_failures = report_metrics(
            seed, paths[seed], args.operating_point
        )
        failures.extend(seed_failures)

    reference = qids_by_seed[EXPECTED_SEEDS[0]]
    for seed in EXPECTED_SEEDS[1:]:
        if qids_by_seed[seed] != reference:
            failures.append(f"seed {seed}: qid order differs from seed 42")

    aggregate = {
        metric: mean_std(
            [float(per_seed[seed]["metrics"][metric]) for seed in EXPECTED_SEEDS]
        )
        for metric in SUMMARY_METRICS
    }
    result = {
        "status": "PASS" if not failures else "FAIL",
        "scope": (
            f"v27 Stage-5 {args.operating_point.title()} end-to-end "
            "multiseed audit"
        ),
        "operating_point": args.operating_point,
        "seeds": list(EXPECTED_SEEDS),
        "identical_qid_order": all(
            qids_by_seed[seed] == reference for seed in EXPECTED_SEEDS
        ),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "rank_reversal_status": (
            "not computable from end-to-end reports; use fixed-pool state "
            "intervention records"
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
