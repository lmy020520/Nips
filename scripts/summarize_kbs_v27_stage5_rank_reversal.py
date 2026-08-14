#!/usr/bin/env python3
"""Summarize v27 fixed-pool rank reversal across seeds 42, 43, and 44."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = ("42", "43", "44")
MODES = ("correct", "query_only", "frozen", "previous_evidence_only")
METRICS = (
    "current_preference_accuracy",
    "next_preference_accuracy",
    "conditional_rank_reversal_accuracy",
)
PAIRED_BASELINES = ("query_only", "frozen", "previous_evidence_only")
PAIRED_METRICS = ("step_at_1", "step_at_5", "mrr", "gold_margin")
NECESSITY_FIELDS = (
    "rescued_at_1",
    "harmed_at_1",
    "rescued_at_5",
    "harmed_at_5",
    "rank_improved",
    "rank_degraded",
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("summary must use SEED=PATH")
    seed, path = value.split("=", 1)
    return seed.strip(), Path(path.strip())


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean_std(values: list[float]) -> dict:
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(statistics.mean(values), 6),
        "sample_std": round(statistics.stdev(values), 6),
    }


def pair_key(record: dict) -> tuple:
    return (
        str(record.get("qid")),
        int(record.get("t", 0)),
        int(record.get("next_t", 0)),
        str(record.get("current_positive")),
        str(record.get("next_positive")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=parse_named_path, required=True)
    parser.add_argument("--expected-qids", type=int, default=3000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = dict(args.summary)
    if set(paths) != set(SEEDS):
        raise ValueError(f"summaries must contain exactly seeds {SEEDS}")

    failures = []
    per_seed = {}
    records_by_seed = {}
    for seed in SEEDS:
        path = paths[seed]
        obj = json.loads(path.read_text(encoding="utf-8"))
        reversal = obj.get("conditional_rank_reversal") or {}
        modes = reversal.get("modes") or {}
        if obj.get("status") != "OK":
            failures.append(f"seed {seed}: summary status is not OK")
        if int((obj.get("data_diagnostics") or {}).get("qids", 0)) != args.expected_qids:
            failures.append(f"seed {seed}: qids != {args.expected_qids}")
        if set(MODES) - set(modes):
            failures.append(f"seed {seed}: missing rank-reversal modes")
        record_path = path.parent / "rank_reversal_records.jsonl"
        if not record_path.is_file():
            failures.append(f"seed {seed}: missing {record_path}")
            records = []
        else:
            records = read_jsonl(record_path)
        if len(records) != int(reversal.get("eligible_pairs", -1)):
            failures.append(f"seed {seed}: rank-reversal record count mismatch")
        records_by_seed[seed] = records
        per_seed[seed] = {
            "source_summary": str(path),
            "checkpoint": obj.get("checkpoint"),
            "qids": (obj.get("data_diagnostics") or {}).get("qids"),
            "evaluated_states": obj.get("evaluated_states"),
            "eligible_pairs": reversal.get("eligible_pairs"),
            "modes": {mode: modes.get(mode) for mode in MODES},
            "state_necessity": obj.get("state_necessity"),
            "paired_bootstrap_t_ge_1_given_pool": obj.get(
                "paired_bootstrap_t_ge_1_given_pool"
            ),
        }

    reference_keys = [pair_key(record) for record in records_by_seed["42"]]
    for seed in ("43", "44"):
        if [pair_key(record) for record in records_by_seed[seed]] != reference_keys:
            failures.append(f"seed {seed}: eligible pair order differs from seed 42")

    aggregate = {
        mode: {
            metric: mean_std(
                [float(per_seed[seed]["modes"][mode][metric]) for seed in SEEDS]
            )
            for metric in METRICS
        }
        for mode in MODES
    }
    deltas = {}
    for baseline in MODES[1:]:
        values = [
            float(per_seed[seed]["modes"]["correct"]["conditional_rank_reversal_accuracy"])
            - float(per_seed[seed]["modes"][baseline]["conditional_rank_reversal_accuracy"])
            for seed in SEEDS
        ]
        deltas[f"correct_minus_{baseline}"] = {
            **mean_std(values),
            "positive_for_every_seed": all(value > 0 for value in values),
        }

    state_necessity = {
        field: mean_std(
            [
                float(per_seed[seed]["state_necessity"][field]["rate"])
                for seed in SEEDS
            ]
        )
        for field in NECESSITY_FIELDS
    }
    state_necessity["mean_query_rank_minus_correct_rank"] = mean_std(
        [
            float(
                per_seed[seed]["state_necessity"][
                    "mean_query_rank_minus_correct_rank"
                ]
            )
            for seed in SEEDS
        ]
    )

    paired_deltas = {}
    for baseline in PAIRED_BASELINES:
        paired_deltas[f"correct_minus_{baseline}"] = {}
        for metric in PAIRED_METRICS:
            entries = [
                per_seed[seed]["paired_bootstrap_t_ge_1_given_pool"][baseline][
                    "metrics"
                ][metric]
                for seed in SEEDS
            ]
            values = [float(entry["observed_delta"]) for entry in entries]
            paired_deltas[f"correct_minus_{baseline}"][metric] = {
                **mean_std(values),
                "ci95_by_seed": {
                    seed: [
                        float(entry["ci95_low"]),
                        float(entry["ci95_high"]),
                    ]
                    for seed, entry in zip(SEEDS, entries)
                },
                "positive_ci_for_every_seed": all(
                    float(entry["ci95_low"]) > 0 for entry in entries
                ),
            }

    result = {
        "status": "PASS" if not failures else "FAIL",
        "scope": "v27 Stage-5 fixed-pool rank-reversal multiseed audit",
        "seeds": list(SEEDS),
        "qids_per_seed": args.expected_qids,
        "api_calls": 0,
        "identical_pair_order": all(
            [pair_key(record) for record in records_by_seed[seed]] == reference_keys
            for seed in SEEDS
        ),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "rank_reversal_deltas": deltas,
        "state_necessity": state_necessity,
        "paired_state_deltas_t_ge_1_given_pool": paired_deltas,
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
