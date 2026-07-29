#!/usr/bin/env python3
"""Paired offline diagnosis for Full state versus a trained anchor policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}: {exc}") from exc


def load_report(path: Path) -> tuple[dict, dict[str, dict]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    results = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report must contain summary/results: {path}")
    by_qid = {}
    for row in results:
        qid = str(row.get("qid") or "")
        if not qid or qid in by_qid:
            raise ValueError(f"missing or duplicate qid in {path}: {qid!r}")
        by_qid[qid] = row
    return summary, by_qid


def load_question_types(path: Path) -> dict[str, str]:
    result = {}
    for row in read_jsonl(path):
        qid = str(row.get("qid") or "")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        qtype = str(metadata.get("type") or row.get("type") or "unknown")
        if qid:
            result[qid] = qtype
    return result


def full_unit(record: dict) -> float:
    gold = {str(value) for value in record.get("gold_unit_ids") or []}
    selected = {str(value) for value in record.get("selected_unit_ids") or []}
    return float(bool(gold) and gold.issubset(selected))


def aggregate(
    records: dict[str, dict],
    qids: list[str],
    step_filter: Callable[[dict], bool],
) -> dict:
    steps = [
        step
        for qid in qids
        for step in records[qid].get("steps") or []
        if step_filter(step)
    ]
    ranks = [int(step.get("positive_rank") or 0) for step in steps]
    if any(rank <= 0 for rank in ranks):
        raise ValueError("positive_rank must be a positive integer")
    return {
        "qids": len(qids),
        "states": len(steps),
        "step_at_1": round(sum(rank == 1 for rank in ranks) / max(len(ranks), 1), 6),
        "step_at_5": round(sum(rank <= 5 for rank in ranks) / max(len(ranks), 1), 6),
        "mrr": round(sum(1.0 / rank for rank in ranks) / max(len(ranks), 1), 6),
        "answer_em": round(
            sum(float(records[qid].get("answer_em") or 0.0) for qid in qids)
            / max(len(qids), 1),
            6,
        ),
        "answer_f1": round(
            sum(float(records[qid].get("answer_f1") or 0.0) for qid in qids)
            / max(len(qids), 1),
            6,
        ),
        "full_unit_coverage": round(
            sum(full_unit(records[qid]) for qid in qids) / max(len(qids), 1),
            6,
        ),
    }


def paired_rank_outcomes(
    full: dict[str, dict],
    anchor: dict[str, dict],
    qids: list[str],
    step_filter: Callable[[dict], bool],
) -> dict:
    counts = Counter()
    for qid in qids:
        full_steps = full[qid].get("steps") or []
        anchor_steps = anchor[qid].get("steps") or []
        if len(full_steps) != len(anchor_steps):
            raise ValueError(f"step count mismatch: qid={qid}")
        for full_step, anchor_step in zip(full_steps, anchor_steps):
            if (
                int(full_step.get("t") or 0) != int(anchor_step.get("t") or 0)
                or full_step.get("positive_unit_id") != anchor_step.get("positive_unit_id")
            ):
                raise ValueError(f"paired step mismatch: qid={qid}")
            if not step_filter(full_step):
                continue
            full_rank = int(full_step["positive_rank"])
            anchor_rank = int(anchor_step["positive_rank"])
            if full_rank < anchor_rank:
                counts["full_better"] += 1
            elif anchor_rank < full_rank:
                counts["anchor_better"] += 1
            else:
                counts["tie"] += 1
    total = sum(counts.values())
    return {
        "states": total,
        "full_better": counts["full_better"],
        "anchor_better": counts["anchor_better"],
        "tie": counts["tie"],
        "anchor_net_wins": counts["anchor_better"] - counts["full_better"],
    }


def anchor_conditioning(anchor: dict[str, dict], qids: list[str]) -> dict:
    groups = {True: [], False: []}
    for qid in qids:
        previous_positive = None
        for step in anchor[qid].get("steps") or []:
            t = int(step.get("t") or 0)
            if t >= 1:
                groups[step.get("context_anchor_unit_id") == previous_positive].append(step)
            previous_positive = step.get("positive_unit_id")

    result = {}
    total = sum(len(values) for values in groups.values())
    for is_gold, steps in groups.items():
        ranks = [int(step["positive_rank"]) for step in steps]
        result["previous_anchor_gold" if is_gold else "previous_anchor_not_gold"] = {
            "states": len(steps),
            "share": round(len(steps) / max(total, 1), 6),
            "step_at_1": round(sum(rank == 1 for rank in ranks) / max(len(ranks), 1), 6),
            "step_at_5": round(sum(rank <= 5 for rank in ranks) / max(len(ranks), 1), 6),
            "mrr": round(sum(1.0 / rank for rank in ranks) / max(len(ranks), 1), 6),
        }
    return result


def qid_step_totals(record: dict, later_only: bool) -> tuple[float, float, float]:
    steps = [
        step
        for step in record.get("steps") or []
        if not later_only or int(step.get("t") or 0) >= 1
    ]
    return (
        float(sum(int(step["positive_rank"]) == 1 for step in steps)),
        float(sum(1.0 / int(step["positive_rank"]) for step in steps)),
        float(len(steps)),
    )


def clustered_delta_ci(
    full: dict[str, dict],
    anchor: dict[str, dict],
    qids: list[str],
    *,
    later_only: bool,
    n_bootstrap: int,
    seed: int,
) -> dict:
    full_rows = np.asarray([qid_step_totals(full[qid], later_only) for qid in qids])
    anchor_rows = np.asarray([qid_step_totals(anchor[qid], later_only) for qid in qids])
    rng = np.random.default_rng(seed)

    def metric(rows: np.ndarray, column: int) -> float:
        return float(rows[:, column].sum() / rows[:, 2].sum())

    result = {}
    for name, column in (("step_at_1", 0), ("mrr", 1)):
        observed = metric(anchor_rows, column) - metric(full_rows, column)
        samples = np.empty(n_bootstrap, dtype=np.float64)
        for index in range(n_bootstrap):
            selected = rng.integers(0, len(qids), len(qids))
            samples[index] = metric(anchor_rows[selected], column) - metric(
                full_rows[selected], column
            )
        low, high = np.percentile(samples, [2.5, 97.5])
        result[name] = {
            "anchor_minus_full": round(observed, 6),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "bootstrap_samples": n_bootstrap,
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-report", type=Path, required=True)
    parser.add_argument("--anchor-report", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    full_summary, full = load_report(args.full_report)
    anchor_summary, anchor = load_report(args.anchor_report)
    if set(full) != set(anchor):
        raise ValueError(
            f"qid sets differ: full_only={len(set(full) - set(anchor))}, "
            f"anchor_only={len(set(anchor) - set(full))}"
        )
    qids = sorted(full)
    question_types = load_question_types(args.queries)
    missing_types = [qid for qid in qids if qid not in question_types]
    if missing_types:
        raise ValueError(
            f"query metadata misses {len(missing_types)} evaluated qids; "
            f"examples={missing_types[:5]}"
        )

    filters = {
        "all_states": lambda step: True,
        "initial_t0": lambda step: int(step.get("t") or 0) == 0,
        "hop2_plus_t1_or_later": lambda step: int(step.get("t") or 0) >= 1,
    }
    methods = {}
    for method_name, records in (("full_v22", full), ("acra_anchor", anchor)):
        methods[method_name] = {
            slice_name: aggregate(records, qids, predicate)
            for slice_name, predicate in filters.items()
        }

    type_slices = {}
    for qtype in sorted(set(question_types.values())):
        selected_qids = [qid for qid in qids if question_types[qid] == qtype]
        type_slices[qtype] = {
            method_name: aggregate(
                records,
                selected_qids,
                filters["hop2_plus_t1_or_later"],
            )
            for method_name, records in (("full_v22", full), ("acra_anchor", anchor))
        }

    result = {
        "status": "OK",
        "full_report": str(args.full_report),
        "anchor_report": str(args.anchor_report),
        "queries": str(args.queries),
        "qids": len(qids),
        "protocol": {
            "full_checkpoint": full_summary.get("checkpoint"),
            "anchor_checkpoint": anchor_summary.get("checkpoint"),
            "full_context": full_summary.get("policy_context_source"),
            "anchor_context": anchor_summary.get("policy_context_source"),
            "candidate_top_k": anchor_summary.get("candidate_top_k"),
            "select_top_k": anchor_summary.get("select_top_k"),
            "policy_blend_weight": anchor_summary.get("policy_blend_weight"),
        },
        "methods": methods,
        "question_type_hop2_plus": type_slices,
        "paired_rank_outcomes": {
            slice_name: paired_rank_outcomes(full, anchor, qids, predicate)
            for slice_name, predicate in filters.items()
        },
        "anchor_previous_evidence_conditioning": anchor_conditioning(anchor, qids),
        "paired_cluster_bootstrap": {
            "all_states": clustered_delta_ci(
                full,
                anchor,
                qids,
                later_only=False,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            ),
            "hop2_plus": clustered_delta_ci(
                full,
                anchor,
                qids,
                later_only=True,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + 1,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
