#!/usr/bin/env python3
"""Validate and summarize a pre-registered policy-blend alpha sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    "answer_em",
    "answer_f1",
    "step_acc@1",
    "step_acc@5",
    "full_gold_doc_coverage",
    "full_gold_unit_coverage",
    "avg_answer_tokens",
)
FIXED_CONFIG_KEYS = (
    "samples",
    "memory",
    "queries",
    "checkpoint",
    "state_mode",
    "policy_context_source",
    "selector",
    "dense_model",
    "dense_query_mode",
    "front_pool_k",
    "front_fusion",
    "local_expansion_window",
    "mmr_lambda",
    "mmr_same_doc_similarity",
    "candidate_top_k",
    "select_top_k",
    "policy_score_mode",
    "seed",
)


def load_report(path: Path) -> tuple[dict[str, Any], set[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    results = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"{path}: expected summary object and results list")
    qids = {str(row.get("qid") or "") for row in results if isinstance(row, dict)}
    qids.discard("")
    if len(qids) != len(results) or len(qids) != summary.get("qids"):
        raise ValueError(
            f"{path}: qid mismatch: summary={summary.get('qids')}, "
            f"results={len(results)}, unique={len(qids)}"
        )
    return summary, qids


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.2,0.35,0.5,0.8,1.0")
    parser.add_argument("--expected-qids", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    args = parser.parse_args()

    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if len(set(alphas)) != len(alphas):
        raise ValueError("alpha list contains duplicates")

    rows = []
    reference_summary = None
    reference_qids = None
    for alpha in alphas:
        path = args.input_dir / f"alpha_{alpha_tag(alpha)}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing alpha report: {path}")
        summary, qids = load_report(path)
        if len(qids) != args.expected_qids:
            raise ValueError(f"{path}: expected {args.expected_qids} qids, found {len(qids)}")
        reported_alpha = summary.get("policy_blend_weight")
        if not isinstance(reported_alpha, (int, float)) or abs(float(reported_alpha) - alpha) > 1e-9:
            raise ValueError(f"{path}: expected alpha={alpha}, found {reported_alpha!r}")
        if reference_qids is not None and qids != reference_qids:
            raise ValueError(f"{path}: qid set differs from the first alpha report")
        if reference_summary is not None:
            mismatches = [
                key for key in FIXED_CONFIG_KEYS if summary.get(key) != reference_summary.get(key)
            ]
            if mismatches:
                raise ValueError(f"{path}: fixed configuration differs for keys: {mismatches}")
        else:
            reference_summary = summary
            reference_qids = qids

        row = {"alpha": alpha, "qids": len(qids), "report": str(path)}
        row.update({metric: summary.get(metric) for metric in METRICS})
        rows.append(row)

    # Pre-registered rule: maximize Step@5, then full-unit coverage, then Step@1.
    selected = max(
        rows,
        key=lambda row: (
            float(row["step_acc@5"]),
            float(row["full_gold_unit_coverage"]),
            float(row["step_acc@1"]),
            -float(row["alpha"]),
        ),
    )
    output = {
        "selection_split": "validation",
        "selection_rule": [
            "maximize step_acc@5",
            "tie-break by full_gold_unit_coverage",
            "tie-break by step_acc@1",
            "final tie-break by smaller alpha",
        ],
        "expected_qids": args.expected_qids,
        "selected_alpha": selected["alpha"],
        "selected_report": selected["report"],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.tsv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.tsv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    print(f"tsv: {args.tsv_output}")


if __name__ == "__main__":
    main()
