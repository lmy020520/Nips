#!/usr/bin/env python3
"""Validate and summarize a pre-registered policy-blend alpha sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
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


def load_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
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
    return summary, results, qids


def load_memory(path: Path) -> dict[str, str]:
    memory = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            unit_id = str(row.get("unit_id") or "")
            if not unit_id:
                raise ValueError(f"{path}:{line_number}: missing unit_id")
            title = str(row.get("title") or row.get("doc_id") or "")
            sent_id = row.get("sent_id")
            text = str(row.get("text") or "").strip()
            memory[unit_id] = f"{title} [{sent_id}] {text}".strip()
    return memory


def derive_metrics(
    results: list[dict[str, Any]],
    memory: dict[str, str],
    report_path: Path,
) -> dict[str, float | int]:
    reciprocal_ranks = []
    context_units = []
    context_chars = []
    context_tokens = []
    missing_units = set()

    for result in results:
        steps = result.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"{report_path}: result {result.get('qid')!r} has no steps list")
        for step in steps:
            rank = step.get("positive_rank") if isinstance(step, dict) else None
            if not isinstance(rank, int) or rank <= 0:
                raise ValueError(
                    f"{report_path}: result {result.get('qid')!r} "
                    f"has invalid positive_rank={rank!r}"
                )
            reciprocal_ranks.append(1.0 / rank)

        unit_ids = result.get("selected_unit_ids")
        if not isinstance(unit_ids, list):
            raise ValueError(
                f"{report_path}: result {result.get('qid')!r} "
                "has no selected_unit_ids list"
            )
        texts = []
        for unit_id in unit_ids:
            unit_id = str(unit_id)
            text = memory.get(unit_id)
            if text is None:
                missing_units.add(unit_id)
                continue
            texts.append(text)
        context = "\n".join(texts)
        context_units.append(len(texts))
        context_chars.append(len(context))
        context_tokens.append(len(re.findall(r"\S+", context)))

    if missing_units:
        examples = sorted(missing_units)[:5]
        raise ValueError(
            f"{report_path}: {len(missing_units)} selected units are absent "
            f"from memory; examples={examples}"
        )
    if not reciprocal_ranks:
        raise ValueError(f"{report_path}: no state ranks available for MRR")

    qid_count = len(results)
    return {
        "state_mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 6),
        "evaluated_steps": len(reciprocal_ranks),
        "avg_selected_context_units": round(sum(context_units) / qid_count, 6),
        "avg_selected_context_chars": round(sum(context_chars) / qid_count, 6),
        "avg_selected_context_lexical_tokens": round(sum(context_tokens) / qid_count, 6),
    }


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.2,0.35,0.5,0.8,1.0")
    parser.add_argument("--expected-qids", type=int, default=1000)
    parser.add_argument("--near-tie-threshold", type=float, default=0.002)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    args = parser.parse_args()
    if args.near_tie_threshold <= 0:
        raise ValueError("--near-tie-threshold must be positive")

    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if len(set(alphas)) != len(alphas):
        raise ValueError("alpha list contains duplicates")

    rows = []
    reference_summary = None
    reference_qids = None
    memory_cache: dict[Path, dict[str, str]] = {}
    for alpha in alphas:
        path = args.input_dir / f"alpha_{alpha_tag(alpha)}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing alpha report: {path}")
        summary, results, qids = load_report(path)
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

        memory_path = Path(str(summary.get("memory") or ""))
        if not memory_path.is_file():
            raise FileNotFoundError(f"{path}: memory file not found: {memory_path}")
        if memory_path not in memory_cache:
            memory_cache[memory_path] = load_memory(memory_path)

        row = {"alpha": alpha, "qids": len(qids), "report": str(path)}
        row.update({metric: summary.get(metric) for metric in METRICS})
        row.update(derive_metrics(results, memory_cache[memory_path], path))
        rows.append(row)

    max_step_at_5 = max(float(row["step_acc@5"]) for row in rows)
    near_best_rows = [
        row
        for row in rows
        if max_step_at_5 - float(row["step_acc@5"]) < args.near_tie_threshold
    ]
    # Near-equal Step@5 values favor the shorter selected context.
    selected = min(
        near_best_rows,
        key=lambda row: (
            float(row["avg_selected_context_lexical_tokens"]),
            float(row["avg_selected_context_chars"]),
            -float(row["full_gold_unit_coverage"]),
            -float(row["step_acc@1"]),
            float(row["alpha"]),
        ),
    )
    output = {
        "selection_split": "validation",
        "selection_rule": [
            "maximize step_acc@5",
            (
                "when step_acc@5 is within strictly less than "
                f"{args.near_tie_threshold} of the maximum, choose the lower "
                "avg_selected_context_lexical_tokens operating point"
            ),
            "tie-break by avg_selected_context_chars",
            "tie-break by full_gold_unit_coverage",
            "tie-break by step_acc@1",
            "final tie-break by smaller alpha",
        ],
        "derived_metric_definitions": {
            "state_mrr": "mean reciprocal positive_rank over all teacher states",
            "avg_selected_context_units": (
                "mean number of unique selected evidence units per qid"
            ),
            "avg_selected_context_chars": (
                "mean character length of title, sentence id, and evidence text "
                "for the final selected context"
            ),
            "avg_selected_context_lexical_tokens": (
                "mean whitespace-token count of the same selected context; "
                "this is not an LLM API token count"
            ),
        },
        "near_tie_threshold": args.near_tie_threshold,
        "max_step_acc@5": max_step_at_5,
        "near_best_alphas": [row["alpha"] for row in near_best_rows],
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
