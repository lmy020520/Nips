#!/usr/bin/env python3
"""Summarize final-protocol strong baselines and their paired uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINES = (
    ("bm25", "BM25-RAG"),
    ("dense", "Dense-RAG"),
    ("hybrid", "Hybrid-RAG"),
    ("iterative_hybrid", "Iterative-Hybrid-RAG"),
    ("bge_reranker", "BGE-Reranker-RAG"),
)
PRIMARY_METHODS = ("KSG-EA-Compact", "KSG-EA-Recall")
METRICS = (
    "answer_em",
    "answer_f1",
    "supporting_fact_f1",
    "supporting_fact_em",
    "joint_f1",
    "joint_em",
    "full_support_coverage",
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("answer audit must use METHOD=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("answer audit must use METHOD=PATH")
    return name.strip(), Path(raw_path)


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def significance(entry: dict[str, Any]) -> str:
    low = float(entry["ci95_low"])
    high = float(entry["ci95_high"])
    if low > 0:
        return "primary_higher"
    if high < 0:
        return "primary_lower"
    return "interval_includes_zero"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-summary", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--compact-bootstrap", type=Path, required=True)
    parser.add_argument("--recall-bootstrap", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument(
        "--answer-audit", action="append", type=parse_named_path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_keys = {key for key, _ in BASELINES}
    audit_paths = dict(args.answer_audit)
    if set(audit_paths) != expected_keys:
        raise ValueError(f"answer audits must contain exactly {sorted(expected_keys)}")

    standard = load_json(args.standard_summary)
    selection = load_json(args.selection_summary)
    cache = load_json(args.cache_summary)
    compact_bootstrap = load_json(args.compact_bootstrap)
    recall_bootstrap = load_json(args.recall_bootstrap)
    failures: list[str] = []

    if standard.get("status") != "OK" or standard.get("failures"):
        failures.append("standard metric summary is not a clean OK")
    if (
        selection.get("status") != "OK"
        or selection.get("mode") != "strong_baseline_selection"
        or selection.get("failures")
    ):
        failures.append("selection summary is not a clean Stage 9.3 OK")
    if cache.get("status") != "OK" or cache.get("failures"):
        failures.append("final cache-propagation summary is not a clean OK")

    methods = standard.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("standard summary lacks methods")
    expected_method_names = set(PRIMARY_METHODS) | {
        name for _, name in BASELINES
    } | {"Gold-Oracle"}
    for name in expected_method_names:
        method = methods.get(name)
        if not isinstance(method, dict):
            failures.append(f"missing standard method: {name}")
        elif int(method.get("qids") or 0) != 3000:
            failures.append(f"{name}: qids != 3000")

    bootstraps = {
        "KSG-EA-Compact": compact_bootstrap,
        "KSG-EA-Recall": recall_bootstrap,
    }
    for primary, bootstrap in bootstraps.items():
        if bootstrap.get("status") != "OK" or bootstrap.get("primary") != primary:
            failures.append(f"{primary}: paired bootstrap is not a clean matching OK")
        for _, baseline_name in BASELINES:
            comparison_name = f"{primary}-minus-{baseline_name}"
            comparison = (bootstrap.get("comparisons") or {}).get(comparison_name)
            if not isinstance(comparison, dict):
                failures.append(f"missing paired comparison: {comparison_name}")

    audits: dict[str, dict[str, Any]] = {}
    for key, _ in BASELINES:
        audit = load_json(audit_paths[key])
        audits[key] = audit
        if (
            audit.get("status") != "OK"
            or audit.get("method") != key
            or int(audit.get("qids") or 0) != 3000
            or audit.get("failures")
        ):
            failures.append(f"{key}: final answer audit is not a clean matching OK")

    if failures:
        result: dict[str, Any] = {
            "status": "FAIL",
            "stage": 9,
            "step": "9.3",
            "mode": "same_generator_strong_baseline_final",
            "api_calls_during_finalization": 0,
            "failures": failures,
        }
    else:
        method_metrics: dict[str, Any] = {}
        for key, name in BASELINES:
            standard_metrics = methods[name]["metrics"]
            selection_metrics = selection["method_metrics"][key]
            answer_metrics = audits[key]["metrics"]
            method_metrics[name] = {
                "qids": 3000,
                **{metric: standard_metrics[metric] for metric in METRICS},
                "closure_success_at_10": standard_metrics["closure_success_at_10"],
                "closure_success_at_50": standard_metrics["closure_success_at_50"],
                "step_at_1": selection_metrics["step_acc@1"],
                "step_at_5": selection_metrics["step_acc@5"],
                "full_gold_doc_coverage": selection_metrics[
                    "full_gold_doc_coverage"
                ],
                "avg_answer_tokens": answer_metrics["avg_answer_tokens"],
                "avg_answer_latency_seconds": answer_metrics["avg_answer_latency"],
                "selection_ms_per_qid": selection_metrics[
                    "selection_ms_per_qid"
                ],
                "selection_throughput_qids_per_second": selection_metrics[
                    "selection_throughput_qids_per_second"
                ],
                "peak_gpu_allocated_mb": selection_metrics[
                    "peak_gpu_allocated_mb"
                ],
            }

        paired: dict[str, Any] = {}
        significance_counts: dict[str, dict[str, int]] = {}
        for primary, bootstrap in bootstraps.items():
            paired[primary] = {}
            counts = {
                "primary_higher": 0,
                "primary_lower": 0,
                "interval_includes_zero": 0,
            }
            for _, baseline_name in BASELINES:
                comparison_name = f"{primary}-minus-{baseline_name}"
                source = bootstrap["comparisons"][comparison_name]
                paired[primary][baseline_name] = {}
                for metric, entry in source.items():
                    label = significance(entry)
                    counts[label] += 1
                    paired[primary][baseline_name][metric] = {
                        **entry,
                        "significance_direction": label,
                    }
            significance_counts[primary] = counts

        strongest = {}
        for metric in METRICS:
            winner = max(
                (name for _, name in BASELINES),
                key=lambda name: float(methods[name]["metrics"][metric]),
            )
            strongest[metric] = {
                "method": winner,
                "value": methods[winner]["metrics"][metric],
            }

        ksg_reference_metrics = {}
        for primary in PRIMARY_METHODS:
            primary_metrics = methods[primary]["metrics"]
            budget = 10 if primary == "KSG-EA-Compact" else 50
            ksg_reference_metrics[primary] = {
                "qids": 3000,
                **{metric: primary_metrics[metric] for metric in METRICS},
                f"closure_success_at_{budget}": primary_metrics[
                    f"closure_success_at_{budget}"
                ],
            }

        reused = sum(int(audits[key]["cache"]["reused_exact_context"]) for key in expected_keys)
        fresh = sum(int(audits[key]["cache"]["fresh_answers"]) for key in expected_keys)
        result = {
            "status": "OK",
            "stage": 9,
            "step": "9.3",
            "mode": "same_generator_strong_baseline_final",
            "api_calls_during_finalization": 0,
            "protocol": {
                "qids": 3000,
                "answer_model": "deepseek-v4-flash",
                "answer_thinking_mode": "disabled",
                "answer_temperature": 0.0,
                "answer_prompt_version": "kbs_extractive_answer_json_v1",
                "paired_bootstrap_samples": 10000,
            },
            "metric_semantics": standard.get("metric_semantics"),
            "ksg_reference_metrics": ksg_reference_metrics,
            "method_metrics": method_metrics,
            "strongest_baseline_by_metric": strongest,
            "paired_kbs_minus_baseline": paired,
            "paired_significance_counts": significance_counts,
            "answer_api_accounting": {
                "target_reports": 15000,
                "exact_context_reuses": reused,
                "fresh_api_answers": fresh,
                "accounting_matches_target_reports": reused + fresh == 15000,
                "duplicate_source_raw_answer_disagreements": cache[
                    "duplicate_source_raw_answer_disagreements"
                ],
            },
            "interpretation_gate": {
                "decision": "FINAL_PROTOCOL_BASELINES_COMPLETE",
                "note": (
                    "Scientific claims must use the metric-specific paired intervals. "
                    "Selection alignment and downstream quality are reported separately."
                ),
            },
            "standard_metrics": str(args.standard_summary),
            "paired_bootstraps": {
                "compact": str(args.compact_bootstrap),
                "recall": str(args.recall_bootstrap),
            },
            "failures": [],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if result["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
