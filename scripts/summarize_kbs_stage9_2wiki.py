#!/usr/bin/env python3
"""Summarize final-policy 2Wiki transfer and paired uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = {
    "compact_seed42": "KSG-EA-Compact",
    "balanced_knee_seed42": "KSG-EA-Balanced",
    "recall_seed42": "KSG-EA-Recall",
    "hybrid": "Hybrid-RAG",
    "bge_reranker": "BGE-Reranker-RAG",
}
PRIMARY_BUDGETS = {
    "KSG-EA-Compact": 10,
    "KSG-EA-Balanced": 15,
    "KSG-EA-Recall": 50,
}
BASELINES = ("Hybrid-RAG", "BGE-Reranker-RAG")
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
    parser.add_argument("--balanced-bootstrap", type=Path, required=True)
    parser.add_argument("--recall-bootstrap", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument(
        "--answer-audit", action="append", type=parse_named_path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit_paths = dict(args.answer_audit)
    if set(audit_paths) != set(METHODS):
        raise ValueError(f"answer audits must contain exactly {sorted(METHODS)}")

    standard = load_json(args.standard_summary)
    selection = load_json(args.selection_summary)
    cache = load_json(args.cache_summary)
    bootstraps = {
        "KSG-EA-Compact": load_json(args.compact_bootstrap),
        "KSG-EA-Balanced": load_json(args.balanced_bootstrap),
        "KSG-EA-Recall": load_json(args.recall_bootstrap),
    }
    failures: list[str] = []

    if standard.get("status") != "OK" or standard.get("failures"):
        failures.append("standard metric summary is not a clean OK")
    if (
        selection.get("status") != "OK"
        or selection.get("mode") != "final_policy_2wiki_selection"
        or selection.get("qids") != 1000
        or selection.get("failures")
    ):
        failures.append("selection summary is not a clean Stage 9.4 OK")
    if (
        cache.get("status") != "OK"
        or cache.get("mode")
        != "final_policy_2wiki_exact_context_answer_cache_preparation"
        or cache.get("failures")
    ):
        failures.append("final cache-propagation summary is not a clean OK")
    if int(cache.get("total_target_contexts") or 0) != 5000:
        failures.append("final cache summary does not cover 5,000 target contexts")
    if int(cache.get("fresh_target_answers_without_cross_method_propagation") or 0):
        failures.append("final cache summary still has unresolved target contexts")

    methods = standard.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("standard summary lacks methods")
    expected_method_names = set(METHODS.values()) | {"Gold-Oracle"}
    for name in expected_method_names:
        method = methods.get(name)
        if not isinstance(method, dict):
            failures.append(f"missing standard method: {name}")
        elif int(method.get("qids") or 0) != 1000:
            failures.append(f"{name}: qids != 1000")

    for primary, bootstrap in bootstraps.items():
        if (
            bootstrap.get("status") != "OK"
            or bootstrap.get("primary") != primary
            or bootstrap.get("baselines") != list(BASELINES)
            or int(bootstrap.get("n_bootstrap") or 0) != 10000
        ):
            failures.append(f"{primary}: paired bootstrap is not a clean registered OK")
        for baseline in BASELINES:
            comparison_name = f"{primary}-minus-{baseline}"
            comparison = (bootstrap.get("comparisons") or {}).get(comparison_name)
            if not isinstance(comparison, dict):
                failures.append(f"missing paired comparison: {comparison_name}")

    audits: dict[str, dict[str, Any]] = {}
    for key in METHODS:
        audit = load_json(audit_paths[key])
        audits[key] = audit
        if (
            audit.get("status") != "OK"
            or audit.get("mode") != "final_policy_2wiki_answer_report"
            or audit.get("method") != key
            or int(audit.get("qids") or 0) != 1000
            or audit.get("failures")
            or int((audit.get("cache") or {}).get("invalid") or 0) != 0
        ):
            failures.append(f"{key}: final answer audit is not a clean matching OK")

    if failures:
        result: dict[str, Any] = {
            "status": "FAIL",
            "stage": 9,
            "step": "9.4",
            "mode": "final_policy_2wiki_zero_shot_final",
            "api_calls_during_finalization": 0,
            "failures": failures,
        }
    else:
        selection_metrics = selection["method_metrics"]
        method_metrics: dict[str, Any] = {}
        for key, name in METHODS.items():
            standard_metrics = methods[name]["metrics"]
            select = selection_metrics[key]
            answer = audits[key]["metrics"]
            method_metrics[name] = {
                "qids": 1000,
                **{metric: standard_metrics[metric] for metric in METRICS},
                "closure_success_at_10": standard_metrics["closure_success_at_10"],
                "closure_success_at_15": standard_metrics["closure_success_at_15"],
                "closure_success_at_50": standard_metrics["closure_success_at_50"],
                "step_at_1": select["step_acc@1"],
                "step_at_5": select["step_acc@5"],
                "full_gold_doc_coverage": select["full_gold_doc_coverage"],
                "avg_answer_tokens": answer["avg_answer_tokens"],
                "avg_answer_latency_seconds": answer["avg_answer_latency"],
                "selection_ms_per_qid": select["selection_ms_per_qid"],
                "selection_throughput_qids_per_second": select[
                    "selection_throughput_qids_per_second"
                ],
                "peak_gpu_allocated_mb": select["peak_gpu_allocated_mb"],
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
            for baseline in BASELINES:
                comparison_name = f"{primary}-minus-{baseline}"
                source = bootstrap["comparisons"][comparison_name]
                paired[primary][baseline] = {}
                for metric, entry in source.items():
                    label = significance(entry)
                    counts[label] += 1
                    paired[primary][baseline][metric] = {
                        **entry,
                        "significance_direction": label,
                    }
            significance_counts[primary] = counts

        reused = sum(
            int(audits[key]["cache"]["reused_exact_context"]) for key in METHODS
        )
        fresh = sum(int(audits[key]["cache"]["fresh_answers"]) for key in METHODS)
        result = {
            "status": "OK",
            "stage": 9,
            "step": "9.4",
            "mode": "final_policy_2wiki_zero_shot_final",
            "api_calls_during_finalization": 0,
            "protocol": {
                "transfer": "HotpotQA-trained v27; no 2Wiki fine-tuning",
                "qids": 1000,
                "primary_seed": 42,
                "robustness_seeds": [43, 44],
                "answer_model": "deepseek-v4-flash",
                "answer_thinking_mode": "disabled",
                "answer_temperature": 0.0,
                "answer_prompt_version": "kbs_extractive_answer_json_v1",
                "paired_bootstrap_samples": 10000,
            },
            "metric_semantics": standard.get("metric_semantics"),
            "method_metrics": method_metrics,
            "ksg_multiseed_selection_robustness": selection[
                "ksg_multiseed_robustness"
            ],
            "paired_kbs_minus_baseline": paired,
            "paired_significance_counts": significance_counts,
            "answer_api_accounting": {
                "target_reports": 5000,
                "exact_context_reuses": reused,
                "fresh_api_answers": fresh,
                "accounting_matches_target_reports": reused + fresh == 5000,
                "duplicate_source_raw_answer_disagreements": cache[
                    "duplicate_source_raw_answer_disagreements"
                ],
            },
            "registered_closure_budgets": PRIMARY_BUDGETS,
            "interpretation_gate": {
                "decision": "FINAL_POLICY_2WIKI_COMPLETE",
                "note": (
                    "This is zero-shot policy transfer. Scientific claims must use "
                    "the metric-specific paired intervals; selection robustness and "
                    "seed-42 downstream quality are reported separately."
                ),
            },
            "standard_metrics": str(args.standard_summary),
            "paired_bootstraps": {
                "compact": str(args.compact_bootstrap),
                "balanced": str(args.balanced_bootstrap),
                "recall": str(args.recall_bootstrap),
            },
            "failures": [],
        }
        if not result["answer_api_accounting"]["accounting_matches_target_reports"]:
            result["status"] = "FAIL"
            result["failures"].append("answer API accounting does not total 5,000")

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
