#!/usr/bin/env python3
"""Summarize Stage 9.5 task-level state-rollout interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = {
    "online_state": "Online-State",
    "query_only": "Query-Only",
    "frozen_initial_state": "Frozen-Initial",
    "other_question_state": "Other-Question",
    "previous_evidence_only": "Previous-Evidence-Only",
}
COMPARISONS = tuple(name for name in METHODS if name != "online_state")
METRICS = (
    "answer_em",
    "answer_f1",
    "supporting_fact_f1",
    "supporting_fact_em",
    "joint_f1",
    "joint_em",
    "full_support_coverage",
    "closure_success_at_10",
)
DIRECT_STATE_METRICS = ("supporting_fact_f1", "full_support_coverage")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def significant_positive(entry: dict[str, Any]) -> bool:
    return float(entry["observed_delta"]) > 0 and float(entry["ci95_low"]) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-summary", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument("--answer-audit", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    standard = load(args.standard_summary)
    selection = load(args.selection_summary)
    bootstrap = load(args.bootstrap)
    cache = load(args.cache_summary)
    for label, report in (
        ("standard summary", standard),
        ("selection summary", selection),
        ("bootstrap", bootstrap),
        ("cache summary", cache),
    ):
        if report.get("status") != "OK" or report.get("failures"):
            failures.append(f"{label} is not a clean OK")

    audits: dict[str, Any] = {}
    for value in args.answer_audit:
        if "=" not in value:
            raise ValueError("answer audit must use CONDITION=PATH")
        condition, raw_path = value.split("=", 1)
        condition = condition.strip()
        if condition not in METHODS or condition in audits:
            raise ValueError(f"invalid or duplicate answer-audit condition: {condition}")
        audit = load(Path(raw_path))
        audits[condition] = audit
        if (
            audit.get("status") != "OK"
            or audit.get("method") != condition
            or audit.get("qids") != 3000
            or audit.get("failures")
        ):
            failures.append(f"answer audit is not clean for {condition}")
    if set(audits) != set(METHODS):
        failures.append("answer audits do not cover all five conditions")

    methods = standard.get("methods") or {}
    comparisons = bootstrap.get("comparisons") or {}
    method_metrics: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for condition, label in METHODS.items():
        method = methods.get(label)
        if not isinstance(method, dict) or int(method.get("qids") or 0) != 3000:
            failures.append(f"missing 3000-qid standard metrics for {label}")
            continue
        method_metrics[condition] = {
            metric: method["metrics"][metric] for metric in METRICS
        }

    for condition in COMPARISONS:
        key = f"Online-State-minus-{METHODS[condition]}"
        comparison = comparisons.get(key)
        if not isinstance(comparison, dict):
            failures.append(f"missing paired comparison: {key}")
            continue
        paired[condition] = {metric: comparison[metric] for metric in METRICS}

    degraded = ("query_only", "frozen_initial_state", "other_question_state")
    downstream_supported = not failures and all(
        significant_positive(paired[condition][metric])
        for condition in degraded
        for metric in DIRECT_STATE_METRICS
    )
    previous_supported = not failures and all(
        significant_positive(paired["previous_evidence_only"][metric])
        for metric in DIRECT_STATE_METRICS
    )

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.5",
        "mode": "downstream_state_rollout_final",
        "api_calls": 0,
        "qids": 3000,
        "n_bootstrap": bootstrap.get("n_bootstrap"),
        "method_metrics": method_metrics,
        "online_state_minus_condition": paired,
        "selection_diagnostics": {
            "source": str(args.selection_summary),
            "method_metrics": selection.get("method_metrics"),
            "paired_deltas": selection.get("paired_deltas"),
        },
        "answer_cache_accounting": {
            "source": str(args.cache_summary),
            "answer_audits": {
                condition: (audit.get("cache") or {})
                for condition, audit in audits.items()
            },
            "total_fresh_answers": sum(
                int((audit.get("cache") or {}).get("fresh_answers") or 0)
                for audit in audits.values()
            ),
            "total_reused_answers": sum(
                int((audit.get("cache") or {}).get("reused_exact_context") or 0)
                for audit in audits.values()
            ),
        },
        "interpretation_gate": {
            "direct_state_metrics": list(DIRECT_STATE_METRICS),
            "degraded_state_conditions": list(degraded),
            "state_relevance_downstream_supported": downstream_supported,
            "full_history_superiority_supported": previous_supported,
            "decision": (
                "STATE_RELEVANCE_DOWNSTREAM_SUPPORTED"
                if downstream_supported
                else "DOWNSTREAM_STATE_EVIDENCE_MIXED"
            ),
            "claim_boundary": (
                "Full-history superiority is a separate comparison against "
                "previous-evidence-only and is not implied by state relevance."
            ),
        },
        "artifacts": {
            "standard_summary": str(args.standard_summary),
            "bootstrap": str(args.bootstrap),
            "cache_summary": str(args.cache_summary),
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
