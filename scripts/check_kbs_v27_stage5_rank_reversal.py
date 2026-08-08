#!/usr/bin/env python3
"""Audit v27 Stage-5 fixed-pool rank-reversal prerequisites and reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHECKPOINTS = {
    "42": Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"),
    "43": Path(
        "outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt"
    ),
    "44": Path(
        "outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt"
    ),
}
REQUIRED_MODES = {
    "correct",
    "query_only",
    "frozen",
    "previous_evidence_only",
}


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def prerequisite_audit() -> tuple[dict, list[str]]:
    required = [
        Path("md/kbs_three_review_execution_plan.md"),
        Path("scripts/analyze_kbs_state_mechanism.py"),
        Path("scripts/run_kbs_state_phase1.sh"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl"),
        Path(
            "data/hotpotqa_distractor_eval_3000_cand50/"
            "unit_registry/raw_units_test.jsonl"
        ),
        Path("models/deberta-v3-large"),
        Path("models/bge-large-en-v1.5"),
        *CHECKPOINTS.values(),
    ]
    failures = [f"missing required path: {path}" for path in required if not path.exists()]
    return {
        "required_paths": [str(path) for path in required],
        "checkpoints": {seed: str(path) for seed, path in CHECKPOINTS.items()},
    }, failures


def report_audit(
    report_path: Path,
    expected_seed: str,
    expected_qids: int,
    expected_states: int,
) -> tuple[dict, list[str]]:
    failures = []
    obj = json.loads(report_path.read_text(encoding="utf-8"))
    expected_checkpoint = str(CHECKPOINTS[expected_seed])
    diagnostics = obj.get("data_diagnostics") or {}
    protocol = obj.get("fixed_pool_protocol") or {}
    reversal = obj.get("conditional_rank_reversal") or {}
    modes = reversal.get("modes") or {}

    expected_values = {
        "status": "OK",
        "checkpoint": expected_checkpoint,
        "qids": expected_qids,
        "candidate_top_k": 10,
        "policy_score_mode": "policy_only_for_mechanism_isolation",
        "front_context": "correct dataset K_t",
    }
    observed_values = {
        "status": obj.get("status"),
        "checkpoint": obj.get("checkpoint"),
        "qids": diagnostics.get("qids"),
        "candidate_top_k": protocol.get("candidate_top_k"),
        "policy_score_mode": protocol.get("policy_score_mode"),
        "front_context": protocol.get("front_context"),
    }
    for key, expected in expected_values.items():
        if observed_values[key] != expected:
            failures.append(
                f"{key}: expected {expected!r}, found {observed_values[key]!r}"
            )

    evaluated_states = int(obj.get("evaluated_states", 0))
    if evaluated_states <= 0:
        failures.append("evaluated_states must be positive")
    if expected_states and evaluated_states != expected_states:
        failures.append(
            f"evaluated_states: expected {expected_states}, found {evaluated_states}"
        )
    eligible_pairs = int(reversal.get("eligible_pairs", 0))
    if eligible_pairs <= 0:
        failures.append("conditional rank reversal has no eligible pairs")
    if not REQUIRED_MODES.issubset(modes):
        failures.append(
            f"rank-reversal modes missing: {sorted(REQUIRED_MODES - set(modes))}"
        )
    for mode in REQUIRED_MODES & set(modes):
        value = modes[mode].get("conditional_rank_reversal_accuracy")
        if value is None or not 0.0 <= float(value) <= 1.0:
            failures.append(f"invalid conditional rank-reversal accuracy for {mode}")

    output_dir = report_path.parent
    intervention_records = output_dir / "state_intervention_records.jsonl"
    reversal_records = output_dir / "rank_reversal_records.jsonl"
    if not intervention_records.is_file():
        failures.append(f"missing record file: {intervention_records}")
    elif count_jsonl(intervention_records) != evaluated_states:
        failures.append("state intervention record count does not match summary")
    if not reversal_records.is_file():
        failures.append(f"missing record file: {reversal_records}")
    elif count_jsonl(reversal_records) != eligible_pairs:
        failures.append("rank reversal record count does not match summary")

    audit = {
        "report": str(report_path),
        "seed": expected_seed,
        "checkpoint": obj.get("checkpoint"),
        "qids": diagnostics.get("qids"),
        "evaluated_states": evaluated_states,
        "eligible_pairs": eligible_pairs,
        "correct_rank_reversal": (
            modes.get("correct") or {}
        ).get("conditional_rank_reversal_accuracy"),
        "query_only_rank_reversal": (
            modes.get("query_only") or {}
        ).get("conditional_rank_reversal_accuracy"),
    }
    return audit, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-seed", choices=tuple(CHECKPOINTS))
    parser.add_argument("--expected-qids", type=int, default=3000)
    parser.add_argument("--expected-states", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/kbs_v27_stage5_rank_reversal/readiness.json"),
    )
    args = parser.parse_args()

    prerequisites, failures = prerequisite_audit()
    report = None
    if not args.check_only:
        if args.report is None or args.expected_seed is None:
            parser.error("--report and --expected-seed are required without --check-only")
        if not args.report.is_file():
            failures.append(f"missing report: {args.report}")
        else:
            report, report_failures = report_audit(
                args.report,
                args.expected_seed,
                args.expected_qids,
                args.expected_states,
            )
            failures.extend(report_failures)

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 5,
        "mode": "readiness" if args.check_only else "report_audit",
        "api_calls": 0,
        "prerequisites": prerequisites,
        "report_audit": report,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
