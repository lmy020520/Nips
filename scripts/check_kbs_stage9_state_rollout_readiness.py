#!/usr/bin/env python3
"""Audit prerequisites for the Stage 9.5 downstream state-rollout experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "qids": 3000,
    "steps": 7296,
    "candidate_top_k": 10,
    "front_pool_k": 30,
    "select_top_k": 5,
    "state_update_top_k": 1,
    "policy_blend_weight": 0.5,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_qids(path: Path) -> tuple[int, set[str]]:
    rows = 0
    qids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            qid = str(row.get("qid") or "")
            if qid:
                qids.add(qid)
    return rows, qids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correct-report",
        default="outputs/rag/kbs_v27_final_hotpot/full_compact.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/analysis/kbs_stage9_state_rollout/readiness.json",
    )
    args = parser.parse_args()

    required = [
        Path("md/kbs_three_review_execution_plan.md"),
        Path("md/kbs_review_75_85_execution_plan.md"),
        Path("scripts/run_hotpotqa_policy_rag.py"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl"),
        Path("models/deberta-v3-large"),
        Path("models/bge-large-en-v1.5"),
        Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"),
        Path(args.correct_report),
    ]
    failures = [f"missing required path: {path}" for path in required if not path.exists()]

    runtime_source = Path("scripts/run_hotpotqa_policy_rag.py").read_text(encoding="utf-8")
    runtime_contract = {
        name: name in runtime_source
        for name in [
            "online_state",
            "query_only",
            "frozen_initial_state",
            "other_question_state",
            "previous_evidence_only",
            "--external-policy-state-report",
            "cyclic_next_qid",
        ]
    }
    failures.extend(
        f"runtime does not expose required state condition: {name}"
        for name, present in runtime_contract.items()
        if not present
    )

    samples_path = Path("data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl")
    queries_path = Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl")
    dataset_audit = {}
    evaluation_qids: set[str] = set()
    if samples_path.is_file() and queries_path.is_file():
        sample_rows, sample_qids = jsonl_qids(samples_path)
        query_rows, query_qids = jsonl_qids(queries_path)
        evaluation_qids = query_qids
        dataset_audit = {
            "sample_rows": sample_rows,
            "sample_qids": len(sample_qids),
            "query_rows": query_rows,
            "query_qids": len(query_qids),
            "qid_sets_match": sample_qids == query_qids,
        }
        if sample_rows != EXPECTED["steps"]:
            failures.append(f"unexpected sample rows: {sample_rows}")
        if len(sample_qids) != EXPECTED["qids"]:
            failures.append(f"unexpected sample qids: {len(sample_qids)}")
        if query_rows != EXPECTED["qids"] or sample_qids != query_qids:
            failures.append("sample/query qids do not match the frozen evaluation protocol")

    correct_report_audit = {}
    report_path = Path(args.correct_report)
    if report_path.is_file():
        report = read_json(report_path)
        summary = report.get("summary") or {}
        results = report.get("results") or report.get("records") or []
        checks = {
            "selector": summary.get("selector") == "hybrid_policy",
            "policy_context_source": summary.get("policy_context_source") == "online_state",
            "save_online_states": summary.get("save_online_states") is True,
            "checkpoint": str(summary.get("checkpoint") or "").endswith(
                "deberta_v3_large_v27_counterfactual_dual/best_model.pt"
            ),
        }
        for field, expected in EXPECTED.items():
            checks[field] = summary.get(field) == expected

        result_qids = [str(row.get("qid") or "") for row in results]
        duplicate_qids = len(result_qids) - len(set(result_qids))
        missing_state_steps = 0
        saved_state_steps = 0
        for record in results:
            for step in record.get("steps") or []:
                saved_state_steps += 1
                if not isinstance(step.get("online_state_before"), dict):
                    missing_state_steps += 1
        checks["result_qids"] = len(results) == EXPECTED["qids"] and duplicate_qids == 0
        checks["evaluation_qids"] = not evaluation_qids or set(result_qids) == evaluation_qids
        checks["saved_state_steps"] = (
            saved_state_steps == EXPECTED["steps"] and missing_state_steps == 0
        )
        failures.extend(
            f"correct online-state report failed protocol check: {name}"
            for name, passed in checks.items()
            if not passed
        )
        correct_report_audit = {
            "path": str(report_path),
            "checks": checks,
            "result_qids": len(results),
            "duplicate_qids": duplicate_qids,
            "saved_state_steps": saved_state_steps,
            "missing_online_state_before_steps": missing_state_steps,
        }

    output = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.5",
        "mode": "downstream_state_rollout_readiness",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "protocol": {
            "dataset": "HotpotQA fixed 3,000-qid evaluation subset",
            "checkpoint": "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
            "candidate_top_k": 10,
            "front_pool_k": 30,
            "select_top_k": 5,
            "state_update_top_k": 1,
            "policy_blend_weight": 0.5,
            "conditions": [
                "online_state",
                "query_only",
                "frozen_initial_state",
                "other_question_state",
                "previous_evidence_only",
            ],
            "other_question_pairing": "cyclic next qid in the frozen ordered evaluation set",
            "other_question_state_source": "the matched correct online-state rollout at the same step; latest prior step only when the paired trajectory is shorter",
            "answers_generated_by_readiness": False,
        },
        "runtime_contract": runtime_contract,
        "dataset_audit": dataset_audit,
        "correct_report_audit": correct_report_audit,
        "claim_boundary": (
            "This is a complete task-level rollout intervention. Earlier teacher-relative "
            "fixed-pool state diagnostics remain mechanism evidence only."
        ),
        "next_gate": "Run a 20-qid five-condition selection-only smoke; do not call the answer API.",
        "failures": failures,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
