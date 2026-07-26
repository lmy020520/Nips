#!/usr/bin/env python3
"""Validate one frozen-protocol v22 Stage 2.2 HotpotQA report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_CONFIGS = {
    "full_compact": ("online_state", 30, 10),
    "full_recall": ("online_state", 50, 50),
    "query_only_compact": ("query_only", 30, 10),
    "query_only_recall": ("query_only", 50, 50),
    "previous_only_compact": ("previous_evidence_only", 30, 10),
    "previous_only_recall": ("previous_evidence_only", 50, 50),
}


def require_equal(summary: dict[str, Any], key: str, expected: Any) -> None:
    actual = summary.get(key)
    if actual != expected:
        raise ValueError(f"{key}: expected {expected!r}, found {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", choices=sorted(RUN_CONFIGS), required=True)
    parser.add_argument("--expected-qids", type=int, required=True)
    args = parser.parse_args()

    obj = json.loads(args.report.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    results = obj.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError("report must contain a summary object and results list")

    context_source, front_pool_k, candidate_top_k = RUN_CONFIGS[args.run]
    expected = {
        "qids": args.expected_qids,
        "answer_judged": args.expected_qids,
        "answer_errors": 0,
        "state_mode": "policy",
        "policy_context_source": context_source,
        "selector": "hybrid_policy",
        "dense_query_mode": "state",
        "hybrid_alpha": 0.5,
        "front_pool_k": front_pool_k,
        "front_fusion": "rrf",
        "local_expansion_window": 1,
        "candidate_top_k": candidate_top_k,
        "select_top_k": 5,
        "policy_score_mode": "front_policy_blend",
        "policy_blend_weight": 0.35,
        "answer_mode": "json",
        "generate_answers": True,
        "answer_generator": "deepseek",
        "answer_temperature": 0.0,
        "answer_prompt_version": "kbs_extractive_answer_json_v1",
        "refresh_answer_cache": True,
        "save_online_states": True,
        "seed": 20260608,
    }
    for key, value in expected.items():
        require_equal(summary, key, value)

    checkpoint = str(summary.get("checkpoint") or "")
    if not checkpoint.endswith("deberta_v3_large_v22_state_focused/best_model.pt"):
        raise ValueError(f"unexpected checkpoint: {checkpoint!r}")
    if len(results) != args.expected_qids:
        raise ValueError(
            f"result count: expected {args.expected_qids}, found {len(results)}"
        )

    qids = [str(row.get("qid") or "") for row in results if isinstance(row, dict)]
    if "" in qids or len(set(qids)) != args.expected_qids:
        raise ValueError("results contain missing or duplicate qids")
    empty_answers = [
        qid
        for qid, row in zip(qids, results)
        if not str(row.get("raw_answer") or "").strip()
    ]
    if empty_answers:
        raise ValueError(
            f"{len(empty_answers)} results have empty raw answers; "
            f"examples={empty_answers[:5]}"
        )
    avg_answer_tokens = summary.get("avg_answer_tokens")
    if not isinstance(avg_answer_tokens, (int, float)) or avg_answer_tokens <= 0:
        raise ValueError(f"avg_answer_tokens must be positive, found {avg_answer_tokens!r}")
    avg_answer_latency = summary.get("avg_answer_latency")
    if not isinstance(avg_answer_latency, (int, float)) or avg_answer_latency <= 0:
        raise ValueError(
            f"avg_answer_latency must be positive, found {avg_answer_latency!r}"
        )
    if not isinstance(summary.get("runtime_profile"), dict):
        raise ValueError("runtime_profile is missing")

    audit = {
        "status": "PASS",
        "run": args.run,
        "report": str(args.report),
        "qids": args.expected_qids,
        "steps": summary.get("steps"),
        "answer_judged": summary.get("answer_judged"),
        "answer_errors": summary.get("answer_errors"),
        "answer_em": summary.get("answer_em"),
        "answer_f1": summary.get("answer_f1"),
        "step_acc@1": summary.get("step_acc@1"),
        "step_acc@5": summary.get("step_acc@5"),
        "full_gold_unit_coverage": summary.get("full_gold_unit_coverage"),
        "avg_answer_tokens": summary.get("avg_answer_tokens"),
        "avg_answer_latency": summary.get("avg_answer_latency"),
        "selection_avg_ms_per_qid": summary["runtime_profile"].get(
            "selection_avg_ms_per_qid"
        ),
        "peak_gpu_allocated_mb": summary["runtime_profile"].get(
            "peak_gpu_allocated_mb"
        ),
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
