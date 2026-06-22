#!/usr/bin/env python3
"""Validate online KBS state alignment in policy-RAG reports.

This script checks the execution trace produced by:

    run_hotpotqa_policy_rag.py --save-online-states

It does not evaluate model quality. It only verifies that online states are
internally consistent and aligned with selected evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def add_issue(issues: list[dict], *, qid: str, step_index: int | None, issue: str, **extra: Any) -> None:
    item = {"qid": qid, "issue": issue}
    if step_index is not None:
        item["step_index"] = step_index
    item.update(extra)
    issues.append(item)


def state_signature(state: dict | None) -> dict:
    if not isinstance(state, dict):
        return {}
    a_t = state.get("A_t") if isinstance(state.get("A_t"), dict) else {}
    s_t = state.get("S_t") if isinstance(state.get("S_t"), dict) else {}
    raw_refs = s_t.get("raw_refs") if isinstance(s_t.get("raw_refs"), list) else []
    return {
        "H_t": list(state.get("H_t") or []),
        "A_raw": list(a_t.get("raw_unit_ids") or []),
        "A_docs": list(a_t.get("doc_ids") or []),
        "S_raw": [str(ref.get("unit_id") or "") for ref in raw_refs if isinstance(ref, dict)],
        "K_t": str(state.get("K_t") or ""),
    }


def validate_state_shape(issues: list[dict], *, qid: str, step_index: int | None, state: Any, name: str) -> None:
    if not isinstance(state, dict):
        add_issue(issues, qid=qid, step_index=step_index, issue=f"{name}_not_object")
        return
    for key in ("H_t", "A_t", "S_t", "K_t"):
        if key not in state:
            add_issue(issues, qid=qid, step_index=step_index, issue=f"{name}_missing_{key}")
    if not isinstance(state.get("H_t"), list):
        add_issue(issues, qid=qid, step_index=step_index, issue=f"{name}_H_t_not_list")
    if not isinstance(state.get("A_t"), dict):
        add_issue(issues, qid=qid, step_index=step_index, issue=f"{name}_A_t_not_object")
    if not isinstance(state.get("S_t"), dict):
        add_issue(issues, qid=qid, step_index=step_index, issue=f"{name}_S_t_not_object")
    if not isinstance(state.get("K_t"), str):
        add_issue(issues, qid=qid, step_index=step_index, issue=f"{name}_K_t_not_string")


def validate_result(result: dict, issues: list[dict], counters: Counter) -> None:
    qid = str(result.get("qid") or "")
    steps = result.get("steps")
    if not qid:
        add_issue(issues, qid="", step_index=None, issue="result_missing_qid")
    if not isinstance(steps, list):
        add_issue(issues, qid=qid, step_index=None, issue="steps_not_list")
        return

    expected_h: list[str] = []
    previous_after = None
    counters["qids"] += 1
    counters["steps"] += len(steps)

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            add_issue(issues, qid=qid, step_index=idx, issue="step_not_object")
            continue

        before = step.get("online_state_before")
        after = step.get("online_state_after")
        validate_state_shape(issues, qid=qid, step_index=idx, state=before, name="online_state_before")
        validate_state_shape(issues, qid=qid, step_index=idx, state=after, name="online_state_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue

        before_sig = state_signature(before)
        after_sig = state_signature(after)

        if idx == 0:
            if before_sig.get("H_t"):
                add_issue(
                    issues,
                    qid=qid,
                    step_index=idx,
                    issue="first_before_H_t_not_empty",
                    H_t=before_sig.get("H_t"),
                )
        elif previous_after is not None and before_sig != previous_after:
            add_issue(issues, qid=qid, step_index=idx, issue="before_state_not_equal_previous_after")

        selected = step.get("selected_unit_ids")
        if not isinstance(selected, list):
            add_issue(issues, qid=qid, step_index=idx, issue="selected_unit_ids_not_list")
            selected = []
        for unit_id in selected:
            unit_id = str(unit_id)
            if unit_id not in expected_h:
                expected_h.append(unit_id)

        h_after = after_sig.get("H_t") or []
        a_after = after_sig.get("A_raw") or []
        s_after = after_sig.get("S_raw") or []

        if h_after != expected_h:
            add_issue(
                issues,
                qid=qid,
                step_index=idx,
                issue="H_t_not_equal_cumulative_selected_units",
                expected=expected_h,
                actual=h_after,
            )
        if a_after != h_after:
            add_issue(issues, qid=qid, step_index=idx, issue="A_t_raw_unit_ids_not_equal_H_t")
        if s_after != h_after:
            add_issue(issues, qid=qid, step_index=idx, issue="S_t_raw_refs_not_equal_H_t")
        if h_after and not after_sig.get("K_t"):
            add_issue(issues, qid=qid, step_index=idx, issue="K_t_empty_after_selection")

        previous_after = after_sig

    final_state = result.get("final_online_state")
    if final_state is not None:
        validate_state_shape(issues, qid=qid, step_index=None, state=final_state, name="final_online_state")
        if isinstance(final_state, dict):
            final_sig = state_signature(final_state)
            if final_sig.get("H_t") != expected_h:
                add_issue(
                    issues,
                    qid=qid,
                    step_index=None,
                    issue="final_H_t_not_equal_cumulative_selected_units",
                    expected=expected_h,
                    actual=final_sig.get("H_t"),
                )
    else:
        add_issue(issues, qid=qid, step_index=None, issue="missing_final_online_state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-issues", type=int, default=30)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("results")
    issues: list[dict] = []
    counters: Counter = Counter()

    if not isinstance(results, list):
        issues.append({"qid": "", "issue": "report_results_not_list"})
    else:
        for result in results:
            if isinstance(result, dict):
                validate_result(result, issues, counters)
            else:
                issues.append({"qid": "", "issue": "result_not_object"})

    summary = {
        "report": str(report_path),
        "qids": counters["qids"],
        "steps": counters["steps"],
        "issue_count": len(issues),
        "issues": issues[: args.max_issues],
        "status": "PASS" if not issues else "FAILED",
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
