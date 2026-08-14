#!/usr/bin/env python3
"""Inventory Stage-6 teacher, compression, and cost evidence without API calls."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


TEACHER_CONSTANTS = (
    "ETA_BR",
    "ETA_DIS",
    "ETA_SUP",
    "ETA_CTX",
    "ETA_REPEAT_PENALTY",
    "ETA_RETRIEVAL_ORDER_BONUS",
    "STALL_WINDOW",
    "FALSE_STOP_LIMIT",
    "T_MAX",
    "MAX_REPAIR_CONTINUATIONS",
)
TEACHER_FUNCTIONS = (
    "compute_U_for_candidate",
    "compute_repeat_penalty",
    "candidate_priority_key",
    "teacher_select",
    "rollout_one_qid",
)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_teacher_contract(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    constants = {}
    functions = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in TEACHER_CONSTANTS:
            continue
        try:
            constants[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            constants[target.id] = "dynamic"
    return {
        "path": str(path),
        "sha256": sha256(path),
        "constants": {name: constants.get(name) for name in TEACHER_CONSTANTS},
        "functions": {name: name in functions for name in TEACHER_FUNCTIONS},
    }


def first_unit(entries: list[dict], key: str) -> str:
    if not entries:
        return ""
    return str(entries[0].get(key) or "")


def coverage_proxy(step: dict, score_entries: list[dict]) -> str:
    """Choose visible uncovered target first, then preserve C_t order."""
    candidate_order = {
        str(unit_id): index for index, unit_id in enumerate(step.get("C_t") or [])
    }
    ranked = sorted(
        score_entries,
        key=lambda item: (
            -float(item.get("uncovered_target_bonus") or 0.0),
            candidate_order.get(str(item.get("unit_id") or ""), 10**9),
            str(item.get("unit_id") or ""),
        ),
    )
    return first_unit(ranked, "unit_id")


def summarize_trajectory(path: Path) -> dict:
    terminal_status = Counter()
    abort_reasons = Counter()
    trajectory_lengths = Counter()
    qids = set()
    totals = Counter()
    for row in read_jsonl(path):
        qid = str(row.get("qid") or "")
        qids.add(qid)
        steps = row.get("steps") or []
        terminal_status[str(row.get("terminal_status"))] += 1
        abort_reasons[str(row.get("abort_reason"))] += 1
        trajectory_lengths[str(len(steps))] += 1
        totals["steps"] += len(steps)
        totals["stop_probe_count"] += int(row.get("stop_probe_count") or 0)
        totals["false_stop_count"] += int(row.get("final_false_stop_count") or 0)
        totals["repair_attempt_count"] += int(row.get("repair_attempt_count") or 0)
        totals["repair_effective_qids"] += int(bool(row.get("repair_effective")))
        build_time = (row.get("build_meta") or {}).get("build_time_seconds")
        totals["build_time_records"] += int(isinstance(build_time, (int, float)))
        for step in steps:
            totals["derived_selected_steps"] += int(
                str(step.get("selected_provenance") or "") == "derived"
            )
            debug = ((step.get("candidate_debug") or {}).get("teacher_select_debug") or {})
            entries = debug.get("all_candidate_scores") or []
            if not entries:
                continue
            totals["score_debug_steps"] += 1
            selected = str(debug.get("selected_candidate") or step.get("positive_unit_id") or "")
            base = first_unit(debug.get("top_candidates_before_repair_bias") or [], "unit_id")
            relevance = str((step.get("R_t") or [""])[0])
            coverage = coverage_proxy(step, entries)
            totals["full_matches_base_closure"] += int(selected == base)
            totals["full_matches_relevance_top1"] += int(selected == relevance)
            totals["full_matches_target_coverage_proxy"] += int(selected == coverage)

    qid_count = len(qids)
    step_count = totals["steps"]
    debug_count = totals["score_debug_steps"]
    agreement = {}
    for key in (
        "full_matches_base_closure",
        "full_matches_relevance_top1",
        "full_matches_target_coverage_proxy",
    ):
        agreement[key] = {
            "count": totals[key],
            "rate": round(totals[key] / debug_count, 6) if debug_count else None,
        }
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "qids": qid_count,
        "steps": step_count,
        "avg_steps_per_qid": round(step_count / qid_count, 6) if qid_count else None,
        "trajectory_length_distribution": dict(sorted(trajectory_lengths.items(), key=lambda x: int(x[0]))),
        "terminal_status": dict(terminal_status),
        "abort_reasons": dict(abort_reasons),
        "stop_probe_count": totals["stop_probe_count"],
        "false_stop_count": totals["false_stop_count"],
        "repair_attempt_count": totals["repair_attempt_count"],
        "repair_effective_qids": totals["repair_effective_qids"],
        "derived_selected_steps": totals["derived_selected_steps"],
        "build_time_records": totals["build_time_records"],
        "score_debug_steps": debug_count,
        "offline_label_agreement": agreement,
    }


def report_info(path: Path) -> dict:
    info = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return info
    info["bytes"] = path.stat().st_size
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        info["parse_error"] = str(exc)
        return info
    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else obj
    info["checkpoint"] = summary.get("checkpoint")
    info["qids"] = summary.get("qids")
    info["candidate_top_k"] = summary.get("candidate_top_k")
    info["policy_blend_weight"] = summary.get("policy_blend_weight")
    runtime = summary.get("runtime_profile") or {}
    info["has_runtime_profile"] = bool(runtime)
    info["selection_ms_per_qid"] = runtime.get("selection_avg_ms_per_qid")
    info["peak_gpu_allocated_mb"] = runtime.get("peak_gpu_allocated_mb")
    return info


def discovered_reports(root: Path, patterns: tuple[str, ...]) -> list[dict]:
    found = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            found.append(report_info(path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, default=Path("data/hotpotqa_distractor_v7_10k_llm_prestep"))
    parser.add_argument("--teacher-code", type=Path, default=Path("scripts/build_hotpotqa_full_trajectories_v4.py"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/kbs_stage6_readiness/artifact_inventory.json"))
    args = parser.parse_args()

    trajectory_paths = {
        split: args.teacher_root / "trajectories" / f"full_{split}.jsonl"
        for split in ("train", "val", "test")
    }
    mandatory = [
        Path("md/kbs_three_review_execution_plan.md"),
        args.teacher_code,
        *trajectory_paths.values(),
        Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl"),
    ]
    failures = [f"missing mandatory path: {path}" for path in mandatory if not path.exists()]

    teacher_contract = None
    trajectories = {}
    if args.teacher_code.is_file():
        teacher_contract = static_teacher_contract(args.teacher_code)
        missing_functions = [
            name for name, present in teacher_contract["functions"].items() if not present
        ]
        if missing_functions:
            failures.append(f"teacher code missing functions: {missing_functions}")
    for split, path in trajectory_paths.items():
        if path.is_file():
            trajectories[split] = summarize_trajectory(path)

    total_steps = sum(item["steps"] for item in trajectories.values())
    total_repairs = sum(item["repair_attempt_count"] for item in trajectories.values())
    total_false_stops = sum(item["false_stop_count"] for item in trajectories.values())
    total_derived = sum(item["derived_selected_steps"] for item in trajectories.values())
    output_root = Path("outputs")
    artifacts = {
        "final_v27_answer_reports": discovered_reports(
            output_root,
            (
                "rag/kbs_v27_final_hotpot/*.json",
                "rag/kbs_v27_stage5*/**/*.json",
            ),
        ),
        "compression_failure_reports": discovered_reports(
            output_root,
            (
                "analysis/*failure*summary*.json",
                "analysis/*frontend*diagnos*.json",
                "analysis/*compression*.json",
            ),
        ),
        "candidate_budget_reports": discovered_reports(
            output_root,
            (
                "rag/candidate_size/*.json",
                "rag/kbs_v21_unified/*compact*.json",
                "rag/kbs_v21_unified/*recall*.json",
            ),
        ),
        "runtime_profiles": discovered_reports(
            output_root,
            (
                "analysis/runtime_profiles*/*.json",
                "analysis/*runtime*/*.json",
            ),
        ),
        "agentic_reports": discovered_reports(
            output_root,
            ("rag/*agentic*.json", "rag/**/agentic*.json"),
        ),
    }

    final_checkpoint = "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"
    final_runtime = [
        item
        for item in artifacts["runtime_profiles"]
        if item.get("checkpoint") == final_checkpoint and item.get("has_runtime_profile")
    ]
    final_budget_reports = [
        item
        for item in artifacts["candidate_budget_reports"]
        if item.get("checkpoint") == final_checkpoint
    ]
    final_compression_reports = [
        item
        for item in artifacts["compression_failure_reports"]
        if item.get("checkpoint") == final_checkpoint
    ]

    gaps = []
    if total_repairs == 0 and total_false_stops == 0 and total_derived == 0:
        gaps.append(
            "Final v7 trajectories contain no repair, false-stop, or derived-selection events; "
            "a Full-Repair advantage cannot be claimed from the training trajectories."
        )
    if not final_compression_reports:
        gaps.append("No compression-funnel report is tied to the final v27 checkpoint.")
    if not final_runtime:
        gaps.append("No unified runtime profile is tied to the final v27 checkpoint and alpha=0.5 protocol.")
    final_budgets = sorted(
        {
            int(item["candidate_top_k"])
            for item in final_budget_reports
            if isinstance(item.get("candidate_top_k"), int)
        }
    )
    if not {10, 15, 20, 50}.issubset(final_budgets):
        gaps.append(
            f"Final-v27 cost frontier lacks candidate budgets {sorted({10, 15, 20, 50} - set(final_budgets))}."
        )
    gaps.append(
        "A four-way teacher trajectory comparison is not present; stored score debug supports "
        "offline one-step label agreement only, not counterfactual rollout success/abort statistics."
    )
    gaps.append(
        "Existing agentic reports do not establish complete selection-call/token/cost accounting unless "
        "their per-qid selection records contain those fields."
    )

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 6,
        "mode": "offline_artifact_inventory",
        "api_calls": 0,
        "training_runs": 0,
        "teacher_contract": teacher_contract,
        "trajectories": trajectories,
        "trajectory_totals": {
            "steps": total_steps,
            "repair_attempt_count": total_repairs,
            "false_stop_count": total_false_stops,
            "derived_selected_steps": total_derived,
        },
        "artifacts": artifacts,
        "final_v27_coverage": {
            "compression_reports": len(final_compression_reports),
            "runtime_profiles": len(final_runtime),
            "candidate_budgets": final_budgets,
        },
        "readiness_decision": {
            "teacher_current_trajectory_stats": "available_offline",
            "teacher_one_step_label_agreement": "available_offline",
            "teacher_counterfactual_rollout_ablation": "missing_controlled_outputs",
            "compression_funnel_final_v27": (
                "available" if final_compression_reports else "requires_no_api_gpu_run"
            ),
            "runtime_profile_final_v27": (
                "available" if final_runtime else "requires_no_api_gpu_run"
            ),
            "cost_frontier_answer_metrics": "partially_available; final-v27 budgets 15/20 may require answer generation",
            "agentic_complete_cost_accounting": "not_verified; optional unless paper retains cost comparison",
        },
        "experiment_gaps": gaps,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
