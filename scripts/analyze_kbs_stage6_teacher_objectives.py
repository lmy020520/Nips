#!/usr/bin/env python3
"""Compare observable Stage-6 teacher objectives on identical stored states."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


METHODS = (
    "relevance_front",
    "coverage_greedy",
    "closure_utility",
    "full_repair",
)
MATERIAL_DISAGREEMENT_RATE = 0.05
MATERIAL_DISAGREEMENT_COUNT = 25


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


def first_scored_candidate(candidate_ids: list, score_by_id: dict[str, dict]) -> str:
    for candidate_id in candidate_ids:
        candidate_id = str(candidate_id)
        if candidate_id in score_by_id:
            return candidate_id
    return ""


def coverage_candidate(candidate_ids: list, score_by_id: dict[str, dict]) -> str:
    available = [str(candidate_id) for candidate_id in candidate_ids if str(candidate_id) in score_by_id]
    if not available:
        return ""
    return max(
        available,
        key=lambda candidate_id: (
            float(score_by_id[candidate_id].get("uncovered_target_bonus") or 0.0),
            -available.index(candidate_id),
        ),
    )


def state_slice(t: int) -> str:
    if t == 0:
        return "t0"
    if t == 1:
        return "t1"
    return "t2_plus"


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def empty_counter() -> Counter:
    return Counter(
        states=0,
        selected_in_pool=0,
        uncovered_target_selected=0,
        derived_selected=0,
    )


def summarize_method(counter: Counter) -> dict:
    states = counter["states"]
    return {
        "states": states,
        "selected_in_pool": counter["selected_in_pool"],
        "selected_in_pool_rate": safe_rate(counter["selected_in_pool"], states),
        "uncovered_target_selected": counter["uncovered_target_selected"],
        "uncovered_target_selected_rate": safe_rate(counter["uncovered_target_selected"], states),
        "derived_selected": counter["derived_selected"],
        "derived_selected_rate": safe_rate(counter["derived_selected"], states),
    }


def summarize_pair(counter: Counter) -> dict:
    states = counter["states"]
    disagreements = counter["disagreements"]
    return {
        "states": states,
        "agreement_count": counter["agreements"],
        "agreement_rate": safe_rate(counter["agreements"], states),
        "disagreement_count": disagreements,
        "disagreement_rate": safe_rate(disagreements, states),
        "left_only_uncovered_target": counter["left_only_uncovered_target"],
        "right_only_uncovered_target": counter["right_only_uncovered_target"],
        "both_uncovered_target": counter["both_uncovered_target"],
        "neither_uncovered_target": counter["neither_uncovered_target"],
    }


def analyze_split(path: Path) -> tuple[dict, list[dict]]:
    qids = set()
    trajectory_lengths = Counter()
    terminal_status = Counter()
    abort_reasons = Counter()
    method_counters = {method: empty_counter() for method in METHODS}
    method_slice_counters = {
        method: defaultdict(empty_counter) for method in METHODS
    }
    pair_counters: dict[str, Counter] = defaultdict(Counter)
    pair_slice_counters: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    invalid_states = []
    repair_events = Counter()

    for row in read_jsonl(path):
        qid = str(row.get("qid") or "")
        qids.add(qid)
        steps = row.get("steps") or []
        trajectory_lengths[str(len(steps))] += 1
        terminal_status[str(row.get("terminal_status"))] += 1
        abort_reasons[str(row.get("abort_reason"))] += 1
        repair_events["repair_attempt_count"] += int(row.get("repair_attempt_count") or 0)
        repair_events["false_stop_count"] += int(row.get("final_false_stop_count") or 0)
        repair_events["repair_effective_qids"] += int(bool(row.get("repair_effective")))

        for step in steps:
            t = int(step.get("t") or 0)
            slice_name = state_slice(t)
            debug = ((step.get("candidate_debug") or {}).get("teacher_select_debug") or {})
            entries = debug.get("all_candidate_scores") or []
            score_by_id = {
                str(entry.get("unit_id") or ""): entry
                for entry in entries
                if str(entry.get("unit_id") or "")
            }
            candidate_ids = [str(item) for item in (step.get("C_t") or [])]
            relevance_ids = [str(item) for item in (step.get("R_t") or [])]
            closure_ranked = debug.get("top_candidates_before_repair_bias") or []
            selections = {
                "relevance_front": first_scored_candidate(relevance_ids, score_by_id),
                "coverage_greedy": coverage_candidate(candidate_ids, score_by_id),
                "closure_utility": str((closure_ranked[0] if closure_ranked else {}).get("unit_id") or ""),
                "full_repair": str(debug.get("selected_candidate") or step.get("positive_unit_id") or ""),
            }
            missing = [method for method, candidate_id in selections.items() if not candidate_id]
            if missing:
                invalid_states.append({"qid": qid, "t": t, "missing_methods": missing})
                continue

            for method, candidate_id in selections.items():
                entry = score_by_id.get(candidate_id, {})
                target_selected = float(entry.get("uncovered_target_bonus") or 0.0) > 0.0
                derived_selected = str(entry.get("provenance") or "") == "derived"
                for counter in (method_counters[method], method_slice_counters[method][slice_name]):
                    counter["states"] += 1
                    counter["selected_in_pool"] += int(candidate_id in candidate_ids)
                    counter["uncovered_target_selected"] += int(target_selected)
                    counter["derived_selected"] += int(derived_selected)

            for left_index, left in enumerate(METHODS):
                for right in METHODS[left_index + 1 :]:
                    pair_name = f"{left}-vs-{right}"
                    left_id = selections[left]
                    right_id = selections[right]
                    left_target = float(score_by_id.get(left_id, {}).get("uncovered_target_bonus") or 0.0) > 0.0
                    right_target = float(score_by_id.get(right_id, {}).get("uncovered_target_bonus") or 0.0) > 0.0
                    for counter in (pair_counters[pair_name], pair_slice_counters[pair_name][slice_name]):
                        counter["states"] += 1
                        same = left_id == right_id
                        counter["agreements"] += int(same)
                        counter["disagreements"] += int(not same)
                        counter["left_only_uncovered_target"] += int(left_target and not right_target)
                        counter["right_only_uncovered_target"] += int(right_target and not left_target)
                        counter["both_uncovered_target"] += int(left_target and right_target)
                        counter["neither_uncovered_target"] += int(not left_target and not right_target)

            repair_debug = step.get("repair_debug") or {}
            repair_events["repair_attempt_steps"] += int(bool(repair_debug.get("repair_attempt_active")))
            repair_events["repair_bias_steps"] += int(bool(repair_debug.get("teacher_select_bias_to_repair_applied")))
            repair_events["repair_continuation_steps"] += int(bool(repair_debug.get("closure_continuation_applied")))
            repair_events["derived_selected_steps"] += int(str(step.get("selected_provenance") or "") == "derived")

    methods = {}
    for method in METHODS:
        methods[method] = summarize_method(method_counters[method])
        methods[method]["by_t"] = {
            slice_name: summarize_method(counter)
            for slice_name, counter in sorted(method_slice_counters[method].items())
        }
    comparisons = {}
    for pair_name, counter in sorted(pair_counters.items()):
        comparisons[pair_name] = summarize_pair(counter)
        comparisons[pair_name]["by_t"] = {
            slice_name: summarize_pair(slice_counter)
            for slice_name, slice_counter in sorted(pair_slice_counters[pair_name].items())
        }

    result = {
        "path": str(path),
        "qids": len(qids),
        "states": sum(trajectory_lengths[length] * int(length) for length in trajectory_lengths),
        "trajectory_length_distribution": dict(sorted(trajectory_lengths.items(), key=lambda item: int(item[0]))),
        "terminal_status": dict(terminal_status),
        "abort_reasons": dict(abort_reasons),
        "methods": methods,
        "pairwise_comparisons": comparisons,
        "repair_events": dict(repair_events),
        "invalid_state_count": len(invalid_states),
    }
    return result, invalid_states[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=Path("data/hotpotqa_distractor_v7_10k_llm_prestep"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/kbs_stage6_teacher_objectives/objective_audit.json"),
    )
    args = parser.parse_args()

    paths = {
        split: args.teacher_root / "trajectories" / f"full_{split}.jsonl"
        for split in ("train", "val", "test")
    }
    failures = [f"missing trajectory: {path}" for path in paths.values() if not path.is_file()]
    splits = {}
    invalid_examples = {}
    if not failures:
        for split, path in paths.items():
            splits[split], invalid_examples[split] = analyze_split(path)
            if splits[split]["invalid_state_count"]:
                failures.append(
                    f"{split} has {splits[split]['invalid_state_count']} states without complete observable selections"
                )

    validation_comparison = (
        splits.get("val", {})
        .get("pairwise_comparisons", {})
        .get("coverage_greedy-vs-closure_utility", {})
    )
    disagreement_rate = validation_comparison.get("disagreement_rate")
    disagreement_count = int(validation_comparison.get("disagreement_count") or 0)
    labels_materially_differ = bool(
        disagreement_rate is not None
        and disagreement_rate >= MATERIAL_DISAGREEMENT_RATE
        and disagreement_count >= MATERIAL_DISAGREEMENT_COUNT
    )
    full_closure_identical = all(
        split_result.get("pairwise_comparisons", {})
        .get("closure_utility-vs-full_repair", {})
        .get("agreement_rate")
        == 1.0
        for split_result in splits.values()
    ) if splits else False

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 6,
        "step": "6.1",
        "mode": "same_state_teacher_objective_audit",
        "api_calls": 0,
        "gpu_runs": 0,
        "training_runs": 0,
        "protocol": {
            "comparison_unit": "identical stored teacher state and candidate pool",
            "relevance_front": "first R_t candidate present in scored C_t",
            "coverage_greedy": "first C_t candidate with maximum visible-uncovered-target indicator",
            "closure_utility": "top candidate before repair-specific bias",
            "full_repair": "actual final teacher selection after all enabled bias terms",
            "materiality_gate": {
                "decision_split": "validation",
                "minimum_disagreement_rate": MATERIAL_DISAGREEMENT_RATE,
                "minimum_disagreement_count": MATERIAL_DISAGREEMENT_COUNT,
            },
            "causal_scope": "one-step label comparison only; counterfactual rollout outcomes are not inferred",
        },
        "splits": splits,
        "decision": {
            "validation_coverage_vs_closure_disagreement_rate": disagreement_rate,
            "validation_coverage_vs_closure_disagreement_count": disagreement_count,
            "coverage_labels_materially_differ": labels_materially_differ,
            "full_repair_identical_to_closure_on_stored_states": full_closure_identical,
            "coverage_student_pilot": "authorize_matched_pilot" if labels_materially_differ else "not_justified",
            "repair_claim": "unsupported" if full_closure_identical else "requires_controlled_rollout",
        },
        "limitations": [
            "Success, abort, stall, and trajectory length are observed only for the stored Full trajectory.",
            "Alternative-objective selections are evaluated on the same states; they are not counterfactual rollouts.",
            "Coverage uses the stored visible-uncovered-target indicator and deterministic C_t order.",
            "Construction time was not recorded in the stored trajectory rows.",
        ],
        "invalid_state_examples": invalid_examples,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
