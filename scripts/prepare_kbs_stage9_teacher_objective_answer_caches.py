#!/usr/bin/env python3
"""Prepare exact-context answer caches for Coverage-teacher evaluation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
ANSWER_PROTOCOL = {
    "answer_model": "deepseek-v4-flash",
    "answer_thinking_mode": "disabled",
    "answer_mode": "json",
    "answer_temperature": 0.0,
    "answer_prompt_version": "kbs_extractive_answer_json_v1",
}
CLOSURE_REPORTS = {
    42: Path("outputs/rag/kbs_v27_final_hotpot/full_compact.json"),
    43: Path("outputs/rag/kbs_v27_stage5_multiseed/seed43/full_compact.json"),
    44: Path("outputs/rag/kbs_v27_stage5_multiseed/seed44/full_compact.json"),
}
CLOSURE_CHECKPOINTS = {
    42: "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
    43: "outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt",
    44: "outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt",
}
COVERAGE_CHECKPOINTS = {
    42: "outputs/ranker/deberta_v3_large_v29_coverage_greedy/best_model.pt",
    43: "outputs/ranker/deberta_v3_large_v29_coverage_greedy_seed43/best_model.pt",
    44: "outputs/ranker/deberta_v3_large_v29_coverage_greedy_seed44/best_model.pt",
}


def read_report(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary") if isinstance(obj, dict) else None
    results = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report lacks summary/results: {path}")
    records: dict[str, dict[str, Any]] = {}
    for record in results:
        if not isinstance(record, dict):
            raise ValueError(f"non-object record in {path}")
        qid = str(record.get("qid") or "")
        if not qid or qid in records:
            raise ValueError(f"empty or duplicate qid in {path}: {qid!r}")
        records[qid] = record
    return summary, records


def checkpoint(seed: int, family: str) -> str:
    return (
        CLOSURE_CHECKPOINTS if family == "closure" else COVERAGE_CHECKPOINTS
    )[seed]


def selection_report(root: Path, seed: int, family: str) -> Path:
    return root / f"{family}_seed{seed}" / "alpha_0p50.json"


def cache_file(cache_dir: Path, qid: str) -> Path:
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
    return cache_dir / f"{safe_qid}.json"


def selected_units(record: dict[str, Any], path: Path) -> list[str]:
    units = record.get("selected_unit_ids")
    if not isinstance(units, list) or not units:
        raise ValueError(f"missing selected_unit_ids for {record.get('qid')} in {path}")
    return [str(value) for value in units]


def audit_summary(
    summary: dict[str, Any],
    path: Path,
    *,
    expected_checkpoint: str,
    answers: bool,
) -> list[str]:
    expected: dict[str, Any] = {
        "checkpoint": expected_checkpoint,
        "qids": 3000,
        "steps": 7296,
        "policy_context_source": "online_state",
        "selector": "hybrid_policy",
        "front_pool_k": 30,
        "front_fusion": "rrf",
        "local_expansion_window": 1,
        "candidate_top_k": 10,
        "select_top_k": 5,
        "state_update_top_k": 1,
        "policy_score_mode": "front_policy_blend",
        "policy_blend_weight": 0.5,
        "generate_answers": answers,
        "answer_judged": 3000 if answers else 0,
        "answer_errors": 0,
    }
    if answers:
        expected.update(ANSWER_PROTOCOL)
    return [
        f"{path}: {key}={summary.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]


def cache_payload(
    seed: int,
    qid: str,
    source_path: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qid": qid,
        "answer": str(source.get("answer") or ""),
        "raw_answer": str(source.get("raw_answer") or ""),
        "answer_tokens": int(source.get("answer_tokens") or 0),
        "answer_latency": float(source.get("answer_latency") or 0.0),
        **ANSWER_PROTOCOL,
        "cache_reuse": {
            "type": "exact_ordered_selected_context",
            "source_family": "closure",
            "target_family": "coverage",
            "seed": seed,
            "source_report": str(source_path),
            "selected_unit_ids": selected_units(source, source_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_teacher_objective/selection3000"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("outputs/rag/cache_kbs_stage9_teacher_objective"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/kbs_stage9_teacher_objective/answer_cache_readiness.json"
        ),
    )
    args = parser.parse_args()

    required = [args.selection_root / "multiseed_summary.json"]
    for seed in SEEDS:
        required.extend(
            [
                CLOSURE_REPORTS[seed],
                selection_report(args.selection_root, seed, "closure"),
                selection_report(args.selection_root, seed, "coverage"),
            ]
        )
    failures = [f"missing required path: {path}" for path in required if not path.exists()]
    per_seed: dict[str, Any] = {}
    intended: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in SEEDS}

    if not failures:
        multiseed = json.loads(required[0].read_text(encoding="utf-8"))
        if multiseed.get("status") != "OK" or multiseed.get("failures"):
            failures.append("multiseed selection summary is not a clean OK")

    for seed in SEEDS:
        if failures:
            break
        closure_answer_path = CLOSURE_REPORTS[seed]
        closure_selection_path = selection_report(args.selection_root, seed, "closure")
        coverage_selection_path = selection_report(args.selection_root, seed, "coverage")
        try:
            answer_summary, answer_records = read_report(closure_answer_path)
            closure_summary, closure_records = read_report(closure_selection_path)
            coverage_summary, coverage_records = read_report(coverage_selection_path)
            failures.extend(
                audit_summary(
                    answer_summary,
                    closure_answer_path,
                    expected_checkpoint=checkpoint(seed, "closure"),
                    answers=True,
                )
            )
            failures.extend(
                audit_summary(
                    closure_summary,
                    closure_selection_path,
                    expected_checkpoint=checkpoint(seed, "closure"),
                    answers=False,
                )
            )
            failures.extend(
                audit_summary(
                    coverage_summary,
                    coverage_selection_path,
                    expected_checkpoint=checkpoint(seed, "coverage"),
                    answers=False,
                )
            )
            ordered_qids = list(coverage_records)
            if not (
                ordered_qids == list(closure_records) == list(answer_records)
                and len(ordered_qids) == 3000
            ):
                failures.append(f"seed {seed}: report qid order/count differs")
                continue

            exact_matches = 0
            for qid in ordered_qids:
                answer_record = answer_records[qid]
                closure_record = closure_records[qid]
                coverage_record = coverage_records[qid]
                for key in ("question", "gold_answer"):
                    if not (
                        answer_record.get(key)
                        == closure_record.get(key)
                        == coverage_record.get(key)
                    ):
                        failures.append(f"seed {seed} qid {qid}: {key} differs")
                        break
                if failures:
                    break
                answer_units = selected_units(answer_record, closure_answer_path)
                closure_units = selected_units(closure_record, closure_selection_path)
                if answer_units != closure_units:
                    failures.append(
                        f"seed {seed} qid {qid}: historical Closure answer context "
                        "differs from current Closure selection"
                    )
                    break
                raw_answer = str(answer_record.get("raw_answer") or "").strip()
                if not raw_answer or raw_answer.startswith("ERROR:"):
                    failures.append(f"seed {seed} qid {qid}: invalid Closure answer")
                    break
                if answer_units == selected_units(
                    coverage_record, coverage_selection_path
                ):
                    intended[seed][qid] = cache_payload(
                        seed, qid, closure_answer_path, answer_record
                    )
                    exact_matches += 1

            per_seed[str(seed)] = {
                "closure_answer_report": str(closure_answer_path),
                "closure_selection_report": str(closure_selection_path),
                "coverage_selection_report": str(coverage_selection_path),
                "qids": len(ordered_qids),
                "reused_exact_context": exact_matches,
                "fresh_api_answers_required": len(ordered_qids) - exact_matches,
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"seed {seed}: {exc}")

    existing_identical = {seed: 0 for seed in SEEDS}
    if not failures:
        for seed in SEEDS:
            cache_dir = args.cache_root / f"coverage_seed{seed}"
            expected = {
                cache_file(cache_dir, qid): payload
                for qid, payload in intended[seed].items()
            }
            if cache_dir.exists():
                for path in cache_dir.glob("*.json"):
                    if path not in expected:
                        failures.append(f"unexpected pre-existing cache: {path}")
                        continue
                    try:
                        current = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        failures.append(f"cannot read pre-existing cache {path}: {exc}")
                        continue
                    if current != expected[path]:
                        failures.append(f"pre-existing cache payload differs: {path}")
                    else:
                        existing_identical[seed] += 1

    written = {seed: 0 for seed in SEEDS}
    if not failures:
        for seed in SEEDS:
            cache_dir = args.cache_root / f"coverage_seed{seed}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            for qid, payload in intended[seed].items():
                path = cache_file(cache_dir, qid)
                if path.exists():
                    continue
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                written[seed] += 1

    total_fresh = sum(
        int(report.get("fresh_api_answers_required") or 0)
        for report in per_seed.values()
    )
    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.1",
        "mode": "exact_context_answer_cache_preparation",
        "api_calls": 0,
        "protocol": {
            **ANSWER_PROTOCOL,
            "reuse_rule": (
                "seed, question, gold answer, and full ordered selected-unit "
                "sequence must match exactly"
            ),
            "cache_root": str(args.cache_root),
        },
        "per_seed": per_seed,
        "existing_identical_cache_files": {
            str(seed): count for seed, count in existing_identical.items()
        },
        "written_cache_files": {
            str(seed): count for seed, count in written.items()
        },
        "total_fresh_api_answers_required": total_fresh,
        "next_gate": (
            "Review API-call requirement before one 20-qid answer smoke."
            if not failures
            else "Resolve failures; do not call the answer API."
        ),
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
