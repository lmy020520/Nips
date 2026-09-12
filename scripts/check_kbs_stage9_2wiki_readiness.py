#!/usr/bin/env python3
"""Audit Stage 9.4 final-policy 2Wiki zero-shot prerequisites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ANSWER_PROTOCOL = {
    "answer_model": "deepseek-v4-flash",
    "answer_thinking_mode": "disabled",
    "answer_mode": "json",
    "answer_temperature": 0.0,
    "answer_prompt_version": "kbs_extractive_answer_json_v1",
}
CHECKPOINTS = {
    "42": Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"),
    "43": Path(
        "outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt"
    ),
    "44": Path(
        "outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt"
    ),
}
OPERATING_POINTS = {
    "compact": {"candidate_top_k": 10, "front_pool_k": 30},
    "balanced_knee": {"candidate_top_k": 15, "front_pool_k": 30},
    "recall": {"candidate_top_k": 50, "front_pool_k": 50},
}
BASELINES = {
    "hybrid": {
        "reported_name": "Hybrid-RAG",
        "selector": "hybrid",
        "candidate_top_k": 8,
        "state_update_top_k": 5,
    },
    "bge_reranker": {
        "reported_name": "BGE-Reranker-RAG",
        "selector": "generic_reranker",
        "candidate_top_k": 8,
        "state_update_top_k": 5,
    },
}
HISTORICAL_REPORTS = {
    "v22_compact": Path("outputs/rag/kbs_v22_stage2_2wiki/full_compact.json"),
    "v22_recall": Path("outputs/rag/kbs_v22_stage2_2wiki/full_recall.json"),
    "legacy_hybrid": Path("outputs/rag/2wiki_generalization_1000/hybrid_rag.json"),
    "legacy_bge": Path(
        "outputs/rag/2wiki_generalization_1000/bge_reranker_rag.json"
    ),
    "gold_oracle": Path("outputs/rag/2wiki_generalization_1000/gold_oracle.json"),
}


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def qid_set(rows: list[dict[str, Any]], path: Path) -> set[str]:
    values = {str(row.get("qid") or "") for row in rows}
    if "" in values:
        raise ValueError(f"missing qid in {path}")
    return values


def protocol_differences(summary: dict[str, Any]) -> dict[str, Any]:
    expected = {
        **ANSWER_PROTOCOL,
        "qids": 1000,
        "answer_judged": 1000,
        "answer_errors": 0,
        "generate_answers": True,
    }
    return {
        key: {"observed": summary.get(key), "expected": value}
        for key, value in expected.items()
        if summary.get(key) != value
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_2wiki"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.output_root / "readiness.json"

    data_root = Path("data/2wiki_multihopqa_eval_1000_cand50")
    manifest = data_root / "manifest.json"
    samples_path = data_root / "samples/test.jsonl"
    queries_path = data_root / "queries/test.jsonl"
    memory_path = data_root / "unit_registry/raw_units_test.jsonl"
    required = [
        Path("md/kbs_three_review_execution_plan.md"),
        Path("md/kbs_review_75_85_execution_plan.md"),
        Path("scripts/run_hotpotqa_policy_rag.py"),
        Path("scripts/evaluate_kbs_standard_metrics.py"),
        Path("scripts/bootstrap_kbs_stage4_metrics.py"),
        manifest,
        samples_path,
        queries_path,
        memory_path,
        Path("models/deberta-v3-large"),
        Path("models/bge-large-en-v1.5"),
        Path("models/bge-reranker-large"),
        HISTORICAL_REPORTS["gold_oracle"],
        *CHECKPOINTS.values(),
    ]
    failures = [f"missing required path: {path}" for path in required if not path.exists()]

    data_audit: dict[str, Any] = {
        "root": str(data_root),
        "queries": None,
        "samples": None,
        "memory_rows": None,
        "query_qids": None,
        "sample_qids": None,
        "sample_qids_match_queries": None,
    }
    manifest_summary: dict[str, Any] | None = None
    if not failures:
        try:
            manifest_obj = load_json(manifest)
            manifest_summary = {
                key: manifest_obj.get(key)
                for key in (
                    "dataset",
                    "source_split",
                    "output_split",
                    "size",
                    "seed",
                    "max_candidates",
                )
            }
            queries = load_jsonl(queries_path)
            samples = load_jsonl(samples_path)
            memory = load_jsonl(memory_path)
            query_qids = qid_set(queries, queries_path)
            sample_qids = qid_set(samples, samples_path)
            data_audit.update(
                {
                    "queries": len(queries),
                    "samples": len(samples),
                    "memory_rows": len(memory),
                    "query_qids": len(query_qids),
                    "sample_qids": len(sample_qids),
                    "sample_qids_match_queries": sample_qids == query_qids,
                }
            )
            if len(queries) != 1000 or len(query_qids) != 1000:
                failures.append(
                    f"2Wiki queries/qids must both be 1000, found {len(queries)}/{len(query_qids)}"
                )
            if len(samples) != 2416:
                failures.append(f"2Wiki sample states must be 2416, found {len(samples)}")
            if sample_qids != query_qids:
                failures.append("2Wiki sample qids do not exactly match query qids")
            if not memory:
                failures.append("2Wiki memory is empty")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"2Wiki data audit failed: {exc}")

    runtime_path = Path("scripts/run_hotpotqa_policy_rag.py")
    runtime_text = runtime_path.read_text(encoding="utf-8") if runtime_path.is_file() else ""
    implementation_audit = {}
    for key, spec in BASELINES.items():
        selector = str(spec["selector"])
        implemented = f'args.selector == "{selector}"' in runtime_text
        implementation_audit[key] = {"selector": selector, "implemented": implemented}
        if not implemented:
            failures.append(f"runtime does not implement selector={selector}")
    hybrid_policy_implemented = 'args.selector == "hybrid_policy"' in runtime_text
    if not hybrid_policy_implemented:
        failures.append("runtime does not implement selector=hybrid_policy")

    historical_audit = {}
    for key, path in HISTORICAL_REPORTS.items():
        entry: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "eligible_as_final_result": False,
            "eligible_as_exact_context_cache_source": False,
            "protocol_differences": {},
        }
        if path.is_file():
            try:
                obj = load_json(path)
                summary = obj.get("summary")
                if not isinstance(summary, dict):
                    raise ValueError("missing summary")
                differences = protocol_differences(summary)
                entry["protocol_differences"] = differences
                entry["eligible_as_exact_context_cache_source"] = not differences
                entry["qids"] = summary.get("qids")
                entry["checkpoint"] = summary.get("checkpoint")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                entry["read_error"] = str(exc)
        historical_audit[key] = entry

    proposed_outputs = {}
    for seed in CHECKPOINTS:
        proposed_outputs[seed] = {}
        for operating_point in OPERATING_POINTS:
            path = args.output_root / "selection1000" / f"{operating_point}_seed{seed}.json"
            proposed_outputs[seed][operating_point] = str(path)
            if path.exists():
                failures.append(f"proposed selection output already exists: {path}")
    for baseline in BASELINES:
        path = args.output_root / "selection1000" / f"{baseline}.json"
        proposed_outputs[baseline] = str(path)
        if path.exists():
            failures.append(f"proposed selection output already exists: {path}")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.4",
        "mode": "final_policy_2wiki_zero_shot_readiness",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "training_runs": 0,
        "transfer_contract": {
            "student_training_dataset": "HotpotQA only",
            "2wiki_fine_tuning": False,
            "primary_seed": 42,
            "robustness_selection_only_seeds": [43, 44],
            "policy_context_source": "online_state",
            "state_update_top_k": 1,
            "select_top_k": 5,
            "policy_blend_weight": 0.5,
        },
        "frozen_answer_protocol": ANSWER_PROTOCOL,
        "operating_points": OPERATING_POINTS,
        "baselines": BASELINES,
        "data_audit": data_audit,
        "manifest": manifest_summary,
        "checkpoint_paths": {seed: str(path) for seed, path in CHECKPOINTS.items()},
        "implementation_audit": {
            "hybrid_policy": hybrid_policy_implemented,
            **implementation_audit,
        },
        "historical_report_audit": historical_audit,
        "historical_policy": (
            "Historical v22 or legacy reports cannot serve as final v27 results. "
            "They may seed answer caches only after complete protocol and exact ordered-context matching."
        ),
        "proposed_selection_outputs": proposed_outputs,
        "next_gate": (
            "Run a 20-qid selection-only smoke for three seed-42 operating points, "
            "Hybrid, and BGE-Reranker; do not call the answer API."
            if not failures
            else "Resolve readiness failures before GPU inference or API calls."
        ),
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
