#!/usr/bin/env python3
"""Readiness audit for the final-v27 Stage 6.2 compression funnel."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path


EXPECTED_VARIANTS = {
    "rrf_top30",
    "rrf_local_mmr10",
    "bge_top10",
    "no_compression",
    "oracle_target_preserving_mmr10",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc


def candidate_ids(row: dict) -> list[str]:
    candidates = row.get("candidates") or {}
    return [str(value) for value in (candidates.get("C_t") or candidates.get("R_t") or [])]


def positive_id(row: dict) -> str:
    return str((((row.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id")) or "")


def static_audit(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    constants = set()
    functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            constants.add(node.value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
    forbidden = [
        token
        for token in ("DEEPSEEK_API_KEY", "answer_with_llm", "deepseek_chat")
        if token in source
    ]
    return {
        "syntax": "OK",
        "functions": sorted(functions),
        "variants": sorted(EXPECTED_VARIANTS & constants),
        "forbidden_api_tokens": forbidden,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl"))
    parser.add_argument("--memory", type=Path, default=Path("data/hotpotqa_distractor_eval_3000_cand50/unit_registry/raw_units_test.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/deberta-v3-large"))
    parser.add_argument("--dense-model", type=Path, default=Path("models/bge-large-en-v1.5"))
    parser.add_argument("--analyzer", type=Path, default=Path("scripts/analyze_kbs_stage6_compression_funnel.py"))
    parser.add_argument("--expected-qids", type=int, default=3000)
    parser.add_argument("--expected-states", type=int, default=7296)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/kbs_stage6_compression_funnel/readiness.json"))
    args = parser.parse_args()

    required = [
        Path("md/kbs_three_review_execution_plan.md"),
        args.samples,
        args.queries,
        args.memory,
        args.checkpoint,
        args.model_dir,
        args.dense_model,
        args.analyzer,
        Path("scripts/run_hotpotqa_policy_rag.py"),
        Path("src/online_state.py"),
    ]
    failures = [f"missing required path: {path}" for path in required if not path.exists()]
    audit = static_audit(args.analyzer) if args.analyzer.is_file() else {}
    if audit.get("forbidden_api_tokens"):
        failures.append(f"analyzer contains forbidden API tokens: {audit['forbidden_api_tokens']}")
    if set(audit.get("variants") or []) != EXPECTED_VARIANTS:
        failures.append(f"analyzer variant contract mismatch: {audit.get('variants')}")

    sample_summary = {}
    if args.samples.is_file() and args.queries.is_file() and args.memory.is_file():
        query_qids = {str(row.get("qid") or "") for row in read_jsonl(args.queries)}
        memory_ids = {str(row.get("unit_id") or "") for row in read_jsonl(args.memory)}
        qids = set()
        states = 0
        missing_positive = 0
        missing_memory = 0
        candidate_sizes = Counter()
        for row in read_jsonl(args.samples):
            qid = str(row.get("qid") or "")
            pool = candidate_ids(row)
            positive = positive_id(row)
            states += 1
            qids.add(qid)
            candidate_sizes[len(pool)] += 1
            missing_positive += int(not positive or positive not in pool)
            missing_memory += sum(unit_id not in memory_ids for unit_id in pool)
        sample_summary = {
            "qids": len(qids),
            "states": states,
            "query_qids": len(query_qids),
            "ordered_qids_match_queries": qids == query_qids,
            "missing_or_out_of_pool_positive": missing_positive,
            "candidate_ids_missing_from_memory": missing_memory,
            "candidate_size_distribution": dict(sorted(candidate_sizes.items())),
        }
        if len(qids) != args.expected_qids or len(query_qids) != args.expected_qids:
            failures.append(f"expected {args.expected_qids} qids, observed samples={len(qids)} queries={len(query_qids)}")
        if states != args.expected_states:
            failures.append(f"expected {args.expected_states} states, observed {states}")
        if qids != query_qids:
            failures.append("sample and query qid sets differ")
        if missing_positive:
            failures.append(f"{missing_positive} states lack an in-pool positive")
        if missing_memory:
            failures.append(f"{missing_memory} candidate references are missing from memory")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 6,
        "step": "6.2",
        "mode": "compression_funnel_readiness",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "policy_context_source": "online_state",
            "front_fusion": "rrf",
            "front_pool_k": 30,
            "candidate_top_k": 10,
            "select_top_k": 5,
            "state_update_top_k": 1,
            "policy_blend_weight": 0.5,
            "local_expansion_window": 1,
            "mmr_lambda": 0.7,
            "mmr_same_doc_similarity": 0.35,
            "variants": sorted(EXPECTED_VARIANTS),
            "raw_pool_is_membership_ceiling_only": True,
            "independent_online_state_per_variant": True,
            "answers_generated": False,
        },
        "sample_audit": sample_summary,
        "static_analyzer_audit": audit,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
