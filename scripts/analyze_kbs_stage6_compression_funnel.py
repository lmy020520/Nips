#!/usr/bin/env python3
"""Evaluate compression interventions with the final v27 online policy.

This is a selection-only analysis. It never calls an answer-generation API.
Each non-oracle intervention maintains its own online state so that later-hop
scores reflect the evidence actually written by that intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts import run_hotpotqa_policy_rag as rag


VARIANTS = (
    "rrf_top30",
    "rrf_local_mmr10",
    "bge_top10",
    "no_compression",
    "oracle_target_preserving_mmr10",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def build_front(
    context: str,
    candidate_ids: list[str],
    candidate_texts: list[str],
    memory: dict[str, dict],
    dense: rag.DenseScorer,
    args: argparse.Namespace,
) -> dict:
    dense_scores = dense.score(context, candidate_texts)
    lexical_scores = rag.bm25_scores(context, candidate_texts)
    dense_order = np.argsort(dense_scores)[::-1].tolist()
    lexical_order = np.argsort(lexical_scores)[::-1].tolist()
    rrf_scores = rag.reciprocal_rank_fusion(
        [lexical_order, dense_order], len(candidate_ids)
    )
    rrf_order = np.argsort(rrf_scores)[::-1].tolist()

    front_k = min(args.front_pool_k, len(candidate_ids))
    seed_indices = unique(lexical_order[:front_k] + dense_order[:front_k])
    expanded_indices = rag.local_expanded_pool(
        seed_indices,
        candidate_ids,
        memory,
        window=args.local_expansion_window,
    )
    expanded_indices.sort(key=lambda index: float(rrf_scores[index]), reverse=True)
    mmr_indices = rag.mmr_select(
        expanded_indices,
        rrf_scores,
        candidate_texts,
        candidate_ids,
        memory,
        limit=min(args.candidate_top_k, len(candidate_ids)),
        lambda_=args.mmr_lambda,
        same_doc_similarity=args.mmr_same_doc_similarity,
    )
    return {
        "dense_scores": dense_scores,
        "dense_order": dense_order,
        "rrf_scores": rrf_scores,
        "rrf_order": rrf_order,
        "seed_indices": seed_indices,
        "expanded_indices": expanded_indices,
        "mmr_indices": mmr_indices,
    }


def retained_pool(
    variant: str,
    front: dict,
    positive_index: int,
    args: argparse.Namespace,
) -> tuple[list[int], np.ndarray]:
    if variant == "rrf_top30":
        indices = front["rrf_order"][: min(args.front_pool_k, len(front["rrf_order"]))]
        scores = front["rrf_scores"]
    elif variant == "rrf_local_mmr10":
        indices = list(front["mmr_indices"])
        scores = front["rrf_scores"]
    elif variant == "bge_top10":
        indices = front["dense_order"][: min(args.candidate_top_k, len(front["dense_order"]))]
        scores = front["dense_scores"]
    elif variant == "no_compression":
        indices = list(front["rrf_order"])
        scores = front["rrf_scores"]
    elif variant == "oracle_target_preserving_mmr10":
        indices = list(front["mmr_indices"])
        if positive_index not in indices:
            if indices:
                indices[-1] = positive_index
            else:
                indices = [positive_index]
        indices = unique(indices)
        scores = front["rrf_scores"]
    else:
        raise ValueError(f"unknown compression variant: {variant}")
    return indices, np.asarray(scores, dtype=float)


def blended_order(
    context: str,
    retained: list[int],
    front_scores: np.ndarray,
    candidate_texts: list[str],
    policy: rag.PolicyModel,
    policy_weight: float,
) -> list[int]:
    retained_texts = [candidate_texts[index] for index in retained]
    policy_scores = policy.score(context, retained_texts)
    local_front = np.array([front_scores[index] for index in retained], dtype=float)
    final_scores = (
        (1.0 - policy_weight) * rag.minmax_normalize(local_front)
        + policy_weight * rag.minmax_normalize(policy_scores)
    )
    local_order = np.argsort(final_scores)[::-1].tolist()
    return [retained[index] for index in local_order]


def render_context(question: str, state: dict) -> str:
    return f"Question: {question}\nNotebook:\n{state['K_t']}"


def evaluate_variant(
    variant: str,
    qids: list[str],
    grouped: dict[str, list[dict]],
    memory: dict[str, dict],
    dense: rag.DenseScorer,
    policy: rag.PolicyModel,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    totals = Counter()
    qid_totals = Counter()
    records = []
    started = time.perf_counter()

    for qid in tqdm(qids, desc=f"compression:{variant}"):
        rows = sorted(grouped[qid], key=lambda row: int(row.get("t", 0)))
        target_units = [rag.sample_positive_id(row) for row in rows]
        target_units = [unit_id for unit_id in target_units if unit_id in memory]
        selected_units: list[str] = []
        selected_set: set[str] = set()
        state_written: set[str] = set()
        online_state = rag.init_online_state()
        qid_records = []

        for row in rows:
            question = str(row.get("question") or "")
            positive_id = rag.sample_positive_id(row)
            source_ids = rag.sample_candidate_ids(row)
            usable_ids = [unit_id for unit_id in source_ids if unit_id in memory]
            if not question or positive_id not in usable_ids:
                totals["skipped"] += 1
                continue
            candidate_texts = [rag.format_candidate_text(memory[unit_id]) for unit_id in usable_ids]
            positive_index = usable_ids.index(positive_id)
            context = render_context(question, online_state)
            front = build_front(context, usable_ids, candidate_texts, memory, dense, args)
            retained, retained_front_scores = retained_pool(
                variant, front, positive_index, args
            )
            order = blended_order(
                context,
                retained,
                retained_front_scores,
                candidate_texts,
                policy,
                args.policy_blend_weight,
            )
            selected_indices = order[: min(args.select_top_k, len(order))]
            selected_ids = [usable_ids[index] for index in selected_indices]
            state_ids = selected_ids[: min(args.state_update_top_k, len(selected_ids))]

            totals["steps"] += 1
            totals["target_retained"] += int(positive_index in retained)
            totals["alignment_at_1"] += int(bool(selected_ids) and selected_ids[0] == positive_id)
            totals["alignment_at_5"] += int(positive_id in selected_ids)
            totals["retained_units"] += len(retained)
            totals["selected_units"] += len(selected_ids)
            for unit_id in selected_ids:
                if unit_id not in selected_set:
                    selected_set.add(unit_id)
                    selected_units.append(unit_id)
            for unit_id in state_ids:
                if unit_id in state_written:
                    continue
                online_state = rag.update_online_state(
                    online_state,
                    unit_id,
                    memory[unit_id],
                    memory,
                    step_id=int(row.get("t", 0)),
                    max_raw=args.online_state_max_raw,
                    max_chars_per_item=args.online_state_max_chars,
                )
                state_written.add(unit_id)
            qid_records.append(
                {
                    "t": int(row.get("t", 0)),
                    "positive_unit_id": positive_id,
                    "raw_pool_size": len(usable_ids),
                    "retained_pool_size": len(retained),
                    "target_retained": positive_index in retained,
                    "selected_unit_ids": selected_ids,
                    "selected_contains_target": positive_id in selected_ids,
                }
            )

        if not qid_records:
            continue
        qid_totals["qids"] += 1
        qid_totals["full_unit_coverage"] += int(set(target_units).issubset(selected_set))
        records.append(
            {
                "variant": variant,
                "qid": qid,
                "gold_unit_ids": target_units,
                "selected_unit_ids": selected_units,
                "full_unit_coverage": set(target_units).issubset(selected_set),
                "steps": qid_records,
            }
        )

    elapsed = time.perf_counter() - started
    step_count = totals["steps"]
    qid_count = qid_totals["qids"]
    summary = {
        "qids": qid_count,
        "steps": step_count,
        "skipped": totals["skipped"],
        "target_recall": round(totals["target_retained"] / step_count, 6) if step_count else None,
        "policy_alignment_at_1": round(totals["alignment_at_1"] / step_count, 6) if step_count else None,
        "policy_alignment_at_5": round(totals["alignment_at_5"] / step_count, 6) if step_count else None,
        "full_unit_coverage": round(qid_totals["full_unit_coverage"] / qid_count, 6) if qid_count else None,
        "avg_retained_pool_size": round(totals["retained_units"] / step_count, 3) if step_count else None,
        "avg_selected_units_per_step": round(totals["selected_units"] / step_count, 3) if step_count else None,
        "selection_seconds": round(elapsed, 3),
        "selection_ms_per_qid": round(1000.0 * elapsed / qid_count, 3) if qid_count else None,
        "qids_per_second": round(qid_count / elapsed, 6) if elapsed else None,
    }
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dense-model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--front-pool-k", type=int, default=30)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--select-top-k", type=int, default=5)
    parser.add_argument("--state-update-top-k", type=int, default=1)
    parser.add_argument("--policy-blend-weight", type=float, default=0.5)
    parser.add_argument("--local-expansion-window", type=int, default=1)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--mmr-same-doc-similarity", type=float, default=0.35)
    parser.add_argument("--online-state-max-raw", type=int, default=8)
    parser.add_argument("--online-state-max-chars", type=int, default=260)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    args = parser.parse_args()

    if args.policy_blend_weight != 0.5:
        raise ValueError("Stage 6.2 is locked to final v27 alpha=0.5")
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    samples = list(rag.read_jsonl(args.samples))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        qid = str(row.get("qid") or "")
        if qid:
            grouped[qid].append(row)
    qids = sorted(grouped)
    if args.max_qids > 0:
        qids = qids[: args.max_qids]
    memory = rag.load_memory(args.memory)
    dense = rag.DenseScorer(args.dense_model, device=args.device, batch_size=args.dense_batch_size)
    policy = rag.PolicyModel(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    if policy.architecture != "dual_state_interaction":
        raise ValueError(f"expected final v27 dual-state checkpoint, got {policy.architecture}")

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    summaries = {}
    all_records = []
    for variant in variants:
        summaries[variant], records = evaluate_variant(
            variant, qids, grouped, memory, dense, policy, args
        )
        all_records.extend(records)
    peak_mb = None
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)

    raw_steps = sum(
        1
        for qid in qids
        for row in grouped[qid]
        if rag.sample_positive_id(row) in rag.sample_candidate_ids(row)
    )
    result = {
        "status": "OK",
        "stage": 6,
        "step": "6.2",
        "mode": "compression_funnel",
        "api_calls": 0,
        "protocol": {
            "samples": str(args.samples),
            "memory": str(args.memory),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "dense_model": args.dense_model,
            "qids": len(qids),
            "front_pool_k": args.front_pool_k,
            "candidate_top_k": args.candidate_top_k,
            "select_top_k": args.select_top_k,
            "state_update_top_k": args.state_update_top_k,
            "policy_blend_weight": args.policy_blend_weight,
            "local_expansion_window": args.local_expansion_window,
            "mmr_lambda": args.mmr_lambda,
            "mmr_same_doc_similarity": args.mmr_same_doc_similarity,
            "online_state_max_raw": args.online_state_max_raw,
            "online_state_max_chars": args.online_state_max_chars,
            "independent_online_state_per_variant": True,
        },
        "raw_pool": {
            "steps": raw_steps,
            "target_recall": 1.0 if raw_steps else None,
            "policy_alignment_at_1": None,
            "policy_alignment_at_5": None,
            "full_unit_coverage": None,
            "note": "Membership ceiling only; selection metrics are reported by no_compression.",
        },
        "variants": summaries,
        "peak_gpu_allocated_mb": peak_mb,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(all_records, args.records_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"records: {args.records_output}")


if __name__ == "__main__":
    main()
