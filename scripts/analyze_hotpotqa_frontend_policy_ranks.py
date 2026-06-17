#!/usr/bin/env python3
"""Analyze where gold evidence is lost in the hybrid-front-end + policy pipeline."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from run_hotpotqa_policy_rag import (
    DenseScorer,
    PolicyModel,
    bm25_scores,
    format_candidate_text,
    format_notebook_evidence,
    load_memory,
    local_expanded_pool,
    mmr_select,
    read_jsonl,
    reciprocal_rank_fusion,
    sample_candidate_ids,
    sample_k_t,
    sample_positive_id,
)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rank_of(order: list[int], index: int) -> int | None:
    try:
        return order.index(index) + 1
    except ValueError:
        return None


def hit_at(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def bucket_rank(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 1:
        return "1"
    if rank <= 2:
        return "<=2"
    if rank <= 3:
        return "<=3"
    if rank <= 5:
        return "<=5"
    if rank <= 10:
        return "<=10"
    if rank <= 20:
        return "<=20"
    if rank <= 30:
        return "<=30"
    if rank <= 50:
        return "<=50"
    return ">50"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dense-model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--state-mode", choices=["dataset", "policy"], default="policy")
    parser.add_argument("--dense-query-mode", choices=["question", "state"], default="state")
    parser.add_argument("--front-pool-k", type=int, default=30)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--local-expansion-window", type=int, default=1)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--mmr-same-doc-similarity", type=float, default=0.35)
    parser.add_argument("--select-top-k", type=int, default=5)
    parser.add_argument("--ks", default="1,2,3,5,10,15,20,30,50")
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--output", default="outputs/analysis/frontend_policy_rank_summary.json")
    parser.add_argument("--records-output", default="outputs/analysis/frontend_policy_rank_records.jsonl")
    args = parser.parse_args()

    samples = list(read_jsonl(Path(args.samples)))
    memory = load_memory(Path(args.memory))
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        qid = str(row.get("qid") or "")
        if qid:
            grouped[qid].append(row)
    qids = sorted(grouped)
    if args.max_qids > 0:
        qids = qids[: args.max_qids]

    dense = DenseScorer(args.dense_model, device=args.device, batch_size=args.dense_batch_size)
    policy = PolicyModel(
        model_dir=Path(args.model_dir),
        checkpoint=Path(args.checkpoint),
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    totals = Counter()
    rank_buckets: dict[str, Counter] = defaultdict(Counter)
    records = []

    for qid in tqdm(qids, desc="rank-analysis"):
        rows = sorted(grouped[qid], key=lambda item: int(item.get("t", 0)))
        selected_evidence = []

        for row in rows:
            question = str(row.get("question") or "")
            candidate_ids = sample_candidate_ids(row)
            positive_id = sample_positive_id(row)
            if not question or not candidate_ids or positive_id not in candidate_ids:
                totals["skipped"] += 1
                continue

            usable_candidate_ids = []
            candidate_texts = []
            for unit_id in candidate_ids:
                item = memory.get(unit_id)
                if not item:
                    continue
                usable_candidate_ids.append(unit_id)
                candidate_texts.append(format_candidate_text(item))
            if positive_id not in usable_candidate_ids:
                totals["skipped"] += 1
                continue

            if args.state_mode == "policy":
                notebook = "\n".join(
                    format_notebook_evidence(item, index + 1)
                    for index, item in enumerate(selected_evidence)
                )
                context = f"Question: {question}\nNotebook:\n{notebook}"
            else:
                context = f"Question: {question}\nNotebook:\n{sample_k_t(row)}"

            positive_index = usable_candidate_ids.index(positive_id)
            front_query = context
            dense_query = front_query if args.dense_query_mode == "state" else question
            dense_scores = dense.score(dense_query, candidate_texts)
            bm25_score_values = bm25_scores(front_query, candidate_texts)

            dense_order = np.argsort(dense_scores)[::-1].tolist()
            bm25_order = np.argsort(bm25_score_values)[::-1].tolist()
            rrf_scores = reciprocal_rank_fusion([bm25_order, dense_order], len(usable_candidate_ids))
            rrf_order = np.argsort(rrf_scores)[::-1].tolist()

            front_pool_k = min(max(1, args.front_pool_k), len(rrf_order))
            seed_indices = list(dict.fromkeys(bm25_order[:front_pool_k] + dense_order[:front_pool_k]))
            expanded_indices = local_expanded_pool(
                seed_indices,
                usable_candidate_ids,
                memory,
                window=max(0, args.local_expansion_window),
            )
            expanded_indices.sort(key=lambda index: float(rrf_scores[index]), reverse=True)
            candidate_top_k = min(max(1, args.candidate_top_k), len(rrf_order))
            mmr_indices = mmr_select(
                expanded_indices,
                rrf_scores,
                candidate_texts,
                usable_candidate_ids,
                memory,
                limit=candidate_top_k,
                lambda_=args.mmr_lambda,
                same_doc_similarity=args.mmr_same_doc_similarity,
            )

            policy_order = []
            policy_rank = None
            if mmr_indices:
                policy_scores = policy.score(context, [candidate_texts[index] for index in mmr_indices])
                policy_local_order = np.argsort(policy_scores)[::-1].tolist()
                policy_order = [mmr_indices[index] for index in policy_local_order]
                policy_rank = rank_of(policy_order, positive_index)

            bm25_rank = rank_of(bm25_order, positive_index)
            dense_rank = rank_of(dense_order, positive_index)
            rrf_rank = rank_of(rrf_order, positive_index)
            seed_rank = rank_of(seed_indices, positive_index)
            expanded_rank = rank_of(expanded_indices, positive_index)
            mmr_rank = rank_of(mmr_indices, positive_index)

            selected_indices = policy_order[: max(1, args.select_top_k)]
            selected_contains_gold = positive_index in selected_indices

            stage_ranks = {
                "bm25": bm25_rank,
                "dense": dense_rank,
                "rrf": rrf_rank,
                "seed_pool": seed_rank,
                "expanded_pool": expanded_rank,
                "mmr_compressed": mmr_rank,
                "policy": policy_rank,
            }
            totals["steps"] += 1
            totals["selected_contains_gold"] += int(selected_contains_gold)
            for stage, rank in stage_ranks.items():
                rank_buckets[stage][bucket_rank(rank)] += 1
                for k in ks:
                    totals[f"{stage}@{k}"] += hit_at(rank, k)

            predicted_id = usable_candidate_ids[policy_order[0]] if policy_order else ""
            for index in selected_indices:
                selected_evidence.append(memory[usable_candidate_ids[index]])

            records.append(
                {
                    "qid": qid,
                    "t": int(row.get("t", 0)),
                    "question": question,
                    "positive_unit_id": positive_id,
                    "positive_doc_id": memory[positive_id]["doc_id"],
                    "predicted_unit_id": predicted_id,
                    "predicted_doc_id": memory[predicted_id]["doc_id"] if predicted_id in memory else "",
                    "selected_contains_gold": selected_contains_gold,
                    "ranks": stage_ranks,
                    "mmr_candidate_unit_ids": [usable_candidate_ids[index] for index in mmr_indices],
                    "policy_top5_unit_ids": [usable_candidate_ids[index] for index in policy_order[:5]],
                }
            )

    step_total = max(1, totals["steps"])
    summary = {
        "samples": args.samples,
        "memory": args.memory,
        "checkpoint": args.checkpoint,
        "dense_model": args.dense_model,
        "qids": len(qids),
        "steps": totals["steps"],
        "skipped": totals["skipped"],
        "front_pool_k": args.front_pool_k,
        "candidate_top_k": args.candidate_top_k,
        "local_expansion_window": args.local_expansion_window,
        "mmr_lambda": args.mmr_lambda,
        "mmr_same_doc_similarity": args.mmr_same_doc_similarity,
        "selected_contains_gold": round(totals["selected_contains_gold"] / step_total, 6),
        "stage_hit_rates": {},
        "rank_buckets": {stage: dict(counter) for stage, counter in rank_buckets.items()},
    }
    for stage in ["bm25", "dense", "rrf", "seed_pool", "expanded_pool", "mmr_compressed", "policy"]:
        summary["stage_hit_rates"][stage] = {
            f"hit@{k}": round(totals[f"{stage}@{k}"] / step_total, 6)
            for k in ks
        }

    write_json(summary, Path(args.output))
    write_jsonl(records, Path(args.records_output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"records: {args.records_output}")


if __name__ == "__main__":
    main()
