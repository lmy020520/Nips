#!/usr/bin/env python3
"""Trace where full-candidate RAG loses gold evidence.

This diagnostic is intentionally stage-oriented. It does not train a new model;
it records how a gold evidence unit moves through:

raw candidates -> BM25/Dense/RRF -> front pool -> local expansion ->
MMR compression -> student policy ranking.

The goal is to distinguish front-end recall failure from policy ranking failure.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queries(path: str) -> dict[str, dict]:
    if not path:
        return {}
    query_path = Path(path)
    if not query_path.exists():
        return {}
    return {str(row.get("qid")): row for row in read_jsonl(query_path) if row.get("qid")}


def rank_of(order: list[int], target_index: int) -> int | None:
    try:
        return order.index(target_index) + 1
    except ValueError:
        return None


def hit_at(rank: int | None, k: int) -> int:
    return int(rank is not None and rank <= k)


def pool_hit(order: list[int], target_index: int, k: int) -> bool:
    return target_index in order[: min(k, len(order))]


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    for bound in (1, 2, 3, 5, 10, 20, 30, 50):
        if rank <= bound:
            return "1" if bound == 1 else f"<= {bound}"
    return "> 50"


def doc_token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.lower().replace("_", " ").split() if token}
    right_tokens = {token for token in right.lower().replace("_", " ").split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def answer_visible(query_row: dict | None, text: str) -> bool:
    if not query_row:
        return False
    answer = str(query_row.get("answer") or "").strip().lower()
    if not answer or answer in {"yes", "no"}:
        return False
    return answer in str(text).lower()


def classify_top1_error(
    *,
    positive_id: str,
    predicted_id: str,
    positive_in_compressed: bool,
    memory: dict[str, dict],
    query_row: dict | None,
) -> str:
    if not predicted_id:
        return "no_prediction"
    if predicted_id == positive_id:
        return "correct"
    if not positive_in_compressed:
        return "gold_missing_after_compression"

    positive_item = memory.get(positive_id) or {}
    predicted_item = memory.get(predicted_id) or {}
    positive_doc = str(positive_item.get("doc_id") or positive_item.get("title") or "")
    predicted_doc = str(predicted_item.get("doc_id") or predicted_item.get("title") or "")
    predicted_text = str(predicted_item.get("text") or "")

    if positive_doc and positive_doc == predicted_doc:
        return "same_doc_wrong_sentence"
    if doc_token_overlap(positive_doc, predicted_doc) >= 0.35:
        return "same_entity_or_title_overlap"
    if answer_visible(query_row, predicted_text):
        return "answer_string_visible_but_not_gold"
    return "other_distractor"


def summarize_top_items(order: list[int], candidate_ids: list[str], memory: dict[str, dict], limit: int) -> list[dict]:
    items = []
    for rank, index in enumerate(order[:limit], start=1):
        unit_id = candidate_ids[index]
        item = memory.get(unit_id) or {}
        items.append(
            {
                "rank": rank,
                "unit_id": unit_id,
                "doc_id": str(item.get("doc_id") or item.get("title") or ""),
                "sent_id": item.get("sent_id"),
                "text": str(item.get("text") or "")[:240],
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", default="")
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dense-model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--state-mode", choices=["dataset", "policy"], default="policy")
    parser.add_argument(
        "--policy-context-source",
        choices=["legacy", "online_state"],
        default="online_state",
    )
    parser.add_argument("--online-state-max-raw", type=int, default=8)
    parser.add_argument("--online-state-max-chars", type=int, default=260)
    parser.add_argument("--dense-query-mode", choices=["question", "state"], default="state")
    parser.add_argument("--front-pool-k", type=int, default=30)
    parser.add_argument("--front-fusion", choices=["rrf", "score"], default="rrf")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--select-top-k", type=int, default=5)
    parser.add_argument(
        "--policy-score-mode",
        choices=["rank", "front_policy_blend"],
        default="rank",
        help="Use plain student policy ranking or front-end/student score blending.",
    )
    parser.add_argument(
        "--policy-blend-weight",
        type=float,
        default=0.35,
        help=(
            "For front_policy_blend, weight assigned to the student policy score. "
            "Final score = (1-w)*front_score + w*policy_score."
        ),
    )
    parser.add_argument("--local-expansion-window", type=int, default=1)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--mmr-same-doc-similarity", type=float, default=0.35)
    parser.add_argument("--ks", default="1,2,3,5,10,20,30,50")
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--trace-top-n", type=int, default=10)
    parser.add_argument("--output", default="outputs/analysis/frontend_diagnosis_report.json")
    parser.add_argument("--trace-output", default="outputs/analysis/candidate_trace.jsonl")
    args = parser.parse_args()

    # Delay model-dependent imports so --help and static checks do not require
    # the server training environment.
    from run_hotpotqa_policy_rag import (
        DenseScorer,
        PolicyModel,
        bm25_scores,
        format_candidate_text,
        format_notebook_evidence,
        init_online_state,
        load_memory,
        local_expanded_pool,
        mmr_select,
        reciprocal_rank_fusion,
        sample_candidate_ids,
        sample_k_t,
        sample_positive_id,
        update_online_state,
    )

    samples = list(read_jsonl(Path(args.samples)))
    memory = load_memory(Path(args.memory))
    queries = load_queries(args.queries)
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
    stage_hits: dict[str, Counter] = defaultdict(Counter)
    rank_buckets: dict[str, Counter] = defaultdict(Counter)
    top1_errors = Counter()
    conditional = Counter()
    traces = []

    for qid in tqdm(qids, desc="frontend-trace"):
        rows = sorted(grouped[qid], key=lambda item: int(item.get("t", 0)))
        selected_evidence = []
        selected_unit_ids = set()
        online_state = init_online_state()

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
                if args.policy_context_source == "online_state":
                    notebook = online_state["K_t"]
                else:
                    notebook = "\n".join(
                        format_notebook_evidence(item, index + 1)
                        for index, item in enumerate(selected_evidence)
                    )
                context = f"Question: {question}\nNotebook:\n{notebook}"
            else:
                context = f"Question: {question}\nNotebook:\n{sample_k_t(row)}"

            positive_index = usable_candidate_ids.index(positive_id)
            dense_query = context if args.dense_query_mode == "state" else question

            bm25_score_values = bm25_scores(context, candidate_texts)
            dense_score_values = dense.score(dense_query, candidate_texts)
            bm25_order = np.argsort(bm25_score_values)[::-1].tolist()
            dense_order = np.argsort(dense_score_values)[::-1].tolist()

            if args.front_fusion == "score":
                bm25_norm = (bm25_score_values - np.min(bm25_score_values)) / (
                    np.ptp(bm25_score_values) + 1e-12
                )
                dense_norm = (dense_score_values - np.min(dense_score_values)) / (
                    np.ptp(dense_score_values) + 1e-12
                )
                fused_scores = args.hybrid_alpha * dense_norm + (1.0 - args.hybrid_alpha) * bm25_norm
            else:
                fused_scores = reciprocal_rank_fusion([bm25_order, dense_order], len(usable_candidate_ids))
            fused_order = np.argsort(fused_scores)[::-1].tolist()

            front_pool_k = min(max(1, args.front_pool_k), len(fused_order))
            seed_indices = list(dict.fromkeys(bm25_order[:front_pool_k] + dense_order[:front_pool_k]))
            seed_indices.sort(key=lambda index: float(fused_scores[index]), reverse=True)
            expanded_indices = local_expanded_pool(
                seed_indices,
                usable_candidate_ids,
                memory,
                window=max(0, args.local_expansion_window),
            )
            expanded_indices.sort(key=lambda index: float(fused_scores[index]), reverse=True)
            compressed_indices = mmr_select(
                expanded_indices,
                fused_scores,
                candidate_texts,
                usable_candidate_ids,
                memory,
                limit=min(max(1, args.candidate_top_k), len(usable_candidate_ids)),
                lambda_=args.mmr_lambda,
                same_doc_similarity=args.mmr_same_doc_similarity,
            )

            policy_scores = np.array([])
            rerank_scores = np.array([])
            policy_order = []
            if compressed_indices:
                policy_scores = policy.score(context, [candidate_texts[index] for index in compressed_indices])
                if args.policy_score_mode == "front_policy_blend":
                    front_local_scores = np.array([fused_scores[index] for index in compressed_indices], dtype=float)
                    policy_weight = min(1.0, max(0.0, args.policy_blend_weight))
                    rerank_scores = (
                        (1.0 - policy_weight)
                        * ((front_local_scores - np.min(front_local_scores)) / (np.ptp(front_local_scores) + 1e-12))
                        + policy_weight
                        * ((policy_scores - np.min(policy_scores)) / (np.ptp(policy_scores) + 1e-12))
                    )
                else:
                    rerank_scores = policy_scores
                policy_local_order = np.argsort(rerank_scores)[::-1].tolist()
                policy_order = [compressed_indices[index] for index in policy_local_order]

            stage_orders = {
                "bm25": bm25_order,
                "dense": dense_order,
                "fused": fused_order,
                "seed_pool": seed_indices,
                "expanded_pool": expanded_indices,
                "compressed_pool": compressed_indices,
                "policy": policy_order,
            }
            stage_ranks = {stage: rank_of(order, positive_index) for stage, order in stage_orders.items()}

            totals["steps"] += 1
            in_pool50 = pool_hit(fused_order, positive_index, 50)
            in_compressed = positive_index in compressed_indices
            policy_hit5 = pool_hit(policy_order, positive_index, args.select_top_k)
            conditional["gold_in_pool50"] += int(in_pool50)
            conditional["gold_in_compressed"] += int(in_compressed)
            conditional["policy_hit5"] += int(policy_hit5)
            if in_pool50:
                conditional["pool50_den"] += 1
                conditional["compressed_given_pool50"] += int(in_compressed)
            if in_compressed:
                conditional["compressed_den"] += 1
                conditional["policy_hit5_given_compressed"] += int(policy_hit5)

            for stage, rank in stage_ranks.items():
                rank_buckets[stage][rank_bucket(rank)] += 1
                for k in ks:
                    stage_hits[stage][k] += hit_at(rank, k)

            predicted_id = usable_candidate_ids[policy_order[0]] if policy_order else ""
            error_type = classify_top1_error(
                positive_id=positive_id,
                predicted_id=predicted_id,
                positive_in_compressed=in_compressed,
                memory=memory,
                query_row=queries.get(qid),
            )
            top1_errors[error_type] += 1

            selected_indices = policy_order[: max(1, args.select_top_k)]
            selected_ids = [usable_candidate_ids[index] for index in selected_indices]
            for selected_id in selected_ids:
                item = memory.get(selected_id)
                if not item or selected_id in selected_unit_ids:
                    continue
                selected_unit_ids.add(selected_id)
                selected_evidence.append(item)
                online_state = update_online_state(
                    online_state,
                    selected_id,
                    item,
                    memory,
                    step_id=int(row.get("t", 0)),
                    max_raw=args.online_state_max_raw,
                    max_chars_per_item=args.online_state_max_chars,
                )

            positive_item = memory[positive_id]
            traces.append(
                {
                    "qid": qid,
                    "t": int(row.get("t", 0)),
                    "question": question,
                    "positive_unit_id": positive_id,
                    "positive_doc_id": positive_item["doc_id"],
                    "positive_text": positive_item["text"],
                    "predicted_unit_id": predicted_id,
                    "predicted_doc_id": memory[predicted_id]["doc_id"] if predicted_id in memory else "",
                    "top1_error_type": error_type,
                    "selected_contains_gold": policy_hit5,
                    "ranks": stage_ranks,
                    "pool_flags": {
                        "gold_in_pool50": in_pool50,
                        "gold_in_compressed": in_compressed,
                        "policy_hit_at_select_top_k": policy_hit5,
                    },
                    "top_pools": {
                        "bm25": summarize_top_items(bm25_order, usable_candidate_ids, memory, args.trace_top_n),
                        "dense": summarize_top_items(dense_order, usable_candidate_ids, memory, args.trace_top_n),
                        "fused": summarize_top_items(fused_order, usable_candidate_ids, memory, args.trace_top_n),
                        "expanded_pool": summarize_top_items(
                            expanded_indices,
                            usable_candidate_ids,
                            memory,
                            args.trace_top_n,
                        ),
                        "compressed_pool": summarize_top_items(
                            compressed_indices,
                            usable_candidate_ids,
                            memory,
                            args.trace_top_n,
                        ),
                        "policy": summarize_top_items(policy_order, usable_candidate_ids, memory, args.trace_top_n),
                    },
                }
            )

    steps = max(1, totals["steps"])
    compressed_den = max(1, conditional["compressed_den"])
    pool50_den = max(1, conditional["pool50_den"])
    report = {
        "samples": args.samples,
        "memory": args.memory,
        "queries": args.queries,
        "checkpoint": args.checkpoint,
        "state_mode": args.state_mode,
        "policy_context_source": args.policy_context_source,
        "dense_model": args.dense_model,
        "qids": len(qids),
        "steps": totals["steps"],
        "skipped": totals["skipped"],
        "front_pool_k": args.front_pool_k,
        "front_fusion": args.front_fusion,
        "hybrid_alpha": args.hybrid_alpha,
        "candidate_top_k": args.candidate_top_k,
        "select_top_k": args.select_top_k,
        "policy_score_mode": args.policy_score_mode,
        "policy_blend_weight": args.policy_blend_weight,
        "local_expansion_window": args.local_expansion_window,
        "mmr_lambda": args.mmr_lambda,
        "mmr_same_doc_similarity": args.mmr_same_doc_similarity,
        "core_breakdown": {
            "p_gold_in_pool50": round(conditional["gold_in_pool50"] / steps, 6),
            "p_gold_in_compressed": round(conditional["gold_in_compressed"] / steps, 6),
            "p_gold_in_compressed_given_pool50": round(
                conditional["compressed_given_pool50"] / pool50_den,
                6,
            ),
            "p_policy_hit_at_select_top_k": round(conditional["policy_hit5"] / steps, 6),
            "p_policy_hit_given_compressed": round(
                conditional["policy_hit5_given_compressed"] / compressed_den,
                6,
            ),
        },
        "stage_hit_rates": {
            stage: {f"hit@{k}": round(counter[k] / steps, 6) for k in ks}
            for stage, counter in stage_hits.items()
        },
        "rank_buckets": {stage: dict(counter) for stage, counter in rank_buckets.items()},
        "top1_error_types": dict(top1_errors),
        "trace_output": args.trace_output,
    }

    write_json(report, Path(args.output))
    write_jsonl(traces, Path(args.trace_output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"trace: {args.trace_output}")


if __name__ == "__main__":
    main()
