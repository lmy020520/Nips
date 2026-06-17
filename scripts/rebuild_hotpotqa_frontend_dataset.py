#!/usr/bin/env python3
"""Rebuild HotpotQA policy training samples with the fixed hybrid front-end.

The generated samples follow the front-end plan:

    q_t = question + SlotSummary(K_t)
    BM25 top-k + Dense top-k
        -> merge/dedup
        -> local expansion
        -> RRF ranking
        -> MMR/diversity compression to cand10/cand15
        -> policy model learns over the compressed candidate pool

This script keeps the existing teacher positive sequence as the trajectory
backbone, but regenerates every step's candidate pool and ranking labels under
the new front-end distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def tokenize_for_retrieval(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def format_candidate_text(item: dict) -> str:
    title = str(item.get("title") or item.get("doc_id") or "")
    sent_id = item.get("sent_id")
    if sent_id is None:
        try:
            sent_id = int(str(item["unit_id"]).rsplit("::", 1)[-1])
        except Exception:
            sent_id = 0
    return f"{title} [{sent_id}] {str(item.get('text') or '').strip()}"


def memory_doc_id(item: dict) -> str:
    return str(item.get("doc_id") or item.get("title") or "")


def memory_sent_id(item: dict) -> int:
    sent_id = item.get("sent_id")
    if sent_id is None:
        try:
            return int(str(item["unit_id"]).rsplit("::", 1)[-1])
        except Exception:
            return 0
    return int(sent_id)


def positive_id(row: dict) -> str:
    labels = row.get("labels") or {}
    ranking = labels.get("ranking_label") or {}
    if ranking.get("positive_unit_id"):
        return str(ranking["positive_unit_id"])
    if (labels.get("u_t_plus") or {}).get("unit_id"):
        return str(labels["u_t_plus"]["unit_id"])
    if row.get("positive_unit_id"):
        return str(row["positive_unit_id"])
    return ""


def candidate_ids_from_row(row: dict) -> list[str]:
    candidates = row.get("candidates") or {}
    value = candidates.get("C_t") or candidates.get("R_t") or []
    return [str(unit_id) for unit_id in value]


def get_k_t(row: dict) -> str:
    if row.get("K_t") is not None:
        return str(row["K_t"])
    return str((row.get("state") or {}).get("K_t") or "")


def set_k_t(row: dict, k_t: str) -> dict:
    row = dict(row)
    state = dict(row.get("state") or {})
    state["K_t"] = k_t
    row["state"] = state
    if "K_t" in row:
        row["K_t"] = k_t
    return row


def notebook_from_unit_ids(unit_ids: list[str], memory_by_unit: dict[str, dict]) -> str:
    lines = []
    for idx, unit_id in enumerate(unit_ids, start=1):
        item = memory_by_unit.get(unit_id)
        if not item:
            continue
        lines.append(f"[{idx}] {memory_doc_id(item)}: {str(item.get('text') or '').strip()}")
    return "\n".join(lines)


def reciprocal_rank_fusion(orders: list[list[int]], size: int, k: int = 60) -> np.ndarray:
    scores = np.zeros(size, dtype=float)
    for order in orders:
        for rank, index in enumerate(order, start=1):
            scores[index] += 1.0 / (k + rank)
    return scores


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score - min_score < 1e-12:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def bm25_scores(query: str, candidate_texts: list[str]) -> np.ndarray:
    from rank_bm25 import BM25Okapi

    tokenized_candidates = [tokenize_for_retrieval(text) for text in candidate_texts]
    return np.array(BM25Okapi(tokenized_candidates).get_scores(tokenize_for_retrieval(query)))


class DenseScorer:
    def __init__(self, model_name_or_path: str, device: str, batch_size: int):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.batch_size = max(1, batch_size)

    def score(self, query: str, candidate_texts: list[str]) -> np.ndarray:
        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        candidate_embeddings = self.model.encode(
            candidate_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.dot(candidate_embeddings, query_embedding)


def group_memory_by_qid(memory_rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for item in memory_rows:
        unit_id = str(item.get("unit_id") or "")
        if not unit_id:
            continue
        qid = str(item.get("qid") or unit_id.split("::", 1)[0])
        item = dict(item)
        item["qid"] = qid
        item["unit_id"] = unit_id
        item["doc_id"] = memory_doc_id(item)
        item["sent_id"] = memory_sent_id(item)
        grouped[qid].append(item)
    for qid in grouped:
        grouped[qid].sort(key=lambda row: (memory_doc_id(row), memory_sent_id(row), str(row["unit_id"])))
    return grouped


def local_expand(seed_indices: list[int], pool_items: list[dict], *, window: int) -> list[int]:
    by_doc = defaultdict(list)
    for idx, item in enumerate(pool_items):
        by_doc[(str(item["qid"]), memory_doc_id(item))].append((memory_sent_id(item), idx))
    for key in by_doc:
        by_doc[key].sort()

    expanded = []
    seen = set()

    def add(index: int) -> None:
        if index not in seen:
            seen.add(index)
            expanded.append(index)

    for index in seed_indices:
        add(index)
        if window <= 0:
            continue
        item = pool_items[index]
        doc_key = (str(item["qid"]), memory_doc_id(item))
        sent_id = memory_sent_id(item)
        for neighbor_sent_id, neighbor_index in by_doc.get(doc_key, []):
            if abs(neighbor_sent_id - sent_id) <= window:
                add(neighbor_index)
    return expanded


def text_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize_for_retrieval(left))
    right_tokens = set(tokenize_for_retrieval(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def candidate_similarity(
    left_idx: int,
    right_idx: int,
    candidate_texts: list[str],
    pool_items: list[dict],
    same_doc_similarity: float,
) -> float:
    similarity = text_jaccard(candidate_texts[left_idx], candidate_texts[right_idx])
    if memory_doc_id(pool_items[left_idx]) == memory_doc_id(pool_items[right_idx]):
        similarity = max(similarity, same_doc_similarity)
    return similarity


def mmr_select(
    pool_indices: list[int],
    relevance_scores: np.ndarray,
    candidate_texts: list[str],
    pool_items: list[dict],
    *,
    limit: int,
    lambda_: float,
    same_doc_similarity: float,
) -> list[int]:
    if limit <= 0 or len(pool_indices) <= limit:
        return pool_indices[:limit] if limit > 0 else pool_indices

    relevance = minmax_normalize(relevance_scores)
    lambda_ = min(1.0, max(0.0, lambda_))
    selected = []
    remaining = list(dict.fromkeys(pool_indices))
    while remaining and len(selected) < limit:
        best_index = None
        best_score = None
        for index in remaining:
            redundancy = 0.0
            if selected:
                redundancy = max(
                    candidate_similarity(
                        index,
                        selected_idx,
                        candidate_texts,
                        pool_items,
                        same_doc_similarity,
                    )
                    for selected_idx in selected
                )
            score = lambda_ * float(relevance[index]) - (1.0 - lambda_) * redundancy
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        selected.append(int(best_index))
        remaining.remove(int(best_index))
    return selected


def interleave_orders(orders: list[list[int]]) -> list[int]:
    """Interleave ranked lists while preserving each list's internal order."""
    merged = []
    seen = set()
    max_len = max((len(order) for order in orders), default=0)
    for rank in range(max_len):
        for order in orders:
            if rank >= len(order):
                continue
            index = int(order[rank])
            if index in seen:
                continue
            seen.add(index)
            merged.append(index)
    return merged


def select_frontend_hard_negatives(
    *,
    positive_index: int,
    source: str,
    count: int,
    expanded_indices: list[int],
    front_order: list[int],
    dense_order: list[int],
    lexical_order: list[int],
) -> list[int]:
    if count <= 0 or source == "none":
        return []

    expanded_set = set(expanded_indices)
    if source == "rrf":
        order = front_order
    elif source == "dense":
        order = dense_order
    elif source == "bm25":
        order = lexical_order
    elif source == "mixed":
        order = interleave_orders([front_order, dense_order, lexical_order])
    else:
        raise ValueError(f"unknown hard-negative source: {source}")

    selected = []
    seen = set()
    for index in order:
        index = int(index)
        if index == positive_index or index not in expanded_set or index in seen:
            continue
        selected.append(index)
        seen.add(index)
        if len(selected) >= count:
            break
    return selected


def compose_final_candidates(
    *,
    hard_negative_indices: list[int],
    mmr_indices: list[int],
    front_order: list[int],
    positive_index: int,
    target_count: int,
    require_positive: bool,
) -> list[int]:
    final = []
    seen = set()

    def add(index: int) -> None:
        index = int(index)
        if index in seen:
            return
        seen.add(index)
        final.append(index)

    for index in hard_negative_indices:
        add(index)
    for index in mmr_indices:
        add(index)

    if require_positive and positive_index not in seen:
        if len(final) >= target_count and final:
            removed = final.pop()
            seen.remove(removed)
        add(positive_index)

    for index in front_order:
        if len(final) >= target_count:
            break
        add(index)

    if require_positive and positive_index not in seen:
        if len(final) >= target_count and final:
            removed = final.pop()
            seen.remove(removed)
        add(positive_index)

    return final[:target_count]


def rebuild_candidates(
    row: dict,
    pool_items: list[dict],
    dense: DenseScorer,
    *,
    front_pool_k: int,
    candidate_top_k: int,
    local_expansion_window: int,
    mmr_lambda: float,
    mmr_same_doc_similarity: float,
    force_positive: bool,
    hard_negative_source: str,
    hard_negative_count: int,
) -> tuple[list[str], dict]:
    question = str(row.get("question") or "")
    front_query = f"Question: {question}\nNotebook:\n{get_k_t(row)}"
    positive = positive_id(row)

    pool_items = [item for item in pool_items if item.get("unit_id")]
    candidate_texts = [format_candidate_text(item) for item in pool_items]
    unit_ids = [str(item["unit_id"]) for item in pool_items]
    unit_index = {unit_id: idx for idx, unit_id in enumerate(unit_ids)}
    if positive not in unit_index:
        raise ValueError(f"positive not found in memory: qid={row.get('qid')} positive={positive}")

    dense_scores = dense.score(front_query, candidate_texts)
    lexical_scores = bm25_scores(front_query, candidate_texts)
    dense_order = np.argsort(dense_scores)[::-1].tolist()
    lexical_order = np.argsort(lexical_scores)[::-1].tolist()
    front_scores = reciprocal_rank_fusion([lexical_order, dense_order], len(unit_ids))
    front_order = np.argsort(front_scores)[::-1].tolist()

    top_k = min(max(1, front_pool_k), len(unit_ids))
    seed_indices = list(dict.fromkeys(lexical_order[:top_k] + dense_order[:top_k]))
    expanded_indices = local_expand(seed_indices, pool_items, window=max(0, local_expansion_window))
    expanded_indices.sort(key=lambda index: float(front_scores[index]), reverse=True)
    compressed = mmr_select(
        expanded_indices,
        front_scores,
        candidate_texts,
        pool_items,
        limit=min(max(1, candidate_top_k), len(unit_ids)),
        lambda_=mmr_lambda,
        same_doc_similarity=mmr_same_doc_similarity,
    )

    positive_index = unit_index[positive]
    natural_positive_in_candidates = positive_index in compressed
    hard_negative_indices = select_frontend_hard_negatives(
        positive_index=positive_index,
        source=hard_negative_source,
        count=hard_negative_count,
        expanded_indices=expanded_indices,
        front_order=front_order,
        dense_order=dense_order,
        lexical_order=lexical_order,
    )
    forced_positive = False
    if force_positive and not natural_positive_in_candidates:
        forced_positive = True
        if len(compressed) < candidate_top_k:
            compressed.append(positive_index)
        elif compressed:
            compressed[-1] = positive_index
        else:
            compressed = [positive_index]

    target_count = min(max(1, candidate_top_k), len(unit_ids))
    compressed = compose_final_candidates(
        hard_negative_indices=hard_negative_indices,
        mmr_indices=compressed,
        front_order=front_order,
        positive_index=positive_index,
        target_count=target_count,
        require_positive=force_positive,
    )

    candidate_ids = [unit_ids[index] for index in compressed]
    dense_rank = dense_order.index(positive_index) + 1
    bm25_rank = lexical_order.index(positive_index) + 1
    rrf_rank = front_order.index(positive_index) + 1
    meta = {
        "front_end": "BM25_topK+Dense_topK+RRF+LocalExpansion+MMR",
        "front_query": "question_plus_slot_summary",
        "front_pool_k": front_pool_k,
        "candidate_top_k": candidate_top_k,
        "local_expansion_window": local_expansion_window,
        "mmr_lambda": mmr_lambda,
        "mmr_same_doc_similarity": mmr_same_doc_similarity,
        "bm25_rank": bm25_rank,
        "dense_rank": dense_rank,
        "rrf_rank": rrf_rank,
        "natural_positive_in_candidates": natural_positive_in_candidates,
        "forced_positive": forced_positive,
        "hard_negative_source": hard_negative_source,
        "hard_negative_count": hard_negative_count,
        "hard_negative_unit_ids": [unit_ids[index] for index in hard_negative_indices],
        "candidate_count": len(candidate_ids),
    }
    return candidate_ids, meta


def provenance_for(unit_id: str, memory_by_unit: dict[str, dict]) -> dict:
    item = memory_by_unit[unit_id]
    parent_chunk_id = str(item.get("parent_chunk_id") or unit_id.rsplit("::", 1)[0])
    return {
        "chunk_id": parent_chunk_id,
        "doc_id": memory_doc_id(item),
        "parent_chunk_id": parent_chunk_id,
    }


def rebuild_split(
    split: str,
    source_root: Path,
    output_root: Path,
    dense: DenseScorer,
    args: argparse.Namespace,
) -> dict:
    samples_path = source_root / "samples" / f"{split}.jsonl"
    memory_path = source_root / "unit_registry" / f"raw_units_{split}.jsonl"
    if not samples_path.exists() or not memory_path.exists():
        raise FileNotFoundError(f"missing source split files for {split}: {samples_path}, {memory_path}")

    source_samples = list(read_jsonl(samples_path))
    memory_rows = list(read_jsonl(memory_path))
    memory_by_qid = group_memory_by_qid(memory_rows)
    memory_by_unit = {str(row["unit_id"]): row for row in memory_rows if row.get("unit_id")}

    rebuilt_samples = []
    stats = Counter()
    rank_buckets = Counter()

    rows_by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in source_samples:
        rows_by_qid[str(row.get("qid") or "")].append(row)
    for qid in rows_by_qid:
        rows_by_qid[qid].sort(key=lambda item: int(item.get("t", 0)))
    ordered_samples = [
        row
        for qid in sorted(rows_by_qid)
        for row in rows_by_qid[qid]
    ]

    previous_rebuilt_by_key: dict[tuple[str, int], dict] = {}

    def build_one(row: dict, *, variant: str, variant_meta: dict | None = None) -> dict | None:
        qid = str(row.get("qid") or "")
        if not qid or qid not in memory_by_qid:
            stats["skipped_missing_qid_memory"] += 1
            return None

        candidate_ids, meta = rebuild_candidates(
            row,
            memory_by_qid[qid],
            dense,
            front_pool_k=args.front_pool_k,
            candidate_top_k=args.candidate_top_k,
            local_expansion_window=args.local_expansion_window,
            mmr_lambda=args.mmr_lambda,
            mmr_same_doc_similarity=args.mmr_same_doc_similarity,
            force_positive=True,
            hard_negative_source=args.hard_negative_source,
            hard_negative_count=args.hard_negative_count,
        )
        if args.natural_only and not meta["natural_positive_in_candidates"]:
            stats["skipped_not_natural_positive"] += 1
            return None
        positive = positive_id(row)
        if positive not in candidate_ids:
            raise AssertionError(f"positive missing after force: qid={qid} positive={positive}")

        provenance = {unit_id: provenance_for(unit_id, memory_by_unit) for unit_id in candidate_ids}
        rebuilt = dict(row)
        rebuilt["build_meta"] = {
            **(rebuilt.get("build_meta") or {}),
            "rebuilt_by": "rebuild_hotpotqa_frontend_dataset.py",
            "rebuilt_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "front_end_plan": "Hybrid + Local Expansion + MMR",
            "sample_variant": variant,
        }
        if variant_meta:
            rebuilt["build_meta"]["variant_meta"] = variant_meta
        rebuilt["candidates"] = {
            **(rebuilt.get("candidates") or {}),
            "R_t": candidate_ids,
            "C_t": candidate_ids,
            "candidate_provenance": provenance,
        }
        rebuilt.setdefault("labels", {})
        rebuilt["labels"]["ranking_label"] = {
            "positive_unit_id": positive,
            "negative_unit_ids": [unit_id for unit_id in candidate_ids if unit_id != positive],
            "positive_provenance": provenance[positive],
            "negative_provenance": {
                unit_id: provenance[unit_id] for unit_id in candidate_ids if unit_id != positive
            },
        }
        rebuilt["frontend_meta"] = meta

        stats["samples"] += 1
        stats["natural_positive"] += int(meta["natural_positive_in_candidates"])
        stats["forced_positive"] += int(meta["forced_positive"])
        stats[f"hard_negative_source_{meta['hard_negative_source']}"] += 1
        stats[f"hard_negative_count_{len(meta['hard_negative_unit_ids'])}"] += 1
        stats[f"candidate_count_{len(candidate_ids)}"] += 1
        rank_bucket = int(math.ceil(min(meta["rrf_rank"], 100) / 10.0) * 10)
        rank_buckets[f"rrf_rank<={rank_bucket}"] += 1
        stats[f"variant_{variant}"] += 1
        return rebuilt

    for row in tqdm(ordered_samples, desc=f"rebuild-{split}"):
        rebuilt = build_one(row, variant="teacher_state")
        if rebuilt is None:
            continue
        rebuilt_samples.append(rebuilt)
        previous_rebuilt_by_key[(str(row.get("qid") or ""), int(row.get("t", 0)))] = rebuilt

        if args.corrupt_state_variants <= 0 or int(row.get("t", 0)) <= 0:
            continue

        qid = str(row.get("qid") or "")
        t = int(row.get("t", 0))
        previous = previous_rebuilt_by_key.get((qid, t - 1))
        if not previous:
            continue
        previous_positive = positive_id(previous)
        previous_candidates = [
            unit_id for unit_id in candidate_ids_from_row(previous)
            if unit_id != previous_positive and unit_id in memory_by_unit
        ]
        if not previous_candidates:
            continue

        for variant_idx, wrong_unit_id in enumerate(previous_candidates[: args.corrupt_state_variants], start=1):
            corrupted = set_k_t(
                row,
                notebook_from_unit_ids([wrong_unit_id], memory_by_unit),
            )
            rebuilt_corrupt = build_one(
                corrupted,
                variant="corrupted_state",
                variant_meta={
                    "variant_idx": variant_idx,
                    "wrong_previous_unit_id": wrong_unit_id,
                    "wrong_previous_doc_id": memory_doc_id(memory_by_unit[wrong_unit_id]),
                },
            )
            if rebuilt_corrupt is not None:
                rebuilt_samples.append(rebuilt_corrupt)

    output_samples_path = output_root / "samples" / f"{split}.jsonl"
    output_memory_path = output_root / "unit_registry" / f"raw_units_{split}.jsonl"
    write_jsonl(rebuilt_samples, output_samples_path)
    write_jsonl(memory_rows, output_memory_path)

    for folder, name in [
        ("queries", f"{split}.jsonl"),
        ("targets", f"{split}.jsonl"),
        ("raw", f"{split}.json"),
        ("processed", f"{split}.jsonl"),
    ]:
        source_path = source_root / folder / name
        if source_path.exists():
            target_path = output_root / folder / name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

    return {
        "split": split,
        "source_samples": len(source_samples),
        "rebuilt_samples": len(rebuilt_samples),
        "raw_units": len(memory_rows),
        "natural_positive_rate": round(stats["natural_positive"] / max(1, stats["samples"]), 6),
        "forced_positive_rate": round(stats["forced_positive"] / max(1, stats["samples"]), 6),
        "candidate_count": {
            key.replace("candidate_count_", ""): value
            for key, value in stats.items()
            if key.startswith("candidate_count_")
        },
        "hard_negative_count": {
            key.replace("hard_negative_count_", ""): value
            for key, value in stats.items()
            if key.startswith("hard_negative_count_")
        },
        "hard_negative_source": {
            key.replace("hard_negative_source_", ""): value
            for key, value in stats.items()
            if key.startswith("hard_negative_source_")
        },
        "rank_buckets": dict(rank_buckets),
        "variants": {
            key.replace("variant_", ""): value
            for key, value in stats.items()
            if key.startswith("variant_")
        },
        "skipped": {
            "missing_qid_memory": stats["skipped_missing_qid_memory"],
            "not_natural_positive": stats["skipped_not_natural_positive"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/hotpotqa_distractor_v7_10k_llm_prestep"))
    parser.add_argument("--output-root", type=Path, default=Path("data/hotpotqa_distractor_v10_hybrid_frontend"))
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--dense-model", default="models/bge-large-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--front-pool-k", type=int, default=30)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--local-expansion-window", type=int, default=1)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--mmr-same-doc-similarity", type=float, default=0.35)
    parser.add_argument(
        "--hard-negative-source",
        choices=["none", "rrf", "dense", "bm25", "mixed"],
        default="none",
        help=(
            "Preserve high-ranked front-end near-miss negatives in the final candidate pool. "
            "Use 'mixed' to interleave RRF, dense, and BM25 ranks."
        ),
    )
    parser.add_argument(
        "--hard-negative-count",
        type=int,
        default=0,
        help="Number of front-end hard negatives to force-preserve before MMR fill.",
    )
    parser.add_argument(
        "--natural-only",
        action="store_true",
        help="Keep only samples where the fixed front-end naturally includes the positive evidence.",
    )
    parser.add_argument(
        "--corrupt-state-variants",
        type=int,
        default=0,
        help="Add N rollout-style variants with near-miss previous evidence as K_t for t>0 samples.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to rebuild: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    dense = DenseScorer(args.dense_model, device=args.device, batch_size=args.dense_batch_size)
    split_summaries = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        split_summaries.append(rebuild_split(split, args.source_root, args.output_root, dense, args))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "purpose": "rebuild_training_data_after_frontend_change",
        "front_end": {
            "query": "q_t = question + SlotSummary(S_t/K_t)",
            "bm25_top_k": args.front_pool_k,
            "dense_top_k": args.front_pool_k,
            "merge": "dedup",
            "fusion": "RRF",
            "local_expansion_window": args.local_expansion_window,
            "compression": "MMR/diversity filter",
            "candidate_top_k": args.candidate_top_k,
            "dense_model": args.dense_model,
            "mmr_lambda": args.mmr_lambda,
            "mmr_same_doc_similarity": args.mmr_same_doc_similarity,
            "hard_negative_source": args.hard_negative_source,
            "hard_negative_count": args.hard_negative_count,
            "natural_only": args.natural_only,
            "corrupt_state_variants": args.corrupt_state_variants,
        },
        "teacher": {
            "trajectory_backbone": "existing teacher positive sequence",
            "regenerated": [
                "R_t",
                "C_t",
                "ranking labels",
                "hard negatives from fixed hybrid front-end",
                "optional natural-only filtering",
                "optional corrupted-state rollout-style variants",
            ],
            "note": (
                "Positive evidence is forced into C_t when not naturally selected unless --natural-only is set, "
                "matching C_t = R_t union teacher gold action for teacher-forced variants."
            ),
        },
        "splits": split_summaries,
    }
    write_json(manifest, args.output_root / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
