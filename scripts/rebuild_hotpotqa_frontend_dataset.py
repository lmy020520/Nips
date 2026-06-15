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


def get_k_t(row: dict) -> str:
    if row.get("K_t") is not None:
        return str(row["K_t"])
    return str((row.get("state") or {}).get("K_t") or "")


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
    forced_positive = False
    if force_positive and not natural_positive_in_candidates:
        forced_positive = True
        if len(compressed) < candidate_top_k:
            compressed.append(positive_index)
        elif compressed:
            compressed[-1] = positive_index
        else:
            compressed = [positive_index]

    # Preserve front-end/MMR order while ensuring no duplicate after replacement.
    deduped = []
    seen = set()
    for index in compressed:
        if index in seen:
            continue
        seen.add(index)
        deduped.append(index)
    compressed = deduped
    if force_positive and positive_index not in compressed:
        if len(compressed) >= candidate_top_k:
            compressed[-1] = positive_index
        else:
            compressed.append(positive_index)

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

    for row in tqdm(source_samples, desc=f"rebuild-{split}"):
        qid = str(row.get("qid") or "")
        if not qid or qid not in memory_by_qid:
            stats["skipped_missing_qid_memory"] += 1
            continue

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
        )
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
        }
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
        rebuilt_samples.append(rebuilt)

        stats["samples"] += 1
        stats["natural_positive"] += int(meta["natural_positive_in_candidates"])
        stats["forced_positive"] += int(meta["forced_positive"])
        stats[f"candidate_count_{len(candidate_ids)}"] += 1
        rank_bucket = int(math.ceil(min(meta["rrf_rank"], 100) / 10.0) * 10)
        rank_buckets[f"rrf_rank<={rank_bucket}"] += 1

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
        "rank_buckets": dict(rank_buckets),
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
        },
        "teacher": {
            "trajectory_backbone": "existing teacher positive sequence",
            "regenerated": [
                "R_t",
                "C_t",
                "ranking labels",
                "hard negatives from fixed hybrid front-end",
            ],
            "note": "Positive evidence is forced into C_t when not naturally selected, matching C_t = R_t union teacher gold action.",
        },
        "splits": split_summaries,
    }
    write_json(manifest, args.output_root / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
