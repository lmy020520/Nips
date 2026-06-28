#!/usr/bin/env python3
"""Prepare a 2WikiMultihopQA evaluation set for KBS/Policy-RAG.

This script converts 2WikiMultihopQA examples into the lightweight evaluator
format used by scripts/run_hotpotqa_policy_rag.py:

  queries/test.jsonl
  unit_registry/raw_units_test.jsonl
  targets/test.jsonl
  samples/test.jsonl

It is intentionally evaluation-only. No student retraining data is produced.
The gold evidence trajectory is derived from supporting_facts order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_local_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return list(read_jsonl(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data]
    if isinstance(data, dict):
        for key in ("data", "examples", "validation", "dev", "test", "train"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value]
    raise ValueError(f"unsupported local input format: {path}")


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{value}".encode("utf-8")).hexdigest()


def qid_of(row: dict) -> str:
    for key in ("qid", "_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("row has no qid/_id/id")


def normalize_context(context) -> list[dict]:
    """Normalize common 2Wiki/Hotpot-like context variants."""
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("context") or []
        if len(titles) != len(sentences):
            raise ValueError("context title/sentences length mismatch")
        return [
            {"title": str(title), "sentences": [str(sent) for sent in sents if str(sent).strip()]}
            for title, sents in zip(titles, sentences)
        ]

    if isinstance(context, list):
        out = []
        for item in context:
            if isinstance(item, dict):
                title = item.get("title") or item.get("doc_title") or item.get("name")
                sents = item.get("sentences") or item.get("paragraph_text") or item.get("text") or []
                if isinstance(sents, str):
                    sents = split_sentences(sents)
                out.append({"title": str(title), "sentences": [str(sent) for sent in sents if str(sent).strip()]})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = item[0]
                sents = item[1]
                if isinstance(sents, str):
                    sents = split_sentences(sents)
                out.append({"title": str(title), "sentences": [str(sent) for sent in sents if str(sent).strip()]})
            else:
                raise ValueError(f"unsupported context item: {type(item)}")
        return [block for block in out if block["title"] and block["sentences"]]

    raise ValueError(f"unsupported context type: {type(context)}")


def split_sentences(text: str) -> list[str]:
    # Conservative fallback for local files that provide paragraph strings.
    import re

    pieces = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def normalize_supporting_facts(row: dict) -> list[tuple[str, int]]:
    facts = row.get("supporting_facts")
    if facts is None:
        facts = row.get("supporting_facts_by_hop") or row.get("supporting_facts_list")
    if facts is None:
        facts = row.get("evidence") or row.get("evidences")

    pairs: list[tuple[str, int]] = []
    if isinstance(facts, dict):
        titles = facts.get("title") or facts.get("titles") or []
        sent_ids = facts.get("sent_id") or facts.get("sent_ids") or facts.get("sentence_id") or []
        if len(titles) != len(sent_ids):
            raise ValueError("supporting_facts title/sent_id length mismatch")
        pairs = [(str(title), int(sent_id)) for title, sent_id in zip(titles, sent_ids)]
    elif isinstance(facts, list):
        for item in facts:
            if isinstance(item, dict):
                title = item.get("title") or item.get("doc_title") or item.get("name")
                sent_id = item.get("sent_id", item.get("sentence_id", item.get("sent_idx", 0)))
                pairs.append((str(title), int(sent_id)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # 2Wiki official variants sometimes include [title, sent_id].
                if isinstance(item[1], int) or str(item[1]).isdigit():
                    pairs.append((str(item[0]), int(item[1])))
                # Some evidence triples do not identify sentence ids. Skip them.
            else:
                continue
    else:
        raise ValueError("missing or unsupported supporting facts")

    seen = set()
    out = []
    for pair in pairs:
        if not pair[0] or pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def sentence_lookup(context: list[dict]) -> dict[tuple[str, int], str]:
    lookup = {}
    for block in context:
        title = str(block["title"])
        for sent_id, text in enumerate(block["sentences"]):
            text = str(text).strip()
            if text:
                lookup[(title, sent_id)] = text
    return lookup


def build_unit_id(qid: str, title: str, sent_id: int) -> str:
    return f"{qid}::{title}::{sent_id}"


def build_parent_chunk_id(qid: str, title: str) -> str:
    return f"{qid}::{title}"


def unit_record(qid: str, title: str, sent_id: int, text: str) -> dict:
    return {
        "unit_id": build_unit_id(qid, title, sent_id),
        "text": text,
        "doc_id": title,
        "parent_chunk_id": build_parent_chunk_id(qid, title),
        "span_start": None,
        "span_end": None,
        "provenance": "raw",
        "candidate_granularity": "sentence",
    }


def project_row(row: dict) -> dict:
    question = str(row.get("question") or row.get("query") or "").strip()
    answer = str(row.get("answer") or row.get("gold_answer") or "").strip()
    context = normalize_context(row.get("context") or row.get("contexts") or row.get("paragraphs"))
    support = normalize_supporting_facts(row)
    return {
        "qid": qid_of(row),
        "question": question,
        "answer": answer,
        "type": str(row.get("type") or row.get("question_type") or "2wiki").strip(),
        "level": str(row.get("level") or row.get("difficulty") or "unknown").strip(),
        "supporting_facts": [[title, sent_id] for title, sent_id in support],
        "context": context,
    }


def valid_projected_row(row: dict) -> tuple[bool, str]:
    if not row["qid"] or not row["question"] or not row["answer"]:
        return False, "missing_qid_question_or_answer"
    lookup = sentence_lookup(row["context"])
    support = [(str(title), int(sent_id)) for title, sent_id in row["supporting_facts"]]
    if not support:
        return False, "no_supporting_facts"
    missing = [pair for pair in support if pair not in lookup]
    if missing:
        return False, "support_sentence_missing"
    if len(lookup) <= len(support):
        return False, "no_distractor_sentences"
    return True, ""


def select_candidates(
    qid: str,
    all_unit_ids: list[str],
    positive_unit_id: str,
    previous_gold: set[str],
    *,
    max_candidates: int,
    seed: int,
) -> list[str]:
    candidates = [unit_id for unit_id in all_unit_ids if unit_id not in previous_gold]
    if positive_unit_id not in candidates:
        candidates.append(positive_unit_id)
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return sorted(candidates, key=lambda unit_id: stable_key(f"{qid}::{unit_id}", seed))

    negatives = [unit_id for unit_id in candidates if unit_id != positive_unit_id]
    negatives.sort(key=lambda unit_id: stable_key(f"{qid}::{unit_id}", seed))
    kept = [positive_unit_id] + negatives[: max_candidates - 1]
    return sorted(kept, key=lambda unit_id: stable_key(f"{qid}::{unit_id}", seed + 1))


def build_outputs(rows: list[dict], output_root: Path, *, max_candidates: int, seed: int) -> dict:
    queries = []
    raw_units = []
    targets = []
    samples = []
    raw_json = []
    processed = []

    for row in rows:
        qid = row["qid"]
        raw_json.append(row)
        processed.append(row)
        queries.append(
            {
                "qid": qid,
                "question": row["question"],
                "answer": row["answer"],
                "metadata": {
                    "dataset": "2wikimultihopqa",
                    "split": "test",
                    "type": row["type"],
                    "level": row["level"],
                },
            }
        )

        lookup = sentence_lookup(row["context"])
        qid_units = []
        unit_by_id = {}
        for (title, sent_id), text in lookup.items():
            unit = unit_record(qid, title, sent_id, text)
            qid_units.append(unit["unit_id"])
            unit_by_id[unit["unit_id"]] = unit
            raw_units.append(unit)

        support = [(str(title), int(sent_id)) for title, sent_id in row["supporting_facts"]]
        target_units = []
        for title, sent_id in support:
            unit_id = build_unit_id(qid, title, sent_id)
            unit = unit_by_id[unit_id]
            target_units.append(
                {
                    "unit_id": unit_id,
                    "chunk_id": unit["parent_chunk_id"],
                    "text": unit["text"],
                    "doc_id": unit["doc_id"],
                    "parent_chunk_id": unit["parent_chunk_id"],
                    "span_start": None,
                    "span_end": None,
                    "provenance": "raw",
                    "weight": 1.0,
                    "primary_role": "support",
                    "role_label_source": "2wiki_supporting_fact",
                }
            )
        targets.append({"qid": qid, "question": row["question"], "T_q_raw": target_units})

        previous_gold: set[str] = set()
        previous_gold_evidence = []
        for t, (title, sent_id) in enumerate(support):
            positive_unit_id = build_unit_id(qid, title, sent_id)
            candidate_ids = select_candidates(
                qid,
                qid_units,
                positive_unit_id,
                previous_gold,
                max_candidates=max_candidates,
                seed=seed + t,
            )
            provenance = {
                unit_id: {
                    "chunk_id": unit_by_id[unit_id]["parent_chunk_id"],
                    "doc_id": unit_by_id[unit_id]["doc_id"],
                    "parent_chunk_id": unit_by_id[unit_id]["parent_chunk_id"],
                }
                for unit_id in candidate_ids
            }
            notebook = "\n".join(
                f"[{idx + 1}] {item['doc_id']}: {item['text']}"
                for idx, item in enumerate(previous_gold_evidence)
            )
            samples.append(
                {
                    "qid": qid,
                    "t": t,
                    "build_meta": {
                        "run_id": "2wiki_policy_rag_eval",
                        "source": "prepare_2wiki_policy_rag_eval.py",
                        "split": "test",
                    },
                    "question": row["question"],
                    "state": {
                        "H_t": [
                            {
                                "step_id": idx,
                                "unit_id": item["unit_id"],
                                "chunk_id": item["parent_chunk_id"],
                                "doc_id": item["doc_id"],
                                "parent_chunk_id": item["parent_chunk_id"],
                            }
                            for idx, item in enumerate(previous_gold_evidence)
                        ],
                        "K_t": notebook,
                    },
                    "candidates": {
                        "R_t": candidate_ids,
                        "C_t": candidate_ids,
                        "G_t_final": [],
                        "G_t_aux": [],
                        "G_t_illegal": [],
                        "candidate_provenance": provenance,
                    },
                    "labels": {
                        "u_t_plus": {
                            "step_id": t,
                            "unit_id": positive_unit_id,
                            "chunk_id": unit_by_id[positive_unit_id]["parent_chunk_id"],
                            "doc_id": unit_by_id[positive_unit_id]["doc_id"],
                            "parent_chunk_id": unit_by_id[positive_unit_id]["parent_chunk_id"],
                        },
                        "ranking_label": {
                            "positive_unit_id": positive_unit_id,
                            "negative_unit_ids": [unit_id for unit_id in candidate_ids if unit_id != positive_unit_id],
                            "positive_provenance": provenance[positive_unit_id],
                            "negative_provenance": {
                                unit_id: provenance[unit_id]
                                for unit_id in candidate_ids
                                if unit_id != positive_unit_id
                            },
                        },
                    },
                }
            )
            previous_gold.add(positive_unit_id)
            previous_gold_evidence.append(unit_by_id[positive_unit_id])

    write_json(raw_json, output_root / "raw" / "test.json")
    write_jsonl(processed, output_root / "processed" / "test.jsonl")
    write_jsonl(queries, output_root / "queries" / "test.jsonl")
    write_jsonl(raw_units, output_root / "unit_registry" / "raw_units_test.jsonl")
    write_jsonl(targets, output_root / "targets" / "test.jsonl")
    write_jsonl(samples, output_root / "samples" / "test.jsonl")

    return {
        "qids": len(rows),
        "samples": len(samples),
        "raw_units": len(raw_units),
        "avg_steps_per_qid": round(len(samples) / max(1, len(rows)), 4),
        "avg_units_per_qid": round(len(raw_units) / max(1, len(rows)), 4),
        "type": dict(Counter(row["type"] for row in rows)),
        "level": dict(Counter(row["level"] for row in rows)),
        "support_steps": dict(Counter(len(row["supporting_facts"]) for row in rows)),
    }


def load_hf_rows(dataset_name: str, config: str, split: str, cache_dir: Path | None) -> list[dict]:
    from datasets import load_dataset

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    if config:
        ds = load_dataset(dataset_name, config, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)
    if split not in ds:
        available = ", ".join(ds.keys())
        raise KeyError(f"split {split!r} not found in {dataset_name}; available: {available}")
    return [dict(row) for row in ds[split]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/2wiki_multihopqa_eval_1000_cand50"))
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--source-json", type=Path, default=None, help="Local 2Wiki JSON/JSONL file.")
    parser.add_argument("--hf-dataset", default="voidful/2WikiMultihopQA")
    parser.add_argument("--hf-config", default="")
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"output already exists; pass --force to rebuild: {output_root}")

    if args.source_json:
        source_rows = read_local_rows(args.source_json)
        source_name = str(args.source_json)
    else:
        source_rows = load_hf_rows(args.hf_dataset, args.hf_config, args.source_split, args.dataset_cache_dir)
        source_name = args.hf_dataset

    source_rows.sort(key=lambda row: stable_key(qid_of(row), args.seed))

    selected = []
    skip_reasons = Counter()
    seen = set()
    for raw in source_rows:
        try:
            qid = qid_of(raw)
        except Exception:
            skip_reasons["missing_qid"] += 1
            continue
        if qid in seen:
            skip_reasons["duplicate_qid"] += 1
            continue
        seen.add(qid)
        try:
            row = project_row(raw)
            ok, reason = valid_projected_row(row)
        except Exception:
            skip_reasons["normalization_error"] += 1
            continue
        if not ok:
            skip_reasons[reason] += 1
            continue
        selected.append(row)
        if len(selected) >= args.size:
            break

    if len(selected) < args.size:
        raise RuntimeError(
            f"not enough usable 2Wiki rows: requested={args.size}, selected={len(selected)}, "
            f"skip_reasons={dict(skip_reasons)}"
        )

    stats = build_outputs(selected, output_root, max_candidates=args.max_candidates, seed=args.seed)
    manifest = {
        "dataset": "2WikiMultihopQA",
        "source": source_name,
        "source_split": args.source_split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size": args.size,
        "seed": args.seed,
        "max_candidates": args.max_candidates,
        "output_root": str(output_root),
        "stats": stats,
        "skip_reasons": dict(skip_reasons),
    }
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
