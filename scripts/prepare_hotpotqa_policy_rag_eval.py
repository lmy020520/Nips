#!/usr/bin/env python3
"""Prepare an independent HotpotQA distractor evaluation set for Policy-RAG.

The output is intentionally lightweight: queries, raw sentence units, targets,
and policy-RAG sample states. It does not create LLM role labels or training
artifacts because this dataset is for evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


DEFAULT_EXCLUDE_ROOTS = [
    Path("data/hotpotqa_distractor_v7_10k_llm_prestep"),
    Path("data/hotpotqa_distractor_v8_15k_llm_prestep"),
]


def qid_of(row: dict) -> str:
    for key in ("qid", "_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("row has no qid/_id/id")


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{value}".encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
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


def normalize_context(context) -> list[dict]:
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        if len(titles) != len(sentences):
            raise ValueError("context title/sentences length mismatch")
        return [
            {"title": str(title), "sentences": [str(sent) for sent in sents]}
            for title, sents in zip(titles, sentences)
        ]
    if isinstance(context, list):
        out = []
        for item in context:
            out.append(
                {
                    "title": str(item["title"]),
                    "sentences": [str(sent) for sent in item["sentences"]],
                }
            )
        return out
    raise ValueError(f"unsupported context type: {type(context)}")


def normalize_supporting_facts(supporting_facts) -> list[tuple[str, int]]:
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title") or []
        sent_ids = supporting_facts.get("sent_id") or []
        if len(titles) != len(sent_ids):
            raise ValueError("supporting_facts title/sent_id length mismatch")
        pairs = [(str(title), int(sent_id)) for title, sent_id in zip(titles, sent_ids)]
    else:
        pairs = [(str(title), int(sent_id)) for title, sent_id in supporting_facts]
    seen = set()
    out = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def build_unit_id(qid: str, title: str, sent_id: int) -> str:
    return f"{qid}::{title}::{sent_id}"


def build_parent_chunk_id(qid: str, title: str) -> str:
    return f"{qid}::{title}"


def sentence_lookup(context: list[dict]) -> dict[tuple[str, int], str]:
    lookup = {}
    for block in context:
        title = str(block["title"])
        for sent_id, text in enumerate(block["sentences"]):
            text = str(text).strip()
            if text:
                lookup[(title, sent_id)] = text
    return lookup


def unit_record(qid: str, title: str, sent_id: int, text: str) -> dict:
    parent_chunk_id = build_parent_chunk_id(qid, title)
    return {
        "unit_id": build_unit_id(qid, title, sent_id),
        "text": text,
        "doc_id": title,
        "parent_chunk_id": parent_chunk_id,
        "span_start": None,
        "span_end": None,
        "provenance": "raw",
        "candidate_granularity": "sentence",
    }


def load_excluded_qids(roots: list[Path]) -> dict[str, int]:
    excluded = set()
    for root in roots:
        for rel in [
            "queries/train.jsonl",
            "queries/val.jsonl",
            "queries/test.jsonl",
            "samples/train.jsonl",
            "samples/val.jsonl",
            "samples/test.jsonl",
        ]:
            path = root / rel
            if not path.exists():
                continue
            for row in read_jsonl(path):
                qid = row.get("qid")
                if qid:
                    excluded.add(str(qid))
    return {"roots": len(roots), "qids": len(excluded), "set": excluded}


def project_row(row: dict) -> dict:
    return {
        "qid": qid_of(row),
        "question": str(row["question"]).strip(),
        "answer": str(row["answer"]).strip(),
        "type": str(row["type"]).strip(),
        "level": str(row["level"]).strip(),
        "supporting_facts": [[title, sent_id] for title, sent_id in normalize_supporting_facts(row["supporting_facts"])],
        "context": normalize_context(row["context"]),
    }


def valid_projected_row(row: dict) -> tuple[bool, str]:
    if not row["qid"] or not row["question"] or not row["answer"]:
        return False, "missing_qid_question_or_answer"
    lookup = sentence_lookup(row["context"])
    support = normalize_supporting_facts(row["supporting_facts"])
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


def build_outputs(
    rows: list[dict],
    output_root: Path,
    *,
    max_candidates: int,
    seed: int,
    output_split: str,
) -> dict:
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
                    "dataset": "hotpotqa_distractor",
                    "split": output_split,
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

        support = normalize_supporting_facts(row["supporting_facts"])
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
                    "role_label_source": "hotpotqa_supporting_fact",
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
                        "run_id": "hotpotqa_policy_rag_eval",
                        "source": "prepare_hotpotqa_policy_rag_eval.py",
                        "split": output_split,
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

    write_json(raw_json, output_root / "raw" / f"{output_split}.json")
    write_jsonl(processed, output_root / "processed" / f"{output_split}.jsonl")
    write_jsonl(queries, output_root / "queries" / f"{output_split}.jsonl")
    write_jsonl(raw_units, output_root / "unit_registry" / f"raw_units_{output_split}.jsonl")
    write_jsonl(targets, output_root / "targets" / f"{output_split}.jsonl")
    write_jsonl(samples, output_root / "samples" / f"{output_split}.jsonl")

    return {
        "qids": len(rows),
        "samples": len(samples),
        "raw_units": len(raw_units),
        "avg_steps_per_qid": round(len(samples) / max(1, len(rows)), 4),
        "avg_units_per_qid": round(len(raw_units) / max(1, len(rows)), 4),
        "type": dict(Counter(row["type"] for row in rows)),
        "level": dict(Counter(row["level"] for row in rows)),
        "type_level": {
            f"{key[0]}/{key[1]}": value
            for key, value in Counter((row["type"], row["level"]) for row in rows).items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/hotpotqa_distractor_eval_3000"))
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--source-split", choices=["validation", "train"], default="validation")
    parser.add_argument(
        "--output-split",
        choices=["val", "test"],
        default="test",
        help="Filename and metadata split used in the generated evaluation artifacts.",
    )
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--exclude-root", type=Path, action="append", default=[])
    parser.add_argument("--no-default-exclude-roots", action="store_true")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="0 means all HotpotQA distractor sentences; otherwise keep positive plus stable distractors.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"output already exists; pass --force to rebuild: {output_root}")

    exclude_roots = [] if args.no_default_exclude_roots else list(DEFAULT_EXCLUDE_ROOTS)
    exclude_roots.extend(args.exclude_root)
    excluded = load_excluded_qids(exclude_roots)

    print("Loading HotpotQA distractor split...", flush=True)
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        cache_dir=str(args.dataset_cache_dir) if args.dataset_cache_dir else None,
    )
    source_rows = [dict(row) for row in dataset[args.source_split]]
    source_rows.sort(key=lambda row: stable_key(qid_of(row), args.seed))

    selected = []
    skip_reasons = Counter()
    seen = set()
    for raw in source_rows:
        qid = qid_of(raw)
        if qid in seen:
            skip_reasons["duplicate_qid"] += 1
            continue
        seen.add(qid)
        if qid in excluded["set"]:
            skip_reasons["excluded_existing_qid"] += 1
            continue
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
        raise RuntimeError(f"not enough usable rows: requested={args.size}, selected={len(selected)}")

    stats = build_outputs(
        selected,
        output_root,
        max_candidates=args.max_candidates,
        seed=args.seed,
        output_split=args.output_split,
    )
    manifest = {
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "purpose": "independent_policy_rag_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_split": args.source_split,
        "output_split": args.output_split,
        "seed": args.seed,
        "output_root": str(output_root),
        "size": args.size,
        "max_candidates": args.max_candidates,
        "excluded_roots": [str(root) for root in exclude_roots],
        "excluded_qids": excluded["qids"],
        "skip_reasons": dict(skip_reasons),
        "stats": stats,
        "files": {
            "queries": str(output_root / "queries" / f"{args.output_split}.jsonl"),
            "samples": str(output_root / "samples" / f"{args.output_split}.jsonl"),
            "raw_units": str(
                output_root / "unit_registry" / f"raw_units_{args.output_split}.jsonl"
            ),
            "targets": str(output_root / "targets" / f"{args.output_split}.jsonl"),
        },
    }
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
