#!/usr/bin/env python3
"""Build a deterministic paragraph-unit MuSiQue-Ans evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{value}".encode("utf-8")).hexdigest()


def qid_of(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("qid") or row.get("question_id") or "")


def paragraph_unit_id(qid: str, paragraph_idx: int) -> str:
    return f"{qid}::paragraph::{paragraph_idx}"


def parent_chunk_id(qid: str, paragraph_idx: int) -> str:
    return f"{qid}::paragraph::{paragraph_idx}"


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    qid = qid_of(row)
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not qid or not question or not answer or row.get("answerable", True) is not True:
        raise ValueError("row is not a complete answerable MuSiQue instance")

    paragraphs_raw = row.get("paragraphs")
    decomposition = row.get("question_decomposition")
    if not isinstance(paragraphs_raw, list) or not paragraphs_raw:
        raise ValueError("missing paragraphs")
    if not isinstance(decomposition, list) or not decomposition:
        raise ValueError("missing question_decomposition")

    paragraphs = []
    by_idx: dict[int, dict[str, Any]] = {}
    for raw in paragraphs_raw:
        if not isinstance(raw, dict) or isinstance(raw.get("idx"), bool):
            raise ValueError("invalid paragraph object")
        index = int(raw["idx"])
        if index in by_idx:
            raise ValueError(f"duplicate paragraph idx={index}")
        text = str(raw.get("paragraph_text") or "").strip()
        if not text:
            raise ValueError(f"empty paragraph idx={index}")
        paragraph = {
            "idx": index,
            "title": str(raw.get("title") or f"paragraph_{index}").strip(),
            "text": text,
            "is_supporting": raw.get("is_supporting") is True,
        }
        paragraphs.append(paragraph)
        by_idx[index] = paragraph

    support_order: list[int] = []
    for item in decomposition:
        if not isinstance(item, dict) or isinstance(item.get("paragraph_support_idx"), bool):
            raise ValueError("invalid decomposition support mapping")
        index = int(item["paragraph_support_idx"])
        if index not in by_idx:
            raise ValueError(f"decomposition support idx={index} is absent")
        support_order.append(index)

    supporting = {paragraph["idx"] for paragraph in paragraphs if paragraph["is_supporting"]}
    if len(support_order) != len(set(support_order)):
        raise ValueError("decomposition support mapping is not one-to-one")
    if set(support_order) != supporting:
        raise ValueError("decomposition support mapping differs from supporting flags")

    aliases = []
    seen_aliases = set()
    for value in row.get("answer_aliases") or []:
        alias = str(value).strip()
        if alias and alias != answer and alias not in seen_aliases:
            seen_aliases.add(alias)
            aliases.append(alias)

    return {
        "qid": qid,
        "question": question,
        "answer": answer,
        "answer_aliases": aliases,
        "paragraphs": sorted(paragraphs, key=lambda item: item["idx"]),
        "support_order": support_order,
    }


def candidate_ids(row: dict[str, Any], acquired: set[int], seed: int) -> list[str]:
    qid = row["qid"]
    values = [
        paragraph_unit_id(qid, paragraph["idx"])
        for paragraph in row["paragraphs"]
        if paragraph["idx"] not in acquired
    ]
    return sorted(values, key=lambda unit_id: stable_key(f"{qid}::{unit_id}", seed))


def render_notebook(previous_units: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{index + 1}] {unit['doc_id']}: {unit['text']}"
        for index, unit in enumerate(previous_units)
    )


def build_outputs(
    rows: list[dict[str, Any]], output_root: Path, *, source: Path, seed: int
) -> dict[str, Any]:
    queries = []
    units = []
    targets = []
    samples = []
    paragraph_counts: Counter[int] = Counter()
    hop_counts: Counter[int] = Counter()
    alias_counts: Counter[int] = Counter()
    candidate_sizes: Counter[int] = Counter()

    for row in rows:
        qid = row["qid"]
        paragraph_counts[len(row["paragraphs"])] += 1
        hop_counts[len(row["support_order"])] += 1
        alias_counts[len(row["answer_aliases"])] += 1
        queries.append(
            {
                "qid": qid,
                "question": row["question"],
                "answer": row["answer"],
                "answer_aliases": row["answer_aliases"],
                "metadata": {
                    "dataset": "musique_ans_v1.0",
                    "source_split": "dev",
                    "evaluation_split": "test",
                    "evidence_granularity": "paragraph",
                    "hop_count": len(row["support_order"]),
                },
            }
        )

        unit_by_idx: dict[int, dict[str, Any]] = {}
        for paragraph in row["paragraphs"]:
            index = paragraph["idx"]
            unit_id = paragraph_unit_id(qid, index)
            unit = {
                "unit_id": unit_id,
                "text": paragraph["text"],
                "doc_id": paragraph["title"],
                "parent_chunk_id": parent_chunk_id(qid, index),
                "span_start": None,
                "span_end": None,
                "provenance": "raw",
                "candidate_granularity": "paragraph",
                "paragraph_idx": index,
            }
            units.append(unit)
            unit_by_idx[index] = unit

        target_units = []
        for index in row["support_order"]:
            unit = unit_by_idx[index]
            target_units.append(
                {
                    "unit_id": unit["unit_id"],
                    "chunk_id": unit["parent_chunk_id"],
                    "text": unit["text"],
                    "doc_id": unit["doc_id"],
                    "parent_chunk_id": unit["parent_chunk_id"],
                    "span_start": None,
                    "span_end": None,
                    "provenance": "raw",
                    "weight": 1.0,
                    "primary_role": "support",
                    "role_label_source": "musique_paragraph_support",
                    "paragraph_idx": index,
                }
            )
        targets.append(
            {
                "qid": qid,
                "question": row["question"],
                "evidence_granularity": "paragraph",
                "T_q_raw": target_units,
            }
        )

        acquired: set[int] = set()
        previous_units: list[dict[str, Any]] = []
        for step, positive_idx in enumerate(row["support_order"]):
            positive = unit_by_idx[positive_idx]
            candidates = candidate_ids(row, acquired, seed)
            if positive["unit_id"] not in candidates:
                raise AssertionError(f"positive missing from candidates: {qid} step={step}")
            candidate_sizes[len(candidates)] += 1
            provenance = {
                unit_id: {
                    "chunk_id": unit_id,
                    "doc_id": next(
                        unit["doc_id"] for unit in unit_by_idx.values() if unit["unit_id"] == unit_id
                    ),
                    "parent_chunk_id": unit_id,
                }
                for unit_id in candidates
            }
            samples.append(
                {
                    "qid": qid,
                    "t": step,
                    "build_meta": {
                        "run_id": "musique_ans_zero_shot_eval",
                        "source": "prepare_musique_policy_rag_eval.py",
                        "source_split": "dev",
                        "evaluation_split": "test",
                        "evidence_granularity": "paragraph",
                    },
                    "question": row["question"],
                    "state": {
                        "H_t": [
                            {
                                "step_id": index,
                                "unit_id": unit["unit_id"],
                                "chunk_id": unit["parent_chunk_id"],
                                "doc_id": unit["doc_id"],
                                "parent_chunk_id": unit["parent_chunk_id"],
                            }
                            for index, unit in enumerate(previous_units)
                        ],
                        "K_t": render_notebook(previous_units),
                    },
                    "candidates": {
                        "R_t": candidates,
                        "C_t": candidates,
                        "G_t_final": [],
                        "G_t_aux": [],
                        "G_t_illegal": [],
                        "candidate_provenance": provenance,
                    },
                    "labels": {
                        "u_t_plus": {
                            "step_id": step,
                            "unit_id": positive["unit_id"],
                            "chunk_id": positive["parent_chunk_id"],
                            "doc_id": positive["doc_id"],
                            "parent_chunk_id": positive["parent_chunk_id"],
                        },
                        "ranking_label": {
                            "positive_unit_id": positive["unit_id"],
                            "negative_unit_ids": [
                                unit_id for unit_id in candidates if unit_id != positive["unit_id"]
                            ],
                            "positive_provenance": provenance[positive["unit_id"]],
                            "negative_provenance": {
                                unit_id: provenance[unit_id]
                                for unit_id in candidates
                                if unit_id != positive["unit_id"]
                            },
                        },
                    },
                }
            )
            acquired.add(positive_idx)
            previous_units.append(positive)

    write_jsonl(queries, output_root / "queries/test.jsonl")
    write_jsonl(units, output_root / "unit_registry/raw_units_test.jsonl")
    write_jsonl(targets, output_root / "targets/test.jsonl")
    write_jsonl(samples, output_root / "samples/test.jsonl")
    return {
        "queries": len(queries),
        "samples": len(samples),
        "memory_rows": len(units),
        "paragraph_count_distribution": dict(sorted(paragraph_counts.items())),
        "hop_count_distribution": dict(sorted(hop_counts.items())),
        "answer_alias_count_distribution": dict(sorted(alias_counts.items())),
        "candidate_size_distribution": dict(sorted(candidate_sizes.items())),
        "source_sha256": file_sha256(source),
        "ordered_qids_sha256": hashlib.sha256(
            "\n".join(row["qid"] for row in rows).encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/musique_ans_eval_1000_paragraph20"),
    )
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    if not args.source_json.is_file():
        raise FileNotFoundError(args.source_json)
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {args.output_root}")

    normalized = [normalize_row(row) for row in read_jsonl(args.source_json)]
    normalized.sort(key=lambda row: stable_key(row["qid"], args.seed))
    if len(normalized) < args.size:
        raise RuntimeError(f"requested {args.size} rows but only {len(normalized)} are valid")
    selected = normalized[: args.size]
    stats = build_outputs(
        selected, args.output_root, source=args.source_json, seed=args.seed
    )
    manifest = {
        "dataset": "MuSiQue-Ans v1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source_json),
        "source_split": "dev",
        "output_split": "test",
        "size": args.size,
        "seed": args.seed,
        "selection": "ascending sha256(seed::qid), first size rows",
        "student_training": "HotpotQA only; no MuSiQue fine-tuning",
        "unit_granularity": "paragraph",
        "candidate_memory": "all question-local official paragraphs",
        "gold_mapping": "question_decomposition.paragraph_support_idx",
        "decomposition_text_visible_to_policy": False,
        "max_available_candidates": 20,
        "recall_50_supported": False,
        "stats": stats,
    }
    write_json(manifest, args.output_root / "manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"data_root: {args.output_root}")


if __name__ == "__main__":
    main()
