#!/usr/bin/env python3
"""Relabel v27 states with a deterministic coverage-greedy teacher."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}: {exc}") from exc


def load_targets(path: Path) -> dict[str, set[str]]:
    targets = {}
    for row in read_jsonl(path):
        qid = str(row.get("qid") or "")
        target_ids = {
            str(item.get("chunk_id") or item.get("unit_id") or "")
            for item in (row.get("T_q_raw") or [])
        }
        target_ids.discard("")
        targets[qid] = target_ids
    return targets


def load_memory(path: Path) -> dict[str, dict]:
    return {
        str(row["unit_id"]): row
        for row in read_jsonl(path)
        if row.get("unit_id")
    }


def candidate_ids(row: dict) -> list[str]:
    return [str(value) for value in ((row.get("candidates") or {}).get("C_t") or [])]


def old_positive(row: dict) -> str:
    return str((((row.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id")) or "")


def history_ids(row: dict) -> list[str]:
    output = []
    for item in ((row.get("state") or {}).get("H_t") or []):
        if isinstance(item, dict) and item.get("unit_id"):
            output.append(str(item["unit_id"]))
        elif isinstance(item, str):
            output.append(item)
    return output


def candidate_provenance(row: dict, unit_id: str, memory: dict[str, dict]) -> dict:
    stored = (((row.get("candidates") or {}).get("candidate_provenance") or {}).get(unit_id) or {})
    item = memory.get(unit_id) or {}
    parent = str(
        stored.get("parent_chunk_id")
        or stored.get("chunk_id")
        or item.get("parent_chunk_id")
        or ""
    )
    return {
        "chunk_id": parent,
        "doc_id": str(stored.get("doc_id") or item.get("doc_id") or item.get("title") or ""),
        "parent_chunk_id": parent,
    }


def coverage_positive(row: dict, targets: set[str], memory: dict[str, dict]) -> tuple[str, bool]:
    pool = candidate_ids(row)
    if not pool:
        raise ValueError(f"empty candidate pool: qid={row.get('qid')} t={row.get('t')}")
    acquired = set(history_ids(row))
    eligible = [unit_id for unit_id in pool if unit_id not in acquired]
    if not eligible:
        eligible = pool
    covered = {
        str(value)
        for value in (((row.get("state") or {}).get("A_t") or {}).get("covered_target_ids") or [])
    }
    for unit_id in eligible:
        parent = candidate_provenance(row, unit_id, memory)["parent_chunk_id"]
        if parent and parent in targets and parent not in covered:
            return unit_id, True
    return eligible[0], False


def rewrite_row(
    row: dict,
    positive: str,
    memory: dict[str, dict],
    source_positive: str,
    previous_coverage_positive: str | None,
) -> dict:
    output = copy.deepcopy(row)
    pool = candidate_ids(output)
    provenance = candidate_provenance(output, positive, memory)
    negative_ids = [unit_id for unit_id in pool if unit_id != positive]
    negative_provenance = {
        unit_id: candidate_provenance(output, unit_id, memory)
        for unit_id in negative_ids
    }
    labels = output.setdefault("labels", {})
    labels["u_t_plus"] = {
        "step_id": int(output.get("t", 0)),
        "unit_id": positive,
        **provenance,
    }
    labels["ranking_label"] = {
        "positive_unit_id": positive,
        "negative_unit_ids": negative_ids,
        "positive_provenance": provenance,
        "negative_provenance": negative_provenance,
    }
    acquired = [
        unit_id
        for unit_id in history_ids(output)
        if unit_id in pool and unit_id != positive
    ]
    counterfactual = labels.setdefault("counterfactual_ranking", {})
    counterfactual.update(
        {
            "current_positive_unit_id": positive,
            "acquired_negative_unit_ids": acquired,
            "adjacent_previous_positive_unit_id": previous_coverage_positive,
            "preference_switch_required": int(output.get("t", 0)) >= 1,
            "source_closure_positive_unit_id": source_positive,
            "teacher_objective": "coverage_greedy",
        }
    )
    build_meta = output.setdefault("build_meta", {})
    build_meta.update(
        {
            "dataset_version": "v29_coverage_greedy",
            "variant": "v27_matched_coverage_teacher",
            "source_closure_positive_unit_id": source_positive,
            "coverage_label_changed": positive != source_positive,
            "mask_auxiliary_labels": True,
        }
    )
    return output


def build_split(source: Path, output: Path, targets_path: Path, memory_path: Path) -> dict:
    targets = load_targets(targets_path)
    memory = load_memory(memory_path)
    stats = Counter()
    qids = set()
    t_distribution = Counter()
    coverage_by_state = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(source):
            qid = str(row.get("qid") or "")
            if qid not in targets:
                raise ValueError(f"missing target metadata: qid={qid}")
            t = int(row.get("t", -1))
            source_positive = old_positive(row)
            positive, found_uncovered = coverage_positive(row, targets[qid], memory)
            previous_coverage_positive = coverage_by_state.get((qid, t - 1))
            rewritten = rewrite_row(
                row,
                positive,
                memory,
                source_positive,
                previous_coverage_positive,
            )
            handle.write(json.dumps(rewritten, ensure_ascii=False) + "\n")
            coverage_by_state[(qid, t)] = positive
            qids.add(qid)
            t_distribution[str(t)] += 1
            stats["rows"] += 1
            stats["changed_labels"] += int(positive != source_positive)
            stats["unchanged_labels"] += int(positive == source_positive)
            stats["uncovered_target_available"] += int(found_uncovered)
            source_parent = candidate_provenance(row, source_positive, memory)["parent_chunk_id"]
            coverage_parent = candidate_provenance(row, positive, memory)["parent_chunk_id"]
            stats["changed_with_same_parent"] += int(
                positive != source_positive and source_parent == coverage_parent
            )
            stats["changed_with_different_parent"] += int(
                positive != source_positive and source_parent != coverage_parent
            )
    return {
        "source": str(source),
        "output": str(output),
        "rows": stats["rows"],
        "qids": len(qids),
        "t_distribution": dict(t_distribution),
        "changed_labels": stats["changed_labels"],
        "changed_label_rate": round(stats["changed_labels"] / stats["rows"], 6) if stats["rows"] else None,
        "unchanged_labels": stats["unchanged_labels"],
        "uncovered_target_available": stats["uncovered_target_available"],
        "changed_with_same_parent": stats["changed_with_same_parent"],
        "changed_with_different_parent": stats["changed_with_different_parent"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/hotpotqa_distractor_v27_counterfactual_dual"))
    parser.add_argument("--teacher-root", type=Path, default=Path("data/hotpotqa_distractor_v7_10k_llm_prestep"))
    parser.add_argument("--output-root", type=Path, default=Path("data/hotpotqa_distractor_v29_coverage_greedy"))
    args = parser.parse_args()

    required = []
    for split in SPLITS:
        required.extend(
            [
                args.source_root / "samples" / f"{split}.jsonl",
                args.teacher_root / "targets" / f"{split}.jsonl",
                args.teacher_root / "unit_registry" / f"raw_units_{split}.jsonl",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))
    if args.output_root.exists():
        raise FileExistsError(f"output already exists: {args.output_root}")

    split_stats = {
        split: build_split(
            args.source_root / "samples" / f"{split}.jsonl",
            args.output_root / "samples" / f"{split}.jsonl",
            args.teacher_root / "targets" / f"{split}.jsonl",
            args.teacher_root / "unit_registry" / f"raw_units_{split}.jsonl",
        )
        for split in SPLITS
    }
    source_manifest_path = args.source_root / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else {}
    )
    manifest = {
        "dataset": "KBS v29 coverage-greedy matched Student data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root),
        "teacher_root": str(args.teacher_root),
        "output_root": str(args.output_root),
        "seed": source_manifest.get("seed"),
        "later_repeat": source_manifest.get("later_repeat"),
        "candidate_top_k": source_manifest.get("candidate_top_k"),
        "teacher_objective": "first legal unacquired candidate covering a visible uncovered target; otherwise first legal unacquired C_t candidate",
        "matched_contract": [
            "row order and repetitions",
            "qids and split",
            "state and rendered K_t",
            "candidate pool and order",
            "architecture and initialization",
            "optimizer and loss weights",
        ],
        "changed_contract": ["ranking positive and dependent ranking metadata"],
        "split_stats": split_stats,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
