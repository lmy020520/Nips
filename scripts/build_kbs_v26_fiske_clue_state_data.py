#!/usr/bin/env python3
"""Rewrite v22 states as a deterministic FiSKE-inspired textual clue state."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.clue_state import (
    CLUE_STATE_VERSION,
    build_clue_state,
    format_clue_evidence,
    render_clue_state,
)


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_h_ids(row: dict) -> list[str]:
    result = []
    for item in (row.get("state") or {}).get("H_t") or []:
        if isinstance(item, dict) and item.get("unit_id"):
            result.append(str(item["unit_id"]))
        elif isinstance(item, str):
            result.append(item)
    return result


def limit_qids(rows, max_qids: int):
    if max_qids <= 0:
        return list(rows)
    kept = []
    seen = set()
    for row in rows:
        qid = str(row.get("qid") or "")
        if qid not in seen and len(seen) >= max_qids:
            continue
        seen.add(qid)
        kept.append(row)
    return kept


def load_required_memory(path: Path, required_ids: set[str]) -> dict[str, dict]:
    memory = {}
    for row_index, row in enumerate(read_jsonl(path), start=1):
        unit_id = str(row.get("unit_id") or "")
        if unit_id in required_ids:
            memory[unit_id] = row
        if row_index % 100000 == 0:
            print(
                f"[BUILD] memory={path.name} scanned={row_index} "
                f"resolved={len(memory)}/{len(required_ids)}",
                file=sys.stderr,
                flush=True,
            )
    missing = sorted(required_ids - set(memory))
    if missing:
        raise ValueError(f"{len(missing)} prefix units missing from {path}: {missing[:5]}")
    return memory


def rewrite_split(
    source_path: Path,
    memory_path: Path,
    output_path: Path,
    *,
    max_qids: int,
) -> dict:
    rows = limit_qids(read_jsonl(source_path), max_qids)
    required_ids = {
        unit_id
        for row in rows
        for unit_id in state_h_ids(row)
    }
    memory = load_required_memory(memory_path, required_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qids = set()
    t_distribution = Counter()
    clue_counts = Counter()
    covered_by_t = Counter()
    clues_by_t = Counter()
    rendered_changes = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            qid = str(row["qid"])
            t = int(row["t"])
            h_ids = state_h_ids(row)
            prefix_texts = [format_clue_evidence(memory[unit_id]) for unit_id in h_ids]
            clue_state = build_clue_state(str(row["question"]), prefix_texts)
            rendered = render_clue_state(clue_state)

            rewritten = json.loads(json.dumps(row))
            state = rewritten.setdefault("state", {})
            if str(state.get("K_t") or "") != rendered:
                rendered_changes += 1
            state["K_t"] = rendered
            state["clue_state"] = clue_state
            build_meta = rewritten.setdefault("build_meta", {})
            build_meta.update(
                {
                    "source": "build_kbs_v26_fiske_clue_state_data.py",
                    "variant": "fiske_inspired_textual_clue_state",
                    "source_dataset": str(source_path.parent.parent),
                    "clue_generator_version": clue_state["version"],
                    "clue_generator_inputs": ["question"],
                    "clue_coverage_inputs": ["state.H_t", "memory.text"],
                    "mask_auxiliary_labels": True,
                }
            )
            handle.write(json.dumps(rewritten, ensure_ascii=False) + "\n")

            qids.add(qid)
            t_distribution[str(t)] += 1
            clue_counts[str(len(clue_state["clues"]))] += 1
            covered_by_t[str(t)] += sum(clue_state["coverage_vector"])
            clues_by_t[str(t)] += len(clue_state["coverage_vector"])

    return {
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "output_path": str(output_path),
        "rows": len(rows),
        "qids": len(qids),
        "required_prefix_units": len(required_ids),
        "t_distribution": dict(t_distribution),
        "clue_count_distribution": dict(clue_counts),
        "covered_clue_rate_by_t": {
            t: round(covered_by_t[t] / max(clues_by_t[t], 1), 6)
            for t in sorted(clues_by_t, key=int)
        },
        "rendered_changes": rendered_changes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="data/hotpotqa_distractor_v22_state_focused",
    )
    parser.add_argument(
        "--memory-root",
        default="data/hotpotqa_distractor_v7_10k_llm_prestep/unit_registry",
    )
    parser.add_argument(
        "--output-root",
        default="data/hotpotqa_distractor_v26_fiske_clue_state",
    )
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    memory_root = Path(args.memory_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"output exists; pass --force: {output_root}")
        shutil.rmtree(output_root)

    split_stats = {}
    for split in ("train", "val", "test"):
        print(f"[BUILD] split={split} started", file=sys.stderr, flush=True)
        source_path = source_root / "samples" / f"{split}.jsonl"
        memory_path = memory_root / f"raw_units_{split}.jsonl"
        for required in (source_path, memory_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        split_stats[split] = rewrite_split(
            source_path,
            memory_path,
            output_root / "samples" / f"{split}.jsonl",
            max_qids=args.max_qids,
        )
        print(
            f"[BUILD] split={split} rows={split_stats[split]['rows']} "
            f"qids={split_stats[split]['qids']} completed",
            file=sys.stderr,
            flush=True,
        )

    manifest = {
        "dataset": "KBS v26 FiSKE-inspired textual clue-state baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "memory_root": str(memory_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "max_qids": args.max_qids,
        "baseline_name": "FiSKE-inspired textual clue-state baseline",
        "faithful_fiske_reproduction": False,
        "question_clue_generator": {
            "version": CLUE_STATE_VERSION,
            "learned": False,
            "api_calls": 0,
            "inputs": ["question"],
        },
        "coverage_rule": {
            "learned": False,
            "inputs": ["question-derived clues", "current-prefix evidence text"],
            "future_evidence_used": False,
        },
        "forbidden_inputs": [
            "answer",
            "gold supporting facts",
            "teacher roles",
            "future trajectory units",
        ],
        "candidate_pool_policy": "copied exactly from v22",
        "ranking_labels": "copied exactly from v22",
        "auxiliary_labels_masked": True,
        "split_stats": split_stats,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
