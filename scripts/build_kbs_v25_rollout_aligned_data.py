#!/usr/bin/env python3
"""Canonicalize v22 states and build rollout-aligned v25 ranking data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.online_state import init_online_state, render_online_k_t, update_online_state


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def positive_id(row: dict) -> str:
    labels = row.get("labels") or {}
    ranking = labels.get("ranking_label") or {}
    return str(ranking.get("positive_unit_id") or "")


def candidate_ids(row: dict) -> list[str]:
    candidates = row.get("candidates") or {}
    return [str(unit_id) for unit_id in candidates.get("C_t") or []]


def canonical_signature(row: dict) -> tuple:
    return (
        positive_id(row),
        tuple(candidate_ids(row)),
        str((row.get("state") or {}).get("K_t") or ""),
    )


def canonicalize_split(source: Path, output: Path, max_qids: int) -> dict:
    first_by_key: dict[tuple[str, int], dict] = {}
    signatures: dict[tuple[str, int], tuple] = {}
    qid_order: list[str] = []
    retained_qids: set[str] = set()
    seen_qids: set[str] = set()
    repeated = 0
    conflicts = 0

    for row in read_jsonl(source):
        qid = str(row.get("qid") or "")
        t = int(row.get("t", -1))
        if not qid or t < 0:
            raise ValueError(f"invalid qid/t in {source}: qid={qid!r}, t={t}")
        if qid not in seen_qids:
            if max_qids > 0 and len(qid_order) >= max_qids:
                continue
            seen_qids.add(qid)
            qid_order.append(qid)
            retained_qids.add(qid)
        if qid not in retained_qids:
            continue
        key = (qid, t)
        signature = canonical_signature(row)
        if key in first_by_key:
            repeated += 1
            if signature != signatures[key]:
                conflicts += 1
            continue
        first_by_key[key] = row
        signatures[key] = signature

    if conflicts:
        raise ValueError(f"{source} has {conflicts} conflicting repeated (qid,t) rows")

    rows = []
    for qid in qid_order:
        qid_rows = [
            first_by_key[(qid, t)]
            for t in sorted(t for candidate_qid, t in first_by_key if candidate_qid == qid)
        ]
        expected = list(range(len(qid_rows)))
        actual = [int(row["t"]) for row in qid_rows]
        if actual != expected:
            raise ValueError(f"non-contiguous states for qid={qid}: {actual}")
        for row in qid_rows:
            item = copy.deepcopy(row)
            item.setdefault("build_meta", {})["canonicalized_for_v25"] = True
            item["build_meta"]["repetition"] = 0
            rows.append(item)

    written = write_jsonl(output, rows)
    return {
        "source": str(source),
        "output": str(output),
        "qids": len(qid_order),
        "rows": written,
        "removed_repeated_rows": repeated,
        "conflicting_repeated_rows": conflicts,
    }


def canonicalize(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    workspace = Path(args.workspace)
    canonical_root = workspace / "canonical"
    if canonical_root.exists() and args.force:
        shutil.rmtree(canonical_root)
    if canonical_root.exists() and not args.force:
        raise FileExistsError(f"canonical output exists; pass --force: {canonical_root}")

    reports = {}
    for split in SPLITS:
        reports[split] = canonicalize_split(
            source_root / "samples" / f"{split}.jsonl",
            canonical_root / f"{split}.jsonl",
            args.max_qids,
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "workspace": str(workspace),
        "max_qids": args.max_qids,
        "splits": reports,
    }
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "canonical_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_memory(path: Path) -> dict[str, dict]:
    memory = {}
    for row in read_jsonl(path):
        unit_id = str(row.get("unit_id") or "")
        if not unit_id:
            continue
        title = str(row.get("title") or row.get("doc_id") or "")
        memory[unit_id] = {
            **row,
            "unit_id": unit_id,
            "title": title,
            "doc_id": str(row.get("doc_id") or title),
            "text": str(row.get("text") or "").strip(),
        }
    return memory


def h_entries(unit_ids: list[str], row: dict, memory: dict[str, dict]) -> list[dict]:
    provenance_map = ((row.get("candidates") or {}).get("candidate_provenance") or {})
    entries = []
    for step_id, unit_id in enumerate(unit_ids):
        item = memory[unit_id]
        provenance = provenance_map.get(unit_id) or {}
        parent = str(
            provenance.get("parent_chunk_id")
            or provenance.get("chunk_id")
            or item.get("parent_chunk_id")
            or f"{row['qid']}::{item['title']}"
        )
        entries.append(
            {
                "step_id": step_id,
                "unit_id": unit_id,
                "chunk_id": parent,
                "doc_id": item["doc_id"],
                "parent_chunk_id": parent,
            }
        )
    return entries


def dataset_state(runtime_state: dict, row: dict, memory: dict[str, dict]) -> dict:
    h_ids = [str(unit_id) for unit_id in runtime_state.get("H_t") or []]
    return {
        "H_t": h_entries(h_ids, row, memory),
        "A_t": copy.deepcopy(runtime_state.get("A_t") or {}),
        "S_t": copy.deepcopy(runtime_state.get("S_t") or {}),
        "K_t": str(runtime_state.get("K_t") or ""),
    }


def provenance_for(row: dict, unit_id: str, memory: dict[str, dict]) -> dict:
    provenance_map = ((row.get("candidates") or {}).get("candidate_provenance") or {})
    provenance = provenance_map.get(unit_id) or {}
    item = memory[unit_id]
    parent = str(
        provenance.get("parent_chunk_id")
        or provenance.get("chunk_id")
        or item.get("parent_chunk_id")
        or f"{row['qid']}::{item['title']}"
    )
    return {
        "chunk_id": parent,
        "doc_id": str(provenance.get("doc_id") or item["doc_id"]),
        "parent_chunk_id": parent,
    }


def rewrite_row(
    source: dict,
    runtime_state: dict,
    target_id: str,
    *,
    split: str,
    state_source: str,
    original_positive: str,
    memory: dict[str, dict],
) -> dict:
    row = copy.deepcopy(source)
    pool = candidate_ids(row)
    if target_id not in pool:
        raise ValueError(
            f"target missing from pool: qid={row.get('qid')}, t={row.get('t')}, target={target_id}"
        )
    target_provenance = provenance_for(row, target_id, memory)
    negative_ids = [unit_id for unit_id in pool if unit_id != target_id]
    negative_provenance = {
        unit_id: provenance_for(row, unit_id, memory) for unit_id in negative_ids
    }

    row["state"] = dataset_state(runtime_state, row, memory)
    row["labels"]["u_t_plus"] = {
        "step_id": int(row["t"]),
        "unit_id": target_id,
        **target_provenance,
    }
    row["labels"]["ranking_label"] = {
        "positive_unit_id": target_id,
        "negative_unit_ids": negative_ids,
        "positive_provenance": target_provenance,
        "negative_provenance": negative_provenance,
    }
    build_meta = row.setdefault("build_meta", {})
    build_meta.update(
        {
            "run_id": "kbs_v25_rollout_aligned",
            "source": "build_kbs_v25_rollout_aligned_data.py",
            "split": split,
            "variant": "rollout_aligned_full_state",
            "state_source": state_source,
            "original_teacher_positive_unit_id": original_positive,
            "rollout_target_rewritten": target_id != original_positive,
            "repetition": 0,
            "mask_auxiliary_labels": True,
        }
    )
    row.setdefault("meta", {})["state_source"] = state_source
    row["meta"]["rollout_state_length"] = len(runtime_state.get("H_t") or [])
    return row


def result_map(report: dict) -> dict[str, dict]:
    mapped = {}
    for result in report.get("results") or []:
        qid = str(result.get("qid") or "")
        if not qid or qid in mapped:
            raise ValueError(f"invalid/duplicate qid in rollout report: {qid!r}")
        mapped[qid] = result
    return mapped


def build_split(
    split: str,
    canonical_path: Path,
    rollout_path: Path,
    memory_path: Path,
    output_path: Path,
    max_raw: int,
    max_chars: int,
) -> dict:
    memory = load_memory(memory_path)
    source_by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(canonical_path):
        source_by_qid[str(row["qid"])].append(row)
    for rows in source_by_qid.values():
        rows.sort(key=lambda row: int(row["t"]))

    rollout_report = json.loads(rollout_path.read_text(encoding="utf-8"))
    rollouts = result_map(rollout_report)
    if set(source_by_qid) != set(rollouts):
        missing = sorted(set(source_by_qid) - set(rollouts))[:5]
        extra = sorted(set(rollouts) - set(source_by_qid))[:5]
        raise ValueError(f"rollout qid mismatch: missing={missing}, extra={extra}")

    stats = Counter()
    output_rows = []
    for qid, rows in source_by_qid.items():
        result = rollouts[qid]
        steps = sorted(result.get("steps") or [], key=lambda step: int(step.get("t", -1)))
        step_by_t = {int(step["t"]): step for step in steps}
        if len(step_by_t) != len(rows):
            raise ValueError(f"rollout step mismatch: qid={qid}, source={len(rows)}, report={len(step_by_t)}")
        gold_order = [positive_id(row) for row in rows]

        teacher_state = init_online_state()
        for row in rows:
            t = int(row["t"])
            original_positive = positive_id(row)
            if split == "train":
                output_rows.append(
                    rewrite_row(
                        row,
                        teacher_state,
                        original_positive,
                        split=split,
                        state_source="teacher_online",
                        original_positive=original_positive,
                        memory=memory,
                    )
                )
                stats["teacher_rows"] += 1

            step = step_by_t[t]
            rollout_state = step.get("online_state_before")
            if not isinstance(rollout_state, dict):
                raise ValueError(f"online_state_before missing: qid={qid}, t={t}")
            rendered = render_online_k_t(
                rollout_state,
                memory,
                max_raw=max_raw,
                max_chars_per_item=max_chars,
            )
            if rendered != str(rollout_state.get("K_t") or ""):
                raise ValueError(f"rollout renderer mismatch: qid={qid}, t={t}")
            acquired = {str(unit_id) for unit_id in rollout_state.get("H_t") or []}
            unresolved = [unit_id for unit_id in gold_order if unit_id not in acquired]
            if unresolved:
                target_id = unresolved[0]
                if split != "train" or t > 0:
                    output_rows.append(
                        rewrite_row(
                            row,
                            rollout_state,
                            target_id,
                            split=split,
                            state_source="frozen_v22_rollout",
                            original_positive=original_positive,
                            memory=memory,
                        )
                    )
                    stats["rollout_rows"] += 1
                    stats["rewritten_rollout_targets"] += int(target_id != original_positive)
            else:
                stats["rollout_rows_skipped_complete"] += 1

            teacher_state = update_online_state(
                teacher_state,
                original_positive,
                memory[original_positive],
                memory,
                step_id=t,
                max_raw=max_raw,
                max_chars_per_item=max_chars,
            )

    stats["qids"] = len(source_by_qid)
    stats["canonical_rows"] = sum(len(rows) for rows in source_by_qid.values())
    stats["written_rows"] = write_jsonl(output_path, output_rows)
    stats["runtime_checkpoint"] = str((rollout_report.get("summary") or {}).get("checkpoint") or "")
    return dict(stats)


def build(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace)
    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"output exists; pass --force: {output_root}")
        shutil.rmtree(output_root)

    teacher_root = Path(args.teacher_root)
    split_stats = {}
    for split in SPLITS:
        split_stats[split] = build_split(
            split,
            workspace / "canonical" / f"{split}.jsonl",
            workspace / "rollouts" / f"{split}.json",
            teacher_root / "unit_registry" / f"raw_units_{split}.jsonl",
            output_root / "samples" / f"{split}.jsonl",
            args.max_raw,
            args.max_chars,
        )

    manifest = {
        "dataset": "KBS v25 rollout-aligned full-state ranking data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "teacher_root": str(teacher_root),
        "output_root": str(output_root),
        "source_checkpoint": args.source_checkpoint,
        "source_checkpoint_sha256": (
            hashlib.sha256(Path(args.source_checkpoint).read_bytes()).hexdigest()
            if args.hash_checkpoint
            else None
        ),
        "candidate_top_k": 10,
        "select_top_k": 5,
        "state_update_top_k": 1,
        "hybrid_alpha": 0.5,
        "policy_blend_weight": 0.5,
        "front_pool_k": 30,
        "front_fusion": "rrf",
        "local_expansion_window": 1,
        "mmr_lambda": 0.7,
        "mmr_same_doc_similarity": 0.35,
        "online_state_max_raw": args.max_raw,
        "online_state_max_chars": args.max_chars,
        "train_state_mixture": "t0 teacher once; t>=1 teacher_online plus frozen_v22_rollout",
        "validation_state_distribution": "frozen_v22_rollout_only",
        "auxiliary_labels_masked": True,
        "split_stats": split_stats,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("canonicalize", "build"))
    parser.add_argument(
        "--source-root",
        default="data/hotpotqa_distractor_v22_state_focused",
    )
    parser.add_argument(
        "--teacher-root",
        default="data/hotpotqa_distractor_v7_10k_llm_prestep",
    )
    parser.add_argument(
        "--workspace",
        default="outputs/analysis/kbs_v25_rollout_workspace",
    )
    parser.add_argument(
        "--output-root",
        default="data/hotpotqa_distractor_v25_rollout_aligned",
    )
    parser.add_argument(
        "--source-checkpoint",
        default="outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt",
    )
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--max-raw", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=260)
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "canonicalize":
        canonicalize(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
