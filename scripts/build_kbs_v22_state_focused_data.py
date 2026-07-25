#!/usr/bin/env python3
"""Build fixed-pool, multi-prefix ranking data for state-focused v22 training."""

import argparse
import gzip
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROLE_PRIORITY = {
    "bridge": 0,
    "distinguish": 1,
    "disambiguation": 1,
    "support": 2,
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
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


def stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def normalize_text(text: object) -> str:
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def lexical_overlap(question: str, text: str) -> Tuple[int, float]:
    q_tokens = set(TOKEN_RE.findall(question.lower()))
    t_tokens = set(TOKEN_RE.findall(text.lower()))
    if not q_tokens or not t_tokens:
        return 0, 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap, overlap / len(q_tokens | t_tokens)


def load_memory(path: Path) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    by_id: Dict[str, dict] = {}
    by_qid: Dict[str, List[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        unit_id = str(row["unit_id"])
        qid = str(row.get("qid") or unit_id.split("::", 1)[0])
        title = str(row.get("title") or row.get("doc_id") or "")
        sent_id = row.get("sent_id")
        if sent_id is None:
            sent_id = int(unit_id.rsplit("::", 1)[-1])
        item = {
            "qid": qid,
            "unit_id": unit_id,
            "title": title,
            "sent_id": int(sent_id),
            "text": str(row["text"]).strip(),
            "parent_chunk_id": str(row.get("parent_chunk_id") or f"{qid}::{title}"),
        }
        if unit_id in by_id:
            raise ValueError(f"duplicate memory unit: {unit_id}")
        by_id[unit_id] = item
        by_qid[qid].append(item)
    return by_id, by_qid


def load_target_roles(path: Path) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for row in read_jsonl(path):
        qid = str(row["qid"])
        roles: Dict[str, str] = {}
        for target in row.get("T_q_raw") or []:
            title = str(target.get("doc_id") or "").strip()
            role = str(target.get("primary_role") or "support").strip()
            if role == "disambiguation":
                role = "distinguish"
            if title:
                roles[title] = role
        result[qid] = roles
    return result


def resolve_gold_units(
    source_row: dict,
    memory_by_id: Dict[str, dict],
    target_roles: Dict[str, Dict[str, str]],
) -> List[dict]:
    qid = str(source_row["qid"])
    roles = target_roles.get(qid, {})
    resolved = []
    seen = set()
    for source_index, fact in enumerate(source_row.get("supporting_facts") or []):
        if not isinstance(fact, (list, tuple)) or len(fact) < 2:
            continue
        title = str(fact[0])
        sent_id = int(fact[1])
        unit_id = f"{qid}::{title}::{sent_id}"
        if unit_id in seen:
            continue
        item = memory_by_id.get(unit_id)
        if item is None:
            raise ValueError(f"supporting fact missing from memory: qid={qid}, unit={unit_id}")
        seen.add(unit_id)
        role = roles.get(title, "support")
        resolved.append(
            {
                **item,
                "role": role,
                "source_index": source_index,
            }
        )
    if not resolved:
        raise ValueError(f"question has no resolved supporting facts: qid={qid}")
    resolved.sort(
        key=lambda item: (
            ROLE_PRIORITY.get(item["role"], 3),
            item["source_index"],
            item["title"],
            item["sent_id"],
        )
    )
    return resolved


def select_fixed_pool(
    qid: str,
    question: str,
    answer: str,
    memory_items: Sequence[dict],
    gold_units: Sequence[dict],
    candidate_top_k: int,
    seed: int,
) -> List[str]:
    gold_ids = {item["unit_id"] for item in gold_units}
    gold_titles = {item["title"] for item in gold_units}
    answer_norm = normalize_text(answer)
    distractors = []
    for item in memory_items:
        if item["unit_id"] in gold_ids:
            continue
        overlap, jaccard = lexical_overlap(question, item["title"] + " " + item["text"])
        text_norm = normalize_text(item["text"])
        answer_visible = int(bool(answer_norm) and answer_norm in text_norm)
        same_gold_doc = int(item["title"] in gold_titles)
        score = (
            same_gold_doc,
            answer_visible,
            overlap,
            jaccard,
            stable_int(seed, qid, item["unit_id"]),
        )
        distractors.append((score, item["unit_id"]))
    distractors.sort(key=lambda pair: pair[0], reverse=True)

    pool = [item["unit_id"] for item in gold_units]
    target_size = max(candidate_top_k, len(pool))
    pool.extend(unit_id for _, unit_id in distractors[: max(0, target_size - len(pool))])
    rng = random.Random(stable_int(seed, qid, "fixed-pool"))
    rng.shuffle(pool)
    return pool


def render_k_t(prefix: Sequence[dict]) -> str:
    if not prefix:
        return ""
    lines = ["Evidence:"]
    for index, item in enumerate(prefix, start=1):
        lines.append(f"[{index}] {item['title']}: {item['text']}")
    return "\n".join(lines)


def build_progress(prefix: Sequence[dict], gold_units: Sequence[dict]) -> dict:
    counts = Counter(item["role"] for item in prefix)
    covered = [item["unit_id"] for item in prefix]
    return {
        "covered_target_ids": covered,
        "k_bridge": float(counts.get("bridge", 0)),
        "k_distinguish": float(counts.get("distinguish", 0)),
        "k_support": float(counts.get("support", 0)),
        "coverage_trace": {
            item["unit_id"]: {
                "role": item["role"],
                "covered": item["unit_id"] in set(covered),
            }
            for item in gold_units
        },
    }


def provenance(item: dict) -> dict:
    return {
        "chunk_id": item["parent_chunk_id"],
        "doc_id": item["title"],
        "parent_chunk_id": item["parent_chunk_id"],
    }


def build_state_row(
    source_row: dict,
    memory_by_id: Dict[str, dict],
    gold_units: Sequence[dict],
    pool: Sequence[str],
    t: int,
    split: str,
    repetition: int,
    seed: int,
) -> dict:
    qid = str(source_row["qid"])
    prefix = list(gold_units[:t])
    positive = gold_units[t]
    pool_hash = hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest()[:16]
    h_t = [
        {
            "step_id": index,
            "unit_id": item["unit_id"],
            "chunk_id": item["parent_chunk_id"],
            "doc_id": item["title"],
            "parent_chunk_id": item["parent_chunk_id"],
        }
        for index, item in enumerate(prefix)
    ]
    raw_refs = [
        {
            **entry,
            "added_step": entry["step_id"],
            "used_in_summary_count": 0,
            "selected_count": 1,
        }
        for entry in h_t
    ]
    candidate_provenance = {
        unit_id: provenance(memory_by_id[unit_id])
        for unit_id in pool
    }
    negatives = [unit_id for unit_id in pool if unit_id != positive["unit_id"]]
    return {
        "qid": qid,
        "t": t,
        "build_meta": {
            "run_id": f"kbs_v22_state_focused_seed{seed}",
            "source": "build_kbs_v22_state_focused_data.py",
            "split": split,
            "variant": "gold_prefix_fixed_pool",
            "pair_group_id": qid,
            "fixed_pool_id": pool_hash,
            "repetition": repetition,
            "mask_auxiliary_labels": True,
        },
        "question": str(source_row["question"]),
        "state": {
            "H_t": h_t,
            "A_t": build_progress(prefix, gold_units),
            "S_t": {
                "raw_refs": raw_refs,
                "derived_refs": [],
                "last_added_unit_id": prefix[-1]["unit_id"] if prefix else None,
                "last_updated_step": t - 1 if prefix else None,
            },
            "K_t": render_k_t(prefix),
        },
        "candidates": {
            "R_t": list(pool),
            "G_t_final": [],
            "G_t_aux": [],
            "G_t_illegal": [],
            "C_t": list(pool),
            "candidate_provenance": candidate_provenance,
            "aux_candidate_provenance": {},
        },
        "labels": {
            "u_t_plus": {
                "step_id": t,
                "unit_id": positive["unit_id"],
                **provenance(positive),
            },
            "ranking_label": {
                "positive_unit_id": positive["unit_id"],
                "negative_unit_ids": negatives,
                "positive_provenance": provenance(positive),
                "negative_provenance": {
                    unit_id: candidate_provenance[unit_id]
                    for unit_id in negatives
                },
            },
            "stop_label": {
                "should_stop": False,
                "label_type": "continue",
            },
        },
        "derived_payloads": {},
        "meta": {
            "question_type": str(source_row.get("type") or "unknown"),
            "level": str(source_row.get("level") or "unknown"),
            "gold_prefix_length": t,
            "gold_trajectory_length": len(gold_units),
            "counterfactual_fixed_pool": True,
        },
    }


def build_split(
    split: str,
    source_path: Path,
    memory_path: Path,
    targets_path: Path,
    output_path: Path,
    candidate_top_k: int,
    deep_repeat: int,
    seed: int,
    max_qids: int,
) -> dict:
    memory_by_id, memory_by_qid = load_memory(memory_path)
    roles = load_target_roles(targets_path)
    source_rows = list(read_jsonl(source_path))
    if max_qids > 0:
        source_rows = source_rows[:max_qids]

    stats = {
        "qids": 0,
        "base_states": 0,
        "written_rows": 0,
        "t_distribution": Counter(),
        "written_t_distribution": Counter(),
        "trajectory_lengths": Counter(),
        "question_types": Counter(),
        "fixed_pool_violations": 0,
        "positive_missing_from_pool": 0,
        "prior_positive_negative_states": 0,
        "masked_auxiliary_rows": 0,
    }

    def rows() -> Iterable[dict]:
        for source_row in source_rows:
            qid = str(source_row["qid"])
            if qid not in memory_by_qid:
                raise ValueError(f"question missing from memory: split={split}, qid={qid}")
            gold_units = resolve_gold_units(source_row, memory_by_id, roles)
            pool = select_fixed_pool(
                qid=qid,
                question=str(source_row["question"]),
                answer=str(source_row.get("answer") or ""),
                memory_items=memory_by_qid[qid],
                gold_units=gold_units,
                candidate_top_k=candidate_top_k,
                seed=seed,
            )
            stats["qids"] += 1
            stats["trajectory_lengths"][str(len(gold_units))] += 1
            stats["question_types"][str(source_row.get("type") or "unknown")] += 1
            for t in range(len(gold_units)):
                stats["base_states"] += 1
                stats["t_distribution"][str(t)] += 1
                repeat_count = deep_repeat if split == "train" and t >= 2 else 1
                for repetition in range(repeat_count):
                    row = build_state_row(
                        source_row=source_row,
                        memory_by_id=memory_by_id,
                        gold_units=gold_units,
                        pool=pool,
                        t=t,
                        split=split,
                        repetition=repetition,
                        seed=seed,
                    )
                    stats["written_rows"] += 1
                    stats["written_t_distribution"][str(t)] += 1
                    stats["masked_auxiliary_rows"] += 1
                    if t > 0:
                        stats["prior_positive_negative_states"] += 1
                    yield row

    written = write_jsonl(output_path, rows())
    if written != stats["written_rows"]:
        raise RuntimeError(f"write count mismatch: split={split}, {written} != {stats['written_rows']}")
    return {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in stats.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="data_packages/hotpotqa_v22_source_metadata",
    )
    parser.add_argument(
        "--teacher-root",
        default="data/hotpotqa_distractor_v7_10k_llm_prestep",
    )
    parser.add_argument(
        "--output-root",
        default="data/hotpotqa_distractor_v22_state_focused",
    )
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--deep-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def find_source_path(source_root: Path, split: str) -> Path:
    candidates = (
        source_root / f"{split}.jsonl",
        source_root / f"{split}.jsonl.gz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing source metadata for split={split}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def main() -> None:
    args = parse_args()
    if args.candidate_top_k < 2:
        raise ValueError("--candidate-top-k must be >= 2")
    if args.deep_repeat < 1:
        raise ValueError("--deep-repeat must be >= 1")

    source_root = Path(args.source_root)
    teacher_root = Path(args.teacher_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"output exists; pass --force to rebuild: {output_root}")
        shutil.rmtree(output_root)

    split_stats = {}
    for split in ("train", "val", "test"):
        source_path = find_source_path(source_root, split)
        memory_path = teacher_root / "unit_registry" / f"raw_units_{split}.jsonl"
        targets_path = teacher_root / "targets" / f"{split}.jsonl"
        for required in (source_path, memory_path, targets_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        split_stats[split] = build_split(
            split=split,
            source_path=source_path,
            memory_path=memory_path,
            targets_path=targets_path,
            output_path=output_root / "samples" / f"{split}.jsonl",
            candidate_top_k=args.candidate_top_k,
            deep_repeat=args.deep_repeat,
            seed=args.seed,
            max_qids=args.max_qids,
        )

    manifest = {
        "dataset": "KBS v22 state-focused fixed-pool prefixes",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "teacher_root": str(teacher_root),
        "output_root": str(output_root),
        "candidate_top_k": args.candidate_top_k,
        "deep_repeat": args.deep_repeat,
        "seed": args.seed,
        "max_qids": args.max_qids,
        "auxiliary_labels_masked": True,
        "selection_supervision": "gold supporting-fact prefix with fixed candidate pool",
        "split_stats": split_stats,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
