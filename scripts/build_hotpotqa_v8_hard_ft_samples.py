#!/usr/bin/env python3
"""Build a hard-finetuning sample set by oversampling difficult train rows.

This is intentionally conservative: it does not relabel data, does not call an
LLM, and keeps val/test unchanged. It only duplicates train samples that match
the hard patterns found in v7 error analysis: t=1, bridge positives, large
candidate sets, same-role/distractor-heavy candidate lists.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPLITS = ("train", "val", "test")
ROLE_SET = {"bridge", "support", "distinguish"}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_doc_roles(targets_path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for row in read_jsonl(targets_path):
        qid = str(row.get("qid", ""))
        roles = out.setdefault(qid, {})
        for unit in row.get("T_q_raw") or []:
            if not isinstance(unit, dict):
                continue
            doc_id = str(unit.get("doc_id") or "").strip()
            role = str(unit.get("primary_role") or "").strip()
            if doc_id and role in ROLE_SET:
                roles[doc_id] = role
    return out


def candidate_doc(sample: dict, unit_id: str) -> str:
    provenance = ((sample.get("candidates") or {}).get("candidate_provenance") or {}).get(unit_id) or {}
    return str(provenance.get("doc_id") or "")


def positive_role(sample: dict, doc_roles: Dict[str, Dict[str, str]]) -> str:
    qid = str(sample.get("qid", ""))
    positive = str(((sample.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id") or "")
    doc_id = candidate_doc(sample, positive)
    return doc_roles.get(qid, {}).get(doc_id, "unlabeled")


def candidate_role_counts(sample: dict, doc_roles: Dict[str, Dict[str, str]]) -> Counter:
    qid = str(sample.get("qid", ""))
    roles = Counter()
    for unit_id in ((sample.get("candidates") or {}).get("C_t") or []):
        doc_id = candidate_doc(sample, str(unit_id))
        roles[doc_roles.get(qid, {}).get(doc_id, "unlabeled")] += 1
    return roles


def has_same_doc_negative(sample: dict) -> bool:
    ranking = (sample.get("labels") or {}).get("ranking_label") or {}
    positive = str(ranking.get("positive_unit_id") or "")
    positive_doc = candidate_doc(sample, positive)
    if not positive_doc:
        return False
    for unit_id in ranking.get("negative_unit_ids") or []:
        if candidate_doc(sample, str(unit_id)) == positive_doc:
            return True
    return False


def hard_score(sample: dict, doc_roles: Dict[str, Dict[str, str]]) -> Tuple[int, List[str], str]:
    t = int(sample.get("t", -1))
    c_t = list(((sample.get("candidates") or {}).get("C_t") or []))
    count = len(c_t)
    role = positive_role(sample, doc_roles)
    role_counts = candidate_role_counts(sample, doc_roles)
    score = 0
    reasons: List[str] = []

    if t == 1:
        score += 4
        reasons.append("t1_second_step")
    if role == "bridge":
        score += 4
        reasons.append("bridge_positive")
    elif role == "support":
        score += 1
        reasons.append("support_positive")
    elif role == "distinguish":
        score += 2
        reasons.append("distinguish_positive")
    if count >= 10:
        score += 4
        reasons.append("candidate_count_ge10")
    elif count == 9:
        score += 3
        reasons.append("candidate_count_9")
    elif count == 8:
        score += 1
        reasons.append("candidate_count_8")
    if role in ROLE_SET and role_counts[role] >= 2:
        score += 3
        reasons.append("same_role_candidates")
    if role_counts["unlabeled"] >= 3:
        score += 2
        reasons.append("many_unlabeled_distractors")
    if has_same_doc_negative(sample):
        score += 3
        reasons.append("same_doc_negative")

    return score, reasons, role


def mark_duplicate(sample: dict, *, score: int, reasons: List[str], role: str, dup_index: int) -> dict:
    row = json.loads(json.dumps(sample, ensure_ascii=False))
    meta = row.setdefault("meta", {})
    meta["hard_ft"] = {
        "source": "build_hotpotqa_v8_hard_ft_samples.py",
        "oversample_duplicate": True,
        "duplicate_index": dup_index,
        "hard_score": score,
        "reasons": reasons,
        "positive_role": role,
    }
    return row


def build_train_records(
    *,
    source_train_path: Path,
    doc_roles: Dict[str, Dict[str, str]],
    min_score: int,
    oversample_factor: int,
    max_hard_rows: int,
) -> Tuple[List[dict], List[dict], Counter]:
    original_rows = list(read_jsonl(source_train_path))
    scored: List[Tuple[int, List[str], str, dict]] = []
    stats = Counter()
    for row in original_rows:
        score, reasons, role = hard_score(row, doc_roles)
        stats["original_rows"] += 1
        stats[f"role::{role}"] += 1
        stats[f"t::{row.get('t')}"] += 1
        stats[f"candidate_count::{len(((row.get('candidates') or {}).get('C_t') or []))}"] += 1
        if score >= min_score:
            scored.append((score, reasons, role, row))

    scored.sort(
        key=lambda item: (
            -item[0],
            str(item[3].get("qid", "")),
            int(item[3].get("t", -1)),
        )
    )
    if max_hard_rows > 0:
        scored = scored[:max_hard_rows]

    hard_rows = [item[3] for item in scored]
    output_rows = list(original_rows)
    for score, reasons, role, row in scored:
        stats["selected_hard_rows"] += 1
        stats[f"hard_role::{role}"] += 1
        for reason in reasons:
            stats[f"hard_reason::{reason}"] += 1
        for dup_index in range(1, oversample_factor + 1):
            output_rows.append(
                mark_duplicate(row, score=score, reasons=reasons, role=role, dup_index=dup_index)
            )
            stats["oversampled_rows"] += 1
    stats["final_train_rows"] = len(output_rows)
    return output_rows, hard_rows, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v8 hard-finetuning samples by oversampling difficult train rows.")
    parser.add_argument("--source-root", default="data/hotpotqa_distractor_v7_10k_llm_prestep")
    parser.add_argument("--output-root", default="data/hotpotqa_distractor_v8_hard_ft")
    parser.add_argument("--min-score", type=int, default=10)
    parser.add_argument("--oversample-factor", type=int, default=2)
    parser.add_argument("--max-hard-rows", type=int, default=5000)
    parser.add_argument("--summary-output", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    samples_out = output_root / "samples"
    samples_out.mkdir(parents=True, exist_ok=True)

    for subdir in ("unit_registry", "targets"):
        src = source_root / subdir
        if src.exists():
            copy_tree(src, output_root / subdir)

    doc_roles = load_doc_roles(source_root / "targets" / "train.jsonl")
    train_rows, hard_rows, stats = build_train_records(
        source_train_path=source_root / "samples" / "train.jsonl",
        doc_roles=doc_roles,
        min_score=args.min_score,
        oversample_factor=max(args.oversample_factor, 0),
        max_hard_rows=args.max_hard_rows,
    )
    train_count = write_jsonl(samples_out / "train.jsonl", train_rows)
    hard_count = write_jsonl(output_root / "debug" / "hard_train_selected.jsonl", hard_rows)

    split_counts = {"train": train_count}
    for split in ("val", "test"):
        src = source_root / "samples" / f"{split}.jsonl"
        dst = samples_out / f"{split}.jsonl"
        shutil.copy2(src, dst)
        split_counts[split] = sum(1 for _ in read_jsonl(dst))

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "params": {
            "min_score": args.min_score,
            "oversample_factor": args.oversample_factor,
            "max_hard_rows": args.max_hard_rows,
        },
        "samples": split_counts,
        "hard_selected_rows": hard_count,
        "stats": dict(stats),
    }
    summary_path = Path(args.summary_output) if args.summary_output else output_root / "debug" / "hard_ft_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("HOTPOTQA V8 HARD-FT SAMPLES")
    print(f"source_root: {source_root}")
    print(f"output_root: {output_root}")
    print(f"samples: {split_counts}")
    print(f"hard_selected_rows: {hard_count}")
    print(f"oversampled_rows: {stats['oversampled_rows']}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
