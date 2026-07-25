#!/usr/bin/env python3
"""Validate v22 fixed-pool state-focused data and training prerequisites."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


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


def candidate_ids(row: dict):
    candidates = row.get("candidates") or {}
    return list(candidates.get("C_t") or [])


def positive_id(row: dict):
    labels = row.get("labels") or {}
    return str((labels.get("ranking_label") or {}).get("positive_unit_id") or "")


def inspect_split(path: Path) -> dict:
    pools = {}
    states_by_qid = defaultdict(dict)
    t_counts = Counter()
    rows = 0
    duplicate_state_rows = 0
    errors = []
    qids = set()

    for row in read_jsonl(path):
        rows += 1
        qid = str(row.get("qid") or "")
        t = int(row.get("t", -1))
        repetition = int((row.get("build_meta") or {}).get("repetition", 0))
        pool = tuple(candidate_ids(row))
        positive = positive_id(row)
        qids.add(qid)
        t_counts[str(t)] += 1
        if not qid or t < 0:
            errors.append(f"invalid qid/t at row {rows}")
        if not pool or len(pool) != len(set(pool)):
            errors.append(f"invalid candidate pool: qid={qid}, t={t}")
        if positive not in pool:
            errors.append(f"positive missing from pool: qid={qid}, t={t}")
        if not bool((row.get("build_meta") or {}).get("mask_auxiliary_labels", False)):
            errors.append(f"synthetic auxiliary labels are not masked: qid={qid}, t={t}")
        if qid in pools and pools[qid] != pool:
            errors.append(f"candidate pool changed across prefixes: qid={qid}")
        pools[qid] = pool
        state_key = (t, repetition)
        if state_key in states_by_qid[qid]:
            duplicate_state_rows += 1
        states_by_qid[qid][state_key] = positive

    adjacent_label_switches = 0
    adjacent_pairs = 0
    prior_positive_negative_states = 0
    for qid, states in states_by_qid.items():
        first_by_t = {}
        for (t, _), positive in sorted(states.items()):
            first_by_t.setdefault(t, positive)
        for t in sorted(first_by_t):
            if t > 0:
                prior_positive_negative_states += 1
            if t + 1 in first_by_t:
                adjacent_pairs += 1
                if first_by_t[t] != first_by_t[t + 1]:
                    adjacent_label_switches += 1

    return {
        "path": str(path),
        "rows": rows,
        "qids": len(qids),
        "t_distribution": dict(t_counts),
        "deep_state_rows": sum(count for t, count in t_counts.items() if int(t) >= 2),
        "adjacent_pairs": adjacent_pairs,
        "adjacent_label_switches": adjacent_label_switches,
        "prior_positive_negative_states": prior_positive_negative_states,
        "duplicate_state_rows": duplicate_state_rows,
        "errors": errors[:20],
        "error_count": len(errors),
        "_qid_set": qids,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/hotpotqa_distractor_v22_state_focused",
    )
    parser.add_argument(
        "--teacher-root",
        default="data/hotpotqa_distractor_v7_10k_llm_prestep",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt",
    )
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument(
        "--eval-queries",
        default="data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl",
    )
    parser.add_argument("--require-paths", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_path = data_root / "manifest.json"
    required_paths = [
        manifest_path,
        *(data_root / "samples" / f"{split}.jsonl" for split in ("train", "val", "test")),
    ]
    if args.require_paths:
        required_paths.extend(
            [
                Path(args.init_checkpoint),
                Path(args.model_dir),
                *(Path(args.teacher_root) / "unit_registry" / f"raw_units_{split}.jsonl"
                  for split in ("train", "val", "test")),
                *(Path(args.teacher_root) / "targets" / f"{split}.jsonl"
                  for split in ("train", "val", "test")),
            ]
        )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        print(json.dumps({"status": "MISSING", "paths": missing}, indent=2))
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = {
        split: inspect_split(data_root / "samples" / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    train_qids = reports["train"].pop("_qid_set")
    val_qids = reports["val"].pop("_qid_set")
    test_qids = reports["test"].pop("_qid_set")
    overlap = {
        "train_val": len(train_qids & val_qids),
        "train_test": len(train_qids & test_qids),
        "val_test": len(val_qids & test_qids),
    }
    eval_overlap = None
    eval_queries_path = Path(args.eval_queries)
    if eval_queries_path.is_file():
        eval_qids = {str(row.get("qid") or "") for row in read_jsonl(eval_queries_path)}
        eval_overlap = {
            "eval_qids": len(eval_qids),
            "train_eval": len(train_qids & eval_qids),
            "val_eval": len(val_qids & eval_qids),
            "test_eval": len(test_qids & eval_qids),
        }

    failures = []
    for split, report in reports.items():
        if report["error_count"]:
            failures.append(f"{split}: {report['error_count']} row validation errors")
        if report["adjacent_pairs"] != report["adjacent_label_switches"]:
            failures.append(f"{split}: adjacent state labels do not always switch")
    if reports["train"]["deep_state_rows"] == 0:
        failures.append("train split has no t>=2 state")
    if any(overlap.values()):
        failures.append(f"split qid overlap detected: {overlap}")
    if eval_overlap and eval_overlap["train_eval"]:
        failures.append(f"training/evaluation qid overlap detected: {eval_overlap}")
    if not manifest.get("auxiliary_labels_masked"):
        failures.append("manifest does not declare masked auxiliary labels")

    result = {
        "status": "OK" if not failures else "FAILED",
        "data_root": str(data_root),
        "manifest": {
            "candidate_top_k": manifest.get("candidate_top_k"),
            "deep_repeat": manifest.get("deep_repeat"),
            "seed": manifest.get("seed"),
            "auxiliary_labels_masked": manifest.get("auxiliary_labels_masked"),
        },
        "splits": reports,
        "qid_overlap": overlap,
        "evaluation_overlap": eval_overlap,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
