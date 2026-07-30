#!/usr/bin/env python3
"""Audit v25 rollout-aligned data before training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.online_state import render_online_k_t


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


def load_memory(path: Path) -> dict[str, dict]:
    memory = {}
    for row in read_jsonl(path):
        unit_id = str(row.get("unit_id") or "")
        title = str(row.get("title") or row.get("doc_id") or "")
        if unit_id:
            memory[unit_id] = {
                **row,
                "title": title,
                "doc_id": str(row.get("doc_id") or title),
            }
    return memory


def positive_id(row: dict) -> str:
    return str(
        (((row.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id"))
        or ""
    )


def candidate_ids(row: dict) -> list[str]:
    return [str(unit_id) for unit_id in ((row.get("candidates") or {}).get("C_t") or [])]


def h_ids(row: dict) -> list[str]:
    values = []
    for item in ((row.get("state") or {}).get("H_t") or []):
        if isinstance(item, dict):
            values.append(str(item.get("unit_id") or ""))
        else:
            values.append(str(item))
    return [value for value in values if value]


def inspect_split(
    split: str,
    samples_path: Path,
    memory_path: Path,
    max_raw: int,
    max_chars: int,
) -> tuple[dict, set[str]]:
    memory = load_memory(memory_path)
    qids = set()
    state_keys = set()
    pools = {}
    source_counts = Counter()
    t_counts = Counter()
    errors = []
    rows = 0
    renderer_matches = 0
    rollout_positive_already_acquired = 0

    for row in read_jsonl(samples_path):
        rows += 1
        qid = str(row.get("qid") or "")
        t = int(row.get("t", -1))
        source = str((row.get("build_meta") or {}).get("state_source") or "")
        key = (qid, t, source)
        pool = candidate_ids(row)
        positive = positive_id(row)
        state = row.get("state") or {}
        state_h_ids = h_ids(row)
        qids.add(qid)
        source_counts[source] += 1
        t_counts[str(t)] += 1

        if key in state_keys:
            errors.append(f"duplicate (qid,t,state_source): {key}")
        state_keys.add(key)
        if not pool or len(pool) != len(set(pool)):
            errors.append(f"invalid pool: qid={qid}, t={t}, source={source}")
        if positive not in pool:
            errors.append(f"positive missing from pool: qid={qid}, t={t}, source={source}")
        if qid in pools and pools[qid] != tuple(pool):
            errors.append(f"pool changes within qid={qid}")
        pools[qid] = tuple(pool)
        if any(unit_id not in memory for unit_id in pool):
            errors.append(f"pool unit missing from memory: qid={qid}, t={t}")
        rendered = render_online_k_t(
            state,
            memory,
            max_raw=max_raw,
            max_chars_per_item=max_chars,
        )
        if rendered == str(state.get("K_t") or ""):
            renderer_matches += 1
        else:
            errors.append(f"renderer mismatch: qid={qid}, t={t}, source={source}")
        if source == "frozen_v22_rollout" and positive in set(state_h_ids):
            rollout_positive_already_acquired += 1
            errors.append(f"rollout target already acquired: qid={qid}, t={t}, positive={positive}")
        if not bool((row.get("build_meta") or {}).get("mask_auxiliary_labels", False)):
            errors.append(f"auxiliary labels not masked: qid={qid}, t={t}")

    if split == "train":
        if source_counts["teacher_online"] == 0 or source_counts["frozen_v22_rollout"] == 0:
            errors.append(f"train split lacks required state mixture: {dict(source_counts)}")
    elif set(source_counts) != {"frozen_v22_rollout"}:
        errors.append(f"{split} must be rollout-only: {dict(source_counts)}")

    return (
        {
            "path": str(samples_path),
            "rows": rows,
            "qids": len(qids),
            "state_sources": dict(source_counts),
            "t_distribution": dict(t_counts),
            "renderer_matches": renderer_matches,
            "rollout_positive_already_acquired": rollout_positive_already_acquired,
            "error_count": len(errors),
            "errors": errors[:30],
        },
        qids,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/hotpotqa_distractor_v25_rollout_aligned",
    )
    parser.add_argument(
        "--workspace",
        default="outputs/analysis/kbs_v25_rollout_workspace",
    )
    parser.add_argument(
        "--teacher-root",
        default="data/hotpotqa_distractor_v7_10k_llm_prestep",
    )
    parser.add_argument(
        "--source-checkpoint",
        default="outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt",
    )
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--dense-model", default="models/bge-large-en-v1.5")
    parser.add_argument(
        "--eval-queries",
        default="data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl",
    )
    parser.add_argument("--require-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    workspace = Path(args.workspace)
    teacher_root = Path(args.teacher_root)
    manifest_path = data_root / "manifest.json"
    required = [
        manifest_path,
        workspace / "canonical_manifest.json",
        *(workspace / "canonical" / f"{split}.jsonl" for split in SPLITS),
        *(workspace / "rollouts" / f"{split}.json" for split in SPLITS),
        *(data_root / "samples" / f"{split}.jsonl" for split in SPLITS),
    ]
    if args.require_paths:
        required.extend(
            [
                Path(args.source_checkpoint),
                Path(args.model_dir),
                Path(args.dense_model),
                *(teacher_root / "unit_registry" / f"raw_units_{split}.jsonl"
                  for split in SPLITS),
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(json.dumps({"status": "MISSING", "missing_paths": missing}, indent=2))
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = {}
    qid_sets = {}
    failures = []
    max_raw = int(manifest.get("online_state_max_raw", 8))
    max_chars = int(manifest.get("online_state_max_chars", 260))
    for split in SPLITS:
        reports[split], qid_sets[split] = inspect_split(
            split,
            data_root / "samples" / f"{split}.jsonl",
            teacher_root / "unit_registry" / f"raw_units_{split}.jsonl",
            max_raw,
            max_chars,
        )
        if reports[split]["error_count"]:
            failures.append(f"{split}: {reports[split]['error_count']} data errors")

    overlaps = {
        "train_val": len(qid_sets["train"] & qid_sets["val"]),
        "train_test": len(qid_sets["train"] & qid_sets["test"]),
        "val_test": len(qid_sets["val"] & qid_sets["test"]),
    }
    if any(overlaps.values()):
        failures.append(f"split qid overlap: {overlaps}")

    evaluation_overlap = None
    eval_path = Path(args.eval_queries)
    if eval_path.is_file():
        eval_qids = {str(row.get("qid") or "") for row in read_jsonl(eval_path)}
        evaluation_overlap = {
            "eval_qids": len(eval_qids),
            "train_eval": len(qid_sets["train"] & eval_qids),
            "val_eval": len(qid_sets["val"] & eval_qids),
            "test_eval": len(qid_sets["test"] & eval_qids),
        }
        if evaluation_overlap["train_eval"]:
            failures.append(f"training/evaluation leakage: {evaluation_overlap}")

    frozen = {
        "candidate_top_k": 10,
        "select_top_k": 5,
        "state_update_top_k": 1,
        "policy_blend_weight": 0.5,
        "online_state_max_raw": 8,
        "online_state_max_chars": 260,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in frozen.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        failures.append(f"manifest protocol mismatch: {mismatches}")
    if manifest.get("source_checkpoint") != args.source_checkpoint:
        failures.append(
            "manifest source checkpoint mismatch: "
            f"{manifest.get('source_checkpoint')} != {args.source_checkpoint}"
        )

    rollout_protocol = {}
    for split in SPLITS:
        report = json.loads((workspace / "rollouts" / f"{split}.json").read_text(encoding="utf-8"))
        summary = report.get("summary") or {}
        rollout_protocol[split] = {
            "qids": summary.get("qids"),
            "checkpoint": summary.get("checkpoint"),
            "selector": summary.get("selector"),
            "policy_context_source": summary.get("policy_context_source"),
            "state_update_top_k": summary.get("state_update_top_k"),
            "save_online_states": summary.get("save_online_states"),
            "answer_judged": summary.get("answer_judged"),
        }
        expected_values = {
            "checkpoint": args.source_checkpoint,
            "selector": "hybrid_policy",
            "policy_context_source": "online_state",
            "state_update_top_k": 1,
            "save_online_states": True,
            "answer_judged": 0,
        }
        for key, expected in expected_values.items():
            if summary.get(key) != expected:
                failures.append(
                    f"{split} rollout {key} mismatch: {summary.get(key)!r} != {expected!r}"
                )

    result = {
        "status": "OK" if not failures else "FAILED",
        "data_root": str(data_root),
        "workspace": str(workspace),
        "manifest_protocol": {key: manifest.get(key) for key in frozen},
        "splits": reports,
        "qid_overlap": overlaps,
        "evaluation_overlap": evaluation_overlap,
        "rollout_protocol": rollout_protocol,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
