#!/usr/bin/env python3
"""Build a v27-matched previous-evidence-only Anchor training set."""

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
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
    return [str(value) for value in ((row.get("candidates") or {}).get("C_t") or [])]


def positive_id(row: dict):
    ranking = (row.get("labels") or {}).get("ranking_label") or {}
    return str(ranking.get("positive_unit_id") or "")


def history_ids(row: dict):
    values = []
    for item in ((row.get("state") or {}).get("H_t") or []):
        if isinstance(item, dict) and item.get("unit_id"):
            values.append(str(item["unit_id"]))
        elif isinstance(item, str):
            values.append(item)
    return values


def visible_acquired_ids(row: dict):
    pool = candidate_ids(row)
    positive = positive_id(row)
    history = history_ids(row)
    if not history:
        return []
    previous = history[-1]
    return [previous] if previous in pool and previous != positive else []


def full_acquired_ids(row: dict):
    pool = set(candidate_ids(row))
    positive = positive_id(row)
    return [value for value in history_ids(row) if value in pool and value != positive]


def build_split(source_path: Path, output_path: Path):
    stats = {
        "rows": 0,
        "qids": set(),
        "t_distribution": Counter(),
        "source_acquired_negative_pairs": 0,
        "visible_acquired_negative_pairs": 0,
        "hidden_history_pairs_removed": 0,
        "rows_with_visible_acquired_negative": 0,
        "rewritten_rows": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(source_path):
            qid = str(row.get("qid") or "")
            t = int(row.get("t", -1))
            if not qid or t < 0:
                raise ValueError(f"invalid state identity in {source_path}: qid={qid!r}, t={t}")
            if not positive_id(row):
                raise ValueError(f"missing positive label: qid={qid}, t={t}")

            source_acquired = full_acquired_ids(row)
            visible_acquired = visible_acquired_ids(row)
            output = copy.deepcopy(row)
            counterfactual = output.setdefault("labels", {}).setdefault(
                "counterfactual_ranking", {}
            )
            previous_value = [
                str(value)
                for value in (counterfactual.get("acquired_negative_unit_ids") or [])
            ]
            counterfactual["acquired_negative_unit_ids"] = visible_acquired
            counterfactual["acquired_negative_scope"] = "visible_previous_evidence_only"
            output["build_meta"] = {
                **(output.get("build_meta") or {}),
                "dataset_version": "v28_matched_previous_anchor",
                "variant": "v27_matched_previous_evidence_only",
                "source_dataset": str(source_path),
                "acquired_negative_scope": "visible_previous_evidence_only",
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

            stats["rows"] += 1
            stats["qids"].add(qid)
            stats["t_distribution"][str(t)] += 1
            stats["source_acquired_negative_pairs"] += len(source_acquired)
            stats["visible_acquired_negative_pairs"] += len(visible_acquired)
            stats["hidden_history_pairs_removed"] += len(source_acquired) - len(visible_acquired)
            stats["rows_with_visible_acquired_negative"] += int(bool(visible_acquired))
            stats["rewritten_rows"] += int(previous_value != visible_acquired)

    return {
        **stats,
        "qids": len(stats["qids"]),
        "t_distribution": dict(stats["t_distribution"]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="data/hotpotqa_distractor_v27_counterfactual_dual",
    )
    parser.add_argument(
        "--output-root",
        default="data/hotpotqa_distractor_v28_matched_anchor",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    staging_root = output_root.with_name(output_root.name + ".building")
    required = [source_root / "manifest.json"] + [
        source_root / "samples" / f"{split}.jsonl" for split in ("train", "val", "test")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing v27 source files: " + ", ".join(missing))
    if output_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output/staging directory: {output_root}, {staging_root}"
        )

    split_stats = {
        split: build_split(
            source_root / "samples" / f"{split}.jsonl",
            staging_root / "samples" / f"{split}.jsonl",
        )
        for split in ("train", "val", "test")
    }
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "dataset": "KBS v28 v27-matched previous-evidence-only Anchor data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "source_manifest_seed": source_manifest.get("seed"),
        "later_repeat": source_manifest.get("later_repeat"),
        "candidate_top_k": source_manifest.get("candidate_top_k"),
        "context_mode": "previous_evidence_only",
        "architecture": "dual_state_interaction",
        "training_objective": (
            "v27-matched ranking plus acquired-evidence reversal restricted "
            "to the previous evidence visible to the Anchor"
        ),
        "comparison_contract": {
            "unchanged": [
                "ordered rows",
                "qids and splits",
                "candidate pools",
                "ranking positives",
                "row repetitions",
                "initialization",
                "dual-state architecture",
                "optimizer and losses",
            ],
            "changed": [
                "policy context: full accumulated state -> previous evidence only",
                "acquired-negative scope: full history -> visible previous evidence only",
            ],
        },
        "split_stats": split_stats,
    }
    (staging_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    staging_root.rename(output_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
