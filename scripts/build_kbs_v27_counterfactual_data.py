#!/usr/bin/env python3
"""Build v27 fixed-pool counterfactual state-ranking data from audited v22 rows."""

import argparse
import copy
import json
import shutil
from collections import Counter, defaultdict
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
    candidates = row.get("candidates") or {}
    return [str(unit_id) for unit_id in (candidates.get("C_t") or [])]


def positive_id(row: dict):
    labels = row.get("labels") or {}
    ranking = labels.get("ranking_label") or {}
    return str(ranking.get("positive_unit_id") or "")


def history_ids(row: dict):
    state = row.get("state") or {}
    output = []
    for item in state.get("H_t") or []:
        if isinstance(item, dict) and item.get("unit_id"):
            output.append(str(item["unit_id"]))
        elif isinstance(item, str):
            output.append(item)
    return output


def canonical_rows(path: Path, max_qids: int):
    selected_qids = []
    selected_qid_set = set()
    by_state = {}
    for row in read_jsonl(path):
        qid = str(row.get("qid") or "")
        if qid not in selected_qid_set:
            if max_qids > 0 and len(selected_qids) >= max_qids:
                continue
            selected_qids.append(qid)
            selected_qid_set.add(qid)
        if qid not in selected_qid_set:
            continue
        key = (qid, int(row.get("t", -1)))
        repetition = int((row.get("build_meta") or {}).get("repetition", 0))
        if key not in by_state or repetition < by_state[key][0]:
            by_state[key] = (repetition, row)
    qid_order = {qid: index for index, qid in enumerate(selected_qids)}
    return [
        by_state[key][1]
        for key in sorted(by_state, key=lambda item: (qid_order[item[0]], item[1]))
    ]


def build_split(source_path: Path, output_path: Path, split: str, later_repeat: int, max_qids: int):
    rows = canonical_rows(source_path, max_qids=max_qids)
    by_qid = defaultdict(list)
    for row in rows:
        by_qid[str(row["qid"])].append(row)

    stats = {
        "qids": len(by_qid),
        "canonical_rows": len(rows),
        "written_rows": 0,
        "canonical_t_distribution": Counter(),
        "written_t_distribution": Counter(),
        "acquired_negative_pairs": 0,
        "adjacent_preference_switches": 0,
        "adjacent_pairs": 0,
        "fixed_pool_violations": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for qid, qid_rows in by_qid.items():
            qid_rows.sort(key=lambda row: int(row["t"]))
            expected_pool = None
            previous_positive = None
            for row in qid_rows:
                t = int(row["t"])
                pool = candidate_ids(row)
                positive = positive_id(row)
                acquired = [
                    unit_id
                    for unit_id in history_ids(row)
                    if unit_id in pool and unit_id != positive
                ]
                stats["canonical_t_distribution"][str(t)] += 1
                stats["acquired_negative_pairs"] += len(acquired)
                if expected_pool is None:
                    expected_pool = pool
                elif pool != expected_pool:
                    stats["fixed_pool_violations"] += 1
                if previous_positive is not None:
                    stats["adjacent_pairs"] += 1
                    stats["adjacent_preference_switches"] += int(previous_positive != positive)

                repeat_count = later_repeat if split == "train" and t >= 1 else 1
                for repetition in range(repeat_count):
                    output = copy.deepcopy(row)
                    output.setdefault("labels", {})["counterfactual_ranking"] = {
                        "pair_group_id": qid,
                        "current_positive_unit_id": positive,
                        "acquired_negative_unit_ids": acquired,
                        "adjacent_previous_positive_unit_id": previous_positive,
                        "preference_switch_required": t >= 1,
                    }
                    output["build_meta"] = {
                        **(output.get("build_meta") or {}),
                        "dataset_version": "v27_counterfactual_dual",
                        "variant": "fixed_pool_state_preference_switch",
                        "source_dataset": str(source_path),
                        "repetition": repetition,
                        "later_state_oversampling": repeat_count,
                        "mask_auxiliary_labels": True,
                    }
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    stats["written_rows"] += 1
                    stats["written_t_distribution"][str(t)] += 1
                previous_positive = positive

    return {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in stats.items()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/hotpotqa_distractor_v22_state_focused")
    parser.add_argument("--output-root", default="data/hotpotqa_distractor_v27_counterfactual_dual")
    parser.add_argument("--later-repeat", type=int, default=2)
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.later_repeat < 1:
        raise ValueError("--later-repeat must be >= 1")
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    required = [source_root / "manifest.json"] + [
        source_root / "samples" / f"{split}.jsonl" for split in ("train", "val", "test")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing v22 source files: " + ", ".join(missing))
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"output exists; pass --force to rebuild: {output_root}")
        shutil.rmtree(output_root)

    split_stats = {
        split: build_split(
            source_path=source_root / "samples" / f"{split}.jsonl",
            output_path=output_root / "samples" / f"{split}.jsonl",
            split=split,
            later_repeat=args.later_repeat,
            max_qids=args.max_qids,
        )
        for split in ("train", "val", "test")
    }
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "dataset": "KBS v27 counterfactual fixed-pool state-ranking data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "source_manifest_seed": source_manifest.get("seed"),
        "output_root": str(output_root),
        "seed": args.seed,
        "max_qids": args.max_qids,
        "later_repeat": args.later_repeat,
        "candidate_top_k": source_manifest.get("candidate_top_k"),
        "auxiliary_labels_masked": True,
        "training_objective": "state-conditioned ranking plus acquired-evidence reversal margin",
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
