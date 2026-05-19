#!/usr/bin/env python
"""Build the full HotpotQA v5 teacher dataset under a fresh data folder."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


SPLITS = ("train", "val", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "hotpotqa_distractor_v5"
DEFAULT_QUERY_SOURCE = Path("/home/lmy/study/queries")


def run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def copy_queries(query_source: Path, data_root: Path) -> None:
    queries_dir = data_root / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = query_source / f"{split}.full.bak.jsonl"
        dst = queries_dir / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"missing query source: {src}")
        shutil.copyfile(src, dst)
        count = sum(1 for _ in dst.open("r", encoding="utf-8"))
        print(f"queries/{split}.jsonl: {count}", flush=True)


def write_manifest(data_root: Path, args: argparse.Namespace) -> None:
    debug_dir = data_root / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_version": "hotpotqa_distractor_v5",
        "data_root": str(data_root),
        "query_source": str(args.query_source),
        "role_label_mode": args.role_label_mode,
        "full_max_qids": args.full_max_qids,
        "full_only_split": args.full_only_split,
        "notes": [
            "V5 reuses the v4 teacher construction policy with a configurable data root.",
            "Target role labels use DeepSeek when available; otherwise rule_fallback is used.",
        ],
    }
    with (debug_dir / "build_manifest_v5.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HotpotQA v5 full dataset pipeline.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--query-source", type=Path, default=DEFAULT_QUERY_SOURCE)
    parser.add_argument("--force", action="store_true", help="Remove an existing v5 folder before rebuilding.")
    parser.add_argument(
        "--role-label-mode",
        choices=("auto", "rule"),
        default="auto",
        help="auto uses DeepSeek if DEEPSEEK_API_KEY is set, otherwise rule fallback.",
    )
    parser.add_argument("--full-max-qids", default="0", help="Debug limit; 0 means all qids.")
    parser.add_argument("--full-only-split", choices=("", *SPLITS), default="")
    parser.add_argument("--full-max-workers", default="4")
    parser.add_argument("--skip-full", action="store_true", help="Only build v5 inputs, not trajectories/samples/debug.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if data_root.exists() and args.force:
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HOTPOTQA_DATA_ROOT"] = str(data_root)
    env["HOTPOTQA_RUN_SUFFIX"] = "hotpotqa_v5"
    env["HOTPOTQA_ROLE_LABEL_MODE"] = args.role_label_mode
    env["FULL_MAX_QIDS"] = str(args.full_max_qids)
    env["FULL_ONLY_SPLIT"] = args.full_only_split
    env["FULL_MAX_WORKERS"] = str(args.full_max_workers)

    copy_queries(args.query_source.resolve(), data_root)
    write_manifest(data_root, args)

    build_input_steps = [
        ["python", "scripts/rebuild_hotpotqa_raw_units_v2.py"],
        ["python", "scripts/rebuild_hotpotqa_index_store_v2.py"],
        ["python", "scripts/rebuild_hotpotqa_targets_v2.py"],
        ["python", "scripts/build_hotpotqa_seed_v2.py"],
        ["python", "scripts/build_hotpotqa_init_state_v2.py"],
    ]
    for cmd in build_input_steps:
        run(cmd, env=env)

    if args.skip_full:
        print(f"V5 inputs built at {data_root}", flush=True)
        return

    build_dataset_steps = [
        ["python", "scripts/build_hotpotqa_full_trajectories_v4.py", "--force"],
        ["python", "scripts/build_hotpotqa_derived_harvest_v4.py"],
        ["python", "scripts/build_hotpotqa_states_v4.py"],
        ["python", "scripts/build_hotpotqa_candidates_v4.py"],
        ["python", "scripts/build_hotpotqa_ranking_labels_v4.py"],
        ["python", "scripts/build_hotpotqa_stop_labels_v4.py"],
        ["python", "scripts/build_hotpotqa_deficit_labels_v4.py"],
        ["python", "scripts/build_hotpotqa_contribution_labels_v4.py"],
        ["python", "scripts/build_hotpotqa_samples_v4.py", "--force"],
        ["python", "scripts/build_hotpotqa_success_debug_v4.py", "--force"],
        ["python", "scripts/validate_hotpotqa_dataset_release.py", "--data_root", str(data_root), "--strict"],
    ]
    for cmd in build_dataset_steps:
        run(cmd, env=env)

    print(f"V5 dataset built and validated at {data_root}", flush=True)


if __name__ == "__main__":
    main()
