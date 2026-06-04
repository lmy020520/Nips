#!/usr/bin/env python3
"""Build and validate an expanded HotpotQA dataset with cached LLM roles."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/hotpotqa_distractor_v8_15k_source"))
    parser.add_argument("--output-root", type=Path, default=Path("data/hotpotqa_distractor_v8_15k_llm_prestep"))
    parser.add_argument("--reuse-root", type=Path, default=Path("data/hotpotqa_distractor_v7_10k_llm_prestep"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trajectory-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-llm", action="store_true", help="Use rule labels; intended only for pipeline smoke tests.")
    args = parser.parse_args()

    source = args.source_root.resolve()
    output = args.output_root.resolve()
    reuse = args.reuse_root.resolve()
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        required = source / "processed" / f"{split}.jsonl"
        if not required.exists():
            raise FileNotFoundError(required)

    env = os.environ.copy()
    env.update(
        {
            "HOTPOTQA_DATA_ROOT": str(output),
            "HOTPOTQA_PROCESSED_BASE": str(source / "processed"),
            "HOTPOTQA_QUERY_BASE": str(output / "queries"),
            "HOTPOTQA_RUN_SUFFIX": "hotpotqa_v8_15k_llm_prestep",
            "FULL_MAX_QIDS": "0",
            "FULL_ONLY_SPLIT": "",
            "FULL_MAX_WORKERS": str(args.trajectory_workers),
        }
    )

    run([sys.executable, "scripts/rebuild_hotpotqa_queries_v2.py"], env)
    run([sys.executable, "scripts/rebuild_hotpotqa_raw_units_v2.py"], env)
    run([sys.executable, "scripts/rebuild_hotpotqa_index_store_v2.py"], env)

    if args.skip_llm:
        env["HOTPOTQA_ROLE_LABEL_MODE"] = "rule"
        run([sys.executable, "scripts/rebuild_hotpotqa_targets_v2.py"], env)
    else:
        if not env.get("DEEPSEEK_API_KEY", "").strip():
            raise RuntimeError("DEEPSEEK_API_KEY must be set unless --skip-llm is used")
        old_cache = reuse / "llm_role_cache"
        new_cache = output / "llm_role_cache"
        if old_cache.exists() and not new_cache.exists():
            print(f"Reusing LLM cache from {old_cache}", flush=True)
            shutil.copytree(old_cache, new_cache)
        run(
            [
                sys.executable,
                "scripts/rebuild_hotpotqa_targets_llm_parallel.py",
                "--query-dir",
                str(output / "queries"),
                "--processed-dir",
                str(source / "processed"),
                "--output-root",
                str(output),
                "--workers",
                str(args.workers),
            ],
            env,
        )

    run([sys.executable, "scripts/build_hotpotqa_seed_v2.py"], env)
    run([sys.executable, "scripts/build_hotpotqa_init_state_v2.py"], env)
    for script, extra in (
        ("build_hotpotqa_full_trajectories_v4.py", ["--force"]),
        ("build_hotpotqa_derived_harvest_v4.py", []),
        ("build_hotpotqa_states_v4.py", []),
        ("build_hotpotqa_candidates_v4.py", []),
        ("build_hotpotqa_ranking_labels_v4.py", []),
        ("build_hotpotqa_stop_labels_v4.py", []),
        ("build_hotpotqa_deficit_labels_v4.py", []),
        ("build_hotpotqa_contribution_labels_v4.py", []),
        ("build_hotpotqa_samples_v4.py", ["--force"]),
        ("build_hotpotqa_success_debug_v4.py", ["--force"]),
    ):
        run([sys.executable, f"scripts/{script}", *extra], env)
    run(
        [
            sys.executable,
            "scripts/validate_hotpotqa_dataset_release.py",
            "--data_root",
            str(output),
            "--strict",
        ],
        env,
    )
    print(f"Expanded dataset is ready: {output}", flush=True)


if __name__ == "__main__":
    main()
