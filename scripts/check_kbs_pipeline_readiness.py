#!/usr/bin/env python3
"""Check whether the KBS evidence-acquisition pipeline assets are present.

This script does not run experiments. It only verifies that a data root,
checkpoint, model directory, and front-end assets are sufficient to execute the
current pipeline stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train", "val", "test")


def file_status(path: Path, *, required: bool = True) -> dict:
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else 0
    return {
        "path": str(path),
        "required": required,
        "exists": exists,
        "size": size,
        "ok": exists if required else True,
    }


def dir_status(path: Path, *, required: bool = True) -> dict:
    exists = path.exists() and path.is_dir()
    return {
        "path": str(path),
        "required": required,
        "exists": exists,
        "ok": exists if required else True,
    }


def add_file(checks: list[dict], name: str, path: Path, *, required: bool = True) -> None:
    item = file_status(path, required=required)
    item["name"] = name
    checks.append(item)


def add_dir(checks: list[dict], name: str, path: Path, *, required: bool = True) -> None:
    item = dir_status(path, required=required)
    item["name"] = name
    checks.append(item)


def collect_data_checks(data_root: Path) -> list[dict]:
    checks = []
    add_dir(checks, "data_root", data_root)
    for split in SPLITS:
        add_file(checks, f"samples/{split}", data_root / "samples" / f"{split}.jsonl")
        add_file(checks, f"memory/{split}", data_root / "unit_registry" / f"raw_units_{split}.jsonl")
        add_file(checks, f"queries/{split}", data_root / "queries" / f"{split}.jsonl", required=False)
        add_file(checks, f"targets/{split}", data_root / "targets" / f"{split}.jsonl", required=False)
    add_file(checks, "manifest", data_root / "manifest.json", required=False)
    return checks


def collect_model_checks(model_dir: Path, checkpoint: Path, dense_model: Path | None) -> list[dict]:
    checks = []
    add_dir(checks, "model_dir", model_dir)
    add_file(checks, "model_config", model_dir / "config.json", required=False)
    add_file(checks, "checkpoint", checkpoint)
    if dense_model is not None:
        add_dir(checks, "dense_model", dense_model, required=False)
    return checks


def collect_script_checks(project_root: Path) -> list[dict]:
    checks = []
    scripts = [
        "scripts/rebuild_hotpotqa_frontend_dataset.py",
        "scripts/run_hotpotqa_policy_rag.py",
        "scripts/analyze_hotpotqa_frontend_policy_ranks.py",
        "src/datasets/prefix_dataset.py",
        "src/models/ranker.py",
        "src/train/train_ranker.py",
    ]
    for script in scripts:
        add_file(checks, script, project_root / script)
    return checks


def summarize_stage(name: str, checks: list[dict]) -> dict:
    required = [item for item in checks if item["required"]]
    missing = [item for item in required if not item["exists"]]
    return {
        "name": name,
        "ok": not missing,
        "required": len(required),
        "missing": [item["name"] for item in missing],
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--dense-model", default="")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root)
    checkpoint = Path(args.checkpoint)
    model_dir = Path(args.model_dir)
    dense_model = Path(args.dense_model) if args.dense_model else None

    stages = [
        summarize_stage("data", collect_data_checks(data_root)),
        summarize_stage("model", collect_model_checks(model_dir, checkpoint, dense_model)),
        summarize_stage("scripts", collect_script_checks(project_root)),
    ]

    summary = {
        "pipeline_ready": all(stage["ok"] for stage in stages),
        "data_root": str(data_root),
        "checkpoint": str(checkpoint),
        "model_dir": str(model_dir),
        "dense_model": str(dense_model) if dense_model is not None else "",
        "stages": stages,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
