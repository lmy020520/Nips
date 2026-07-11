#!/usr/bin/env python3
"""Validate the controlled KBS full-model and loss-ablation configs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


CONFIGS = {
    "full": Path("configs/train_ranker_deberta_v21_unified_full.yaml"),
    "ranking_only": Path("configs/train_ranker_deberta_v21_unified_ranking_only.yaml"),
    "no_deficit": Path("configs/train_ranker_deberta_v21_unified_no_deficit.yaml"),
    "no_contribution": Path("configs/train_ranker_deberta_v21_unified_no_contribution.yaml"),
}

EXPECTED_WEIGHTS = {
    "full": (0.20, 0.20),
    "ranking_only": (0.0, 0.0),
    "no_deficit": (0.0, 0.20),
    "no_contribution": (0.20, 0.0),
}

IGNORED_TRAIN_KEYS = {"deficit_aux_weight", "contribution_aux_weight"}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def shared_signature(config: dict) -> dict:
    train = {
        key: value
        for key, value in config["train"].items()
        if key not in IGNORED_TRAIN_KEYS
    }
    return {
        "seed": config["seed"],
        "model": config["model"],
        "data": config["data"],
        "train": train,
    }


def label_stats(path: Path) -> dict:
    counts = Counter()
    for row in read_jsonl(path):
        counts["rows"] += 1
        labels = row.get("labels") or {}
        deficit = labels.get("d_t_star") or row.get("d_t_star")
        contribution = labels.get("c_t_star") or row.get("c_t_star")
        counts["typed_deficit"] += int(
            isinstance(deficit, dict)
            and all(isinstance(deficit.get(key), (int, float)) for key in ("d_br", "d_dis", "d_sup", "d_der"))
        )
        counts["typed_contribution"] += int(
            isinstance(contribution, dict)
            and all(
                isinstance(contribution.get(key), (int, float))
                for key in ("c_br", "c_dis", "c_sup", "c_der")
            )
        )
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-paths", action="store_true")
    args = parser.parse_args()

    issues = []
    configs = {}
    for name, path in CONFIGS.items():
        if not path.exists():
            issues.append(f"missing config: {path}")
            continue
        configs[name] = load_config(path)

    if len(configs) == len(CONFIGS):
        reference = shared_signature(configs["full"])
        for name, config in configs.items():
            if shared_signature(config) != reference:
                issues.append(f"{name}: shared data/init/hyperparameters differ from full")
            expected_d, expected_c = EXPECTED_WEIGHTS[name]
            actual_d = float(config["train"].get("deficit_aux_weight", 0.0))
            actual_c = float(config["train"].get("contribution_aux_weight", 0.0))
            if (actual_d, actual_c) != (expected_d, expected_c):
                issues.append(
                    f"{name}: expected (lambda_d, lambda_c)={(expected_d, expected_c)}, "
                    f"got {(actual_d, actual_c)}"
                )

        full = configs["full"]
        label_summary = {
            split: label_stats(Path(full["data"][f"{split}_samples"]))
            for split in ("train", "val", "test")
        }
        for split, stats in label_summary.items():
            rows = stats.get("rows", 0)
            if stats.get("typed_deficit") != rows or stats.get("typed_contribution") != rows:
                issues.append(f"{split}: incomplete d_t*/c_t* labels: {stats}")

        required_paths = [
            Path(full["model"]["pretrained_name"]),
            Path(full["train"]["init_checkpoint"]),
        ]
        if args.require_paths:
            for path in required_paths:
                if not path.exists():
                    issues.append(f"missing required model path: {path}")
    else:
        label_summary = {}

    report = {
        "status": "OK" if not issues else "FAILED",
        "configs": {name: str(path) for name, path in CONFIGS.items()},
        "expected_loss_weights": {
            name: {"lambda_d": values[0], "lambda_c": values[1]}
            for name, values in EXPECTED_WEIGHTS.items()
        },
        "label_summary": label_summary,
        "issues": issues,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
