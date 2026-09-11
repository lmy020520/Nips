#!/usr/bin/env python3
"""Summarize matched Stage 9.2 training artifacts without model inference."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
VARIANTS = ("ranking_only", "ce_margin", "ce_acquired", "full")
METRICS = ("acc", "acquired_pair_acc")
SPLITS = ("validation", "internal_test")


def output_dir(variant: str, seed: int) -> Path:
    if variant == "full":
        suffix = "" if seed == 42 else f"_seed{seed}"
        return Path(
            f"outputs/ranker/deberta_v3_large_v27_counterfactual_dual{suffix}"
        )
    return Path(f"outputs/ranker/deberta_v3_large_v30_{variant}_seed{seed}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rounded_stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(statistics.mean(values), 6),
        "sample_std": round(statistics.stdev(values), 6),
    }


def selected_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": round(float(metrics["loss"]), 6),
        "acc": round(float(metrics["acc"]), 6),
        "acquired_pair_acc": round(float(metrics["acquired_pair_acc"]), 6),
        "acquired_pairs": int(metrics["acquired_pairs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/kbs_stage9_acquired_loss/training_metrics_summary.json"
        ),
    )
    args = parser.parse_args()

    failures: list[str] = []
    per_seed: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in SEEDS:
            directory = output_dir(variant, seed)
            paths = {
                "validation": directory / "best_val_metrics.json",
                "internal_test": directory / "test_metrics.json",
                "history": directory / "train_history.json",
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                failures.extend(f"missing path: {path}" for path in missing)
                continue
            try:
                validation = load_json(paths["validation"])
                internal_test = load_json(paths["internal_test"])
                history = load_json(paths["history"])
                if not isinstance(validation, dict) or not isinstance(internal_test, dict):
                    raise ValueError("metric artifact is not a JSON object")
                if not isinstance(history, list) or not history:
                    raise ValueError("training history is empty or not a JSON list")
                for split, metrics in (
                    ("validation", validation),
                    ("internal_test", internal_test),
                ):
                    for key in ("loss", "acc", "acquired_pair_acc", "acquired_pairs"):
                        if key not in metrics:
                            raise ValueError(f"{split} metrics missing {key}")
                best = max(history, key=lambda row: float(row["val"]["acc"]))
                per_seed[variant][str(seed)] = {
                    "output_dir": str(directory),
                    "best_epoch": int(best["epoch"]),
                    "validation": selected_metrics(validation),
                    "internal_test": selected_metrics(internal_test),
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"{variant} seed {seed}: {exc}")

    aggregate: dict[str, Any] = {}
    if not failures:
        aggregate = {
            variant: {
                split: {
                    metric: rounded_stats(
                        [
                            float(per_seed[variant][str(seed)][split][metric])
                            for seed in SEEDS
                        ]
                    )
                    for metric in METRICS
                }
                for split in SPLITS
            }
            for variant in VARIANTS
        }

    contrast_pairs = {
        "full_minus_ce_margin": ("full", "ce_margin"),
        "full_minus_ce_acquired": ("full", "ce_acquired"),
        "ce_margin_minus_ranking_only": ("ce_margin", "ranking_only"),
        "ce_acquired_minus_ranking_only": ("ce_acquired", "ranking_only"),
    }
    contrasts: dict[str, Any] = {}
    if not failures:
        for name, (left, right) in contrast_pairs.items():
            contrasts[name] = {
                split: {
                    metric: rounded_stats(
                        [
                            float(per_seed[left][str(seed)][split][metric])
                            - float(per_seed[right][str(seed)][split][metric])
                            for seed in SEEDS
                        ]
                    )
                    for metric in METRICS
                }
                for split in SPLITS
            }

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.2",
        "mode": "acquired_loss_training_metrics",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "primary_contrast": "full_minus_ce_margin",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "paired_seed_descriptive_contrasts": contrasts,
        "interpretation": (
            "Training metrics are diagnostic only; the scientific loss claim requires "
            "matched online selection and downstream evaluation."
        ),
        "next_gate": (
            "Review training metrics before the 20-qid no-answer selection smoke."
            if not failures
            else "Resolve failures; do not run inference or call an API."
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
