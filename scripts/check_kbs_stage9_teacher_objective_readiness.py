#!/usr/bin/env python3
"""Audit the matched three-seed Closure-versus-Coverage experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


SEEDS = (42, 43, 44)
SPLITS = ("train", "val", "test")
PLAN_FILES = (
    Path("md/kbs_three_review_execution_plan.md"),
    Path("md/kbs_review_75_85_execution_plan.md"),
)
COVERAGE_CONFIGS = {
    42: Path("configs/train_ranker_deberta_v29_coverage_greedy.yaml"),
    43: Path("configs/train_ranker_deberta_v29_coverage_greedy_seed43.yaml"),
    44: Path("configs/train_ranker_deberta_v29_coverage_greedy_seed44.yaml"),
}
CLOSURE_CONFIGS = {
    42: Path("configs/train_ranker_deberta_v27_counterfactual_dual.yaml"),
    43: Path("configs/train_ranker_deberta_v27_counterfactual_dual_seed43.yaml"),
    44: Path("configs/train_ranker_deberta_v27_counterfactual_dual_seed44.yaml"),
}
COVERAGE_OUTPUTS = {
    42: Path("outputs/ranker/deberta_v3_large_v29_coverage_greedy"),
    43: Path("outputs/ranker/deberta_v3_large_v29_coverage_greedy_seed43"),
    44: Path("outputs/ranker/deberta_v3_large_v29_coverage_greedy_seed44"),
}
CLOSURE_OUTPUTS = {
    42: Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual"),
    43: Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43"),
    44: Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44"),
}
COVERAGE_ROOT = Path("data/hotpotqa_distractor_v29_coverage_greedy")
TEACHER_ROOT = Path("data/hotpotqa_distractor_v7_10k_llm_prestep")
SOURCE_READINESS = Path(
    "outputs/analysis/kbs_stage6_coverage_student/readiness.json"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"config is not an object: {path}")
    return value


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, item in value.items():
        child = f"{prefix}.{key}" if prefix else str(key)
        output.update(flatten(item, child))
    return output


def config_differences(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    allowed: set[str],
) -> list[dict[str, Any]]:
    left_flat = flatten(left)
    right_flat = flatten(right)
    differences = []
    for key in sorted(set(left_flat) | set(right_flat)):
        if key in allowed:
            continue
        if left_flat.get(key) != right_flat.get(key):
            differences.append(
                {
                    "field": key,
                    "left": left_flat.get(key),
                    "right": right_flat.get(key),
                }
            )
    return differences


def artifact_status(directory: Path) -> dict[str, Any]:
    files = {}
    for name in (
        "best_model.pt",
        "best_val_metrics.json",
        "test_metrics.json",
        "train_history.json",
    ):
        path = directory / name
        files[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("pretrain", "posttrain"), default="pretrain"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/kbs_stage9_teacher_objective/pretrain_readiness.json"
        ),
    )
    args = parser.parse_args()

    required = [*PLAN_FILES, SOURCE_READINESS, COVERAGE_ROOT / "manifest.json"]
    required.extend(COVERAGE_CONFIGS.values())
    required.extend(CLOSURE_CONFIGS.values())
    required.extend(COVERAGE_ROOT / "samples" / f"{split}.jsonl" for split in SPLITS)
    required.extend(
        TEACHER_ROOT / "unit_registry" / f"raw_units_{split}.jsonl"
        for split in SPLITS
    )
    required.extend(
        [
            Path("models/deberta-v3-large"),
            Path("outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt"),
            COVERAGE_OUTPUTS[42] / "best_model.pt",
        ]
    )
    required.extend(directory / "best_model.pt" for directory in CLOSURE_OUTPUTS.values())
    if args.mode == "posttrain":
        for directory in COVERAGE_OUTPUTS.values():
            required.extend(
                directory / name
                for name in (
                    "best_model.pt",
                    "best_val_metrics.json",
                    "test_metrics.json",
                    "train_history.json",
                )
            )

    missing = [str(path) for path in required if not path.exists()]
    failures = [f"missing path: {path}" for path in missing]
    source_readiness: dict[str, Any] = {}
    configs: dict[str, Any] = {}

    if SOURCE_READINESS.is_file():
        try:
            source_readiness = load_json(SOURCE_READINESS)
            if source_readiness.get("status") != "OK":
                failures.append("Stage 6 Coverage data readiness did not pass")
            if source_readiness.get("data_root") != str(COVERAGE_ROOT):
                failures.append("Stage 6 readiness refers to a different Coverage root")
            for split in SPLITS:
                report = (source_readiness.get("splits") or {}).get(split) or {}
                if int(report.get("error_count", -1)) != 0:
                    failures.append(f"Stage 6 readiness has {split} row errors")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"cannot audit Stage 6 readiness: {exc}")

    if all(path.is_file() for path in (*COVERAGE_CONFIGS.values(), *CLOSURE_CONFIGS.values())):
        coverage = {seed: load_config(path) for seed, path in COVERAGE_CONFIGS.items()}
        closure = {seed: load_config(path) for seed, path in CLOSURE_CONFIGS.items()}
        seed_parity = {}
        objective_parity = {}
        for seed in SEEDS:
            expected_output = str(COVERAGE_OUTPUTS[seed])
            if coverage[seed].get("seed") != seed:
                failures.append(f"Coverage seed {seed} config has wrong seed")
            if coverage[seed].get("output_dir") != expected_output:
                failures.append(f"Coverage seed {seed} has wrong output_dir")
            seed_diff = config_differences(
                coverage[42], coverage[seed], allowed={"seed", "output_dir"}
            )
            seed_parity[str(seed)] = seed_diff
            if seed_diff:
                failures.append(f"Coverage seed {seed} config is not seed-matched")
            objective_diff = config_differences(
                closure[seed],
                coverage[seed],
                allowed={
                    "output_dir",
                    "data.train_samples",
                    "data.val_samples",
                    "data.test_samples",
                },
            )
            objective_parity[str(seed)] = objective_diff
            if objective_diff:
                failures.append(
                    f"Closure/Coverage seed {seed} config differs outside the objective data"
                )
        configs = {
            "coverage_seed_parity_unexpected_differences": seed_parity,
            "closure_coverage_unexpected_differences": objective_parity,
        }

    artifacts = {
        "coverage": {
            str(seed): artifact_status(directory)
            for seed, directory in COVERAGE_OUTPUTS.items()
        },
        "closure": {
            str(seed): artifact_status(directory)
            for seed, directory in CLOSURE_OUTPUTS.items()
        },
    }

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.1",
        "mode": f"teacher_objective_{args.mode}_readiness",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "training_runs": 0,
        "protocol": {
            "closure_teacher": "v27 answer-facing Closure",
            "coverage_teacher": "v29 coverage-greedy",
            "seeds": list(SEEDS),
            "architecture": "dual_state_interaction",
            "initialization": (
                "outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt"
            ),
            "primary_operating_point": {
                "front_pool_k": 30,
                "candidate_top_k": 10,
                "select_top_k": 5,
                "state_update_top_k": 1,
                "policy_blend_weight": 0.5,
            },
        },
        "source_data_readiness": {
            "path": str(SOURCE_READINESS),
            "status": source_readiness.get("status"),
        },
        "config_audit": configs,
        "artifacts": artifacts,
        "missing_paths": missing,
        "next_gate": (
            "Authorize v29 seed-43/44 training only."
            if not failures and args.mode == "pretrain"
            else (
                "Authorize a no-answer selection smoke only."
                if not failures
                else "Resolve failures; do not train or call an API."
            )
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
