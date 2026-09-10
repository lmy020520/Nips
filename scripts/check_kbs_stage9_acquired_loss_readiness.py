#!/usr/bin/env python3
"""Audit and resolve the matched Stage 9.2 acquired-loss ablation matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


SEEDS = (42, 43, 44)
SPLITS = ("train", "val", "test")
VARIANTS = {
    "ranking_only": {
        "margin_loss_weight": 0.0,
        "acquired_negative_margin_weight": 0.0,
    },
    "ce_margin": {
        "margin_loss_weight": 0.20,
        "acquired_negative_margin_weight": 0.0,
    },
    "ce_acquired": {
        "margin_loss_weight": 0.0,
        "acquired_negative_margin_weight": 0.50,
    },
}
BASE_CONFIGS = {
    42: Path("configs/train_ranker_deberta_v27_counterfactual_dual.yaml"),
    43: Path("configs/train_ranker_deberta_v27_counterfactual_dual_seed43.yaml"),
    44: Path("configs/train_ranker_deberta_v27_counterfactual_dual_seed44.yaml"),
}
FULL_OUTPUTS = {
    42: Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual"),
    43: Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43"),
    44: Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44"),
}
DATA_ROOT = Path("data/hotpotqa_distractor_v27_counterfactual_dual")
MEMORY_ROOT = Path("data/hotpotqa_distractor_v7_10k_llm_prestep")
CONFIG_ROOT = Path("outputs/analysis/kbs_stage9_acquired_loss/configs")
ARTIFACT_NAMES = (
    "best_model.pt",
    "best_val_metrics.json",
    "test_metrics.json",
    "train_history.json",
)


def load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return obj


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, child in value.items():
        child_key = f"{prefix}.{key}" if prefix else str(key)
        output.update(flatten(child, child_key))
    return output


def differences(
    left: dict[str, Any], right: dict[str, Any], allowed: set[str]
) -> list[dict[str, Any]]:
    left_flat = flatten(left)
    right_flat = flatten(right)
    output = []
    for key in sorted(set(left_flat) | set(right_flat)):
        if key not in allowed and left_flat.get(key) != right_flat.get(key):
            output.append(
                {"field": key, "left": left_flat.get(key), "right": right_flat.get(key)}
            )
    return output


def output_dir(variant: str, seed: int) -> Path:
    return Path(f"outputs/ranker/deberta_v3_large_v30_{variant}_seed{seed}")


def resolved_config_path(variant: str, seed: int) -> Path:
    return CONFIG_ROOT / f"{variant}_seed{seed}.yaml"


def artifact_status(directory: Path) -> dict[str, Any]:
    files = {
        name: {
            "path": str(directory / name),
            "exists": (directory / name).is_file(),
            "bytes": (directory / name).stat().st_size
            if (directory / name).is_file()
            else None,
        }
        for name in ARTIFACT_NAMES
    }
    present = sum(int(item["exists"] and int(item["bytes"] or 0) > 0) for item in files.values())
    return {
        "output_dir": str(directory),
        "complete": present == len(ARTIFACT_NAMES),
        "partial": 0 < present < len(ARTIFACT_NAMES),
        "files": files,
    }


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row {path}:{line_number}")
            yield row


def inspect_split(path: Path) -> tuple[dict[str, Any], set[str]]:
    rows = 0
    qids: set[str] = set()
    acquired_pairs = 0
    unmasked_auxiliary_rows = 0
    for row in read_jsonl(path):
        rows += 1
        qid = str(row.get("qid") or "")
        if not qid:
            raise ValueError(f"empty qid in {path}")
        qids.add(qid)
        counterfactual = (row.get("labels") or {}).get("counterfactual_ranking") or {}
        acquired_pairs += len(counterfactual.get("acquired_negative_unit_ids") or [])
        unmasked_auxiliary_rows += int(
            not bool((row.get("build_meta") or {}).get("mask_auxiliary_labels", False))
        )
    return {
        "path": str(path),
        "rows": rows,
        "qids": len(qids),
        "acquired_negative_pairs": acquired_pairs,
        "unmasked_auxiliary_rows": unmasked_auxiliary_rows,
    }, qids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pretrain", "posttrain"), default="pretrain")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_acquired_loss/pretrain_readiness.json"),
    )
    args = parser.parse_args()

    required = [
        Path("md/kbs_three_review_execution_plan.md"),
        Path("md/kbs_review_75_85_execution_plan.md"),
        Path("models/deberta-v3-large"),
        Path("outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt"),
        DATA_ROOT / "manifest.json",
        Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl"),
        *BASE_CONFIGS.values(),
    ]
    required.extend(DATA_ROOT / "samples" / f"{split}.jsonl" for split in SPLITS)
    required.extend(
        MEMORY_ROOT / "unit_registry" / f"raw_units_{split}.jsonl" for split in SPLITS
    )
    required.extend(directory / name for directory in FULL_OUTPUTS.values() for name in ARTIFACT_NAMES)
    missing = [str(path) for path in required if not path.exists()]
    failures = [f"missing path: {path}" for path in missing]

    data_audit: dict[str, Any] = {}
    qid_sets: dict[str, set[str]] = {}
    if all((DATA_ROOT / "samples" / f"{split}.jsonl").is_file() for split in SPLITS):
        expected_rows = {"train": 37730, "val": 1207, "test": 1247}
        expected_qids = {"train": 10000, "val": 500, "test": 500}
        for split in SPLITS:
            report, qids = inspect_split(DATA_ROOT / "samples" / f"{split}.jsonl")
            data_audit[split] = report
            qid_sets[split] = qids
            if report["rows"] != expected_rows[split]:
                failures.append(f"{split}: rows={report['rows']} != {expected_rows[split]}")
            if report["qids"] != expected_qids[split]:
                failures.append(f"{split}: qids={report['qids']} != {expected_qids[split]}")
            if report["acquired_negative_pairs"] == 0:
                failures.append(f"{split}: no acquired-negative pairs")
            if report["unmasked_auxiliary_rows"]:
                failures.append(f"{split}: auxiliary labels are not fully masked")

    qid_overlap: dict[str, int] = {}
    if len(qid_sets) == len(SPLITS):
        qid_overlap = {
            "train_val": len(qid_sets["train"] & qid_sets["val"]),
            "train_test": len(qid_sets["train"] & qid_sets["test"]),
            "val_test": len(qid_sets["val"] & qid_sets["test"]),
        }
        if any(qid_overlap.values()):
            failures.append(f"split qid overlap: {qid_overlap}")
        eval_path = Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl")
        if eval_path.is_file():
            eval_qids = {str(row.get("qid") or "") for row in read_jsonl(eval_path)}
            qid_overlap.update(
                {
                    f"{split}_primary_eval": len(qids & eval_qids)
                    for split, qids in qid_sets.items()
                }
            )
            if any(qid_overlap[f"{split}_primary_eval"] for split in SPLITS):
                failures.append(
                    "training/validation/internal-test qids overlap primary evaluation"
                )

    config_audit: dict[str, Any] = {}
    resolved_configs: dict[str, Any] = {}
    if all(path.is_file() for path in BASE_CONFIGS.values()):
        base_configs = {seed: load_yaml(path) for seed, path in BASE_CONFIGS.items()}
        for seed in SEEDS:
            if base_configs[seed].get("seed") != seed:
                failures.append(f"Full seed {seed} config has the wrong seed")
            seed_diff = differences(
                base_configs[42], base_configs[seed], {"seed", "output_dir"}
            )
            if seed_diff:
                failures.append(f"Full seed {seed} differs outside seed/output_dir")
            train = base_configs[seed].get("train") or {}
            expected_full = {
                "margin_loss_weight": 0.20,
                "margin": 0.20,
                "acquired_negative_margin_weight": 0.50,
                "acquired_negative_margin": 0.20,
            }
            for key, expected in expected_full.items():
                if float(train.get(key, -1)) != expected:
                    failures.append(f"Full seed {seed}: {key} != {expected}")
            if (base_configs[seed].get("model") or {}).get("architecture") != "dual_state_interaction":
                failures.append(f"Full seed {seed}: architecture is not dual_state_interaction")
            if (base_configs[seed].get("data") or {}).get("context_mode") != "full_state":
                failures.append(f"Full seed {seed}: context_mode is not full_state")

        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        for variant, weights in VARIANTS.items():
            resolved_configs[variant] = {}
            for seed in SEEDS:
                config = copy.deepcopy(base_configs[seed])
                config["output_dir"] = str(output_dir(variant, seed))
                config["train"].update(weights)
                path = resolved_config_path(variant, seed)
                path.write_text(
                    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                unexpected = differences(
                    base_configs[seed],
                    config,
                    {
                        "output_dir",
                        "train.margin_loss_weight",
                        "train.acquired_negative_margin_weight",
                    },
                )
                if unexpected:
                    failures.append(f"{variant} seed {seed}: unexpected config differences")
                resolved_configs[variant][str(seed)] = {
                    "path": str(path),
                    "sha256": sha256(path),
                    "output_dir": str(output_dir(variant, seed)),
                    **weights,
                    "unexpected_differences": unexpected,
                }
        config_audit = {
            "base_seed_parity": {
                str(seed): differences(
                    base_configs[42], base_configs[seed], {"seed", "output_dir"}
                )
                for seed in SEEDS
            },
            "resolved_configs": resolved_configs,
        }

    full_artifacts = {
        str(seed): artifact_status(directory) for seed, directory in FULL_OUTPUTS.items()
    }
    for seed, status in full_artifacts.items():
        if not status["complete"]:
            failures.append(f"Full seed {seed} artifacts are incomplete")

    variant_artifacts: dict[str, Any] = {}
    pending_runs = []
    complete_runs = []
    for variant in VARIANTS:
        variant_artifacts[variant] = {}
        for seed in SEEDS:
            status = artifact_status(output_dir(variant, seed))
            variant_artifacts[variant][str(seed)] = status
            run_name = f"{variant}_seed{seed}"
            if status["complete"]:
                complete_runs.append(run_name)
            else:
                pending_runs.append(run_name)
            if status["partial"]:
                failures.append(f"partial training artifacts: {run_name}")
            if args.mode == "posttrain" and not status["complete"]:
                failures.append(f"missing completed training run: {run_name}")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.2",
        "mode": f"acquired_loss_{args.mode}_readiness",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "training_runs_started": 0,
        "protocol": {
            "data_root": str(DATA_ROOT),
            "architecture": "dual_state_interaction",
            "context_mode": "full_state",
            "initialization": "outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt",
            "seeds": list(SEEDS),
            "loss_matrix": {
                "ranking_only": {"ce": True, "margin": False, "acquired": False},
                "ce_margin": {"ce": True, "margin": True, "acquired": False},
                "ce_acquired": {"ce": True, "margin": False, "acquired": True},
                "full": {"ce": True, "margin": True, "acquired": True},
            },
            "ordinary_margin": 0.20,
            "ordinary_margin_weight": 0.20,
            "acquired_margin": 0.20,
            "acquired_margin_weight": 0.50,
            "primary_contrast": "Full minus CE+Margin",
        },
        "data_audit": data_audit,
        "qid_overlap": qid_overlap,
        "config_audit": config_audit,
        "artifacts": {"full": full_artifacts, "variants": variant_artifacts},
        "complete_training_runs": complete_runs,
        "pending_training_runs": pending_runs,
        "expected_missing_training_runs_at_stage_start": 9,
        "missing_paths": missing,
        "next_gate": (
            "Review readiness before authorizing the nine matched training runs."
            if not failures and args.mode == "pretrain"
            else (
                "Review completed training metrics before selection evaluation."
                if not failures
                else "Resolve failures; do not train, run GPU inference, or call an API."
            )
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
