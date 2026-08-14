#!/usr/bin/env python3
"""Validate matched v27-to-v29 Coverage Student data and configuration."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from scripts.build_kbs_stage6_coverage_student_data import (
    coverage_positive,
    load_memory,
    load_targets,
)


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


def positive(row: dict) -> str:
    return str((((row.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id")) or "")


def pool(row: dict) -> list[str]:
    return [str(value) for value in ((row.get("candidates") or {}).get("C_t") or [])]


def inspect_split(
    source_path: Path,
    output_path: Path,
    targets_path: Path,
    memory_path: Path,
) -> tuple[dict, set[str]]:
    source_rows = list(read_jsonl(source_path))
    output_rows = list(read_jsonl(output_path))
    targets = load_targets(targets_path)
    memory = load_memory(memory_path)
    errors = []
    qids = set()
    stats = Counter()
    expected_by_state = {}
    if len(source_rows) != len(output_rows):
        errors.append(f"row count mismatch: {len(source_rows)} != {len(output_rows)}")
    for index, (source, output) in enumerate(zip(source_rows, output_rows)):
        source_key = (str(source.get("qid") or ""), int(source.get("t", -1)))
        output_key = (str(output.get("qid") or ""), int(output.get("t", -1)))
        qids.add(output_key[0])
        stats["rows"] += 1
        if source_key != output_key:
            errors.append(f"row order mismatch at {index}: {source_key} != {output_key}")
        if pool(source) != pool(output):
            errors.append(f"candidate pool changed at {output_key}")
        if source.get("state") != output.get("state"):
            errors.append(f"state changed at {output_key}")
        new_positive = positive(output)
        old_positive = positive(source)
        expected_positive, _ = coverage_positive(source, targets.get(output_key[0], set()), memory)
        expected_previous = expected_by_state.get((output_key[0], output_key[1] - 1))
        expected_by_state[output_key] = expected_positive
        if not new_positive or new_positive not in pool(output):
            errors.append(f"invalid coverage positive at {output_key}: {new_positive}")
        if new_positive != expected_positive:
            errors.append(
                f"coverage positive mismatch at {output_key}: {new_positive} != {expected_positive}"
            )
        labels = output.get("labels") or {}
        counterfactual = labels.get("counterfactual_ranking") or {}
        meta = output.get("build_meta") or {}
        if counterfactual.get("teacher_objective") != "coverage_greedy":
            errors.append(f"teacher objective missing at {output_key}")
        if str(counterfactual.get("source_closure_positive_unit_id") or "") != old_positive:
            errors.append(f"source positive mismatch at {output_key}")
        if counterfactual.get("adjacent_previous_positive_unit_id") != expected_previous:
            errors.append(f"previous coverage positive mismatch at {output_key}")
        if not bool(meta.get("mask_auxiliary_labels", False)):
            errors.append(f"auxiliary mask disabled at {output_key}")
        stats["changed_labels"] += int(new_positive != old_positive)
        stats["unchanged_labels"] += int(new_positive == old_positive)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "rows": stats["rows"],
        "qids": len(qids),
        "changed_labels": stats["changed_labels"],
        "changed_label_rate": round(stats["changed_labels"] / stats["rows"], 6) if stats["rows"] else None,
        "error_count": len(errors),
        "errors": errors[:30],
    }, qids


def flattened_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    output = {"seed": config.get("seed"), "output_dir": config.get("output_dir")}
    for section in ("model", "data", "train"):
        for key, value in (config.get(section) or {}).items():
            output[f"{section}.{key}"] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/hotpotqa_distractor_v27_counterfactual_dual"))
    parser.add_argument("--data-root", type=Path, default=Path("data/hotpotqa_distractor_v29_coverage_greedy"))
    parser.add_argument("--teacher-root", type=Path, default=Path("data/hotpotqa_distractor_v7_10k_llm_prestep"))
    parser.add_argument("--reference-config", type=Path, default=Path("configs/train_ranker_deberta_v27_counterfactual_dual.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/train_ranker_deberta_v29_coverage_greedy.yaml"))
    parser.add_argument("--eval-queries", type=Path, default=Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl"))
    parser.add_argument("--require-paths", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/kbs_stage6_coverage_student/readiness.json"))
    args = parser.parse_args()

    required = [args.data_root / "manifest.json", args.reference_config, args.config]
    required.extend(args.source_root / "samples" / f"{split}.jsonl" for split in SPLITS)
    required.extend(args.data_root / "samples" / f"{split}.jsonl" for split in SPLITS)
    required.extend(args.teacher_root / "targets" / f"{split}.jsonl" for split in SPLITS)
    required.extend(args.teacher_root / "unit_registry" / f"raw_units_{split}.jsonl" for split in SPLITS)
    if args.require_paths:
        required.extend(
            [
                Path("outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt"),
                Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"),
                Path("models/deberta-v3-large"),
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    failures = [f"missing path: {path}" for path in missing]
    reports = {}
    qids = {}
    if not failures:
        for split in SPLITS:
            reports[split], qids[split] = inspect_split(
                args.source_root / "samples" / f"{split}.jsonl",
                args.data_root / "samples" / f"{split}.jsonl",
                args.teacher_root / "targets" / f"{split}.jsonl",
                args.teacher_root / "unit_registry" / f"raw_units_{split}.jsonl",
            )
            if reports[split]["error_count"]:
                failures.append(f"{split}: {reports[split]['error_count']} row errors")

    overlap = {}
    evaluation_overlap = {}
    if qids:
        overlap = {
            "train_val": len(qids["train"] & qids["val"]),
            "train_test": len(qids["train"] & qids["test"]),
            "val_test": len(qids["val"] & qids["test"]),
        }
        if any(overlap.values()):
            failures.append(f"split overlap: {overlap}")
        if args.eval_queries.is_file():
            eval_qids = {str(row.get("qid") or "") for row in read_jsonl(args.eval_queries)}
            evaluation_overlap = {
                "eval_qids": len(eval_qids),
                **{f"{split}_eval": len(qids[split] & eval_qids) for split in SPLITS},
            }
            if any(evaluation_overlap[f"{split}_eval"] for split in SPLITS):
                failures.append(f"evaluation overlap: {evaluation_overlap}")

    config_mismatches = []
    if args.reference_config.is_file() and args.config.is_file():
        reference = flattened_config(args.reference_config)
        candidate = flattened_config(args.config)
        allowed = {
            "output_dir",
            "data.train_samples",
            "data.val_samples",
            "data.test_samples",
        }
        for key in sorted(set(reference) | set(candidate)):
            if key in allowed:
                continue
            if reference.get(key) != candidate.get(key):
                config_mismatches.append(
                    {"field": key, "reference": reference.get(key), "candidate": candidate.get(key)}
                )
        if config_mismatches:
            failures.append(f"unmatched config fields: {len(config_mismatches)}")
        if candidate.get("output_dir") != "outputs/ranker/deberta_v3_large_v29_coverage_greedy":
            failures.append(f"unexpected Coverage output_dir: {candidate.get('output_dir')}")
        for split in SPLITS:
            expected_samples = str(args.data_root / "samples" / f"{split}.jsonl")
            actual_samples = candidate.get(f"data.{split}_samples")
            if actual_samples != expected_samples:
                failures.append(
                    f"unexpected {split} samples path: {actual_samples} != {expected_samples}"
                )
    if reports and reports.get("val", {}).get("changed_labels", 0) < 25:
        failures.append("validation label changes below the pre-registered 25-state materiality floor")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 6,
        "step": "6.1",
        "mode": "coverage_student_readiness",
        "api_calls": 0,
        "training_runs": 0,
        "source_root": str(args.source_root),
        "data_root": str(args.data_root),
        "reference_checkpoint": "outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt",
        "splits": reports,
        "qid_overlap": overlap,
        "evaluation_overlap": evaluation_overlap,
        "config_mismatches": config_mismatches,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
