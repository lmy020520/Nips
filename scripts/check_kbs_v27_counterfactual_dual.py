#!/usr/bin/env python3
"""Validate v27 counterfactual data, dual architecture config, and prerequisites."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


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
    return str((((row.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id")) or "")


def history_ids(row: dict):
    values = []
    for item in ((row.get("state") or {}).get("H_t") or []):
        if isinstance(item, dict) and item.get("unit_id"):
            values.append(str(item["unit_id"]))
        elif isinstance(item, str):
            values.append(item)
    return values


def canonical_source(path: Path):
    output = {}
    for row in read_jsonl(path):
        key = (str(row.get("qid") or ""), int(row.get("t", -1)))
        repetition = int((row.get("build_meta") or {}).get("repetition", 0))
        if key not in output or repetition < output[key][0]:
            output[key] = (repetition, row)
    return {key: value[1] for key, value in output.items()}


def inspect_split(output_path: Path, source_path: Path, split: str, later_repeat: int):
    source = canonical_source(source_path)
    grouped = defaultdict(list)
    qids = set()
    errors = []
    t_distribution = Counter()
    acquired_pairs = 0
    for row in read_jsonl(output_path):
        qid = str(row.get("qid") or "")
        t = int(row.get("t", -1))
        grouped[(qid, t)].append(row)
        qids.add(qid)
        t_distribution[str(t)] += 1

    by_qid = defaultdict(dict)
    for key, rows in grouped.items():
        qid, t = key
        expected_repeat = later_repeat if split == "train" and t >= 1 else 1
        if len(rows) != expected_repeat:
            errors.append(f"repeat mismatch qid={qid} t={t}: {len(rows)} != {expected_repeat}")
        source_row = source.get(key)
        if source_row is None:
            errors.append(f"state absent from v22 source qid={qid} t={t}")
            continue
        reference_pool = candidate_ids(source_row)
        reference_positive = positive_id(source_row)
        for row in rows:
            if candidate_ids(row) != reference_pool:
                errors.append(f"candidate pool changed qid={qid} t={t}")
            if positive_id(row) != reference_positive:
                errors.append(f"positive changed qid={qid} t={t}")
            if str((row.get("state") or {}).get("K_t") or "") != str((source_row.get("state") or {}).get("K_t") or ""):
                errors.append(f"K_t changed qid={qid} t={t}")
            counterfactual = (row.get("labels") or {}).get("counterfactual_ranking") or {}
            actual_acquired = [str(value) for value in (counterfactual.get("acquired_negative_unit_ids") or [])]
            expected_acquired = [
                value
                for value in history_ids(row)
                if value in reference_pool and value != reference_positive
            ]
            if actual_acquired != expected_acquired:
                errors.append(f"acquired negatives mismatch qid={qid} t={t}")
            if reference_positive in actual_acquired:
                errors.append(f"positive marked acquired-negative qid={qid} t={t}")
            if not bool((row.get("build_meta") or {}).get("mask_auxiliary_labels", False)):
                errors.append(f"auxiliary labels not masked qid={qid} t={t}")
            acquired_pairs += len(actual_acquired)
        by_qid[qid][t] = reference_positive

    adjacent_pairs = 0
    adjacent_switches = 0
    for states in by_qid.values():
        for t, positive in states.items():
            if t + 1 in states:
                adjacent_pairs += 1
                adjacent_switches += int(positive != states[t + 1])
    return {
        "path": str(output_path),
        "rows": sum(len(rows) for rows in grouped.values()),
        "canonical_states": len(grouped),
        "qids": len(qids),
        "t_distribution": dict(t_distribution),
        "acquired_negative_pairs": acquired_pairs,
        "adjacent_pairs": adjacent_pairs,
        "adjacent_preference_switches": adjacent_switches,
        "error_count": len(errors),
        "errors": errors[:30],
        "_qids": qids,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/hotpotqa_distractor_v27_counterfactual_dual")
    parser.add_argument("--source-root", default="data/hotpotqa_distractor_v22_state_focused")
    parser.add_argument("--teacher-root", default="data/hotpotqa_distractor_v7_10k_llm_prestep")
    parser.add_argument("--config", default="configs/train_ranker_deberta_v27_counterfactual_dual.yaml")
    parser.add_argument("--init-checkpoint", default="outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt")
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--eval-queries", default="data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl")
    parser.add_argument("--require-paths", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    source_root = Path(args.source_root)
    manifest_path = data_root / "manifest.json"
    required = [manifest_path, Path(args.config)]
    required += [data_root / "samples" / f"{split}.jsonl" for split in ("train", "val", "test")]
    required += [source_root / "samples" / f"{split}.jsonl" for split in ("train", "val", "test")]
    if args.require_paths:
        required += [Path(args.init_checkpoint), Path(args.model_dir)]
        required += [
            Path(args.teacher_root) / "unit_registry" / f"raw_units_{split}.jsonl"
            for split in ("train", "val", "test")
        ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(json.dumps({"status": "MISSING", "missing_paths": missing}, indent=2))
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    later_repeat = int(manifest.get("later_repeat", 0))
    reports = {
        split: inspect_split(
            data_root / "samples" / f"{split}.jsonl",
            source_root / "samples" / f"{split}.jsonl",
            split,
            later_repeat,
        )
        for split in ("train", "val", "test")
    }
    qid_sets = {split: reports[split].pop("_qids") for split in reports}
    overlap = {
        "train_val": len(qid_sets["train"] & qid_sets["val"]),
        "train_test": len(qid_sets["train"] & qid_sets["test"]),
        "val_test": len(qid_sets["val"] & qid_sets["test"]),
    }
    evaluation_overlap = None
    eval_path = Path(args.eval_queries)
    if eval_path.is_file():
        eval_qids = {str(row.get("qid") or "") for row in read_jsonl(eval_path)}
        evaluation_overlap = {
            "eval_qids": len(eval_qids),
            **{f"{split}_eval": len(qids & eval_qids) for split, qids in qid_sets.items()},
        }

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config_checks = {
        "architecture": (config.get("model") or {}).get("architecture"),
        "context_mode": (config.get("data") or {}).get("context_mode"),
        "init_checkpoint": (config.get("train") or {}).get("init_checkpoint"),
        "role_aux_weight": (config.get("train") or {}).get("role_aux_weight"),
        "deficit_aux_weight": (config.get("train") or {}).get("deficit_aux_weight"),
        "contribution_aux_weight": (config.get("train") or {}).get("contribution_aux_weight"),
        "acquired_negative_margin_weight": (config.get("train") or {}).get("acquired_negative_margin_weight"),
    }
    failures = []
    if later_repeat != 2:
        failures.append(f"later_repeat must be frozen at 2, got {later_repeat}")
    for split, report in reports.items():
        if report["error_count"]:
            failures.append(f"{split}: {report['error_count']} validation errors")
        if report["adjacent_pairs"] != report["adjacent_preference_switches"]:
            failures.append(f"{split}: not every adjacent state changes preference")
    if reports["train"]["acquired_negative_pairs"] == 0:
        failures.append("train split has no acquired-negative counterfactual pairs")
    if any(overlap.values()):
        failures.append(f"split qid overlap: {overlap}")
    if evaluation_overlap and any(
        evaluation_overlap.get(f"{split}_eval", 0) for split in ("train", "val", "test")
    ):
        failures.append(f"evaluation qid overlap: {evaluation_overlap}")
    if config_checks["architecture"] != "dual_state_interaction":
        failures.append("config does not select dual_state_interaction")
    if config_checks["context_mode"] != "full_state":
        failures.append("config does not use full_state")
    for key in ("role_aux_weight", "deficit_aux_weight", "contribution_aux_weight"):
        if float(config_checks[key] or 0.0) != 0.0:
            failures.append(f"{key} must remain disabled in v27")
    if float(config_checks["acquired_negative_margin_weight"] or 0.0) <= 0.0:
        failures.append("acquired-negative reversal margin is disabled")

    result = {
        "status": "OK" if not failures else "FAIL",
        "data_root": str(data_root),
        "source_root": str(source_root),
        "protocol": {
            "later_repeat": later_repeat,
            "candidate_top_k": manifest.get("candidate_top_k"),
            "training_objective": manifest.get("training_objective"),
        },
        "config": config_checks,
        "splits": reports,
        "qid_overlap": overlap,
        "evaluation_overlap": evaluation_overlap,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
