#!/usr/bin/env python3
"""Audit the v27-matched previous-evidence-only Anchor protocol."""

import argparse
import copy
import itertools
import json
from collections import Counter
from pathlib import Path

import yaml


SEEDS = (42, 43, 44)


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
    ranking = (row.get("labels") or {}).get("ranking_label") or {}
    return str(ranking.get("positive_unit_id") or "")


def history_ids(row: dict):
    values = []
    for item in ((row.get("state") or {}).get("H_t") or []):
        if isinstance(item, dict) and item.get("unit_id"):
            values.append(str(item["unit_id"]))
        elif isinstance(item, str):
            values.append(item)
    return values


def explicit_acquired_ids(row: dict):
    counterfactual = (row.get("labels") or {}).get("counterfactual_ranking") or {}
    values = counterfactual.get("acquired_negative_unit_ids") or []
    return [str(value) for value in values]


def expected_visible_acquired_ids(row: dict):
    history = history_ids(row)
    if not history:
        return []
    previous = history[-1]
    return [previous] if previous in candidate_ids(row) and previous != positive_id(row) else []


def inspect_split(source_path: Path, output_path: Path):
    errors = []
    qids = set()
    t_distribution = Counter()
    rows = 0
    visible_pairs = 0
    hidden_pairs_removed = 0
    protected_fields = ("qid", "t", "question", "state", "candidates", "derived_payloads")
    sentinel = object()
    paired_rows = itertools.zip_longest(
        read_jsonl(source_path), read_jsonl(output_path), fillvalue=sentinel
    )
    for row_index, (source, output) in enumerate(paired_rows, start=1):
        if source is sentinel or output is sentinel:
            errors.append(f"row-count/order mismatch at row={row_index}")
            break
        rows += 1
        qid = str(output.get("qid") or "")
        t = int(output.get("t", -1))
        qids.add(qid)
        t_distribution[str(t)] += 1
        for field in protected_fields:
            if source.get(field) != output.get(field):
                errors.append(f"protected field changed row={row_index} qid={qid} t={t}: {field}")
        if positive_id(source) != positive_id(output):
            errors.append(f"ranking positive changed row={row_index} qid={qid} t={t}")
        source_labels = copy.deepcopy(source.get("labels") or {})
        output_labels = copy.deepcopy(output.get("labels") or {})
        source_labels.pop("counterfactual_ranking", None)
        output_labels.pop("counterfactual_ranking", None)
        if source_labels != output_labels:
            errors.append(f"non-counterfactual labels changed row={row_index} qid={qid} t={t}")
        if (source.get("build_meta") or {}).get("repetition") != (
            output.get("build_meta") or {}
        ).get("repetition"):
            errors.append(f"row repetition changed row={row_index} qid={qid} t={t}")

        expected = expected_visible_acquired_ids(source)
        actual = explicit_acquired_ids(output)
        if actual != expected:
            errors.append(
                f"visible acquired-negative mismatch row={row_index} qid={qid} t={t}: "
                f"actual={actual}, expected={expected}"
            )
        if positive_id(output) in actual:
            errors.append(f"positive marked acquired-negative row={row_index} qid={qid} t={t}")
        counterfactual = (output.get("labels") or {}).get("counterfactual_ranking") or {}
        if counterfactual.get("acquired_negative_scope") != "visible_previous_evidence_only":
            errors.append(f"missing acquired-negative scope row={row_index} qid={qid} t={t}")

        source_acquired = explicit_acquired_ids(source)
        visible_pairs += len(actual)
        hidden_pairs_removed += max(0, len(source_acquired) - len(actual))

    return {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "rows": rows,
        "qids": len(qids),
        "t_distribution": dict(t_distribution),
        "visible_acquired_negative_pairs": visible_pairs,
        "hidden_history_pairs_removed": hidden_pairs_removed,
        "error_count": len(errors),
        "errors": errors[:40],
        "_qids": qids,
    }


def config_paths(config_root: Path, prefix: str):
    return {
        42: config_root / f"{prefix}.yaml",
        43: config_root / f"{prefix}_seed43.yaml",
        44: config_root / f"{prefix}_seed44.yaml",
    }


def audit_configs(config_root: Path):
    reference_paths = config_paths(
        config_root, "train_ranker_deberta_v27_counterfactual_dual"
    )
    anchor_paths = config_paths(config_root, "train_ranker_deberta_v28_matched_anchor")
    reports = {}
    failures = []
    for seed in SEEDS:
        reference = yaml.safe_load(reference_paths[seed].read_text(encoding="utf-8"))
        anchor = yaml.safe_load(anchor_paths[seed].read_text(encoding="utf-8"))
        mismatches = []
        if int(reference.get("seed", -1)) != seed or int(anchor.get("seed", -1)) != seed:
            mismatches.append("seed")
        if reference.get("model") != anchor.get("model"):
            mismatches.append("model")
        if reference.get("train") != anchor.get("train"):
            mismatches.append("train")

        reference_data = copy.deepcopy(reference.get("data") or {})
        anchor_data = copy.deepcopy(anchor.get("data") or {})
        reference_context = reference_data.pop("context_mode", None)
        anchor_context = anchor_data.pop("context_mode", None)
        reference_samples = {
            split: reference_data.pop(f"{split}_samples", None)
            for split in ("train", "val", "test")
        }
        anchor_samples = {
            split: anchor_data.pop(f"{split}_samples", None)
            for split in ("train", "val", "test")
        }
        if reference_data != anchor_data:
            mismatches.append("data fields other than context/sample paths")
        if reference_context != "full_state" or anchor_context != "previous_evidence_only":
            mismatches.append("context_mode contract")
        for split in ("train", "val", "test"):
            expected_reference = (
                f"data/hotpotqa_distractor_v27_counterfactual_dual/samples/{split}.jsonl"
            )
            expected_anchor = (
                f"data/hotpotqa_distractor_v28_matched_anchor/samples/{split}.jsonl"
            )
            if reference_samples[split] != expected_reference:
                mismatches.append(f"reference {split}_samples")
            if anchor_samples[split] != expected_anchor:
                mismatches.append(f"anchor {split}_samples")
        expected_output = "outputs/ranker/deberta_v3_large_v28_matched_anchor"
        if seed != 42:
            expected_output += f"_seed{seed}"
        if anchor.get("output_dir") != expected_output:
            mismatches.append("output_dir")
        reports[str(seed)] = {
            "reference_config": str(reference_paths[seed]),
            "anchor_config": str(anchor_paths[seed]),
            "architecture": (anchor.get("model") or {}).get("architecture"),
            "context_mode": anchor_context,
            "init_checkpoint": (anchor.get("train") or {}).get("init_checkpoint"),
            "mismatches": mismatches,
        }
        if mismatches:
            failures.append(f"seed {seed} config mismatch: {mismatches}")
    return reports, failures, list(reference_paths.values()) + list(anchor_paths.values())


def audit_loader_contract():
    from src.datasets.prefix_dataset import get_acquired_negative_unit_ids

    base = {"state": {"H_t": [{"unit_id": "u1"}, {"unit_id": "u2"}]}}
    explicit = copy.deepcopy(base)
    explicit["labels"] = {
        "counterfactual_ranking": {"acquired_negative_unit_ids": ["u2"]}
    }
    explicit_empty = copy.deepcopy(base)
    explicit_empty["labels"] = {
        "counterfactual_ranking": {"acquired_negative_unit_ids": []}
    }
    checks = {
        "explicit_visible_label": get_acquired_negative_unit_ids(explicit) == ["u2"],
        "explicit_empty_overrides_history": get_acquired_negative_unit_ids(explicit_empty) == [],
        "legacy_fallback_to_history": get_acquired_negative_unit_ids(base) == ["u1", "u2"],
    }
    return checks, [name for name, passed in checks.items() if not passed]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default="data/hotpotqa_distractor_v28_matched_anchor"
    )
    parser.add_argument(
        "--source-root", default="data/hotpotqa_distractor_v27_counterfactual_dual"
    )
    parser.add_argument("--config-root", default="configs")
    parser.add_argument(
        "--teacher-root", default="data/hotpotqa_distractor_v7_10k_llm_prestep"
    )
    parser.add_argument(
        "--eval-queries",
        default="data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="outputs/ranker/deberta_v3_large_v21_unified_full/best_model.pt",
    )
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--require-paths", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    source_root = Path(args.source_root)
    config_root = Path(args.config_root)
    config_reports, config_failures, config_files = audit_configs(config_root)
    required = [data_root / "manifest.json", source_root / "manifest.json", *config_files]
    required += [
        root / "samples" / f"{split}.jsonl"
        for root in (source_root, data_root)
        for split in ("train", "val", "test")
    ]
    if args.require_paths:
        required += [Path(args.init_checkpoint), Path(args.model_dir), Path(args.eval_queries)]
        required += [
            Path(args.teacher_root) / "unit_registry" / f"raw_units_{split}.jsonl"
            for split in ("train", "val", "test")
        ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(json.dumps({"status": "MISSING", "missing_paths": missing}, indent=2))
        raise SystemExit(1)

    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    reports = {
        split: inspect_split(
            source_root / "samples" / f"{split}.jsonl",
            data_root / "samples" / f"{split}.jsonl",
        )
        for split in ("train", "val", "test")
    }
    qid_sets = {split: report.pop("_qids") for split, report in reports.items()}
    overlap = {
        "train_val": len(qid_sets["train"] & qid_sets["val"]),
        "train_test": len(qid_sets["train"] & qid_sets["test"]),
        "val_test": len(qid_sets["val"] & qid_sets["test"]),
    }
    eval_qids = {str(row.get("qid") or "") for row in read_jsonl(Path(args.eval_queries))}
    evaluation_overlap = {
        "eval_qids": len(eval_qids),
        **{
            f"{split}_eval": len(qids & eval_qids)
            for split, qids in qid_sets.items()
        },
    }
    loader_checks, loader_failures = audit_loader_contract()
    failures = list(config_failures)
    failures.extend(f"loader contract failed: {name}" for name in loader_failures)
    failures.extend(
        f"{split}: {report['error_count']} row audit errors"
        for split, report in reports.items()
        if report["error_count"]
    )
    if reports["train"]["visible_acquired_negative_pairs"] == 0:
        failures.append("train split has no visible acquired-negative pairs")
    if any(overlap.values()):
        failures.append(f"split qid overlap: {overlap}")
    if any(evaluation_overlap[f"{split}_eval"] for split in ("train", "val", "test")):
        failures.append(f"evaluation qid overlap: {evaluation_overlap}")
    if manifest.get("context_mode") != "previous_evidence_only":
        failures.append("manifest context_mode is not previous_evidence_only")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 5,
        "baseline_name": "v27-matched previous-evidence-only dual Anchor",
        "data_root": str(data_root),
        "source_root": str(source_root),
        "comparison_contract": manifest.get("comparison_contract"),
        "splits": reports,
        "qid_overlap": overlap,
        "evaluation_overlap": evaluation_overlap,
        "loader_contract": loader_checks,
        "configs": config_reports,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
