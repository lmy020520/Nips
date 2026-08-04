#!/usr/bin/env python3
"""Audit Stage 3.3 clue-state data, leakage constraints, and matched config."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src.clue_state import (
    build_clue_state,
    extract_question_clues,
    format_clue_evidence,
    render_clue_state,
)


FORBIDDEN_CLUE_KEYS = {
    "answer",
    "gold",
    "positive",
    "role",
    "supporting_fact",
    "target",
    "teacher",
    "future",
}
MATCHED_TRAIN_FIELDS = (
    "batch_size",
    "num_workers",
    "epochs",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "max_length",
    "grad_accum_steps",
    "max_grad_norm",
    "fp16",
    "margin_loss_weight",
    "margin",
    "role_aux_weight",
    "candidate_role_aux_weight",
    "deficit_aux_weight",
    "contribution_aux_weight",
)


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


def state_h_ids(row: dict) -> list[str]:
    result = []
    for item in (row.get("state") or {}).get("H_t") or []:
        if isinstance(item, dict) and item.get("unit_id"):
            result.append(str(item["unit_id"]))
        elif isinstance(item, str):
            result.append(item)
    return result


def candidate_ids(row: dict) -> list[str]:
    return list(((row.get("candidates") or {}).get("C_t") or []))


def positive_id(row: dict) -> str:
    return str(
        (((row.get("labels") or {}).get("ranking_label") or {}).get("positive_unit_id"))
        or ""
    )


def repetition(row: dict) -> int:
    return int((row.get("build_meta") or {}).get("repetition", 0))


def limit_qids(rows, max_qids: int):
    if max_qids <= 0:
        return list(rows)
    kept = []
    seen = set()
    for row in rows:
        qid = str(row.get("qid") or "")
        if qid not in seen and len(seen) >= max_qids:
            continue
        seen.add(qid)
        kept.append(row)
    return kept


def load_required_memory(path: Path, required_ids: set[str]) -> dict[str, dict]:
    memory = {}
    for row_index, row in enumerate(read_jsonl(path), start=1):
        unit_id = str(row.get("unit_id") or "")
        if unit_id in required_ids:
            memory[unit_id] = row
        if row_index % 100000 == 0:
            print(
                f"[CHECK] memory={path.name} scanned={row_index} "
                f"resolved={len(memory)}/{len(required_ids)}",
                file=sys.stderr,
                flush=True,
            )
    return memory


def forbidden_keys(value, path: str = "clue_state") -> list[str]:
    failures = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_CLUE_KEYS:
                failures.append(f"{path}.{key}")
            failures.extend(forbidden_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(forbidden_keys(child, f"{path}[{index}]"))
    return failures


def inspect_split(
    split: str,
    source_path: Path,
    output_path: Path,
    memory_path: Path,
    max_qids: int,
) -> dict:
    source_rows = limit_qids(read_jsonl(source_path), max_qids)
    output_rows = list(read_jsonl(output_path))
    required_ids = {
        unit_id
        for row in output_rows
        for unit_id in state_h_ids(row)
    }
    memory = load_required_memory(memory_path, required_ids)
    errors = []
    qids = set()
    t_distribution = Counter()
    clue_count_distribution = Counter()
    covered_by_t = Counter()
    clues_by_t = Counter()
    clue_templates = {}
    first_by_state = {}
    renderer_matches = 0
    pair_matches = 0
    forbidden_key_count = 0

    for index, pair in enumerate(
        itertools.zip_longest(source_rows, output_rows),
        start=1,
    ):
        source, output = pair
        if source is None or output is None:
            errors.append(
                f"row count mismatch: source={len(source_rows)}, output={len(output_rows)}"
            )
            break
        qid = str(output.get("qid") or "")
        t = int(output.get("t", -1))
        rep = repetition(output)
        qids.add(qid)
        t_distribution[str(t)] += 1

        source_identity = (
            str(source.get("qid") or ""),
            int(source.get("t", -1)),
            repetition(source),
            str(source.get("question") or ""),
            state_h_ids(source),
            candidate_ids(source),
            positive_id(source),
        )
        output_identity = (
            qid,
            t,
            rep,
            str(output.get("question") or ""),
            state_h_ids(output),
            candidate_ids(output),
            positive_id(output),
        )
        if source_identity != output_identity:
            errors.append(f"source pairing changed at row {index}: qid={qid}, t={t}")
            continue
        pair_matches += 1

        clue_state = (output.get("state") or {}).get("clue_state")
        if not isinstance(clue_state, dict):
            errors.append(f"missing clue_state: qid={qid}, t={t}")
            continue
        bad_keys = forbidden_keys(clue_state)
        forbidden_key_count += len(bad_keys)
        if bad_keys:
            errors.append(f"forbidden clue fields: qid={qid}, fields={bad_keys[:3]}")

        question = str(output["question"])
        template = extract_question_clues(question)
        if qid in clue_templates and clue_templates[qid] != template:
            errors.append(f"question clue template changed across states: qid={qid}")
        clue_templates[qid] = template

        missing_memory = [unit_id for unit_id in state_h_ids(output) if unit_id not in memory]
        if missing_memory:
            errors.append(f"prefix evidence missing from memory: qid={qid}, {missing_memory[:3]}")
            continue
        prefix_texts = [
            format_clue_evidence(memory[unit_id])
            for unit_id in state_h_ids(output)
        ]
        expected = build_clue_state(question, prefix_texts)
        expected_render = render_clue_state(expected)
        if clue_state != expected:
            errors.append(f"clue state is not reproducible: qid={qid}, t={t}")
        if str((output.get("state") or {}).get("K_t") or "") != expected_render:
            errors.append(f"clue renderer mismatch: qid={qid}, t={t}")
        else:
            renderer_matches += 1

        vector = list(clue_state.get("coverage_vector") or [])
        clues = list(clue_state.get("clues") or [])
        if len(vector) != len(clues) or any(value not in (0, 1) for value in vector):
            errors.append(f"invalid clue coverage vector: qid={qid}, t={t}")
        if t == 0 and any(vector):
            errors.append(f"initial clue state is not unresolved: qid={qid}")
        clue_count_distribution[str(len(clues))] += 1
        covered_by_t[str(t)] += sum(vector)
        clues_by_t[str(t)] += len(vector)
        first_by_state.setdefault((qid, t), vector)

    non_monotonic = 0
    by_qid = defaultdict(dict)
    for (qid, t), vector in first_by_state.items():
        by_qid[qid][t] = vector
    for qid, states in by_qid.items():
        for t in sorted(states):
            if t + 1 not in states:
                continue
            before = states[t]
            after = states[t + 1]
            if len(before) != len(after) or any(a < b for a, b in zip(after, before)):
                non_monotonic += 1
                errors.append(f"clue coverage is non-monotonic: qid={qid}, t={t}")

    return {
        "split": split,
        "source_rows": len(source_rows),
        "output_rows": len(output_rows),
        "paired_rows": pair_matches,
        "qids": len(qids),
        "t_distribution": dict(t_distribution),
        "clue_count_distribution": dict(clue_count_distribution),
        "covered_clue_rate_by_t": {
            t: round(covered_by_t[t] / max(clues_by_t[t], 1), 6)
            for t in sorted(clues_by_t, key=int)
        },
        "renderer_matches": renderer_matches,
        "forbidden_key_count": forbidden_key_count,
        "non_monotonic_transitions": non_monotonic,
        "error_count": len(errors),
        "errors": errors[:30],
        "_qid_set": qids,
    }


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def inspect_config(config_path: Path, reference_path: Path, data_root: Path) -> dict:
    config = load_yaml(config_path)
    reference = load_yaml(reference_path)
    mismatches = []
    if int(config.get("seed", -1)) != int(reference.get("seed", -2)):
        mismatches.append("seed")
    for field in MATCHED_TRAIN_FIELDS:
        if (config.get("train") or {}).get(field) != (reference.get("train") or {}).get(field):
            mismatches.append(f"train.{field}")
    expected_init = (reference.get("train") or {}).get("init_checkpoint")
    if (config.get("train") or {}).get("init_checkpoint") != expected_init:
        mismatches.append("train.init_checkpoint")
    for split in ("train", "val", "test"):
        expected_samples = str(data_root / "samples" / f"{split}.jsonl")
        configured_samples = str((config.get("data") or {}).get(f"{split}_samples") or "")
        if Path(configured_samples) != Path(expected_samples):
            mismatches.append(f"data.{split}_samples")
        if (config.get("data") or {}).get(f"{split}_memory") != (
            reference.get("data") or {}
        ).get(f"{split}_memory"):
            mismatches.append(f"data.{split}_memory")
    if (config.get("data") or {}).get("context_mode", "full_state") != "full_state":
        mismatches.append("data.context_mode")
    return {
        "config": str(config_path),
        "reference_config": str(reference_path),
        "seed": config.get("seed"),
        "init_checkpoint": (config.get("train") or {}).get("init_checkpoint"),
        "matched_fields": len(MATCHED_TRAIN_FIELDS) + 8,
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/hotpotqa_distractor_v26_fiske_clue_state",
    )
    parser.add_argument(
        "--source-root",
        default="data/hotpotqa_distractor_v22_state_focused",
    )
    parser.add_argument(
        "--memory-root",
        default="data/hotpotqa_distractor_v7_10k_llm_prestep/unit_registry",
    )
    parser.add_argument(
        "--config",
        default="configs/train_ranker_deberta_v26_fiske_clue_state.yaml",
    )
    parser.add_argument(
        "--config-data-root",
        default="data/hotpotqa_distractor_v26_fiske_clue_state",
        help="Canonical full-data root expected by the training config.",
    )
    parser.add_argument(
        "--reference-config",
        default="configs/train_ranker_deberta_v22_state_focused.yaml",
    )
    parser.add_argument(
        "--eval-queries",
        default="data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/analysis/kbs_stage3_fiske_clue_state_readiness.json",
    )
    parser.add_argument("--require-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    source_root = Path(args.source_root)
    memory_root = Path(args.memory_root)
    config_path = Path(args.config)
    reference_path = Path(args.reference_config)
    manifest_path = data_root / "manifest.json"
    required = [manifest_path, config_path, reference_path]
    for split in ("train", "val", "test"):
        required.extend(
            [
                data_root / "samples" / f"{split}.jsonl",
                source_root / "samples" / f"{split}.jsonl",
                memory_root / f"raw_units_{split}.jsonl",
            ]
        )
    if args.require_paths:
        config = load_yaml(config_path) if config_path.is_file() else {}
        required.extend(
            [
                Path((config.get("train") or {}).get("init_checkpoint", "")),
                Path((config.get("model") or {}).get("pretrained_name", "")),
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        result = {"status": "MISSING", "missing_paths": missing}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    max_qids = int(manifest.get("max_qids", 0))
    reports = {}
    for split in ("train", "val", "test"):
        print(f"[CHECK] split={split} started", file=sys.stderr, flush=True)
        reports[split] = inspect_split(
            split,
            source_root / "samples" / f"{split}.jsonl",
            data_root / "samples" / f"{split}.jsonl",
            memory_root / f"raw_units_{split}.jsonl",
            max_qids,
        )
        print(
            f"[CHECK] split={split} paired={reports[split]['paired_rows']} "
            f"errors={reports[split]['error_count']} completed",
            file=sys.stderr,
            flush=True,
        )
    qid_sets = {split: reports[split].pop("_qid_set") for split in reports}
    overlap = {
        "train_val": len(qid_sets["train"] & qid_sets["val"]),
        "train_test": len(qid_sets["train"] & qid_sets["test"]),
        "val_test": len(qid_sets["val"] & qid_sets["test"]),
    }
    eval_overlap = None
    eval_path = Path(args.eval_queries)
    if eval_path.is_file():
        eval_qids = {str(row.get("qid") or "") for row in read_jsonl(eval_path)}
        eval_overlap = {
            "eval_qids": len(eval_qids),
            "train_eval": len(qid_sets["train"] & eval_qids),
            "val_eval": len(qid_sets["val"] & eval_qids),
            "test_eval": len(qid_sets["test"] & eval_qids),
        }

    config_report = inspect_config(
        config_path,
        reference_path,
        Path(args.config_data_root),
    )
    failures = []
    for split, report in reports.items():
        if report["error_count"]:
            failures.append(f"{split}: {report['error_count']} audit errors")
        if report["source_rows"] != report["output_rows"]:
            failures.append(f"{split}: source/output row count differs")
    if any(overlap.values()):
        failures.append(f"split qid overlap: {overlap}")
    if eval_overlap and any(eval_overlap[key] for key in ("train_eval", "val_eval", "test_eval")):
        failures.append(f"evaluation qid overlap: {eval_overlap}")
    if config_report["mismatches"]:
        failures.append(f"config mismatches: {config_report['mismatches']}")
    if manifest.get("faithful_fiske_reproduction") is not False:
        failures.append("manifest must explicitly mark this as an inspired baseline")
    if (manifest.get("question_clue_generator") or {}).get("inputs") != ["question"]:
        failures.append("question clue generator has non-question inputs")
    if int((manifest.get("question_clue_generator") or {}).get("api_calls", -1)) != 0:
        failures.append("question clue generator is not API-free")
    if not manifest.get("auxiliary_labels_masked"):
        failures.append("auxiliary labels are not declared masked")

    result = {
        "status": "OK" if not failures else "FAIL",
        "baseline_name": manifest.get("baseline_name"),
        "data_root": str(data_root),
        "leakage_contract": {
            "generator_inputs": (manifest.get("question_clue_generator") or {}).get("inputs"),
            "coverage_inputs": (manifest.get("coverage_rule") or {}).get("inputs"),
            "forbidden_inputs": manifest.get("forbidden_inputs"),
            "api_calls": (manifest.get("question_clue_generator") or {}).get("api_calls"),
        },
        "splits": reports,
        "qid_overlap": overlap,
        "evaluation_overlap": eval_overlap,
        "config_match": config_report,
        "failures": failures,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
