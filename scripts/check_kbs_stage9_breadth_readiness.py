#!/usr/bin/env python3
"""Audit Stage 9.6 breadth-enhancement prerequisites without running models."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


KNOWN_MUSIQUE_PATHS = (
    Path("data/musique/musique_ans_v1.0_dev.jsonl"),
    Path("data/musique_v1.0/data/musique_ans_v1.0_dev.jsonl"),
    Path("data/musique_v1.0/musique_ans_v1.0_dev.jsonl"),
    Path("data/musique_ans_v1.0_dev.jsonl"),
    Path("data/musique/musique_ans_dev.jsonl"),
    Path("data/musique_ans_dev.jsonl"),
)
REQUIRED_RUNTIME_PATHS = (
    Path("md/kbs_review_75_85_execution_plan.md"),
    Path("scripts/run_hotpotqa_policy_rag.py"),
    Path("scripts/evaluate_kbs_standard_metrics.py"),
    Path("scripts/bootstrap_kbs_stage4_metrics.py"),
    Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"),
    Path("models/deberta-v3-large"),
    Path("models/bge-large-en-v1.5"),
)
HOTPot_QID_PATHS = (
    Path("data/hotpotqa_distractor_v27_counterfactual_dual/queries/train.jsonl"),
    Path("data/hotpotqa_distractor_v27_counterfactual_dual/queries/val.jsonl"),
    Path("data/hotpotqa_distractor_v27_counterfactual_dual/queries/test.jsonl"),
    Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl"),
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(row)
        return rows

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = next(
            (
                value[key]
                for key in ("data", "examples", "dev", "validation")
                if isinstance(value.get(key), list)
            ),
            None,
        )
        if rows is None:
            raise ValueError(f"unsupported JSON container in {path}")
    else:
        raise ValueError(f"unsupported JSON root in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} contains non-object rows")
    return list(rows)


def discover_musique_files(scan_root: Path) -> tuple[list[Path], list[Path]]:
    data_files = [path for path in KNOWN_MUSIQUE_PATHS if path.is_file()]
    archives: list[Path] = []
    if not scan_root.is_dir():
        return data_files, archives

    root_depth = len(scan_root.resolve().parts)
    for root, dirs, files in os.walk(scan_root):
        depth = len(Path(root).resolve().parts) - root_depth
        if depth >= 4:
            dirs[:] = []
        for name in files:
            lowered = name.lower()
            if "musique" not in lowered:
                continue
            path = Path(root) / name
            if path.suffix.lower() in {".zip", ".gz", ".tgz", ".zst"}:
                archives.append(path)
            elif (
                path.suffix.lower() in {".json", ".jsonl"}
                and ("dev" in lowered or "validation" in lowered)
            ):
                data_files.append(path)

    return sorted(set(data_files), key=lambda path: str(path)), sorted(
        set(archives), key=lambda path: str(path)
    )


def qid_of(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("qid") or row.get("question_id") or "")


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_row(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    qid = qid_of(row)
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    answerable = row.get("answerable", True)
    paragraphs = row.get("paragraphs")
    decomposition = row.get("question_decomposition")

    if not qid:
        reasons.append("missing_id")
    if not question:
        reasons.append("missing_question")
    if answerable is not True:
        reasons.append("not_answerable")
    if not answer:
        reasons.append("missing_answer")
    if not isinstance(paragraphs, list) or not paragraphs:
        reasons.append("missing_paragraphs")
        paragraphs = []
    if not isinstance(decomposition, list) or not decomposition:
        reasons.append("missing_question_decomposition")
        decomposition = []

    paragraph_indices: list[int] = []
    support_indices: list[int] = []
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            reasons.append("invalid_paragraph_object")
            continue
        index = int_value(paragraph.get("idx"))
        if index is None:
            reasons.append("invalid_paragraph_idx")
        else:
            paragraph_indices.append(index)
            if paragraph.get("is_supporting") is True:
                support_indices.append(index)
        if not str(paragraph.get("paragraph_text") or "").strip():
            reasons.append("empty_paragraph_text")

    if len(paragraph_indices) != len(set(paragraph_indices)):
        reasons.append("duplicate_paragraph_idx")
    if not support_indices:
        reasons.append("missing_supporting_paragraphs")

    decomposition_support: list[int] = []
    for item in decomposition:
        if not isinstance(item, dict):
            reasons.append("invalid_decomposition_object")
            continue
        index = int_value(item.get("paragraph_support_idx"))
        if index is None:
            reasons.append("invalid_decomposition_support_idx")
        else:
            decomposition_support.append(index)
        if not str(item.get("question") or "").strip():
            reasons.append("empty_decomposition_question")

    paragraph_index_set = set(paragraph_indices)
    if any(index not in paragraph_index_set for index in decomposition_support):
        reasons.append("decomposition_support_out_of_range")
    if set(decomposition_support) != set(support_indices):
        reasons.append("support_mapping_mismatch")
    if len(decomposition_support) != len(support_indices):
        reasons.append("support_mapping_not_one_to_one")

    return sorted(set(reasons)), {
        "qid": qid,
        "paragraphs": len(paragraphs),
        "supports": len(support_indices),
        "hops": len(decomposition),
        "answer_aliases": len(row.get("answer_aliases") or []),
    }


def load_qids(paths: Iterable[Path]) -> tuple[set[str], list[str]]:
    qids: set[str] = set()
    used: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        used.append(str(path))
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = qid_of(row)
                if qid:
                    qids.add(qid)
    return qids, used


def counter_dict(counter: Counter[int | str]) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter, key=lambda value: (str(type(value)), value))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--musique-path", type=Path)
    parser.add_argument("--scan-root", type=Path, default=Path("data"))
    parser.add_argument("--target-qids", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_breadth/readiness.json"),
    )
    args = parser.parse_args()

    discovered, archives = discover_musique_files(args.scan_root)
    if args.musique_path:
        selected_path = args.musique_path
        if selected_path.is_file() and selected_path not in discovered:
            discovered.insert(0, selected_path)
    else:
        selected_path = next(
            (path for path in KNOWN_MUSIQUE_PATHS if path.is_file()),
            discovered[0] if discovered else None,
        )

    required_paths = {
        str(path): {"exists": path.exists(), "is_file": path.is_file()}
        for path in REQUIRED_RUNTIME_PATHS
    }
    runtime_missing = [path for path in REQUIRED_RUNTIME_PATHS if not path.exists()]
    runtime_path = Path("scripts/run_hotpotqa_policy_rag.py")
    runtime_text = runtime_path.read_text(encoding="utf-8") if runtime_path.is_file() else ""
    second_generator_audit = {
        "independent_backend_registered": False,
        "runtime_uses_deepseek_specific_environment": (
            "DEEPSEEK_API_KEY" in runtime_text and "DEEPSEEK_BASE_URL" in runtime_text
        ),
        "credential_values_inspected": False,
        "ready": False,
        "reason": (
            "The current answer runtime is DeepSeek-specific; no independent non-DeepSeek "
            "generator adapter and frozen protocol are registered."
        ),
    }

    schema_audit: dict[str, Any] | None = None
    data_failures: list[str] = []
    musique_ready = False
    if selected_path is None or not selected_path.is_file():
        data_failures.append("MuSiQue-Ans development data file was not found")
    else:
        try:
            rows = load_rows(selected_path)
            reason_counts: Counter[str] = Counter()
            paragraph_counts: Counter[int] = Counter()
            support_counts: Counter[int] = Counter()
            hop_counts: Counter[int] = Counter()
            alias_counts: Counter[int] = Counter()
            valid_qids: list[str] = []
            all_qids: list[str] = []
            for row in rows:
                reasons, stats = audit_row(row)
                if stats["qid"]:
                    all_qids.append(stats["qid"])
                for reason in reasons:
                    reason_counts[reason] += 1
                if reasons:
                    continue
                valid_qids.append(stats["qid"])
                paragraph_counts[stats["paragraphs"]] += 1
                support_counts[stats["supports"]] += 1
                hop_counts[stats["hops"]] += 1
                alias_counts[stats["answer_aliases"]] += 1

            duplicate_ids = len(all_qids) - len(set(all_qids))
            hotpot_qids, hotpot_sources = load_qids(HOTPot_QID_PATHS)
            overlap = len(set(valid_qids) & hotpot_qids)
            budget_coverage = {
                str(budget): sum(count for size, count in paragraph_counts.items() if size >= budget)
                for budget in (10, 15, 20, 50)
            }
            enough_rows = len(valid_qids) >= args.target_qids
            compact_ready = budget_coverage["10"] >= args.target_qids
            balanced_ready = budget_coverage["15"] >= args.target_qids
            musique_ready = (
                enough_rows
                and compact_ready
                and duplicate_ids == 0
                and overlap == 0
                and not runtime_missing
            )
            if not enough_rows:
                data_failures.append(
                    f"only {len(valid_qids)} schema-valid answerable rows; need {args.target_qids}"
                )
            if not compact_ready:
                data_failures.append(
                    f"only {budget_coverage['10']} valid rows support candidate budget 10"
                )
            if duplicate_ids:
                data_failures.append(f"duplicate MuSiQue ids: {duplicate_ids}")
            if overlap:
                data_failures.append(f"MuSiQue/Hotpot qid overlap: {overlap}")

            schema_audit = {
                "path": str(selected_path),
                "rows": len(rows),
                "valid_answerable_rows": len(valid_qids),
                "excluded_rows": len(rows) - len(valid_qids),
                "exclusion_reasons": dict(sorted(reason_counts.items())),
                "duplicate_ids": duplicate_ids,
                "paragraph_count_distribution": counter_dict(paragraph_counts),
                "support_count_distribution": counter_dict(support_counts),
                "hop_count_distribution": counter_dict(hop_counts),
                "answer_alias_count_distribution": counter_dict(alias_counts),
                "rows_supporting_candidate_budget": budget_coverage,
                "hotpot_qid_sources_checked": hotpot_sources,
                "hotpot_qid_overlap": overlap,
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            data_failures.append(f"MuSiQue schema audit failed: {exc}")

    if runtime_missing:
        data_failures.extend(f"missing runtime prerequisite: {path}" for path in runtime_missing)

    if musique_ready:
        status = "READY_MUSIQUE"
        selected_branch = "musique_zero_shot_transfer"
        next_gate = (
            "Implement and audit a deterministic MuSiQue-Ans dev adapter using paragraph "
            "units; do not run GPU inference or generate answers yet."
        )
    elif second_generator_audit["ready"]:
        status = "READY_SECOND_GENERATOR"
        selected_branch = "non_deepseek_generator_replication"
        next_gate = "Freeze the independent generator protocol before one bounded smoke."
    else:
        status = "BLOCKED_MISSING_PREREQUISITES"
        selected_branch = None
        next_gate = (
            "Provide the official MuSiQue-Ans development file, preferably "
            "musique_ans_v1.0_dev.jsonl, then rerun this readiness audit."
        )

    result = {
        "status": status,
        "stage": 9,
        "step": "9.6",
        "mode": "breadth_validation_readiness",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "training_runs": 0,
        "selected_branch": selected_branch,
        "musique_contract": {
            "source": "official MuSiQue-Ans development split",
            "target_qids": args.target_qids,
            "sampling_seed": args.seed,
            "unit_granularity": "paragraph",
            "candidate_memory": "question-local official paragraphs",
            "gold_units": "paragraphs marked is_supporting",
            "trajectory_order": "question_decomposition.paragraph_support_idx",
            "decomposition_text_visible_to_policy": False,
            "student_training": "HotpotQA only; no MuSiQue fine-tuning",
            "primary_operating_point": "Compact candidate budget 10",
            "metric_boundary": (
                "MuSiQue evidence metrics are paragraph-level and must not be described as "
                "the same sentence-level Step@5 used for HotpotQA/2Wiki."
            ),
        },
        "official_schema_required": {
            "row": [
                "id",
                "question",
                "answerable",
                "answer",
                "answer_aliases",
                "paragraphs",
                "question_decomposition",
            ],
            "paragraph": ["idx", "title", "paragraph_text", "is_supporting"],
            "decomposition": ["question", "answer", "paragraph_support_idx"],
        },
        "discovery": {
            "selected_path": str(selected_path) if selected_path else None,
            "candidate_data_files": [str(path) for path in discovered],
            "candidate_archives": [str(path) for path in archives],
        },
        "schema_audit": schema_audit,
        "runtime_prerequisites": required_paths,
        "second_generator_audit": second_generator_audit,
        "readiness_failures": data_failures,
        "next_gate": next_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
