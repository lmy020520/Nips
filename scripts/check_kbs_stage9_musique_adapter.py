#!/usr/bin/env python3
"""Independently replay and audit the Stage 9.6 MuSiQue adapter outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.prepare_musique_policy_rag_eval import (
        candidate_ids,
        file_sha256,
        normalize_row,
        paragraph_unit_id,
        read_jsonl,
        render_notebook,
        stable_key,
    )
except ModuleNotFoundError:
    from prepare_musique_policy_rag_eval import (  # type: ignore[no-redef]
        candidate_ids,
        file_sha256,
        normalize_row,
        paragraph_unit_id,
        read_jsonl,
        render_notebook,
        stable_key,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def forbidden_keys(value: Any, path: str = "root") -> list[str]:
    forbidden = {"question_decomposition", "paragraph_support_idx", "is_supporting"}
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in forbidden:
                found.append(child_path)
            found.extend(forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_keys(child, f"{path}[{index}]"))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/musique_ans_eval_1000_paragraph20"),
    )
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/kbs_stage9_breadth/adapter_readiness.json"),
    )
    args = parser.parse_args()

    paths = {
        "manifest": args.data_root / "manifest.json",
        "queries": args.data_root / "queries/test.jsonl",
        "memory": args.data_root / "unit_registry/raw_units_test.jsonl",
        "targets": args.data_root / "targets/test.jsonl",
        "samples": args.data_root / "samples/test.jsonl",
    }
    failures = [f"missing adapter output: {path}" for path in paths.values() if not path.is_file()]
    audit: dict[str, Any] = {}

    if not failures:
        try:
            manifest = load_json(paths["manifest"])
            queries = read_jsonl(paths["queries"])
            memory = read_jsonl(paths["memory"])
            targets = read_jsonl(paths["targets"])
            samples = read_jsonl(paths["samples"])
            source_rows = [normalize_row(row) for row in read_jsonl(args.source_json)]
            source_rows.sort(key=lambda row: stable_key(row["qid"], args.seed))
            expected = source_rows[: args.size]
            expected_qids = [row["qid"] for row in expected]
            expected_qid_hash = hashlib.sha256(
                "\n".join(expected_qids).encode("utf-8")
            ).hexdigest()

            expected_manifest = {
                "source": str(args.source_json),
                "source_split": "dev",
                "output_split": "test",
                "size": args.size,
                "seed": args.seed,
                "unit_granularity": "paragraph",
                "decomposition_text_visible_to_policy": False,
                "recall_50_supported": False,
            }
            for key, value in expected_manifest.items():
                if manifest.get(key) != value:
                    failures.append(f"manifest {key}: {manifest.get(key)!r} != {value!r}")
            manifest_stats = manifest.get("stats") or {}
            source_hash = file_sha256(args.source_json)
            if manifest_stats.get("source_sha256") != source_hash:
                failures.append("manifest source SHA-256 does not match the audited source")
            if manifest_stats.get("ordered_qids_sha256") != expected_qid_hash:
                failures.append("manifest ordered-qid SHA-256 does not match replay")

            query_qids = [str(row.get("qid") or "") for row in queries]
            target_qids = [str(row.get("qid") or "") for row in targets]
            if query_qids != expected_qids:
                failures.append("query qid order differs from deterministic source sample")
            if target_qids != expected_qids:
                failures.append("target qid order differs from deterministic source sample")
            if len(set(query_qids)) != len(query_qids):
                failures.append("duplicate query qids")

            memory_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
            memory_by_id = {}
            for unit in memory:
                unit_id = str(unit.get("unit_id") or "")
                qid = unit_id.split("::paragraph::", 1)[0] if "::paragraph::" in unit_id else ""
                memory_by_qid[qid].append(unit)
                if not unit_id or unit_id in memory_by_id:
                    failures.append(f"missing or duplicate memory unit id: {unit_id!r}")
                memory_by_id[unit_id] = unit

            targets_by_qid = {str(row.get("qid") or ""): row for row in targets}
            samples_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for sample in samples:
                samples_by_qid[str(sample.get("qid") or "")].append(sample)

            expected_qid_set = set(expected_qids)
            if set(memory_by_qid) != expected_qid_set:
                failures.append("memory qid set differs from deterministic source sample")
            if set(samples_by_qid) != expected_qid_set:
                failures.append("sample qid set differs from deterministic source sample")
            expected_memory_rows = sum(len(row["paragraphs"]) for row in expected)
            expected_sample_rows = sum(len(row["support_order"]) for row in expected)
            if len(memory) != expected_memory_rows:
                failures.append(
                    f"memory rows: {len(memory)} != expected {expected_memory_rows}"
                )
            if len(samples) != expected_sample_rows:
                failures.append(
                    f"sample rows: {len(samples)} != expected {expected_sample_rows}"
                )

            candidate_size_distribution: Counter[int] = Counter()
            alias_rows = 0
            for source_row, query in zip(expected, queries):
                qid = source_row["qid"]
                if query.get("question") != source_row["question"] or query.get("answer") != source_row["answer"]:
                    failures.append(f"query text/answer mismatch: {qid}")
                if query.get("answer_aliases") != source_row["answer_aliases"]:
                    failures.append(f"answer aliases mismatch: {qid}")
                alias_rows += int(bool(source_row["answer_aliases"]))

                expected_units = {
                    paragraph_unit_id(qid, paragraph["idx"]): paragraph
                    for paragraph in source_row["paragraphs"]
                }
                observed_units = {str(unit.get("unit_id") or ""): unit for unit in memory_by_qid[qid]}
                if set(observed_units) != set(expected_units):
                    failures.append(f"question-local memory membership mismatch: {qid}")
                    continue
                for unit_id, paragraph in expected_units.items():
                    unit = observed_units[unit_id]
                    if unit.get("text") != paragraph["text"] or unit.get("doc_id") != paragraph["title"]:
                        failures.append(f"memory content mismatch: {unit_id}")
                    if unit.get("candidate_granularity") != "paragraph":
                        failures.append(f"non-paragraph memory unit: {unit_id}")

                expected_target_ids = [
                    paragraph_unit_id(qid, index) for index in source_row["support_order"]
                ]
                target = targets_by_qid.get(qid) or {}
                observed_target_ids = [
                    str(item.get("unit_id") or "") for item in target.get("T_q_raw") or []
                ]
                if observed_target_ids != expected_target_ids:
                    failures.append(f"ordered gold mapping mismatch: {qid}")
                if target.get("evidence_granularity") != "paragraph":
                    failures.append(f"target granularity mismatch: {qid}")

                qid_samples = sorted(samples_by_qid[qid], key=lambda row: int(row.get("t", -1)))
                if len(qid_samples) != len(expected_target_ids):
                    failures.append(f"sample-step count mismatch: {qid}")
                    continue
                acquired: set[int] = set()
                previous_units = []
                for step, (sample, positive_idx) in enumerate(
                    zip(qid_samples, source_row["support_order"])
                ):
                    expected_positive = paragraph_unit_id(qid, positive_idx)
                    expected_candidates = candidate_ids(source_row, acquired, args.seed)
                    candidate_size_distribution[len(expected_candidates)] += 1
                    state = sample.get("state") or {}
                    observed_history = [
                        str(item.get("unit_id") or "") for item in state.get("H_t") or []
                    ]
                    expected_history = [str(unit["unit_id"]) for unit in previous_units]
                    if observed_history != expected_history:
                        failures.append(f"history mismatch: {qid} step={step}")
                    if state.get("K_t") != render_notebook(previous_units):
                        failures.append(f"rendered notebook mismatch: {qid} step={step}")
                    candidates = sample.get("candidates") or {}
                    if candidates.get("C_t") != expected_candidates or candidates.get("R_t") != expected_candidates:
                        failures.append(f"candidate order mismatch: {qid} step={step}")
                    labels = sample.get("labels") or {}
                    ranking = labels.get("ranking_label") or {}
                    if ranking.get("positive_unit_id") != expected_positive:
                        failures.append(f"positive mismatch: {qid} step={step}")
                    if expected_positive not in expected_candidates:
                        failures.append(f"positive absent from candidates: {qid} step={step}")
                    expected_negative = [
                        unit_id for unit_id in expected_candidates if unit_id != expected_positive
                    ]
                    if ranking.get("negative_unit_ids") != expected_negative:
                        failures.append(f"negative order mismatch: {qid} step={step}")
                    acquired.add(positive_idx)
                    previous_units.append(observed_units[expected_positive])

            leaked_keys = forbidden_keys(queries) + forbidden_keys(samples)
            if leaked_keys:
                failures.append(
                    f"teacher-only schema keys leaked into policy-facing files: {leaked_keys[:5]}"
                )

            audit = {
                "source_rows": len(source_rows),
                "selected_qids": len(expected_qids),
                "queries": len(queries),
                "samples": len(samples),
                "memory_rows": len(memory),
                "targets": len(targets),
                "answer_alias_rows": alias_rows,
                "candidate_size_distribution": dict(sorted(candidate_size_distribution.items())),
                "ordered_qids_sha256": hashlib.sha256(
                    "\n".join(query_qids).encode("utf-8")
                ).hexdigest(),
                "source_sha256": source_hash,
                "policy_input_teacher_key_leaks": len(leaked_keys),
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append(f"adapter audit failed: {exc}")

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 9,
        "step": "9.6",
        "mode": "musique_paragraph_adapter_readiness",
        "api_calls": 0,
        "gpu_inference_runs": 0,
        "training_runs": 0,
        "protocol": {
            "source": str(args.source_json),
            "data_root": str(args.data_root),
            "source_split": "dev",
            "evaluation_qids": args.size,
            "sampling_seed": args.seed,
            "unit_granularity": "paragraph",
            "candidate_memory": "question-local official paragraphs",
            "gold_order": "question_decomposition.paragraph_support_idx",
            "decomposition_text_visible_to_policy": False,
            "student_training": "HotpotQA only; no MuSiQue fine-tuning",
            "recall_50_supported": False,
        },
        "files": {name: str(path) for name, path in paths.items()},
        "audit": audit,
        "next_gate": (
            "Run a 20-qid Compact selection-only smoke; do not generate answers."
            if not failures
            else "Resolve adapter audit failures before GPU inference or API calls."
        ),
        "failures": failures[:100],
        "failure_count": len(failures),
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
