#!/usr/bin/env python3
"""Check whether the KBS evidence-acquisition pipeline assets are ready.

This script does not run experiments. It verifies that the official KBS online
RAG route has the required data, models, scripts, manifest settings, and
optionally an already-generated online-state report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
OFFICIAL_ROUTE_EXPECTED = {
    "state_mode": "policy",
    "policy_context_source": "online_state",
    "selector": "hybrid_policy",
    "dense_query_mode": "state",
    "front_fusion": "rrf",
    "policy_score_mode": "front_policy_blend",
    "answer_mode": "json",
    "save_online_states": True,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def read_jsonl_preview(path: Path, limit: int = 3) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
            if len(rows) >= limit:
                break
    return rows


def file_status(path: Path, *, required: bool = True) -> dict:
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else 0
    return {
        "path": str(path),
        "required": required,
        "exists": exists,
        "size": size,
        "ok": exists if required else True,
    }


def dir_status(path: Path, *, required: bool = True) -> dict:
    exists = path.exists() and path.is_dir()
    return {
        "path": str(path),
        "required": required,
        "exists": exists,
        "ok": exists if required else True,
    }


def add_file(checks: list[dict], name: str, path: Path, *, required: bool = True) -> None:
    item = file_status(path, required=required)
    item["name"] = name
    checks.append(item)


def add_dir(checks: list[dict], name: str, path: Path, *, required: bool = True) -> None:
    item = dir_status(path, required=required)
    item["name"] = name
    checks.append(item)


def collect_data_checks(data_root: Path) -> list[dict]:
    checks = []
    add_dir(checks, "data_root", data_root)
    for split in SPLITS:
        add_file(checks, f"samples/{split}", data_root / "samples" / f"{split}.jsonl")
        add_file(checks, f"memory/{split}", data_root / "unit_registry" / f"raw_units_{split}.jsonl")
        add_file(checks, f"queries/{split}", data_root / "queries" / f"{split}.jsonl", required=False)
        add_file(checks, f"targets/{split}", data_root / "targets" / f"{split}.jsonl", required=False)
    add_file(checks, "manifest", data_root / "manifest.json", required=False)
    return checks


def collect_single_split_data_checks(samples: Path, memory: Path, queries: Path) -> list[dict]:
    checks = []
    add_file(checks, "official_samples", samples)
    add_file(checks, "official_memory", memory)
    add_file(checks, "official_queries", queries)
    return checks


def collect_model_checks(model_dir: Path, checkpoint: Path, dense_model: Path | None) -> list[dict]:
    checks = []
    add_dir(checks, "model_dir", model_dir)
    add_file(checks, "model_config", model_dir / "config.json", required=False)
    add_file(checks, "checkpoint", checkpoint)
    if dense_model is not None:
        add_dir(checks, "dense_model", dense_model, required=False)
    return checks


def collect_script_checks(project_root: Path) -> list[dict]:
    checks = []
    scripts = [
        "configs/kbs_official_online_rag_v1_manifest.json",
        "scripts/run_kbs_official_online_rag.sh",
        "scripts/rebuild_hotpotqa_frontend_dataset.py",
        "scripts/run_hotpotqa_policy_rag.py",
        "scripts/analyze_hotpotqa_frontend_policy_ranks.py",
        "scripts/diagnose_frontend_trace.py",
        "scripts/validate_kbs_online_state_alignment.py",
        "scripts/check_kbs_pipeline_readiness.py",
        "src/datasets/prefix_dataset.py",
        "src/models/ranker.py",
        "src/train/train_ranker.py",
    ]
    for script in scripts:
        add_file(checks, script, project_root / script)
    return checks


def nested_get(obj: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def collect_manifest_checks(manifest: dict[str, Any], manifest_path: Path) -> list[dict]:
    checks = []

    def add_value(name: str, actual: Any, expected: Any, *, required: bool = True) -> None:
        checks.append(
            {
                "name": name,
                "path": str(manifest_path),
                "required": required,
                "exists": actual is not None,
                "actual": actual,
                "expected": expected,
                "ok": (actual == expected) if required else True,
            }
        )

    for key, expected in OFFICIAL_ROUTE_EXPECTED.items():
        add_value(f"official_route.{key}", nested_get(manifest, ["official_route", key]), expected)

    numeric_expected = {
        "front_pool_k": 30,
        "local_expansion_window": 1,
        "candidate_top_k": 10,
        "select_top_k": 5,
    }
    for key, expected in numeric_expected.items():
        add_value(f"official_route.{key}", nested_get(manifest, ["official_route", key]), expected)

    blend_weight = nested_get(manifest, ["official_route", "policy_blend_weight"])
    checks.append(
        {
            "name": "official_route.policy_blend_weight",
            "path": str(manifest_path),
            "required": True,
            "exists": blend_weight is not None,
            "actual": blend_weight,
            "expected": 0.35,
            "ok": abs(float(blend_weight or 0.0) - 0.35) < 1e-9,
        }
    )
    return checks


def collect_report_checks(report_path: Path, manifest: dict[str, Any]) -> list[dict]:
    checks = []
    if not report_path:
        return checks
    if not report_path.exists():
        checks.append(
            {
                "name": "rag_report",
                "path": str(report_path),
                "required": True,
                "exists": False,
                "ok": False,
            }
        )
        return checks

    report = read_json(report_path)
    checks.append(
        {
            "name": "rag_report",
            "path": str(report_path),
            "required": True,
            "exists": True,
            "ok": True,
        }
    )

    expected_route = manifest.get("official_route") if isinstance(manifest.get("official_route"), dict) else {}
    for key in [
        "state_mode",
        "policy_context_source",
        "selector",
        "dense_query_mode",
        "front_fusion",
        "candidate_top_k",
        "select_top_k",
        "policy_score_mode",
        "policy_blend_weight",
        "answer_mode",
        "save_online_states",
    ]:
        expected = expected_route.get(key)
        actual = report.get(key)
        ok = actual == expected
        if isinstance(expected, float):
            ok = abs(float(actual or 0.0) - expected) < 1e-9
        checks.append(
            {
                "name": f"report.{key}",
                "path": str(report_path),
                "required": True,
                "exists": key in report,
                "actual": actual,
                "expected": expected,
                "ok": ok,
            }
        )

    records = report.get("records")
    has_records = isinstance(records, list) and bool(records)
    checks.append(
        {
            "name": "report.records_non_empty",
            "path": str(report_path),
            "required": True,
            "exists": has_records,
            "ok": has_records,
        }
    )
    if has_records:
        first_record = records[0]
        first_steps = first_record.get("steps") if isinstance(first_record, dict) else None
        first_step = first_steps[0] if isinstance(first_steps, list) and first_steps else {}
        final_state = first_record.get("final_online_state") if isinstance(first_record, dict) else None
        checks.extend(
            [
                {
                    "name": "report.step.online_state_before",
                    "path": str(report_path),
                    "required": True,
                    "exists": isinstance(first_step, dict) and "online_state_before" in first_step,
                    "ok": isinstance(first_step, dict) and isinstance(first_step.get("online_state_before"), dict),
                },
                {
                    "name": "report.step.online_state_after",
                    "path": str(report_path),
                    "required": True,
                    "exists": isinstance(first_step, dict) and "online_state_after" in first_step,
                    "ok": isinstance(first_step, dict) and isinstance(first_step.get("online_state_after"), dict),
                },
                {
                    "name": "report.final_online_state",
                    "path": str(report_path),
                    "required": True,
                    "exists": isinstance(final_state, dict),
                    "ok": isinstance(final_state, dict),
                },
            ]
        )
    return checks


def collect_sample_schema_checks(samples: Path, memory: Path, queries: Path) -> list[dict]:
    checks = []
    sample_rows = read_jsonl_preview(samples)
    memory_rows = read_jsonl_preview(memory)
    query_rows = read_jsonl_preview(queries)
    sample_required = ["qid", "question", "candidates", "labels"]
    memory_required = ["unit_id", "text"]
    query_required = ["qid", "question", "answer"]

    def add_schema_checks(prefix: str, row: dict[str, Any], required_keys: list[str], path: Path) -> None:
        for key in required_keys:
            checks.append(
                {
                    "name": f"{prefix}.{key}",
                    "path": str(path),
                    "required": True,
                    "exists": key in row,
                    "ok": key in row,
                }
            )

    add_schema_checks("sample_preview", sample_rows[0] if sample_rows else {}, sample_required, samples)
    add_schema_checks("memory_preview", memory_rows[0] if memory_rows else {}, memory_required, memory)
    add_schema_checks("query_preview", query_rows[0] if query_rows else {}, query_required, queries)
    return checks


def summarize_stage(name: str, checks: list[dict]) -> dict:
    required = [item for item in checks if item["required"]]
    missing = [item for item in required if not item["exists"]]
    return {
        "name": name,
        "ok": not missing,
        "required": len(required),
        "missing": [item["name"] for item in missing],
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/kbs_official_online_rag_v1_manifest.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--dense-model", default="")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--rag-report", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    data_root = Path(args.data_root or nested_get(manifest, ["data", "data_root"], ""))
    samples = Path(nested_get(manifest, ["data", "samples"], data_root / "samples" / "test.jsonl"))
    memory = Path(nested_get(manifest, ["data", "memory"], data_root / "unit_registry" / "raw_units_test.jsonl"))
    queries = Path(nested_get(manifest, ["data", "queries"], data_root / "queries" / "test.jsonl"))
    checkpoint = Path(args.checkpoint or nested_get(manifest, ["models", "student_checkpoint"], ""))
    model_dir = Path(args.model_dir or nested_get(manifest, ["models", "model_dir"], "models/deberta-v3-large"))
    dense_model_value = args.dense_model or nested_get(manifest, ["models", "dense_model"], "")
    dense_model = Path(dense_model_value) if dense_model_value else None

    stages = [
        summarize_stage("manifest", [file_status(manifest_path) | {"name": "manifest"}] + collect_manifest_checks(manifest, manifest_path)),
        summarize_stage("data", collect_data_checks(data_root)),
        summarize_stage("official_split_data", collect_single_split_data_checks(samples, memory, queries)),
        summarize_stage("sample_schema_preview", collect_sample_schema_checks(samples, memory, queries)),
        summarize_stage("model", collect_model_checks(model_dir, checkpoint, dense_model)),
        summarize_stage("scripts", collect_script_checks(project_root)),
    ]
    if args.rag_report:
        stages.append(summarize_stage("rag_report", collect_report_checks(Path(args.rag_report), manifest)))

    summary = {
        "pipeline_ready": all(stage["ok"] for stage in stages),
        "manifest": str(manifest_path),
        "official_route_name": str(manifest.get("name") or ""),
        "data_root": str(data_root),
        "samples": str(samples),
        "memory": str(memory),
        "queries": str(queries),
        "checkpoint": str(checkpoint),
        "model_dir": str(model_dir),
        "dense_model": str(dense_model) if dense_model is not None else "",
        "rag_report": args.rag_report,
        "stages": stages,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
