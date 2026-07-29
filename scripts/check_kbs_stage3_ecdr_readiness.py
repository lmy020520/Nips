#!/usr/bin/env python3
"""Audit the matched ECDR-inspired textual direct-indirect experiment."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return value


def candidate_ids(row: dict) -> list[str]:
    candidates = row.get("candidates")
    if isinstance(candidates, list):
        return [str(value) for value in candidates]
    if isinstance(candidates, dict):
        values = candidates.get("C_t") or candidates.get("candidates") or []
        return [str(value) for value in values]
    return []


def positive_id(row: dict) -> str:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    ranking = labels.get("ranking_label") if isinstance(labels.get("ranking_label"), dict) else {}
    positive = (
        row.get("positive_unit_id")
        or ranking.get("positive_unit_id")
        or (labels.get("u_t_plus") or {}).get("unit_id")
    )
    return str(positive or "")


def history_ids(row: dict) -> list[str]:
    state = row.get("state") if isinstance(row.get("state"), dict) else {}
    history = state.get("H_t") if isinstance(state.get("H_t"), list) else []
    result = []
    for item in history:
        if isinstance(item, dict) and item.get("unit_id"):
            result.append(str(item["unit_id"]))
        elif isinstance(item, str):
            result.append(item)
    return result


def inspect_samples(path: Path) -> tuple[dict, set[str], set[str]]:
    rows = 0
    qids = set()
    required_memory_ids = set()
    rows_by_qid = defaultdict(lambda: defaultdict(list))
    pools_by_qid = {}
    t_counts = Counter()
    pool_sizes = Counter()
    errors = []
    duplicate_state_rows = 0
    conflicting_duplicate_rows = 0

    for row in read_jsonl(path):
        rows += 1
        qid = str(row.get("qid") or "")
        t = int(row.get("t", -1))
        pool = candidate_ids(row)
        positive = positive_id(row)
        history = history_ids(row)
        qids.add(qid)
        t_counts[str(t)] += 1
        pool_sizes[str(len(pool))] += 1
        required_memory_ids.update(pool)
        required_memory_ids.update(history)

        if not qid or t < 0:
            errors.append(f"invalid qid/t at row={rows}")
        state_record = {
            "positive": positive,
            "history": history,
        }
        existing_states = rows_by_qid[qid][t]
        if existing_states:
            duplicate_state_rows += 1
            if existing_states[0] != state_record:
                conflicting_duplicate_rows += 1
                errors.append(
                    "conflicting repeated qid/t row: "
                    f"qid={qid}, t={t}, first={existing_states[0]}, current={state_record}"
                )
        existing_states.append(state_record)
        if not pool or len(pool) != len(set(pool)):
            errors.append(f"invalid candidate pool: qid={qid}, t={t}")
        if positive not in pool:
            errors.append(f"teacher positive missing from pool: qid={qid}, t={t}")
        if len(history) != t:
            errors.append(f"H_t length does not equal t: qid={qid}, t={t}, len={len(history)}")
        build_meta = row.get("build_meta") if isinstance(row.get("build_meta"), dict) else {}
        if not bool(build_meta.get("mask_auxiliary_labels", False)):
            errors.append(f"auxiliary labels are not masked: qid={qid}, t={t}")
        pool_key = tuple(pool)
        if qid in pools_by_qid and pools_by_qid[qid] != pool_key:
            errors.append(f"candidate pool changes across prefixes: qid={qid}")
        pools_by_qid[qid] = pool_key

    direct_anchors = []
    direct_anchor_matches = 0
    later_states = 0
    for qid, qid_rows in rows_by_qid.items():
        initial_rows = qid_rows.get(0, [])
        if not initial_rows:
            errors.append(f"qid has no t=0 direct stage: qid={qid}")
            continue
        if len(initial_rows) != 1:
            errors.append(f"qid must have exactly one t=0 direct stage: qid={qid}")
        initial = initial_rows[0]
        direct_id = initial["positive"]
        for t, state_rows in sorted(qid_rows.items()):
            if t == 0:
                if initial["history"]:
                    errors.append(f"direct stage contains history: qid={qid}")
                continue
            for item in state_rows:
                later_states += 1
                history = item["history"]
                if not history:
                    errors.append(f"indirect stage has no direct anchor: qid={qid}, t={t}")
                    continue
                anchor_id = history[0]
                direct_anchors.append((qid, t, anchor_id))
                if anchor_id == direct_id:
                    direct_anchor_matches += 1
                else:
                    errors.append(
                        "first H_t unit does not match t=0 teacher-positive direct anchor: "
                        f"qid={qid}, t={t}, expected={direct_id}, actual={anchor_id}"
                    )

    report = {
        "path": str(path),
        "rows": rows,
        "qids": len(qids),
        "direct_stage_rows": int(t_counts.get("0", 0)),
        "indirect_stage_rows": later_states,
        "t_distribution": dict(t_counts),
        "candidate_pool_size_distribution": dict(pool_sizes),
        "repeated_state_rows": duplicate_state_rows,
        "conflicting_repeated_state_rows": conflicting_duplicate_rows,
        "direct_anchor_matches": direct_anchor_matches,
        "direct_anchor_match_rate": (
            round(direct_anchor_matches / later_states, 6) if later_states else None
        ),
        "error_count": len(errors),
        "errors": errors[:20],
        "_direct_anchors": direct_anchors,
    }
    return report, qids, required_memory_ids


def inspect_memory(
    path: Path,
    required_ids: set[str],
    direct_anchors: list[tuple[str, int, str]],
) -> dict:
    retained = {}
    duplicate_ids = 0
    seen = set()
    for row in read_jsonl(path):
        unit_id = str(row.get("unit_id") or "")
        if unit_id in seen:
            duplicate_ids += 1
        seen.add(unit_id)
        if unit_id not in required_ids:
            continue
        retained[unit_id] = {
            "qid": str(row.get("qid") or unit_id.split("::", 1)[0]),
            "title": str(row.get("title") or row.get("doc_id") or ""),
            "text": str(row.get("text") or "").strip(),
        }

    missing = sorted(required_ids - retained.keys())
    invalid_anchors = []
    rendered_examples = []
    for qid, t, unit_id in direct_anchors:
        item = retained.get(unit_id)
        if not item or item["qid"] != qid or not item["title"] or not item["text"]:
            invalid_anchors.append({"qid": qid, "t": t, "unit_id": unit_id})
            continue
        if len(rendered_examples) < 3:
            context = f"[1] {item['title']}: {item['text']}"
            rendered_examples.append(
                {
                    "qid": qid,
                    "t": t,
                    "direct_anchor_unit_id": unit_id,
                    "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                    "context_chars": len(context),
                }
            )

    return {
        "path": str(path),
        "required_units": len(required_ids),
        "resolved_units": len(retained),
        "missing_units": len(missing),
        "missing_examples": missing[:10],
        "duplicate_unit_ids": duplicate_ids,
        "direct_anchor_rows": len(direct_anchors),
        "resolved_direct_anchor_rows": len(direct_anchors) - len(invalid_anchors),
        "invalid_direct_anchor_examples": invalid_anchors[:10],
        "rendered_direct_anchor_examples": rendered_examples,
    }


def compare_configs(config: dict, reference: dict) -> tuple[dict, list[str]]:
    failures = []
    if config.get("seed") != reference.get("seed"):
        failures.append("seed differs from v22 Full")
    if config.get("model") != reference.get("model"):
        failures.append("model configuration differs from v22 Full")
    if config.get("train") != reference.get("train"):
        failures.append("training/initialization/loss configuration differs from v22 Full")

    data = config.get("data") or {}
    reference_data = reference.get("data") or {}
    context_mode = data.get("context_mode")
    if context_mode != "direct_evidence_only":
        failures.append("ECDR-inspired context_mode must be direct_evidence_only")
    for key, value in reference_data.items():
        if data.get(key) != value:
            failures.append(f"data path/config differs from v22 Full: {key}")

    forbidden_output_dirs = {
        reference.get("output_dir"),
        "outputs/ranker/deberta_v3_large_v23_acra_anchor",
    }
    if config.get("output_dir") in forbidden_output_dirs:
        failures.append("ECDR-inspired output_dir must be independent")

    report = {
        "context_mode": context_mode,
        "same_seed": config.get("seed") == reference.get("seed"),
        "same_model": config.get("model") == reference.get("model"),
        "same_training_and_losses": config.get("train") == reference.get("train"),
        "same_data_paths": all(data.get(key) == value for key, value in reference_data.items()),
        "independent_output_dir": config.get("output_dir") not in forbidden_output_dirs,
        "init_checkpoint": (config.get("train") or {}).get("init_checkpoint"),
        "output_dir": config.get("output_dir"),
    }
    return report, failures


def inspect_runtime(path: Path) -> tuple[dict, list[str]]:
    source = path.read_text(encoding="utf-8")
    required_markers = {
        "context source choice": '"direct_evidence_only"',
        "predicted direct anchor initialization": "direct_predicted_unit_id = None",
        "predicted direct anchor context": "context_anchor_unit_id = direct_predicted_unit_id",
        "predicted direct anchor assignment": "direct_predicted_unit_id = pred_id",
    }
    missing = [name for name, marker in required_markers.items() if marker not in source]
    return {
        "path": str(path),
        "required_markers": required_markers,
        "missing_markers": missing,
        "uses_predicted_direct_anchor": not missing,
    }, [f"runtime missing {name}" for name in missing]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_ranker_deberta_v24_ecdr_direct_indirect.yaml",
    )
    parser.add_argument(
        "--reference-config",
        default="configs/train_ranker_deberta_v22_state_focused.yaml",
    )
    parser.add_argument(
        "--data-manifest",
        default="data/hotpotqa_distractor_v22_state_focused/manifest.json",
    )
    parser.add_argument(
        "--runtime",
        default="scripts/run_hotpotqa_policy_rag.py",
    )
    parser.add_argument(
        "--eval-queries",
        default="data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/analysis/kbs_stage3_ecdr_direct_indirect_readiness.json",
    )
    parser.add_argument("--require-paths", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    reference_path = Path(args.reference_config)
    manifest_path = Path(args.data_manifest)
    runtime_path = Path(args.runtime)
    output_path = Path(args.output)
    required = [config_path, reference_path, manifest_path, runtime_path]

    config = load_yaml(config_path) if config_path.is_file() else {}
    data_config = config.get("data") or {}
    train_config = config.get("train") or {}
    required.extend(
        Path(data_config[key])
        for key in (
            "train_samples",
            "val_samples",
            "test_samples",
            "train_memory",
            "val_memory",
            "test_memory",
        )
        if key in data_config
    )
    required.extend(
        Path(value)
        for key, value in data_config.items()
        if key.endswith("_role_targets") and value
    )
    if train_config.get("init_checkpoint"):
        required.append(Path(train_config["init_checkpoint"]))
    if (config.get("model") or {}).get("pretrained_name"):
        required.append(Path(config["model"]["pretrained_name"]))

    missing = sorted({str(path) for path in required if not path.exists()})
    if missing:
        result = {"status": "MISSING", "missing_paths": missing}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise SystemExit(1 if args.require_paths else 0)

    reference = load_yaml(reference_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_report, failures = compare_configs(config, reference)
    runtime_report, runtime_failures = inspect_runtime(runtime_path)
    failures.extend(runtime_failures)

    split_reports = {}
    qids_by_split = {}
    for split in SPLITS:
        samples_path = Path(config["data"][f"{split}_samples"])
        memory_path = Path(config["data"][f"{split}_memory"])
        report, qids, required_ids = inspect_samples(samples_path)
        direct_anchors = report.pop("_direct_anchors")
        report["memory"] = inspect_memory(memory_path, required_ids, direct_anchors)
        split_reports[split] = report
        qids_by_split[split] = qids
        if report["error_count"]:
            failures.append(f"{split} samples contain {report['error_count']} validation errors")
        if report["memory"]["missing_units"]:
            failures.append(f"{split} memory misses {report['memory']['missing_units']} required units")
        if report["memory"]["resolved_direct_anchor_rows"] != report["indirect_stage_rows"]:
            failures.append(f"{split} direct anchors are not fully resolvable")

    overlap = {
        "train_val": len(qids_by_split["train"] & qids_by_split["val"]),
        "train_test": len(qids_by_split["train"] & qids_by_split["test"]),
        "val_test": len(qids_by_split["val"] & qids_by_split["test"]),
    }
    if any(overlap.values()):
        failures.append(f"train/validation/test qid overlap: {overlap}")

    eval_overlap = None
    eval_path = Path(args.eval_queries)
    if eval_path.is_file():
        eval_qids = {str(row.get("qid") or "") for row in read_jsonl(eval_path)}
        eval_overlap = {
            "eval_qids": len(eval_qids),
            "train_eval": len(qids_by_split["train"] & eval_qids),
            "val_eval": len(qids_by_split["val"] & eval_qids),
            "internal_test_eval": len(qids_by_split["test"] & eval_qids),
        }
        if eval_overlap["train_eval"] or eval_overlap["val_eval"]:
            failures.append(f"training/validation leakage into final evaluation: {eval_overlap}")

    if manifest.get("candidate_top_k") != 10:
        failures.append("v22 training manifest candidate_top_k is not the matched Compact value 10")
    if not manifest.get("auxiliary_labels_masked"):
        failures.append("v22 training manifest does not mask unavailable auxiliary labels")

    result = {
        "status": "OK" if not failures else "FAILED",
        "experiment": "Stage 3.2 ECDR-inspired textual direct-indirect baseline",
        "faithfulness_claim": "inspired textual baseline; not a faithful ECDR reproduction",
        "definition": {
            "direct_stage": "(question, candidate)",
            "indirect_stage": "(question, frozen predicted direct evidence, candidate)",
            "full_history_used": False,
            "gold_support_used_at_inference": False,
        },
        "teacher_forcing": (
            "training t=0 is query-only; t>0 uses the first teacher-prefix H_t "
            "unit. Online evaluation must freeze the predicted t=0 top-1 unit."
        ),
        "config_match": config_report,
        "runtime": runtime_report,
        "matched_online_budget": {
            "front_pool_k": 30,
            "candidate_top_k": 10,
            "select_top_k": 5,
            "policy_blend_weight": 0.5,
        },
        "data_manifest": {
            "path": str(manifest_path),
            "candidate_top_k": manifest.get("candidate_top_k"),
            "deep_repeat": manifest.get("deep_repeat"),
            "seed": manifest.get("seed"),
            "selection_supervision": manifest.get("selection_supervision"),
            "auxiliary_labels_masked": manifest.get("auxiliary_labels_masked"),
        },
        "splits": split_reports,
        "qid_overlap": overlap,
        "evaluation_overlap": eval_overlap,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {output_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
