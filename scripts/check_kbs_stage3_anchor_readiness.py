#!/usr/bin/env python3
"""Audit the matched ACRA-style previous-evidence anchor experiment."""

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
    anchors = []
    pools_by_qid = {}
    t_counts = Counter()
    pool_sizes = Counter()
    errors = []

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
        if not pool or len(pool) != len(set(pool)):
            errors.append(f"invalid candidate pool: qid={qid}, t={t}")
        if positive not in pool:
            errors.append(f"teacher positive missing from pool: qid={qid}, t={t}")
        if len(history) != t:
            errors.append(f"H_t length does not equal t: qid={qid}, t={t}, len={len(history)}")
        if t == 0 and history:
            errors.append(f"initial state contains previous evidence: qid={qid}")
        if t > 0:
            if not history:
                errors.append(f"later state has no previous evidence: qid={qid}, t={t}")
            else:
                anchors.append((qid, t, history[-1]))
        build_meta = row.get("build_meta") if isinstance(row.get("build_meta"), dict) else {}
        if not bool(build_meta.get("mask_auxiliary_labels", False)):
            errors.append(f"auxiliary labels are not masked: qid={qid}, t={t}")
        pool_key = tuple(pool)
        if qid in pools_by_qid and pools_by_qid[qid] != pool_key:
            errors.append(f"candidate pool changes across prefixes: qid={qid}")
        pools_by_qid[qid] = pool_key

    report = {
        "path": str(path),
        "rows": rows,
        "qids": len(qids),
        "initial_states": int(t_counts.get("0", 0)),
        "later_states": len(anchors),
        "t_distribution": dict(t_counts),
        "candidate_pool_size_distribution": dict(pool_sizes),
        "teacher_positive_rows": rows - sum("teacher positive" in error for error in errors),
        "anchor_rows": len(anchors),
        "error_count": len(errors),
        "errors": errors[:20],
        "_anchors": anchors,
    }
    return report, qids, required_memory_ids


def inspect_memory(path: Path, required_ids: set[str], anchors: list[tuple[str, int, str]]) -> dict:
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
        title = str(row.get("title") or row.get("doc_id") or "")
        text = str(row.get("text") or "").strip()
        qid = str(row.get("qid") or unit_id.split("::", 1)[0])
        retained[unit_id] = {"qid": qid, "title": title, "text": text}

    missing = sorted(required_ids - retained.keys())
    invalid_anchors = []
    rendered_examples = []
    for qid, t, unit_id in anchors:
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
                    "anchor_unit_id": unit_id,
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
        "anchor_rows": len(anchors),
        "resolved_anchor_rows": len(anchors) - len(invalid_anchors),
        "invalid_anchor_examples": invalid_anchors[:10],
        "rendered_anchor_examples": rendered_examples,
    }


def compare_configs(anchor: dict, reference: dict) -> tuple[dict, list[str]]:
    failures = []
    if anchor.get("seed") != reference.get("seed"):
        failures.append("seed differs from v22 Full")
    if anchor.get("model") != reference.get("model"):
        failures.append("model configuration differs from v22 Full")
    if anchor.get("train") != reference.get("train"):
        failures.append("training/initialization/loss configuration differs from v22 Full")

    anchor_data = anchor.get("data") or {}
    reference_data = reference.get("data") or {}
    context_mode = anchor_data.get("context_mode")
    if context_mode != "previous_evidence_only":
        failures.append("anchor context_mode must be previous_evidence_only")
    for key, value in reference_data.items():
        if anchor_data.get(key) != value:
            failures.append(f"data path/config differs from v22 Full: {key}")

    if anchor.get("output_dir") == reference.get("output_dir"):
        failures.append("anchor output_dir must be independent from v22 Full")

    report = {
        "context_mode": context_mode,
        "same_seed": anchor.get("seed") == reference.get("seed"),
        "same_model": anchor.get("model") == reference.get("model"),
        "same_training_and_losses": anchor.get("train") == reference.get("train"),
        "same_data_paths": all(anchor_data.get(key) == value for key, value in reference_data.items()),
        "independent_output_dir": anchor.get("output_dir") != reference.get("output_dir"),
        "init_checkpoint": (anchor.get("train") or {}).get("init_checkpoint"),
        "output_dir": anchor.get("output_dir"),
    }
    return report, failures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_ranker_deberta_v23_acra_anchor.yaml",
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
        "--eval-queries",
        default="data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/analysis/kbs_stage3_acra_anchor_readiness.json",
    )
    parser.add_argument("--require-paths", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    reference_path = Path(args.reference_config)
    manifest_path = Path(args.data_manifest)
    output_path = Path(args.output)
    required = [config_path, reference_path, manifest_path]

    if config_path.is_file():
        config = load_yaml(config_path)
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
    else:
        config = {}

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

    split_reports = {}
    qids_by_split = {}
    for split in SPLITS:
        samples_path = Path(config["data"][f"{split}_samples"])
        memory_path = Path(config["data"][f"{split}_memory"])
        report, qids, required_ids = inspect_samples(samples_path)
        anchors = report.pop("_anchors")
        report["memory"] = inspect_memory(memory_path, required_ids, anchors)
        split_reports[split] = report
        qids_by_split[split] = qids
        if report["error_count"]:
            failures.append(f"{split} samples contain {report['error_count']} validation errors")
        if report["memory"]["missing_units"]:
            failures.append(f"{split} memory misses {report['memory']['missing_units']} required units")
        if report["memory"]["resolved_anchor_rows"] != report["later_states"]:
            failures.append(f"{split} previous-evidence anchors are not fully resolvable")

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
        "experiment": "Stage 3.1 ACRA-style previous-evidence anchor",
        "definition": "(question, last previous evidence, candidate)",
        "teacher_forcing": "training uses the last teacher-prefix H_t unit; online evaluation must use the last predicted unit",
        "config_match": config_report,
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
