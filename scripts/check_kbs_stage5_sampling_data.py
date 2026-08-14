#!/usr/bin/env python3
"""Audit the full HotpotQA validation dataset for Stage-5 sampling robustness."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


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


def qids_from_path(path: Path):
    qids = set()
    for row in read_jsonl(path):
        qid = str(row.get("qid") or row.get("_id") or row.get("id") or "")
        if qid:
            qids.add(qid)
    return qids


def sample_positive(row: dict):
    ranking = (row.get("labels") or {}).get("ranking_label") or {}
    return str(ranking.get("positive_unit_id") or "")


def sample_candidates(row: dict):
    return [str(value) for value in ((row.get("candidates") or {}).get("C_t") or [])]


def inspect_samples(path: Path, query_qids: set[str]):
    errors = []
    states = set()
    qids = set()
    t_by_qid = defaultdict(list)
    candidate_sizes = Counter()
    rows = 0
    for row in read_jsonl(path):
        rows += 1
        qid = str(row.get("qid") or "")
        t = int(row.get("t", -1))
        key = (qid, t)
        if not qid or t < 0:
            errors.append(f"invalid state identity row={rows}: qid={qid!r}, t={t}")
        if key in states:
            errors.append(f"duplicate state qid={qid} t={t}")
        states.add(key)
        qids.add(qid)
        t_by_qid[qid].append(t)
        candidates = sample_candidates(row)
        positive = sample_positive(row)
        candidate_sizes[str(len(candidates))] += 1
        if not candidates or len(candidates) > 50:
            errors.append(f"invalid candidate count qid={qid} t={t}: {len(candidates)}")
        if len(candidates) != len(set(candidates)):
            errors.append(f"duplicate candidates qid={qid} t={t}")
        if not positive or positive not in candidates:
            errors.append(f"positive absent from candidates qid={qid} t={t}")
        if qid not in query_qids:
            errors.append(f"sample qid absent from queries qid={qid}")
    for qid, values in t_by_qid.items():
        if sorted(values) != list(range(len(values))):
            errors.append(f"non-consecutive state indices qid={qid}: {sorted(values)}")
    return {
        "rows": rows,
        "qids": len(qids),
        "candidate_size_distribution": dict(candidate_sizes),
        "error_count": len(errors),
        "errors": errors[:50],
        "_qids": qids,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default="data/hotpotqa_distractor_validation_full_cand50"
    )
    parser.add_argument(
        "--training-root", default="data/hotpotqa_distractor_v27_counterfactual_dual"
    )
    parser.add_argument(
        "--primary-eval-root", default="data/hotpotqa_distractor_eval_3000_cand50"
    )
    parser.add_argument(
        "--alpha-val-root", default="data/hotpotqa_distractor_alpha_val_1000_cand50"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    manifest_path = root / "manifest.json"
    queries_path = root / "queries/val.jsonl"
    samples_path = root / "samples/val.jsonl"
    memory_path = root / "unit_registry/raw_units_val.jsonl"
    required = [manifest_path, queries_path, samples_path, memory_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        result = {"status": "MISSING", "missing_paths": missing}
        print(json.dumps(result, indent=2))
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    query_rows = list(read_jsonl(queries_path))
    query_order = [str(row.get("qid") or "") for row in query_rows]
    query_qids = set(query_order)
    sample_report = inspect_samples(samples_path, query_qids)
    sample_qids = sample_report.pop("_qids")
    memory_rows = sum(1 for _ in read_jsonl(memory_path))
    failures = []
    expected_manifest = {
        "source_split": "validation",
        "output_split": "val",
        "requested_size": 0,
        "selection_mode": "all_usable_rows",
        "max_candidates": 50,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            failures.append(f"manifest {key}: {manifest.get(key)!r} != {expected!r}")
    if len(query_order) != len(query_qids):
        failures.append("queries contain duplicate qids")
    if len(query_qids) < 7000:
        failures.append(f"full validation set unexpectedly small: {len(query_qids)}")
    if int(manifest.get("size", -1)) != len(query_qids):
        failures.append(f"manifest size {manifest.get('size')} != query qids {len(query_qids)}")
    if int((manifest.get("stats") or {}).get("qids", -1)) != len(query_qids):
        failures.append("manifest stats.qids does not match queries")
    accounted = len(query_qids) + sum(int(value) for value in (manifest.get("skip_reasons") or {}).values())
    if int(manifest.get("source_rows", -1)) != accounted:
        failures.append(
            f"source-row accounting mismatch: source={manifest.get('source_rows')} accounted={accounted}"
        )
    if sample_qids != query_qids:
        failures.append(
            f"sample/query qid sets differ: samples={len(sample_qids)} queries={len(query_qids)}"
        )
    if sample_report["error_count"]:
        failures.append(f"sample audit found {sample_report['error_count']} errors")
    if memory_rows <= len(query_qids):
        failures.append(f"memory row count is unexpectedly small: {memory_rows}")

    training_root = Path(args.training_root)
    training_qids = set()
    for split in ("train", "val", "test"):
        path = training_root / "samples" / f"{split}.jsonl"
        if path.is_file():
            training_qids.update(qids_from_path(path))
    train_overlap = len(query_qids & training_qids)
    if train_overlap:
        failures.append(f"full validation overlaps v27 training/internal qids: {train_overlap}")

    overlap = {"training": train_overlap}
    for name, other_root, split in (
        ("primary_eval", Path(args.primary_eval_root), "test"),
        ("alpha_validation", Path(args.alpha_val_root), "val"),
    ):
        path = other_root / "queries" / f"{split}.jsonl"
        if path.is_file():
            other_qids = qids_from_path(path)
            overlap[name] = len(query_qids & other_qids)
            overlap[f"{name}_total"] = len(other_qids)
            if not other_qids.issubset(query_qids):
                failures.append(f"{name} is not a subset of full validation")
        else:
            overlap[name] = None

    result = {
        "status": "OK" if not failures else "FAIL",
        "stage": 5,
        "mode": "full_validation_sampling_readiness",
        "data_root": str(root),
        "manifest_protocol": {
            key: manifest.get(key)
            for key in (
                "source_split", "output_split", "requested_size", "selection_mode",
                "source_rows", "size", "seed", "max_candidates",
            )
        },
        "queries": len(query_qids),
        "memory_rows": memory_rows,
        "samples": sample_report,
        "qid_overlap": overlap,
        "api_calls": 0,
        "failures": failures,
    }
    output_path = args.output or Path(args.data_root) / "readiness.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
