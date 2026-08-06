#!/usr/bin/env python3
"""Audit prerequisites for v27 Stage-5 multiseed end-to-end evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


CONFIGS = {
    "42": Path("configs/train_ranker_deberta_v27_counterfactual_dual.yaml"),
    "43": Path("configs/train_ranker_deberta_v27_counterfactual_dual_seed43.yaml"),
    "44": Path("configs/train_ranker_deberta_v27_counterfactual_dual_seed44.yaml"),
}
CHECKPOINTS = {
    "42": Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual/best_model.pt"),
    "43": Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed43/best_model.pt"),
    "44": Path("outputs/ranker/deberta_v3_large_v27_counterfactual_dual_seed44/best_model.pt"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_config(path: Path) -> tuple[int, dict]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    seed = int(obj.pop("seed"))
    obj.pop("output_dir", None)
    return seed, obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-root", default="outputs/analysis/kbs_v27_multiseed_validation"
    )
    parser.add_argument(
        "--seed42-compact-report",
        default="outputs/rag/kbs_v27_final_hotpot/full_compact.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/kbs_v27_stage5/readiness.json"),
    )
    args = parser.parse_args()

    gate_root = Path(args.gate_root)
    compact_report = Path(args.seed42_compact_report)
    required = [
        Path("md/kbs_three_review_execution_plan.md"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/samples/test.jsonl"),
        Path("data/hotpotqa_distractor_eval_3000_cand50/queries/test.jsonl"),
        Path(
            "data/hotpotqa_distractor_eval_3000_cand50/"
            "unit_registry/raw_units_test.jsonl"
        ),
        Path("models/deberta-v3-large"),
        Path("models/bge-large-en-v1.5"),
        compact_report,
        gate_root / "summary.json",
        gate_root / "seed43_validation_gate.json",
        gate_root / "seed44_validation_gate.json",
        *CONFIGS.values(),
        *CHECKPOINTS.values(),
    ]
    failures = [f"missing required path: {path}" for path in required if not path.exists()]

    config_audit = {}
    reference = None
    for expected_seed, path in CONFIGS.items():
        if not path.is_file():
            continue
        seed, normalized = normalized_config(path)
        config_audit[expected_seed] = {
            "path": str(path),
            "configured_seed": seed,
            "matches_seed42_protocol": reference is None or normalized == reference,
        }
        if seed != int(expected_seed):
            failures.append(f"{path}: expected seed {expected_seed}, found {seed}")
        if reference is None:
            reference = normalized
        elif normalized != reference:
            failures.append(f"{path}: protocol differs from seed 42")

    gate_audit = {}
    summary_path = gate_root / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        gate_audit = {
            "path": str(summary_path),
            "status": summary.get("status"),
            "seeds": summary.get("seeds"),
            "all_seed_gates_pass": summary.get("all_seed_gates_pass"),
            "failures": summary.get("failures"),
        }
        if summary.get("status") != "PASS":
            failures.append("multiseed validation summary is not PASS")
        if set(map(str, summary.get("seeds") or [])) != {"42", "43", "44"}:
            failures.append("multiseed validation summary does not contain seeds 42/43/44")
        if summary.get("all_seed_gates_pass") is not True:
            failures.append("not all multiseed validation gates passed")
        if summary.get("failures"):
            failures.append("multiseed validation summary contains failures")

    for seed in ("43", "44"):
        path = gate_root / f"seed{seed}_validation_gate.json"
        if path.is_file():
            gate = read_json(path)
            if gate.get("status") != "PASS" or gate.get("protocol_failures"):
                failures.append(f"seed {seed} validation gate is not clean PASS")

    compact_audit = {}
    if compact_report.is_file():
        obj = read_json(compact_report)
        summary = obj.get("summary") or {}
        results = obj.get("results") or []
        compact_audit = {
            "path": str(compact_report),
            "qids": summary.get("qids"),
            "answer_judged": summary.get("answer_judged"),
            "answer_errors": summary.get("answer_errors"),
            "checkpoint": summary.get("checkpoint"),
            "policy_blend_weight": summary.get("policy_blend_weight"),
            "state_update_top_k": summary.get("state_update_top_k"),
            "result_count": len(results),
        }
        expected_checkpoint = str(CHECKPOINTS["42"])
        expected = {
            "qids": 3000,
            "answer_judged": 3000,
            "answer_errors": 0,
            "checkpoint": expected_checkpoint,
            "policy_blend_weight": 0.5,
            "state_update_top_k": 1,
            "result_count": 3000,
        }
        for key, value in expected.items():
            if compact_audit.get(key) != value:
                failures.append(
                    f"seed-42 compact {key}: expected {value!r}, "
                    f"found {compact_audit.get(key)!r}"
                )

    result = {
        "status": "OK" if not failures else "MISSING",
        "stage": 5,
        "configs": config_audit,
        "checkpoints": {seed: str(path) for seed, path in CHECKPOINTS.items()},
        "validation_gate": gate_audit,
        "seed42_compact": compact_audit,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
