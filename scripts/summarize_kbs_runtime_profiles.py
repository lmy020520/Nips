#!/usr/bin/env python3
"""Merge local compute profiles with full-run answer API statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"expected Method=path, received: {spec}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path)


def read_summary(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: missing summary object")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--answer-report", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv-output", type=Path, required=True)
    args = parser.parse_args()

    profiles = {name: read_summary(path) for name, path in map(parse_spec, args.profile)}
    answers = {name: read_summary(path) for name, path in map(parse_spec, args.answer_report)}
    names = list(dict.fromkeys([*profiles, *answers]))
    rows = []
    for name in names:
        profile_summary = profiles.get(name, {})
        runtime = profile_summary.get("runtime_profile") or {}
        answer = answers.get(name, {})
        stage_ms = runtime.get("stage_avg_ms_per_call") or {}
        selection_ms = runtime.get("selection_avg_ms_per_qid")
        answer_latency = answer.get("avg_answer_latency")
        estimated_e2e = None
        if isinstance(answer_latency, (int, float)):
            estimated_e2e = 1000.0 * float(answer_latency) + float(selection_ms or 0.0)
        rows.append(
            {
                "method": name,
                "profile_qids": runtime.get("measured_qids"),
                "profile_steps": runtime.get("measured_steps"),
                "selection_ms_per_qid": selection_ms,
                "selection_qids_per_second": runtime.get("selection_throughput_qids_per_second"),
                "dense_ms_per_step": stage_ms.get("dense_retrieval"),
                "bm25_ms_per_step": stage_ms.get("bm25_retrieval"),
                "fusion_ms_per_step": stage_ms.get("score_fusion"),
                "compression_ms_per_step": stage_ms.get("local_expansion_mmr"),
                "policy_ms_per_step": stage_ms.get("policy_scoring"),
                "reranker_ms_per_step": stage_ms.get("reranker_scoring"),
                "peak_gpu_allocated_mb": runtime.get("peak_gpu_allocated_mb"),
                "peak_gpu_reserved_mb": runtime.get("peak_gpu_reserved_mb"),
                "answer_api_latency_ms": (
                    round(1000.0 * float(answer_latency), 3)
                    if isinstance(answer_latency, (int, float))
                    else None
                ),
                "estimated_end_to_end_ms": round(estimated_e2e, 3) if estimated_e2e is not None else None,
                "avg_answer_api_tokens": answer.get("avg_answer_tokens"),
                "answer_em": answer.get("answer_em"),
                "answer_f1": answer.get("answer_f1"),
            }
        )

    output = {
        "notes": {
            "selection_timing": "Warm-up excluded; CUDA synchronized; answer API excluded.",
            "throughput": "Qids per second for local evidence selection only.",
            "gpu_memory": "Peak PyTorch CUDA allocated/reserved memory after model loading and warm-up.",
            "answer_api_latency": "Mean DeepSeek request latency from the existing full evaluation report.",
            "estimated_end_to_end": "selection_ms_per_qid + answer_api_latency_ms from separate runs.",
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.tsv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.tsv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    print(f"tsv: {args.tsv_output}")


if __name__ == "__main__":
    main()
