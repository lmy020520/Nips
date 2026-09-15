#!/usr/bin/env python3
"""Validate Stage 9.6 MuSiQue answers against frozen paragraph selection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_kbs_standard_metrics import answer_metrics
except ModuleNotFoundError:
    from evaluate_kbs_standard_metrics import answer_metrics


ANSWER_PROTOCOL = {
    "answer_model": "deepseek-v4-flash",
    "answer_thinking_mode": "disabled",
    "answer_mode": "json",
    "answer_temperature": 0.0,
    "answer_prompt_version": "kbs_extractive_answer_json_v1",
    "answer_reference_mode": "max_over_canonical_and_aliases",
}
METHODS = {
    "compact_seed42": {
        "selector": "hybrid_policy",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "reranker_model": "",
        "candidate_top_k": 10,
        "front_pool_k": 30,
        "state_update_top_k": 1,
    },
    "balanced_seed42": {
        "selector": "hybrid_policy",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "reranker_model": "",
        "candidate_top_k": 15,
        "front_pool_k": 30,
        "state_update_top_k": 1,
    },
    "hybrid": {
        "selector": "hybrid",
        "dense_model": "models/bge-large-en-v1.5",
        "dense_query_mode": "state",
        "reranker_model": "",
        "candidate_top_k": 8,
        "front_pool_k": 30,
        "state_update_top_k": 5,
    },
    "bge_reranker": {
        "selector": "generic_reranker",
        "dense_model": "",
        "dense_query_mode": "question",
        "reranker_model": "models/bge-reranker-large",
        "candidate_top_k": 8,
        "front_pool_k": 30,
        "state_update_top_k": 5,
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = obj.get("summary") if isinstance(obj, dict) else None
    results = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise ValueError(f"report lacks summary/results: {path}")
    return summary, results


def index(
    records: list[dict[str, Any]],
    path: Path,
) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        qid = str(record.get("qid") or "")
        if not qid or qid in output:
            raise ValueError(f"empty or duplicate qid in {path}: {qid!r}")
        output[qid] = record
    return output


def cache_file(cache_dir: Path, qid: str) -> Path:
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
    return cache_dir / f"{safe_qid}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-qids", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    metrics: dict[str, Any] = {}
    cache_stats = {
        "evaluated_files": 0,
        "reused_exact_context": 0,
        "fresh_answers": 0,
        "invalid": 0,
    }
    required = (args.report, args.selection_report, args.queries, args.cache_dir)
    failures.extend(
        f"missing required path: {path}" for path in required if not path.exists()
    )

    if not failures:
        try:
            summary, answer_rows = read_report(args.report)
            _, selection_rows = read_report(args.selection_report)
            answers = index(answer_rows, args.report)
            selections_all = index(selection_rows, args.selection_report)
            selections = {
                qid: selections_all[qid]
                for qid in list(selections_all)[: args.expected_qids]
            }
            query_rows = index(read_jsonl(args.queries), args.queries)
            spec = METHODS[args.method]
            expected_summary = {
                "checkpoint": args.checkpoint,
                "qids": args.expected_qids,
                "steps": sum(len(row.get("steps") or []) for row in selections.values()),
                "state_mode": "policy",
                "policy_context_source": "online_state",
                "selector": spec["selector"],
                "dense_model": spec["dense_model"],
                "dense_query_mode": spec["dense_query_mode"],
                "hybrid_alpha": 0.5,
                "front_pool_k": spec["front_pool_k"],
                "reranker_model": spec["reranker_model"],
                "candidate_top_k": spec["candidate_top_k"],
                "select_top_k": 5,
                "state_update_top_k": spec["state_update_top_k"],
                "generate_answers": True,
                "answer_judged": args.expected_qids,
                "answer_errors": 0,
                "refresh_answer_cache": False,
                **ANSWER_PROTOCOL,
            }
            if spec["selector"] == "hybrid_policy":
                expected_summary.update(
                    {
                        "front_fusion": "rrf",
                        "local_expansion_window": 1,
                        "policy_score_mode": "front_policy_blend",
                        "policy_blend_weight": 0.5,
                    }
                )
            for key, expected in expected_summary.items():
                if summary.get(key) != expected:
                    failures.append(
                        f"summary {key}={summary.get(key)!r} != {expected!r}"
                    )
            if list(answers) != list(selections):
                failures.append("answer qid order differs from frozen selection prefix")

            context_mismatches = 0
            alias_mismatches = 0
            score_mismatches = 0
            empty_answers = 0
            error_answers = 0
            cache_mismatches = 0
            alias_qids = 0
            replay_em = 0.0
            replay_f1 = 0.0
            for qid, answer in answers.items():
                selection = selections.get(qid, {})
                query = query_rows.get(qid, {})
                aliases = [str(value) for value in query.get("answer_aliases") or []]
                alias_qids += int(bool(aliases))
                if any(
                    answer.get(key) != selection.get(key)
                    for key in ("question", "gold_answer", "selected_unit_ids")
                ):
                    context_mismatches += 1
                if answer.get("gold_answer_aliases") != aliases:
                    alias_mismatches += 1

                raw_answer = str(answer.get("raw_answer") or "").strip()
                prediction = str(answer.get("answer") or "").strip()
                empty_answers += int(not raw_answer or not prediction)
                error_answers += int(raw_answer.startswith("ERROR:"))
                scores = answer_metrics(
                    prediction,
                    query.get("answer", ""),
                    aliases,
                )
                replay_em += scores["answer_em"]
                replay_f1 += scores["answer_f1"]
                if (
                    float(answer.get("answer_em") or 0.0) != scores["answer_em"]
                    or abs(float(answer.get("answer_f1") or 0.0) - scores["answer_f1"])
                    > 1e-5
                ):
                    score_mismatches += 1

                path = cache_file(args.cache_dir, qid)
                if not path.is_file():
                    cache_mismatches += 1
                    continue
                cached = json.loads(path.read_text(encoding="utf-8"))
                cache_stats["evaluated_files"] += 1
                if (
                    str(cached.get("qid") or "") != qid
                    or str(cached.get("answer") or "") != prediction
                    or str(cached.get("raw_answer") or "") != raw_answer
                    or int(cached.get("answer_tokens") or 0)
                    != int(answer.get("answer_tokens") or 0)
                ):
                    cache_mismatches += 1
                if any(
                    cached.get(key) != ANSWER_PROTOCOL[key]
                    for key in (
                        "answer_model",
                        "answer_thinking_mode",
                        "answer_mode",
                        "answer_prompt_version",
                    )
                ):
                    cache_stats["invalid"] += 1
                if "source_report" in cached:
                    cache_stats["reused_exact_context"] += 1
                    if (
                        cached.get("question") != answer.get("question")
                        or cached.get("gold_answer") != answer.get("gold_answer")
                        or cached.get("gold_answer_aliases") != aliases
                        or cached.get("selected_unit_ids")
                        != answer.get("selected_unit_ids")
                    ):
                        cache_stats["invalid"] += 1
                else:
                    cache_stats["fresh_answers"] += 1

            if context_mismatches:
                failures.append(f"selection context mismatches={context_mismatches}")
            if alias_mismatches:
                failures.append(f"answer alias mismatches={alias_mismatches}")
            if score_mismatches:
                failures.append(f"alias-aware score mismatches={score_mismatches}")
            if empty_answers or error_answers:
                failures.append(
                    f"empty answers={empty_answers}, error answers={error_answers}"
                )
            if cache_mismatches or cache_stats["invalid"]:
                failures.append(
                    f"cache mismatches={cache_mismatches}, "
                    f"invalid metadata={cache_stats['invalid']}"
                )
            if cache_stats["evaluated_files"] != args.expected_qids:
                failures.append(
                    f"evaluated cache files={cache_stats['evaluated_files']} "
                    f"!= {args.expected_qids}"
                )
            replay_em = round(replay_em / args.expected_qids, 6)
            replay_f1 = round(replay_f1 / args.expected_qids, 6)
            if abs(replay_em - float(summary.get("answer_em") or 0.0)) > 1e-6:
                failures.append("summary answer_em differs from alias-aware replay")
            if abs(replay_f1 - float(summary.get("answer_f1") or 0.0)) > 1e-6:
                failures.append("summary answer_f1 differs from alias-aware replay")

            metrics = {
                "answer_em": summary.get("answer_em"),
                "answer_f1": summary.get("answer_f1"),
                "paragraph_alignment_at_1": summary.get("step_acc@1"),
                "paragraph_alignment_at_5": summary.get("step_acc@5"),
                "full_support_title_coverage": summary.get("full_gold_doc_coverage"),
                "full_support_paragraph_coverage": summary.get(
                    "full_gold_unit_coverage"
                ),
                "avg_answer_tokens": summary.get("avg_answer_tokens"),
                "avg_answer_latency": summary.get("avg_answer_latency"),
                "alias_qids": alias_qids,
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append(str(exc))

    result = {
        "status": "SMOKE_OK" if args.smoke and not failures else (
            "OK" if not failures else "FAIL"
        ),
        "stage": 9,
        "step": "9.6",
        "mode": "musique_answer_smoke" if args.smoke else "musique_answer_report",
        "method": args.method,
        "qids": args.expected_qids,
        "report": str(args.report),
        "selection_report": str(args.selection_report),
        "cache_dir": str(args.cache_dir),
        "protocol": ANSWER_PROTOCOL,
        "metric_boundary": (
            "Evidence metrics are paragraph-level; answer EM/F1 maximize over the "
            "canonical answer and annotated aliases."
        ),
        "metrics": metrics,
        "cache": cache_stats,
        "next_gate": (
            "Resolve failures before proceeding."
            if failures
            else (
                "Review the bounded smoke before authorizing complete answers."
                if args.smoke
                else "Finalize MuSiQue answer and paragraph-evidence metrics."
            )
        ),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
