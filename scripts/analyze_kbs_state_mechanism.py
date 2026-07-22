#!/usr/bin/env python3
"""Run fixed-pool diagnostics for state use in the KBS student policy.

The retrieval front-end is executed once with the correct dataset state. Every
policy intervention then reuses the same compressed candidates and front-end
scores. This isolates changes caused by the policy context from changes in
candidate recall.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RAG = None


CONDITIONS = (
    "correct",
    "query_only",
    "empty",
    "frozen",
    "shuffled",
    "other_question",
    "previous_evidence_only",
)


def sample_candidate_ids(row: dict) -> list[str]:
    candidates = row.get("candidates") or {}
    for key in ("C_t", "R_t"):
        value = candidates.get(key)
        if isinstance(value, list) and value:
            return [str(unit_id) for unit_id in value]
    return []


def sample_positive_id(row: dict) -> str:
    labels = row.get("labels") or {}
    ranking = labels.get("ranking_label") or {}
    return str(ranking.get("positive_unit_id") or "")


def sample_k_t(row: dict) -> str:
    if "K_t" in row:
        return str(row["K_t"])
    state = row.get("state") or {}
    return str(state.get("K_t") or "")


def format_candidate_text(memory_item: dict) -> str:
    return f"{memory_item['title']} [{memory_item['sent_id']}] {memory_item['text']}"


def format_notebook_evidence(memory_item: dict, index: int) -> str:
    return f"[{index}] {memory_item['title']}: {memory_item['text']}"


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_query_metadata(path: str) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    return {str(row.get("qid")): row for row in read_jsonl(path) if row.get("qid")}


def question_type(query: dict | None) -> str:
    query = query or {}
    metadata = query.get("metadata") if isinstance(query.get("metadata"), dict) else {}
    return str(metadata.get("type") or query.get("type") or "unknown")


def make_context(question: str, notebook: str, *, query_only: bool = False) -> str:
    if query_only:
        return f"Question: {question}"
    return f"Question: {question}\nNotebook:\n{notebook}".rstrip()


def state_h_ids(row: dict) -> list[str]:
    state = row.get("state") if isinstance(row.get("state"), dict) else {}
    h_t = state.get("H_t") if isinstance(state.get("H_t"), list) else []
    result = []
    for item in h_t:
        if isinstance(item, dict) and item.get("unit_id"):
            result.append(str(item["unit_id"]))
        elif isinstance(item, str):
            result.append(item)
    return result


def previous_evidence_notebook(row: dict, memory: dict[str, dict]) -> str:
    h_ids = state_h_ids(row)
    if not h_ids:
        return ""
    item = memory.get(h_ids[-1])
    return format_notebook_evidence(item, 1) if item else ""


def shuffled_notebook(notebook: str, seed: int, qid: str, t: int) -> str:
    lines = [line for line in str(notebook).splitlines() if line.strip()]
    if len(lines) <= 2:
        return notebook
    header = lines[0] if lines[0].strip().lower().startswith("evidence") else ""
    evidence_lines = lines[1:] if header else lines
    local_seed = seed + sum(ord(char) for char in qid) + 1009 * t
    random.Random(local_seed).shuffle(evidence_lines)
    return "\n".join(([header] if header else []) + evidence_lines)


def prepare_rows(rows: list[dict], max_qids: int) -> tuple[list[dict], dict[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    qid_order = []
    seen = set()
    for row in rows:
        qid = str(row.get("qid") or "")
        if not qid:
            continue
        if qid not in seen:
            seen.add(qid)
            qid_order.append(qid)
        grouped[qid].append(row)
    if max_qids > 0:
        allowed = set(qid_order[:max_qids])
        grouped = {qid: values for qid, values in grouped.items() if qid in allowed}
    for qid in grouped:
        grouped[qid].sort(key=lambda item: int(item.get("t") or 0))
    selected = [row for qid in qid_order if qid in grouped for row in grouped[qid]]
    return selected, grouped


def build_other_question_states(rows: list[dict]) -> dict[tuple[str, int], str]:
    by_t: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        qid = str(row.get("qid") or "")
        t = int(row.get("t") or 0)
        notebook = sample_k_t(row)
        if notebook:
            by_t[t].append((qid, notebook))

    result = {}
    for row in rows:
        qid = str(row.get("qid") or "")
        t = int(row.get("t") or 0)
        candidates = by_t.get(t) or []
        replacement = ""
        if candidates:
            start = sum(ord(char) for char in qid) % len(candidates)
            for offset in range(len(candidates)):
                other_qid, notebook = candidates[(start + offset) % len(candidates)]
                if other_qid != qid:
                    replacement = notebook
                    break
        result[(qid, t)] = replacement
    return result


def context_variants(
    row: dict,
    grouped: dict[str, list[dict]],
    other_states: dict[tuple[str, int], str],
    memory: dict[str, dict],
    seed: int,
) -> dict[str, str]:
    qid = str(row.get("qid") or "")
    t = int(row.get("t") or 0)
    question = str(row.get("question") or "")
    correct = sample_k_t(row)
    frozen = sample_k_t(grouped[qid][0]) if grouped.get(qid) else ""
    return {
        "correct": make_context(question, correct),
        "query_only": make_context(question, "", query_only=True),
        "empty": make_context(question, ""),
        "frozen": make_context(question, frozen),
        "shuffled": make_context(question, shuffled_notebook(correct, seed, qid, t)),
        "other_question": make_context(question, other_states.get((qid, t), "")),
        "previous_evidence_only": make_context(question, previous_evidence_notebook(row, memory)),
    }


def candidate_text(unit_id: str, row: dict, memory: dict[str, dict]) -> str:
    item = memory.get(unit_id)
    if item:
        return format_candidate_text(item)
    payloads = row.get("derived_payloads") or {}
    if isinstance(payloads, list):
        payloads = {
            str(item.get("unit_id")): item
            for item in payloads
            if isinstance(item, dict) and item.get("unit_id")
        }
    payload = payloads.get(unit_id) if isinstance(payloads, dict) else None
    if isinstance(payload, dict):
        note_type = str(payload.get("type") or "derived_note")
        text = str(payload.get("text") or payload.get("unit_text") or "").strip()
        return f"{note_type} {unit_id}: {text}".strip()
    return ""


def fixed_front_pool(
    context: str,
    candidate_ids: list[str],
    candidate_texts: list[str],
    memory: dict[str, dict],
    dense: Any,
    args: argparse.Namespace,
) -> tuple[list[int], np.ndarray]:
    dense_query = context if args.dense_query_mode == "state" else context.split("\nNotebook:", 1)[0]
    dense_scores = dense.score(dense_query, candidate_texts)
    lexical_scores = RAG.bm25_scores(context, candidate_texts)
    dense_order = np.argsort(-dense_scores, kind="mergesort").tolist()
    lexical_order = np.argsort(-lexical_scores, kind="mergesort").tolist()
    if args.front_fusion == "rrf":
        front_scores = RAG.reciprocal_rank_fusion([lexical_order, dense_order], len(candidate_ids))
    else:
        front_scores = (
            args.hybrid_alpha * RAG.minmax_normalize(dense_scores)
            + (1.0 - args.hybrid_alpha) * RAG.minmax_normalize(lexical_scores)
        )
    front_order = np.argsort(-front_scores, kind="mergesort").tolist()
    pool_k = min(max(1, args.front_pool_k), len(front_order))
    seed_indices = list(dict.fromkeys(lexical_order[:pool_k] + dense_order[:pool_k]))
    expanded = RAG.local_expanded_pool(
        seed_indices,
        candidate_ids,
        memory,
        window=max(0, args.local_expansion_window),
    )
    expanded.sort(key=lambda index: float(front_scores[index]), reverse=True)
    compressed = RAG.mmr_select(
        expanded,
        front_scores,
        candidate_texts,
        candidate_ids,
        memory,
        limit=min(max(1, args.candidate_top_k), len(candidate_ids)),
        lambda_=args.mmr_lambda,
        same_doc_similarity=args.mmr_same_doc_similarity,
    )
    return compressed, front_scores


def rank_metrics(scores: np.ndarray, positive_index: int) -> dict[str, float | int]:
    order = np.argsort(-scores, kind="mergesort").tolist()
    rank = order.index(positive_index) + 1
    negative_scores = np.delete(scores, positive_index)
    margin = float(scores[positive_index] - np.max(negative_scores)) if negative_scores.size else 0.0
    return {
        "rank": rank,
        "step_at_1": int(rank <= 1),
        "step_at_3": int(rank <= 3),
        "step_at_5": int(rank <= 5),
        "mrr": 1.0 / rank,
        "gold_margin": margin,
    }


def mean_or_none(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 6) if values else None


def aggregate_condition_records(records: list[dict], *, retained_only: bool) -> dict:
    metrics = ("step_at_1", "step_at_3", "step_at_5", "mrr", "gold_margin")
    result = {}
    for condition in CONDITIONS:
        condition_records = []
        for record in records:
            if retained_only and not record["positive_in_fixed_pool"]:
                continue
            values = record.get("conditions", {}).get(condition)
            if values is None and not retained_only:
                values = {
                    "step_at_1": 0,
                    "step_at_3": 0,
                    "step_at_5": 0,
                    "mrr": 0.0,
                    "gold_margin": None,
                }
            if values is not None:
                condition_records.append(values)
        result[condition] = {
            "states": len(condition_records),
            **{
                metric: mean_or_none(
                    [float(item[metric]) for item in condition_records if item[metric] is not None]
                )
                for metric in metrics
            },
        }
    return result


def aggregate_by_slice(records: list[dict]) -> dict:
    slices: dict[str, list[dict]] = {
        "all": records,
        "t_ge_1": [record for record in records if record["t"] >= 1],
    }
    for t in sorted({record["t"] for record in records}):
        slices[f"t_{t}"] = [record for record in records if record["t"] == t]
    for qtype in sorted({record["question_type"] for record in records}):
        slices[f"type_{qtype}"] = [record for record in records if record["question_type"] == qtype]

    output = {}
    for name, values in slices.items():
        output[name] = {
            "states": len(values),
            "positive_in_fixed_pool": mean_or_none(
                [float(record["positive_in_fixed_pool"]) for record in values]
            ),
            "all_states": aggregate_condition_records(values, retained_only=False),
            "given_positive_in_fixed_pool": aggregate_condition_records(values, retained_only=True),
        }
    return output


def state_necessity(records: list[dict]) -> dict:
    eligible = [
        record
        for record in records
        if record["t"] >= 1
        and record["positive_in_fixed_pool"]
        and record.get("conditions", {}).get("correct")
        and record.get("conditions", {}).get("query_only")
    ]
    counters = Counter()
    rank_deltas = []
    for record in eligible:
        correct = record["conditions"]["correct"]
        query = record["conditions"]["query_only"]
        counters["rescued_at_1"] += int(correct["step_at_1"] == 1 and query["step_at_1"] == 0)
        counters["harmed_at_1"] += int(correct["step_at_1"] == 0 and query["step_at_1"] == 1)
        counters["rescued_at_5"] += int(correct["step_at_5"] == 1 and query["step_at_5"] == 0)
        counters["harmed_at_5"] += int(correct["step_at_5"] == 0 and query["step_at_5"] == 1)
        counters["rank_improved"] += int(correct["rank"] < query["rank"])
        counters["rank_degraded"] += int(correct["rank"] > query["rank"])
        rank_deltas.append(float(query["rank"] - correct["rank"]))
    total = len(eligible)
    return {
        "definition": "t>=1, gold retained in fixed pool; correct-state rescue over query-only",
        "eligible_states": total,
        **{
            key: {
                "count": int(counters[key]),
                "rate": round(counters[key] / total, 6) if total else None,
            }
            for key in (
                "rescued_at_1",
                "harmed_at_1",
                "rescued_at_5",
                "harmed_at_5",
                "rank_improved",
                "rank_degraded",
            )
        },
        "mean_query_rank_minus_correct_rank": mean_or_none(rank_deltas),
    }


def paired_bootstrap(
    records: list[dict],
    baseline: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    metrics = ("step_at_1", "step_at_5", "mrr", "gold_margin")
    eligible = [
        record
        for record in records
        if record["t"] >= 1
        and record["positive_in_fixed_pool"]
        and record.get("conditions", {}).get("correct")
        and record.get("conditions", {}).get(baseline)
    ]
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for record in eligible:
        by_qid[record["qid"]].append(record)
    qids = sorted(by_qid)
    if not qids:
        return {"baseline": baseline, "qids": 0, "states": 0, "metrics": {}}

    qid_deltas = np.array(
        [
            [
                np.mean(
                    [
                        float(record["conditions"]["correct"][metric])
                        - float(record["conditions"][baseline][metric])
                        for record in by_qid[qid]
                    ]
                )
                for metric in metrics
            ]
            for qid in qids
        ],
        dtype=float,
    )
    observed = np.mean(qid_deltas, axis=0)
    rng = np.random.default_rng(seed)
    samples = np.empty((n_bootstrap, len(metrics)), dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(qids), size=len(qids))
        samples[index] = np.mean(qid_deltas[sampled], axis=0)
    return {
        "comparison": f"correct-minus-{baseline}",
        "qids": len(qids),
        "states": len(eligible),
        "n_bootstrap": n_bootstrap,
        "metrics": {
            metric: {
                "observed_delta": round(float(observed[idx]), 6),
                "ci95_low": round(float(np.percentile(samples[:, idx], 2.5)), 6),
                "ci95_high": round(float(np.percentile(samples[:, idx], 97.5)), 6),
            }
            for idx, metric in enumerate(metrics)
        },
    }


def score_rank_reversals(
    grouped: dict[str, list[dict]],
    policy: Any,
    memory: dict[str, dict],
    other_states: dict[tuple[str, int], str],
    seed: int,
) -> tuple[dict, list[dict]]:
    records = []
    modes = ("correct", "query_only", "frozen", "previous_evidence_only")
    for qid, rows in grouped.items():
        for current, following in zip(rows, rows[1:]):
            positive_a = sample_positive_id(current)
            positive_b = sample_positive_id(following)
            if not positive_a or not positive_b or positive_a == positive_b:
                continue
            if positive_b not in sample_candidate_ids(current):
                continue
            text_a = candidate_text(positive_a, current, memory)
            text_b = candidate_text(positive_b, following, memory)
            if not text_a or not text_b:
                continue
            current_contexts = context_variants(current, grouped, other_states, memory, seed)
            following_contexts = context_variants(following, grouped, other_states, memory, seed)
            mode_results = {}
            for mode in modes:
                current_scores = policy.score(current_contexts[mode], [text_a, text_b])
                following_scores = policy.score(following_contexts[mode], [text_a, text_b])
                first_preference = bool(current_scores[0] > current_scores[1])
                second_preference = bool(following_scores[1] > following_scores[0])
                mode_results[mode] = {
                    "current_prefers_current_positive": first_preference,
                    "next_prefers_next_positive": second_preference,
                    "rank_reversal_correct": bool(first_preference and second_preference),
                    "current_margin_a_minus_b": round(float(current_scores[0] - current_scores[1]), 6),
                    "next_margin_b_minus_a": round(float(following_scores[1] - following_scores[0]), 6),
                }
            records.append(
                {
                    "qid": qid,
                    "t": int(current.get("t") or 0),
                    "next_t": int(following.get("t") or 0),
                    "current_positive": positive_a,
                    "next_positive": positive_b,
                    "modes": mode_results,
                }
            )
    summary = {
        "definition": (
            "For consecutive prefixes where the next positive was already available, "
            "prefer g_t under K_t and g_(t+1) under K_(t+1)."
        ),
        "eligible_pairs": len(records),
        "modes": {},
    }
    for mode in modes:
        values = [record["modes"][mode] for record in records]
        summary["modes"][mode] = {
            "current_preference_accuracy": mean_or_none(
                [float(value["current_prefers_current_positive"]) for value in values]
            ),
            "next_preference_accuracy": mean_or_none(
                [float(value["next_prefers_next_positive"]) for value in values]
            ),
            "conditional_rank_reversal_accuracy": mean_or_none(
                [float(value["rank_reversal_correct"]) for value in values]
            ),
        }
    return summary, records


def data_diagnostics(rows: list[dict], grouped: dict[str, list[dict]], query_map: dict[str, dict]) -> dict:
    by_t: dict[int, Counter] = defaultdict(Counter)
    by_type: Counter = Counter()
    for row in rows:
        qid = str(row.get("qid") or "")
        t = int(row.get("t") or 0)
        state = row.get("state") if isinstance(row.get("state"), dict) else {}
        by_t[t]["states"] += 1
        by_t[t]["h_units"] += len(state_h_ids(row))
        by_t[t]["k_chars"] += len(sample_k_t(row))
        by_type[question_type(query_map.get(qid))] += 1
    return {
        "qids": len(grouped),
        "states": len(rows),
        "initial_states": len(grouped),
        "later_states": len(rows) - len(grouped),
        "later_state_rate": round((len(rows) - len(grouped)) / len(rows), 6) if rows else None,
        "by_t": {
            str(t): {
                "states": counts["states"],
                "avg_h_units": round(counts["h_units"] / counts["states"], 6),
                "avg_k_chars": round(counts["k_chars"] / counts["states"], 6),
            }
            for t, counts in sorted(by_t.items())
        },
        "state_counts_by_question_type": dict(sorted(by_type.items())),
    }


def main() -> None:
    global RAG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", default="")
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dense-model", default="models/bge-large-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--front-pool-k", type=int, default=30)
    parser.add_argument("--front-fusion", choices=("rrf", "score"), default="rrf")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--dense-query-mode", choices=("question", "state"), default="state")
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--local-expansion-window", type=int, default=1)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--mmr-same-doc-similarity", type=float, default=0.35)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    rows, grouped = prepare_rows(read_jsonl(args.samples), args.max_qids)
    query_map = load_query_metadata(args.queries)
    output_dir = Path(args.output_dir)
    diagnostics = data_diagnostics(rows, grouped, query_map)
    if args.data_only:
        summary = {
            "status": "DATA_ONLY",
            "samples": args.samples,
            "queries": args.queries,
            "data_diagnostics": diagnostics,
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --data-only is used")

    from scripts import run_hotpotqa_policy_rag as rag_runtime

    RAG = rag_runtime
    memory = RAG.load_memory(Path(args.memory))
    other_states = build_other_question_states(rows)
    policy = RAG.PolicyModel(
        Path(args.model_dir),
        Path(args.checkpoint),
        args.device,
        args.max_length,
        args.batch_size,
    )
    dense = RAG.DenseScorer(args.dense_model, args.device, args.dense_batch_size)

    records = []
    skipped = Counter()
    try:
        from tqdm import tqdm

        iterator = tqdm(rows, desc="state-mechanism")
    except ImportError:
        iterator = rows

    for row in iterator:
        qid = str(row.get("qid") or "")
        t = int(row.get("t") or 0)
        positive_id = sample_positive_id(row)
        raw_candidate_ids = sample_candidate_ids(row)
        usable_ids = []
        usable_texts = []
        for unit_id in raw_candidate_ids:
            text = candidate_text(unit_id, row, memory)
            if text:
                usable_ids.append(unit_id)
                usable_texts.append(text)
        if not positive_id or positive_id not in usable_ids:
            skipped["missing_positive_or_candidate"] += 1
            continue

        contexts = context_variants(row, grouped, other_states, memory, args.seed)
        compressed_indices, _ = fixed_front_pool(
            contexts["correct"], usable_ids, usable_texts, memory, dense, args
        )
        fixed_ids = [usable_ids[index] for index in compressed_indices]
        fixed_texts = [usable_texts[index] for index in compressed_indices]
        positive_in_pool = positive_id in fixed_ids
        condition_results = {}
        if positive_in_pool:
            positive_index = fixed_ids.index(positive_id)
            for condition in CONDITIONS:
                scores = policy.score(contexts[condition], fixed_texts)
                condition_results[condition] = rank_metrics(scores, positive_index)
        records.append(
            {
                "qid": qid,
                "t": t,
                "question_type": question_type(query_map.get(qid)),
                "positive_unit_id": positive_id,
                "raw_candidate_count": len(usable_ids),
                "fixed_pool_count": len(fixed_ids),
                "positive_in_fixed_pool": positive_in_pool,
                "fixed_candidate_ids": fixed_ids,
                "conditions": condition_results,
            }
        )

    rank_reversal_summary, rank_reversal_records = score_rank_reversals(
        grouped, policy, memory, other_states, args.seed
    )
    bootstrap = {
        baseline: paired_bootstrap(records, baseline, args.n_bootstrap, args.seed + index)
        for index, baseline in enumerate(CONDITIONS[1:], start=1)
    }
    summary = {
        "status": "OK",
        "samples": args.samples,
        "memory": args.memory,
        "queries": args.queries,
        "checkpoint": args.checkpoint,
        "model_dir": args.model_dir,
        "dense_model": args.dense_model,
        "fixed_pool_protocol": {
            "front_context": "correct dataset K_t",
            "front_pool_k": args.front_pool_k,
            "front_fusion": args.front_fusion,
            "hybrid_alpha": args.hybrid_alpha,
            "dense_query_mode": args.dense_query_mode,
            "candidate_top_k": args.candidate_top_k,
            "local_expansion_window": args.local_expansion_window,
            "mmr_lambda": args.mmr_lambda,
            "mmr_same_doc_similarity": args.mmr_same_doc_similarity,
            "policy_score_mode": "policy_only_for_mechanism_isolation",
        },
        "conditions": list(CONDITIONS),
        "data_diagnostics": diagnostics,
        "evaluated_states": len(records),
        "skipped": dict(skipped),
        "slices": aggregate_by_slice(records),
        "state_necessity": state_necessity(records),
        "paired_bootstrap_t_ge_1_given_pool": bootstrap,
        "conditional_rank_reversal": rank_reversal_summary,
        "record_files": {
            "state_intervention": str(output_dir / "state_intervention_records.jsonl"),
            "rank_reversal": str(output_dir / "rank_reversal_records.jsonl"),
        },
    }
    write_jsonl(output_dir / "state_intervention_records.jsonl", records)
    write_jsonl(output_dir / "rank_reversal_records.jsonl", rank_reversal_records)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
