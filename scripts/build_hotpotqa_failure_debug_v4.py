import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from build_hotpotqa_success_debug_v4 import (
    DEFAULT_BASE,
    SPLITS,
    build_chunks_block,
    build_history_entry,
    load_derived_harvest,
    load_full,
    load_queries,
    load_raw_chunk_registry,
    load_sample_derived_payloads,
    load_states,
    extract_run_id_from_full_records,
    merge_unique_in_order,
    render_k_t_with_derived_payloads,
    resolve_answer_probe,
    write_debug_manifest,
    write_jsonl,
)

FAILURE_REPEAT_RATIO_THRESHOLD = 0.60


def infer_failure_status(full_rec: dict) -> Dict[str, Optional[str]]:
    abort_reason = full_rec.get("abort_reason")
    ever_progress = bool(full_rec.get("ever_progress", False))
    if abort_reason == "stalled" and not ever_progress:
        status = "failed_stalled"
    elif abort_reason == "stalled" and ever_progress:
        status = "failed_but_progressive"
    elif ever_progress:
        status = "failed_but_progressive"
    else:
        status = "failed_stalled"
    return {
        "status": status,
        "terminal_step": None,
        "abort_reason": None if abort_reason is None else str(abort_reason),
    }


def infer_failure_type(full_rec: dict, answer_probe: dict, last_step: dict) -> str:
    abort_reason = str(full_rec.get("abort_reason") or "")
    final_false_stop_count = int(full_rec.get("final_false_stop_count", 0) or 0)
    repair_attempt_count = int(full_rec.get("repair_attempt_count", 0) or 0)
    repair_effective = bool(full_rec.get("repair_effective", False))
    repeat_ratio = last_step.get("retrieval_repeat_ratio")
    repeat_high = repeat_ratio is not None and float(repeat_ratio) >= FAILURE_REPEAT_RATIO_THRESHOLD
    last_need_derived = bool(last_step.get("need_derived", False))
    last_triggered_propose_derived = bool(last_step.get("triggered_propose_derived", False))

    if bool(answer_probe.get("SupportSufficient_t", False)) and not bool(answer_probe.get("AnswerCorrect_t", False)):
        return "answer_wrong_after_sufficient_evidence"
    if repair_attempt_count > 0 and not repair_effective:
        return "derived_repair_no_effect"
    if final_false_stop_count > 0 and abort_reason in {"stalled", "repeated_false_stop_no_progress"}:
        return "false_stop_then_stalled"
    if int(full_rec.get("stop_probe_count", 0) or 0) == 0 and last_need_derived and last_triggered_propose_derived:
        return "stop_gate_too_conservative"
    if repeat_high and int(last_step.get("delta_covered_targets", 0) or 0) == 0:
        return "retrieval_repeat_stagnation"
    return "retrieval_repeat_stagnation"


def get_latest_state_for_qid(states_by_key: Dict[Tuple[str, int], dict], qid: str) -> dict:
    candidates = [(t, rec) for (rec_qid, t), rec in states_by_key.items() if rec_qid == qid]
    if not candidates:
        raise ValueError(f"找不到 failure trajectory 的最新 state: qid={qid}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def build_failure_debug_record(
    *,
    qid: str,
    query_info: dict,
    full_rec: dict,
    latest_state: dict,
    unit_registry: Dict[str, dict],
    chunk_registry: Dict[str, dict],
    derived_payloads_by_qid: Dict[str, Dict[str, dict]],
    run_id: str,
) -> dict:
    derived_payloads = derived_payloads_by_qid.get(qid, {})
    history = [build_history_entry(item, unit_registry) for item in latest_state.get("H_t", [])]
    k_t = render_k_t_with_derived_payloads(str(latest_state.get("K_t", "")), derived_payloads)
    selected_chunk_ids, chunks = build_chunks_block(history, unit_registry, chunk_registry)
    selected_unit_ids = [item["unit_id"] for item in history]
    answer_probe = resolve_answer_probe(full_rec)
    last_step = full_rec.get("steps", [])[-1] if full_rec.get("steps") else {}

    warnings: List[str] = []
    if "[missing derived payload]" in k_t:
        warnings.append("missing_derived_payload_in_K_t")
    if not chunks:
        warnings.append("no_selected_chunks")
    failure_type = infer_failure_type(full_rec, answer_probe, last_step)

    return {
        "qid": qid,
        "build_meta": {
            "run_id": run_id,
            "source": "build_hotpotqa_failure_debug_v4.py",
        },
        "question": query_info["question"],
        "gold_answer": str(query_info.get("answer", "")),
        "trajectory_status": infer_failure_status(full_rec),
        "terminal_state": {
            "t": int(latest_state["t"]),
            "source_state_t": int(latest_state["t"]),
            "H_t": history,
            "K_t": k_t,
            "selected_unit_ids": selected_unit_ids,
            "selected_chunk_ids": selected_chunk_ids,
        },
        "chunks": chunks,
        "answer_probe": answer_probe,
        "failure_semantic": {
            "failure_type": failure_type,
            "abort_reason": full_rec.get("abort_reason"),
            "ever_progress": bool(full_rec.get("ever_progress", False)),
            "stop_probe_count": int(full_rec.get("stop_probe_count", 0)),
            "final_false_stop_count": int(full_rec.get("final_false_stop_count", 0)),
            "last_positive_unit_id": last_step.get("positive_unit_id"),
            "last_stop_candidate": last_step.get("stop_candidate"),
            "last_need_derived": last_step.get("need_derived"),
            "last_triggered_propose_derived": last_step.get("triggered_propose_derived"),
            "last_retrieval_repeat_ratio": last_step.get("retrieval_repeat_ratio"),
            "last_delta_covered_targets": last_step.get("delta_covered_targets"),
            "repair_attempt_count": int(full_rec.get("repair_attempt_count", 0) or 0),
            "repair_effective": bool(full_rec.get("repair_effective", False)),
            "repair_failure_reason": full_rec.get("repair_failure_reason"),
            "stop_reopened_after_false_stop": bool(full_rec.get("stop_reopened_after_false_stop", False)),
        },
        "debug_warnings": merge_unique_in_order(warnings),
    }


def build_split(base_dir: Path, split: str) -> Tuple[List[dict], dict, str]:
    trajectories_dir = base_dir / "trajectories"
    samples_dir = base_dir / "samples"
    queries_dir = base_dir / "queries"
    unit_registry_dir = base_dir / "unit_registry"
    index_store_dir = base_dir / "index_store"

    queries = load_queries(queries_dir / f"{split}.jsonl")
    full_by_qid = load_full(trajectories_dir / f"full_{split}.jsonl")
    run_id = extract_run_id_from_full_records(full_by_qid, split=split)
    states_by_key = load_states(trajectories_dir / f"states_{split}.jsonl")
    unit_registry, chunk_registry = load_raw_chunk_registry(
        unit_registry_dir / f"raw_units_{split}.jsonl",
        index_store_dir / f"chunks_{split}.jsonl",
    )

    derived_payloads_by_qid: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for qid, payloads in load_sample_derived_payloads(samples_dir / f"{split}.jsonl").items():
        derived_payloads_by_qid[qid].update(payloads)
    for qid, payloads in load_derived_harvest(trajectories_dir / f"derived_harvest_{split}.jsonl").items():
        derived_payloads_by_qid[qid].update(payloads)

    records: List[dict] = []
    stats = {
        "trajectories": len(full_by_qid),
        "failures": 0,
        "warning_counts": Counter(),
    }

    for qid, full_rec in full_by_qid.items():
        if full_rec.get("terminal_status") == "terminal":
            continue
        query_info = queries.get(qid)
        if query_info is None:
            raise ValueError(f"queries 中找不到 qid: split={split}, qid={qid}")
        latest_state = get_latest_state_for_qid(states_by_key, qid)
        record = build_failure_debug_record(
            qid=qid,
            query_info=query_info,
            full_rec=full_rec,
            latest_state=latest_state,
            unit_registry=unit_registry,
            chunk_registry=chunk_registry,
            derived_payloads_by_qid=derived_payloads_by_qid,
            run_id=run_id,
        )
        records.append(record)
        stats["failures"] += 1
        stats["warning_counts"].update(record["debug_warnings"])

    return records, stats, run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Build failure semantic debug v2")
    parser.add_argument("--force", action="store_true", help="Accepted for compatibility; overwrites fixed output files in place")
    parser.parse_args()

    base_dir = DEFAULT_BASE
    debug_dir = base_dir / "debug"
    stats_by_split: Dict[str, dict] = {}
    run_ids_by_split: Dict[str, str] = {}
    for split in SPLITS:
        records, stats, run_id = build_split(base_dir, split)
        out_path = debug_dir / f"failure_semantic_debug_{split}.jsonl"
        write_jsonl(records, out_path)
        stats_by_split[split] = stats
        run_ids_by_split[split] = run_id
    write_debug_manifest(base_dir, debug_type="failure", run_ids_by_split=run_ids_by_split)

    print("failure semantic debug built:")
    for split in SPLITS:
        stats = stats_by_split[split]
        print(f"  {split}: failures={stats['failures']}")
        if stats["warning_counts"]:
            print(f"    warnings={dict(stats['warning_counts'])}")


if __name__ == "__main__":
    main()
