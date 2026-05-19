import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SPLITS = ["train", "val", "test"]
BASE_DIR = Path("/home/lmy/study/project/data/hotpotqa_distractor_v4")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败: file={path}, line={line_idx}, error={e}") from e


def load_by_qid(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in read_jsonl(path):
        out[str(row["qid"])] = row
    return out


def load_samples(path: Path) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for row in read_jsonl(path):
        key = (str(row["qid"]), int(row["t"]))
        out[key] = row
    return out


def build_chunk_text_index(debug_record: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for chunk in debug_record.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("chunk_id")
        if isinstance(chunk_id, str):
            out[chunk_id] = chunk
    return out


def compare_h_t_prefix(sample_h: List[dict], debug_h: List[dict]) -> Tuple[bool, int]:
    for idx, (sample_item, debug_item) in enumerate(zip(sample_h, debug_h)):
        for key in ["step_id", "unit_id", "chunk_id", "doc_id", "parent_chunk_id"]:
            if sample_item.get(key) != debug_item.get(key):
                return False, idx
    return True, -1


def summarize_unit_ids(items: List[dict]) -> List[str]:
    return [str(item.get("unit_id")) for item in items]


def unique_sample_chunk_ids(sample_h: List[dict]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in sample_h:
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str):
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(chunk_id)
    return out


def validate_split(split: str) -> Tuple[Counter, List[dict]]:
    samples = load_samples(BASE_DIR / "samples" / f"{split}.jsonl")
    full_by_qid = load_by_qid(BASE_DIR / "trajectories" / f"full_{split}.jsonl")
    debug_by_qid = load_by_qid(BASE_DIR / "debug" / f"success_semantic_debug_{split}.jsonl")

    stats = Counter()
    stats["samples"] = len(samples)
    errors: List[dict] = []

    sample_qids = {qid for qid, _ in samples.keys()}
    for qid in sample_qids:
        if qid not in full_by_qid:
            errors.append({"split": split, "qid": qid, "t": None, "error_type": "missing_qid_in_full"})
            stats["missing_qid_in_full"] += 1

    for qid, debug_record in debug_by_qid.items():
        terminal_step = debug_record.get("trajectory_status", {}).get("terminal_step")
        if isinstance(terminal_step, int) and terminal_step > 0 and qid not in sample_qids:
            errors.append({"split": split, "qid": qid, "t": None, "error_type": "missing_success_qid_in_samples"})
            stats["missing_success_qid_in_samples"] += 1

    for (qid, t), sample in sorted(samples.items()):
        stats["checked"] += 1
        full_record = full_by_qid.get(qid)
        if full_record is None:
            continue

        trajectory_status = sample.get("meta", {}).get("trajectory_status", {})
        if trajectory_status.get("status") != "success":
            continue

        debug_record = debug_by_qid.get(qid)
        if debug_record is None:
            errors.append({"split": split, "qid": qid, "t": t, "error_type": "missing_success_debug_record"})
            stats["missing_success_debug_record"] += 1
            continue

        sample_terminal_step = trajectory_status.get("terminal_step")
        debug_terminal_step = debug_record.get("trajectory_status", {}).get("terminal_step")
        if sample_terminal_step != debug_terminal_step:
            errors.append(
                {
                    "split": split,
                    "qid": qid,
                    "t": t,
                    "terminal_step_sample": sample_terminal_step,
                    "terminal_step_debug": debug_terminal_step,
                    "error_type": "terminal_step_mismatch",
                }
            )
            stats["terminal_step_mismatch"] += 1

        sample_h = sample.get("state", {}).get("H_t", [])
        debug_h = debug_record.get("terminal_state", {}).get("H_t", [])[: len(sample_h)]
        if len(sample_h) > len(debug_record.get("terminal_state", {}).get("H_t", [])):
            errors.append(
                {
                    "split": split,
                    "qid": qid,
                    "t": t,
                    "sample_H_t_unit_ids": summarize_unit_ids(sample_h),
                    "debug_H_t_unit_ids": summarize_unit_ids(debug_record.get("terminal_state", {}).get("H_t", [])),
                    "first_mismatch_index": len(debug_record.get("terminal_state", {}).get("H_t", [])),
                    "error_type": "h_t_prefix_mismatch",
                }
            )
            stats["h_t_prefix_mismatch"] += 1
        else:
            matched, mismatch_idx = compare_h_t_prefix(sample_h, debug_h)
            if not matched:
                errors.append(
                    {
                        "split": split,
                        "qid": qid,
                        "t": t,
                        "sample_H_t_unit_ids": summarize_unit_ids(sample_h),
                        "debug_H_t_unit_ids": summarize_unit_ids(debug_record.get("terminal_state", {}).get("H_t", [])),
                        "first_mismatch_index": mismatch_idx,
                        "error_type": "h_t_prefix_mismatch",
                    }
                )
                stats["h_t_prefix_mismatch"] += 1

        chunk_index = build_chunk_text_index(debug_record)
        missing_chunk = False
        for item in sample_h:
            chunk_id = item.get("chunk_id")
            unit_id = str(item.get("unit_id"))
            if chunk_id is None:
                continue
            if chunk_id not in chunk_index:
                missing_chunk = True
                errors.append(
                    {
                        "split": split,
                        "qid": qid,
                        "t": t,
                        "unit_id": unit_id,
                        "chunk_id": chunk_id,
                        "error_type": "missing_sample_chunk_in_debug",
                    }
                )
                stats["missing_sample_chunk_in_debug"] += 1
                continue
            selected_unit_ids = chunk_index[chunk_id].get("selected_unit_ids_in_this_chunk", [])
            if unit_id not in selected_unit_ids:
                errors.append(
                    {
                        "split": split,
                        "qid": qid,
                        "t": t,
                        "unit_id": unit_id,
                        "chunk_id": chunk_id,
                        "error_type": "missing_sample_unit_in_debug",
                    }
                )
                stats["missing_sample_unit_in_debug"] += 1
        if not missing_chunk:
            sample_k_t = str(sample.get("state", {}).get("K_t", ""))
            expected_chunk_ids = unique_sample_chunk_ids(sample_h)
            missing_k_t_chunk = False
            for chunk_id in expected_chunk_ids:
                chunk = chunk_index.get(chunk_id, {})
                full_chunk_text = chunk.get("full_chunk_text")
                if isinstance(full_chunk_text, str) and full_chunk_text.strip():
                    if full_chunk_text.strip() not in sample_k_t:
                        missing_k_t_chunk = True
                        break
            if missing_k_t_chunk:
                errors.append(
                    {
                        "split": split,
                        "qid": qid,
                        "t": t,
                        "error_type": "k_t_prefix_mismatch",
                    }
                )
                stats["k_t_prefix_mismatch"] += 1

        full_steps = full_record.get("steps", [])
        if not isinstance(t, int) or t >= len(full_steps):
            errors.append({"split": split, "qid": qid, "t": t, "error_type": "sample_candidate_not_in_current_full"})
            stats["sample_candidate_not_in_current_full"] += 1
            continue

        step = full_steps[t]
        sample_positive = sample.get("labels", {}).get("u_t_plus", {}).get("unit_id")
        full_positive = step.get("positive_unit_id")
        if sample_positive != full_positive:
            errors.append(
                {
                    "split": split,
                    "qid": qid,
                    "t": t,
                    "sample_positive_unit_id": sample_positive,
                    "full_positive_unit_id": full_positive,
                    "error_type": "sample_candidate_not_in_current_full",
                }
            )
            stats["sample_candidate_not_in_current_full"] += 1

        sample_c_t = sample.get("candidates", {}).get("C_t", [])
        full_c_t = step.get("C_t", [])
        if sample_c_t != full_c_t:
            errors.append(
                {
                    "split": split,
                    "qid": qid,
                    "t": t,
                    "error_type": "sample_candidate_not_in_current_full",
                }
            )
            stats["sample_candidate_not_in_current_full"] += 1

    return stats, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate sample/debug alignment")
    parser.add_argument("--split", choices=SPLITS, help="Only validate one split")
    args = parser.parse_args()

    target_splits = [args.split] if args.split else SPLITS
    print("sample-debug alignment:")
    total_errors = 0
    for split in target_splits:
        stats, errors = validate_split(split)
        total_errors += len(errors)
        print(f"  {split}:")
        print(f"    samples: {stats['samples']}")
        print(f"    checked: {stats['checked']}")
        print(f"    errors: {len(errors)}")
        print(f"    h_t_prefix_mismatch: {stats['h_t_prefix_mismatch']}")
        print(f"    terminal_step_mismatch: {stats['terminal_step_mismatch']}")
        print(f"    missing_sample_chunk_in_debug: {stats['missing_sample_chunk_in_debug']}")
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))
    print(f"errors={total_errors}")


if __name__ == "__main__":
    main()
