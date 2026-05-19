import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_BASE = PROJECT_DIR / "data" / "hotpotqa_distractor_v3"
OUTPUT_JSON = "derived_counterfactual_analysis_v3.json"
OUTPUT_MD = "derived_counterfactual_analysis_v3.md"

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for", "and",
    "or", "that", "this", "these", "those", "which", "what", "who", "whom", "when", "where",
    "why", "how", "it", "its", "as", "by", "from", "with", "at", "be", "been", "being",
    "but", "not", "no", "direct", "details", "provided", "current", "raw", "evidence",
    "help", "may", "can", "about", "into", "their", "them", "he", "she", "his", "her",
    "they", "both", "one", "two", "three", "four", "five", "more", "than",
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def normalize_text(text: Any) -> str:
    x = str(text or "").lower().strip()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[\"'“”‘’`.,;:!?()\[\]{}]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def tokenize(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def strip_derived_from_k_t(k_t: str) -> str:
    lines = []
    for line in str(k_t or "").splitlines():
        if line.strip().startswith("[derived_note]"):
            continue
        if line.strip() == "Notes:":
            continue
        if line.strip().startswith("[bridge]") or line.strip().startswith("[verification]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def context_contains_answer(gold_answer: str, k_t: str) -> bool:
    norm_gold = normalize_text(gold_answer)
    norm_ctx = normalize_text(k_t)
    return bool(norm_gold) and norm_gold in norm_ctx


def load_queries(base_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for split in SPLITS:
        path = base_dir / "queries" / f"{split}.jsonl"
        for row in read_jsonl(path):
            out[str(row["qid"])] = {
                "split": split,
                "question": str(row.get("question", "")).strip(),
                "answer": str(row.get("answer", "")).strip(),
            }
    return out


def load_success_debug(base_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for split in SPLITS:
        path = base_dir / "debug" / f"success_semantic_debug_{split}.jsonl"
        for row in read_jsonl(path):
            out[str(row["qid"])] = row
    return out


def load_raw_unit_map(base_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for split in SPLITS:
        path = base_dir / "unit_registry" / f"raw_units_{split}.jsonl"
        for row in read_jsonl(path):
            out[str(row["unit_id"])] = row
    return out


def load_samples(base_dir: Path) -> Dict[Tuple[str, int], dict]:
    out: Dict[Tuple[str, int], dict] = {}
    for split in SPLITS:
        path = base_dir / "samples" / f"{split}.jsonl"
        for row in read_jsonl(path):
            out[(str(row["qid"]), int(row["t"]))] = row
    return out


def load_derived_strategy_analysis(base_dir: Path) -> dict:
    path = base_dir / "debug" / "derived_strategy_analysis_v3.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def filtered_novel_tokens(note_text: str, source_texts: List[str]) -> List[str]:
    note_tokens = set(tokenize(note_text))
    source_tokens = set()
    for text in source_texts:
        source_tokens.update(tokenize(text))
    novel = [
        tok for tok in sorted(note_tokens - source_tokens)
        if len(tok) >= 3 and tok not in STOPWORDS
    ]
    return novel


def infer_counterfactual_role(
    *,
    note_type: str,
    note_mentions_gold: bool,
    source_mentions_gold: bool,
    novel_tokens: List[str],
    next_raw_positive_in_current_c_t: bool,
    next_is_raw: bool,
    next_delta_covered_targets: Optional[int],
) -> str:
    if note_type == "verification_note":
        if note_mentions_gold and not source_mentions_gold:
            return "answer_facing_verification"
        if novel_tokens:
            return "abstractive_verification"
        return "verification_without_new_surface_signal"
    if note_type == "bridge_note":
        if next_is_raw and not next_raw_positive_in_current_c_t:
            return "bridge_scaffold_for_new_raw"
        if next_is_raw and (next_delta_covered_targets or 0) > 0:
            return "bridge_scaffold_for_progress"
        return "bridge_contextualization"
    return "unknown"


def analyze_event(
    *,
    qid: str,
    question: str,
    gold_answer: str,
    split: str,
    event: dict,
    success_debug_by_qid: Dict[str, dict],
    raw_unit_map: Dict[str, dict],
) -> dict:
    source_unit_ids = [str(x) for x in event.get("source_unit_ids", [])]
    source_texts = []
    for unit_id in source_unit_ids:
        row = raw_unit_map.get(unit_id, {})
        text = str(row.get("text", "")).strip()
        if text:
            source_texts.append(text)

    note_text = str(event.get("note_text") or "").strip()
    note_type = str(event.get("note_type") or "unknown").strip() or "unknown"

    success_debug = success_debug_by_qid[qid]
    final_k_t = str(success_debug.get("terminal_state", {}).get("K_t", ""))
    final_k_t_without_derived = strip_derived_from_k_t(final_k_t)

    note_mentions_gold = context_contains_answer(gold_answer, note_text)
    source_mentions_gold = any(context_contains_answer(gold_answer, text) for text in source_texts)
    final_k_t_has_gold = context_contains_answer(gold_answer, final_k_t)
    final_k_t_without_derived_has_gold = context_contains_answer(gold_answer, final_k_t_without_derived)
    novel_tokens = filtered_novel_tokens(note_text, source_texts)

    return {
        "qid": qid,
        "split": split,
        "question": question,
        "gold_answer": gold_answer,
        "t": int(event["t"]),
        "terminal_t": event.get("terminal_t"),
        "position_bucket": event.get("position_bucket"),
        "positive_unit_id": event.get("positive_unit_id"),
        "note_type": note_type,
        "note_text": note_text,
        "source_unit_ids": source_unit_ids,
        "source_texts": source_texts,
        "derive_goal": event.get("derive_goal"),
        "next_is_raw": bool(event.get("next_is_raw")),
        "next_raw_positive_id": event.get("next_raw_positive_id"),
        "next_raw_positive_in_current_c_t": bool(event.get("next_raw_positive_in_current_c_t")),
        "next_delta_covered_targets": event.get("next_delta_covered_targets"),
        "note_mentions_gold": note_mentions_gold,
        "source_mentions_gold": source_mentions_gold,
        "final_k_t_has_gold": final_k_t_has_gold,
        "final_k_t_without_derived_has_gold": final_k_t_without_derived_has_gold,
        "novel_tokens_vs_sources": novel_tokens,
        "counterfactual_role": infer_counterfactual_role(
            note_type=note_type,
            note_mentions_gold=note_mentions_gold,
            source_mentions_gold=source_mentions_gold,
            novel_tokens=novel_tokens,
            next_raw_positive_in_current_c_t=bool(event.get("next_raw_positive_in_current_c_t")),
            next_is_raw=bool(event.get("next_is_raw")),
            next_delta_covered_targets=event.get("next_delta_covered_targets"),
        ),
    }


def summarize(records: List[dict]) -> dict:
    role_counts: Dict[str, int] = {}
    for row in records:
        role = str(row["counterfactual_role"])
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "event_count": len(records),
        "role_counts": role_counts,
        "events_where_note_mentions_gold_but_sources_do_not": sum(
            1 for row in records if row["note_mentions_gold"] and not row["source_mentions_gold"]
        ),
        "events_where_final_k_t_still_has_gold_without_derived": sum(
            1 for row in records if row["final_k_t_without_derived_has_gold"]
        ),
        "events_where_next_raw_not_yet_visible": sum(
            1 for row in records if not row["next_raw_positive_in_current_c_t"] and row["next_is_raw"]
        ),
    }


def build_markdown(run_id: str, overall_summary: dict, historical_summary: dict, historical_records: List[dict]) -> str:
    lines: List[str] = []
    lines.append("# Derived Counterfactual Analysis v2")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- summary: `{json.dumps(overall_summary, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Historical Failure -> Success Focus Set")
    lines.append("")
    lines.append(f"- summary: `{json.dumps(historical_summary, ensure_ascii=False)}`")
    lines.append("")
    for row in historical_records:
        lines.append(f"### {row['split']} / {row['qid']} / t={row['t']}")
        lines.append(f"- question: {row['question']}")
        lines.append(f"- note_type: `{row['note_type']}`")
        lines.append(f"- counterfactual_role: `{row['counterfactual_role']}`")
        lines.append(f"- next_is_raw: `{row['next_is_raw']}`")
        lines.append(f"- next_raw_positive_in_current_c_t: `{row['next_raw_positive_in_current_c_t']}`")
        lines.append(f"- note_mentions_gold: `{row['note_mentions_gold']}`")
        lines.append(f"- source_mentions_gold: `{row['source_mentions_gold']}`")
        lines.append(f"- final_k_t_has_gold: `{row['final_k_t_has_gold']}`")
        lines.append(f"- final_k_t_without_derived_has_gold: `{row['final_k_t_without_derived_has_gold']}`")
        lines.append(f"- novel_tokens_vs_sources: `{row['novel_tokens_vs_sources']}`")
        lines.append(f"- note_text: {row['note_text']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    strategy = load_derived_strategy_analysis(base_dir)
    run_id = str(strategy.get("build_meta", {}).get("run_id", "")).strip()
    queries_by_qid = load_queries(base_dir)
    success_debug_by_qid = load_success_debug(base_dir)
    raw_unit_map = load_raw_unit_map(base_dir)
    samples_by_key = load_samples(base_dir)
    _ = samples_by_key

    all_records: List[dict] = []
    historical_records: List[dict] = []

    for case in strategy.get("all_case_records", []):
        qid = str(case["qid"])
        split = str(case["split"])
        question = queries_by_qid[qid]["question"]
        gold_answer = queries_by_qid[qid]["answer"]
        for event in case.get("derived_positive_events", []):
            row = analyze_event(
                qid=qid,
                question=question,
                gold_answer=gold_answer,
                split=split,
                event=event,
                success_debug_by_qid=success_debug_by_qid,
                raw_unit_map=raw_unit_map,
            )
            all_records.append(row)
            if qid in {
                "5a7630cb554299109176e6af",
                "5a82a36a55429954d2e2eb8d",
                "5a7bbe76554299042af8f7d3",
                "5adbcbca5542996e6852523b",
                "5ae0c9f355429906c02dab51",
                "5ae536065542990ba0bbb227",
                "5a8b42be55429949d91db515",
                "5ab678f455429954757d32d3",
                "5ac2e9c45542990b17b154a0",
                "5ae5fc345542993aec5ec1f8",
            }:
                historical_records.append(row)

    payload = {
        "build_meta": {
            "run_id": run_id,
            "source": "analyze_derived_counterfactual_v3.py",
        },
        "overall_summary": summarize(all_records),
        "historical_failure_to_success_summary": summarize(historical_records),
        "all_records": all_records,
        "historical_failure_to_success_records": historical_records,
    }

    debug_dir = base_dir / "debug"
    json_path = debug_dir / OUTPUT_JSON
    md_path = debug_dir / OUTPUT_MD
    write_json(json_path, payload)
    write_text(
        md_path,
        build_markdown(
            run_id=run_id,
            overall_summary=payload["overall_summary"],
            historical_summary=payload["historical_failure_to_success_summary"],
            historical_records=historical_records,
        ),
    )
    print(f"derived counterfactual analysis written: {json_path}")
    print(f"derived counterfactual analysis written: {md_path}")


if __name__ == "__main__":
    main()
