#!/usr/bin/env python3
"""Validate KBS teacher-student sample schema.

This validator is intentionally lightweight and pure-Python. It checks whether
HotpotQA prefix samples expose the fields required by the KBS pipeline:
knowledge state, candidate pool, ranking labels, deficit labels, contribution
labels, and stop labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = ("qid", "t", "question", "state", "candidates", "labels")
REQUIRED_STATE = ("H_t", "A_t", "S_t", "K_t")
REQUIRED_CANDIDATES = ("C_t", "R_t", "G_t_final", "G_t_aux", "G_t_illegal")
REQUIRED_LABELS = ("u_t_plus", "d_t_star", "c_t_star", "ranking_label", "stop_label")
DEFICIT_KEYS = ("d_raw", "d_br", "d_dis", "d_sup", "d_der")
CONTRIBUTION_KEYS = ("c_raw", "c_br", "c_dis", "c_sup", "c_der")
STOP_TYPES = {"continue", "non-terminal", "near-terminal", "terminal", "abort"}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, {"__json_error__": str(exc)}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def add_issue(issues: list[dict], split: str, line_no: int, issue: str, **extra: Any) -> None:
    item = {
        "split": split,
        "line": line_no,
        "issue": issue,
    }
    qid = extra.pop("qid", None)
    t = extra.pop("t", None)
    if qid is not None:
        item["qid"] = qid
    if t is not None:
        item["t"] = t
    item.update(extra)
    issues.append(item)


def check_numeric_object(
    issues: list[dict],
    *,
    split: str,
    line_no: int,
    qid: str,
    t: Any,
    obj: Any,
    keys: tuple[str, ...],
    name: str,
) -> None:
    if not isinstance(obj, dict):
        add_issue(issues, split, line_no, f"{name}_not_object", qid=qid, t=t)
        return
    missing = [key for key in keys if key not in obj]
    extra = sorted(set(obj) - set(keys))
    if missing:
        add_issue(issues, split, line_no, f"{name}_missing_keys", qid=qid, t=t, keys=missing)
    if extra:
        add_issue(issues, split, line_no, f"{name}_extra_keys", qid=qid, t=t, keys=extra)
    for key in keys:
        value = obj.get(key)
        if not is_number(value):
            add_issue(issues, split, line_no, f"{name}_{key}_not_number", qid=qid, t=t, value=value)
        elif not 0.0 <= float(value) <= 1.0:
            add_issue(issues, split, line_no, f"{name}_{key}_outside_0_1", qid=qid, t=t, value=value)


def check_record(record: dict, *, split: str, line_no: int, issues: list[dict], stats: dict) -> None:
    if "__json_error__" in record:
        add_issue(issues, split, line_no, "invalid_json", detail=record["__json_error__"])
        return

    qid = str(record.get("qid", ""))
    t = record.get("t")
    stats["records"] += 1
    if qid:
        stats["qids"].add(qid)

    for key in REQUIRED_TOP_LEVEL:
        if key not in record:
            add_issue(issues, split, line_no, "missing_top_level_key", qid=qid, t=t, key=key)

    if not isinstance(record.get("qid"), str) or not record.get("qid"):
        add_issue(issues, split, line_no, "qid_invalid", qid=qid, t=t)
    if not isinstance(t, int) or isinstance(t, bool):
        add_issue(issues, split, line_no, "t_not_int", qid=qid, t=t)
    if not isinstance(record.get("question"), str) or not record.get("question", "").strip():
        add_issue(issues, split, line_no, "question_empty", qid=qid, t=t)

    state = record.get("state")
    if not isinstance(state, dict):
        add_issue(issues, split, line_no, "state_not_object", qid=qid, t=t)
        state = {}
    for key in REQUIRED_STATE:
        if key not in state:
            add_issue(issues, split, line_no, "state_missing_key", qid=qid, t=t, key=key)
    if not isinstance(state.get("H_t"), list):
        add_issue(issues, split, line_no, "state_H_t_not_list", qid=qid, t=t)
    if not isinstance(state.get("A_t"), dict):
        add_issue(issues, split, line_no, "state_A_t_not_object", qid=qid, t=t)
    if not isinstance(state.get("S_t"), dict):
        add_issue(issues, split, line_no, "state_S_t_not_object", qid=qid, t=t)
    if not isinstance(state.get("K_t"), str) or not state.get("K_t", "").strip():
        add_issue(issues, split, line_no, "state_K_t_empty", qid=qid, t=t)

    candidates = record.get("candidates")
    if not isinstance(candidates, dict):
        add_issue(issues, split, line_no, "candidates_not_object", qid=qid, t=t)
        candidates = {}
    for key in REQUIRED_CANDIDATES:
        if key not in candidates:
            add_issue(issues, split, line_no, "candidates_missing_key", qid=qid, t=t, key=key)
        elif not is_str_list(candidates.get(key)):
            add_issue(issues, split, line_no, "candidate_list_invalid", qid=qid, t=t, key=key)

    c_t = candidates.get("C_t") if is_str_list(candidates.get("C_t")) else []
    c_t_set = set(c_t)
    stats["candidate_count"][len(c_t)] += 1
    if not c_t:
        add_issue(issues, split, line_no, "C_t_empty", qid=qid, t=t)
    if len(c_t) != len(c_t_set):
        add_issue(issues, split, line_no, "C_t_duplicate_unit_ids", qid=qid, t=t)

    labels = record.get("labels")
    if not isinstance(labels, dict):
        add_issue(issues, split, line_no, "labels_not_object", qid=qid, t=t)
        labels = {}
    for key in REQUIRED_LABELS:
        if key not in labels:
            add_issue(issues, split, line_no, "labels_missing_key", qid=qid, t=t, key=key)

    ranking = labels.get("ranking_label")
    if not isinstance(ranking, dict):
        add_issue(issues, split, line_no, "ranking_label_not_object", qid=qid, t=t)
        ranking = {}
    positive = ranking.get("positive_unit_id")
    negatives = ranking.get("negative_unit_ids")
    if not isinstance(positive, str) or not positive:
        add_issue(issues, split, line_no, "ranking_positive_invalid", qid=qid, t=t)
    elif positive not in c_t_set:
        add_issue(issues, split, line_no, "ranking_positive_not_in_C_t", qid=qid, t=t, positive_unit_id=positive)
    if not is_str_list(negatives):
        add_issue(issues, split, line_no, "ranking_negatives_invalid", qid=qid, t=t)
        negatives = []
    else:
        if len(negatives) != len(set(negatives)):
            add_issue(issues, split, line_no, "ranking_negatives_duplicate", qid=qid, t=t)
        if positive in set(negatives):
            add_issue(issues, split, line_no, "ranking_positive_in_negatives", qid=qid, t=t)
        outside = sorted(set(negatives) - c_t_set)
        if outside:
            add_issue(
                issues,
                split,
                line_no,
                "ranking_negative_outside_C_t",
                qid=qid,
                t=t,
                count=len(outside),
                examples=outside[:5],
            )
    if isinstance(positive, str) and positive:
        u_t_plus = labels.get("u_t_plus")
        if isinstance(u_t_plus, dict) and u_t_plus.get("unit_id") != positive:
            add_issue(
                issues,
                split,
                line_no,
                "u_t_plus_mismatch_ranking_positive",
                qid=qid,
                t=t,
                u_t_plus=u_t_plus.get("unit_id"),
                positive_unit_id=positive,
            )

    check_numeric_object(
        issues,
        split=split,
        line_no=line_no,
        qid=qid,
        t=t,
        obj=labels.get("d_t_star"),
        keys=DEFICIT_KEYS,
        name="d_t_star",
    )
    check_numeric_object(
        issues,
        split=split,
        line_no=line_no,
        qid=qid,
        t=t,
        obj=labels.get("c_t_star"),
        keys=CONTRIBUTION_KEYS,
        name="c_t_star",
    )

    stop = labels.get("stop_label")
    if not isinstance(stop, dict):
        add_issue(issues, split, line_no, "stop_label_not_object", qid=qid, t=t)
    else:
        should_stop = stop.get("should_stop")
        label_type = stop.get("label_type")
        if not isinstance(should_stop, bool):
            add_issue(issues, split, line_no, "stop_label_should_stop_not_bool", qid=qid, t=t)
        if label_type not in STOP_TYPES:
            add_issue(issues, split, line_no, "stop_label_type_invalid", qid=qid, t=t, label_type=label_type)
        elif label_type == "terminal" and should_stop is not True:
            add_issue(issues, split, line_no, "terminal_stop_inconsistent", qid=qid, t=t)
        elif label_type != "terminal" and should_stop is True:
            add_issue(issues, split, line_no, "non_terminal_stop_inconsistent", qid=qid, t=t)


def validate_split(data_root: Path, split: str, max_issues: int) -> dict:
    sample_path = data_root / "samples" / f"{split}.jsonl"
    issues: list[dict] = []
    stats = {
        "records": 0,
        "qids": set(),
        "candidate_count": Counter(),
    }

    if not sample_path.exists():
        return {
            "split": split,
            "sample_path": str(sample_path),
            "records": 0,
            "qids": 0,
            "candidate_count": {},
            "issue_count": 1,
            "issues": [{"split": split, "issue": "missing_samples_file", "path": str(sample_path)}],
        }

    for line_no, record in read_jsonl(sample_path):
        check_record(record, split=split, line_no=line_no, issues=issues, stats=stats)

    shown_issues = issues[:max_issues]
    return {
        "split": split,
        "sample_path": str(sample_path),
        "records": stats["records"],
        "qids": len(stats["qids"]),
        "candidate_count": {str(k): v for k, v in sorted(stats["candidate_count"].items())},
        "issue_count": len(issues),
        "issues": shown_issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-issues", type=int, default=30)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    split_reports = [validate_split(data_root, split, args.max_issues) for split in splits]
    total_issues = sum(report["issue_count"] for report in split_reports)
    total_records = sum(report["records"] for report in split_reports)

    summary = {
        "data_root": str(data_root),
        "splits": split_reports,
        "total_records": total_records,
        "total_issues": total_issues,
        "status": "PASS" if total_issues == 0 else "FAILED",
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
