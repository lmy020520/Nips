#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v4")


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSONL 解析失败: file={path}, line={line_idx}, error={e}"
                ) from e


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return int(x)
        except ValueError:
            return None
    return None


def to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return None


def normalize_status(raw_status: Optional[str], terminal_status: Optional[str]) -> str:
    if raw_status is not None:
        s = str(raw_status).strip()
        if s:
            return s

    if terminal_status is None:
        return "failed_stalled"

    s = str(terminal_status).strip().lower()
    if s in {"terminal", "success", "succeeded"}:
        return "success"
    if s in {"failed_but_progressive", "failed-progressive"}:
        return "failed_but_progressive"
    if s in {"failed_stalled", "failed-stalled"}:
        return "failed_stalled"
    return s


def deep_get(d: Any, *keys: str) -> Any:
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_full_trajectories(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        qid = str(record.get("qid", "")).strip()
        if not qid:
            raise ValueError(f"full trajectory 缺少 qid: file={path}, row={row_idx}")
        if qid in out:
            raise ValueError(f"full trajectory 中重复 qid: {qid}")

        status = normalize_status(
            record.get("status"),
            record.get("terminal_status"),
        )

        terminal_step = to_int(
            first_not_none(
                record.get("terminal_step"),
                record.get("terminal_t"),
            )
        )

        abort_reason = first_not_none(
            record.get("abort_reason"),
            deep_get(record, "trajectory_status", "abort_reason"),
        )

        steps = record.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError(f"steps 必须是 list: qid={qid}")

        normalized_steps = []
        seen_t = set()
        for i, item in enumerate(steps):
            if not isinstance(item, dict):
                raise ValueError(f"steps[{i}] 必须是 dict: qid={qid}")

            t = to_int(item.get("t"))
            if t is None:
                raise ValueError(f"steps[{i}] 缺少合法 t: qid={qid}")

            if t in seen_t:
                raise ValueError(f"steps 中重复 t: qid={qid}, t={t}")
            seen_t.add(t)

            normalized_steps.append(item)

        normalized_steps.sort(key=lambda x: int(x["t"]))

        out[qid] = {
            "qid": qid,
            "status": status,
            "terminal_step": terminal_step,
            "abort_reason": abort_reason,
            "last_progress_step": to_int(
                first_not_none(
                    record.get("last_progress_step"),
                    record.get("t_last_progress"),
                    deep_get(record, "meta", "last_progress_step"),
                )
            ),
            "steps": normalized_steps,
        }
    return out


def extract_keep_prefix(step: dict) -> Optional[bool]:
    return first_not_none(
        to_bool(step.get("keep_prefix")),
        to_bool(deep_get(step, "meta", "keep_prefix")),
    )


def extract_false_stop(step: dict) -> Optional[bool]:
    candidates = [
        step.get("false_stop"),
        step.get("FalseStop_t"),
        step.get("is_false_stop"),
        deep_get(step, "probe", "false_stop"),
        deep_get(step, "probe", "FalseStop_t"),
        deep_get(step, "probe", "should_stop"),
        deep_get(step, "stop_probe", "false_stop"),
        deep_get(step, "stop_probe", "FalseStop_t"),
        deep_get(step, "stop_probe", "should_stop"),
        deep_get(step, "stop_info", "false_stop"),
        deep_get(step, "stop_info", "FalseStop_t"),
        deep_get(step, "labels", "stop_type"),
        deep_get(step, "labels", "label_type"),
        deep_get(step, "meta", "stop_type"),
        deep_get(step, "meta", "label_type"),
    ]

    for v in candidates:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"false-stop", "false_stop", "false_stop_negative"}:
                return True

    direct = [
        step.get("false_stop"),
        step.get("FalseStop_t"),
        step.get("is_false_stop"),
        deep_get(step, "probe", "false_stop"),
        deep_get(step, "probe", "FalseStop_t"),
        deep_get(step, "stop_probe", "false_stop"),
        deep_get(step, "stop_probe", "FalseStop_t"),
        deep_get(step, "stop_info", "false_stop"),
        deep_get(step, "stop_info", "FalseStop_t"),
    ]
    for v in direct:
        b = to_bool(v)
        if b is not None:
            return b

    # 保守回退：
    # 若显式记录了 probe 且 should_stop=False，则视为 false-stop。
    for probe_key in ["probe", "stop_probe"]:
        probe = step.get(probe_key)
        if isinstance(probe, dict):
            should_stop = to_bool(first_not_none(
                probe.get("should_stop"),
                probe.get("TeacherStop_t"),
            ))
            if should_stop is False:
                return True

    return None


def should_keep_prefix(
    traj_status: str,
    step_t: int,
    step: Optional[dict],
    last_progress_step: Optional[int],
) -> bool:
    if step is not None:
        keep_prefix = extract_keep_prefix(step)
        if keep_prefix is not None:
            return keep_prefix

    # 对 failed_stalled，若有最后有效进展步，则只保留到该步为止
    if traj_status == "failed_stalled" and last_progress_step is not None:
        return step_t <= last_progress_step

    return True


def build_prefix_entries(full_traj: dict) -> List[Tuple[int, Optional[dict]]]:
    steps = full_traj["steps"]
    status = full_traj["status"]
    terminal_step = full_traj["terminal_step"]
    last_progress_step = full_traj["last_progress_step"]

    entry_map: Dict[int, Optional[dict]] = {}

    for step in steps:
        t = int(step["t"])
        if should_keep_prefix(status, t, step, last_progress_step):
            entry_map[t] = step

    # 兼容两种 terminal_step 记录方式：
    # 1) terminal_step 就是某个已有 prefix t
    # 2) terminal_step 是最终成功停止时的“额外 terminal prefix”，可能不在 steps 里
    if status == "success" and terminal_step is not None:
        if should_keep_prefix(status, terminal_step, None, last_progress_step):
            entry_map.setdefault(terminal_step, None)

    return sorted(entry_map.items(), key=lambda x: x[0])


def classify_stop_type(
    traj_status: str,
    terminal_step: Optional[int],
    current_t: int,
    step: Optional[dict],
) -> Tuple[int, str]:
    # 1) terminal
    if traj_status == "success" and terminal_step is not None and current_t == terminal_step:
        return 1, "terminal"

    # 2) false-stop
    if step is not None:
        false_stop = extract_false_stop(step)
        if false_stop is True:
            return 0, "false-stop"

    # 3) near-terminal
    if traj_status == "success" and terminal_step is not None:
        if terminal_step - current_t in {1, 2}:
            return 0, "near-terminal"

    # 4) continue
    return 0, "continue"


def build_stop_records_for_qid(qid: str, full_traj: dict) -> List[dict]:
    status = full_traj["status"]
    terminal_step = full_traj["terminal_step"]
    entries = build_prefix_entries(full_traj)

    output_records = []
    seen_t = set()

    for t, step in entries:
        if t in seen_t:
            continue
        seen_t.add(t)

        stop_label, stop_type = classify_stop_type(
            traj_status=status,
            terminal_step=terminal_step,
            current_t=t,
            step=step,
        )

        output_records.append(
            {
                "qid": qid,
                "t": t,
                "stop_label": stop_label,
                "stop_type": stop_type,
            }
        )

    return output_records


def convert_split(full_path: Path, output_path: Path) -> int:
    full_map = load_full_trajectories(full_path)

    def generator():
        for qid in sorted(full_map.keys()):
            for rec in build_stop_records_for_qid(qid, full_map[qid]):
                yield rec

    return write_jsonl(generator(), output_path)


def main():
    parser = argparse.ArgumentParser(description="构建 HotpotQA distractor v2 的 stop labels")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE,
        help=f"数据根目录，默认 {DEFAULT_BASE}",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=SPLITS,
        help="要处理的 split，默认 train val test",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = (project_root / base_dir).resolve()

    trajectories_dir = base_dir / "trajectories"
    labels_dir = base_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    out_name_map = {
        "train": "stop_train.jsonl",
        "val": "stop_val.jsonl",
        "test": "stop_test.jsonl",
    }

    stats = {}
    for split in args.splits:
        if split not in out_name_map:
            raise ValueError(f"不支持的 split: {split}")

        full_path = trajectories_dir / f"full_{split}.jsonl"
        output_path = labels_dir / out_name_map[split]

        if not full_path.exists():
            raise FileNotFoundError(f"找不到必需文件: {full_path}")

        stats[split] = convert_split(
            full_path=full_path,
            output_path=output_path,
        )

    print("stop labels v4 构建完成：")
    for split in args.splits:
        print(f"  {split}: {stats[split]} -> {labels_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()
