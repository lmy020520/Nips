import os
import json
from pathlib import Path
from typing import Dict, Iterable, List


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v4")

DEFICIT_KEYS = [
    "d_br_star",
    "d_dis_star",
    "d_sup_star",
    "d_der_star",
]


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


def zero_deficit() -> dict:
    return {
        "d_br_star": 0.0,
        "d_dis_star": 0.0,
        "d_sup_star": 0.0,
        "d_der_star": 0.0,
    }


def load_full_trajectories(path: Path) -> Dict[str, dict]:
    out = {}

    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "terminal_status", "terminal_t", "abort_reason", "steps"]
        for field in required:
            if field not in record:
                raise ValueError(f"full trajectory 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"full trajectory 中重复 qid: {qid}")

        terminal_status = str(record["terminal_status"]).strip()
        if terminal_status not in {"terminal", "abort"}:
            raise ValueError(f"非法 terminal_status: qid={qid}, terminal_status={terminal_status}")

        steps = record["steps"]
        if not isinstance(steps, list):
            raise ValueError(f"steps 必须是 list: qid={qid}")

        normalized_steps = []
        seen_t = set()

        for i, item in enumerate(steps):
            if not isinstance(item, dict):
                raise ValueError(f"steps[{i}] 必须是 dict: qid={qid}")
            if "t" not in item or "positive_unit_id" not in item:
                raise ValueError(f"steps[{i}] 缺少字段: qid={qid}")

            t = int(item["t"])
            if t in seen_t:
                raise ValueError(f"steps 中重复 t: qid={qid}, t={t}")
            seen_t.add(t)

            normalized_steps.append(
                {
                    "t": t,
                    "positive_unit_id": str(item["positive_unit_id"]),
                }
            )

        normalized_steps.sort(key=lambda x: x["t"])

        out[qid] = {
            "qid": qid,
            "terminal_status": terminal_status,
            "terminal_t": record["terminal_t"],
            "abort_reason": record["abort_reason"],
            "steps": normalized_steps,
        }

    return out


def load_deficit_labels(path: Path) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = {}

    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "t" not in record or "d_t_star" not in record:
            raise ValueError(f"deficit label 缺少字段: file={path}, row={row_idx}")

        qid = str(record["qid"])
        t = int(record["t"])
        d_t_star = record["d_t_star"]

        if not isinstance(d_t_star, dict):
            raise ValueError(f"d_t_star 必须是 dict: qid={qid}, t={t}")

        for key in DEFICIT_KEYS:
            if key not in d_t_star:
                raise ValueError(f"d_t_star 缺少字段 {key}: qid={qid}, t={t}")

        if qid not in out:
            out[qid] = {}

        if t in out[qid]:
            raise ValueError(f"deficit labels 中重复 (qid, t): qid={qid}, t={t}")

        out[qid][t] = {
            "d_br_star": float(d_t_star["d_br_star"]),
            "d_dis_star": float(d_t_star["d_dis_star"]),
            "d_sup_star": float(d_t_star["d_sup_star"]),
            "d_der_star": float(d_t_star["d_der_star"]),
        }

    return out


def compute_contribution(current_def: dict, next_def: dict) -> dict:
    return {
        "c_br_star": round(max(0.0, float(current_def["d_br_star"]) - float(next_def["d_br_star"])), 6),
        "c_dis_star": round(max(0.0, float(current_def["d_dis_star"]) - float(next_def["d_dis_star"])), 6),
        "c_sup_star": round(max(0.0, float(current_def["d_sup_star"]) - float(next_def["d_sup_star"])), 6),
        "c_der_star": round(max(0.0, float(current_def["d_der_star"]) - float(next_def["d_der_star"])), 6),
    }


def build_contribution_records_for_qid(qid: str, full_traj: dict, deficit_map: Dict[int, dict]) -> List[dict]:
    steps = full_traj["steps"]
    terminal_status = full_traj["terminal_status"]

    if not steps:
        return []

    output_records = []

    for idx, step in enumerate(steps):
        t = int(step["t"])
        positive_unit_id = str(step["positive_unit_id"])

        if t not in deficit_map:
            raise ValueError(f"缺少当前步的 deficit label: qid={qid}, t={t}")

        current_def = deficit_map[t]

        if (t + 1) in deficit_map:
            next_def = deficit_map[t + 1]
        else:
            # 最小版规则：
            # - 如果 trajectory 已 terminal，则最后一步后的 deficit 视为 0
            # - 如果 trajectory 是 abort，则最后一步后的 deficit 暂按“不再下降”处理
            if terminal_status == "terminal":
                next_def = zero_deficit()
            else:
                next_def = current_def

        c_t_star = compute_contribution(current_def, next_def)

        output_records.append(
            {
                "qid": qid,
                "t": t,
                "positive_unit_id": positive_unit_id,
                "c_t_star": c_t_star,
            }
        )

    return output_records


def convert_split(
    full_path: Path,
    deficit_path: Path,
    output_path: Path,
) -> int:
    full_map = load_full_trajectories(full_path)
    deficit_map = load_deficit_labels(deficit_path)

    def generator():
        for qid in sorted(full_map.keys()):
            if qid not in deficit_map:
                # 没有 prefix deficit 的题，直接跳过
                continue

            records = build_contribution_records_for_qid(
                qid=qid,
                full_traj=full_map[qid],
                deficit_map=deficit_map[qid],
            )
            for rec in records:
                yield rec

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    trajectories_dir = base_dir / "trajectories"
    labels_dir = base_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    out_name_map = {
        "train": "contribution_train.jsonl",
        "val": "contribution_val.jsonl",
        "test": "contribution_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        full_path = trajectories_dir / f"full_{split}.jsonl"
        deficit_path = labels_dir / f"deficit_{split}.jsonl"
        output_path = labels_dir / out_name_map[split]

        for path in [full_path, deficit_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            full_path=full_path,
            deficit_path=deficit_path,
            output_path=output_path,
        )

    print("contribution labels v2 构建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {labels_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()