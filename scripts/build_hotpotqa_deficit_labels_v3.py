import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v3"
ALPHA = 1.0


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


def canonical_role(role: str) -> str:
    role = str(role).strip().lower()
    if role == "bridge":
        return "bridge"
    if role in {"distinguish", "disambiguation"}:
        return "distinguish"
    if role == "support":
        return "support"
    raise ValueError(f"非法 role: {role}")


def get_role_key(role: str) -> str:
    role = canonical_role(role)
    if role == "bridge":
        return "k_br"
    if role == "distinguish":
        return "k_dis"
    if role == "support":
        return "k_sup"
    raise ValueError(f"非法 role: {role}")


def is_derived_unit_id(unit_id: str) -> bool:
    return "::derived::" in str(unit_id)


def parse_chunk_id_from_unit_id(unit_id: str) -> str:
    parts = str(unit_id).rsplit("::", 1)
    if len(parts) != 2:
        raise ValueError(f"unit_id 格式错误，无法解析 chunk_id: {unit_id}")
    return parts[0]


def load_targets(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "T_q_raw" not in record:
            raise ValueError(f"targets 缺少字段: file={path}, row={row_idx}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"targets 中重复 qid: {qid}")

        t_q_raw = record["T_q_raw"]
        if not isinstance(t_q_raw, list):
            raise ValueError(f"T_q_raw 必须是 list: qid={qid}")

        target_map = {}
        role_totals = {"bridge": 0.0, "distinguish": 0.0, "support": 0.0}

        for i, item in enumerate(t_q_raw):
            for field in ["primary_role"]:
                if field not in item:
                    raise ValueError(f"T_q_raw[{i}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item.get("chunk_id", item.get("unit_id", ""))).strip()
            if not unit_id:
                raise ValueError(f"T_q_raw[{i}] 缺少 chunk_id/unit_id: qid={qid}")
            if unit_id in target_map:
                raise ValueError(f"T_q_raw 中重复 unit_id: qid={qid}, unit_id={unit_id}")

            role = canonical_role(item["primary_role"])
            weight = float(item.get("weight", 1.0))

            target_map[unit_id] = {
                "unit_id": unit_id,
                "chunk_id": unit_id,
                "primary_role": role,
                "weight": weight,
            }
            role_totals[role] += weight

        out[qid] = {
            "qid": qid,
            "target_map": target_map,
            "N_q": role_totals,
        }

    return out


def load_init_states(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "A_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"init_state 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"init_state 中重复 qid: {qid}")

        a_t = record["A_t"]
        if not isinstance(a_t, dict):
            raise ValueError(f"A_t 必须是 dict: qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "A_t": {
                "covered_target_ids": [str(x) for x in a_t.get("covered_target_ids", [])],
                "k_br": float(a_t.get("k_br", 0.0)),
                "k_dis": float(a_t.get("k_dis", 0.0)),
                "k_sup": float(a_t.get("k_sup", 0.0)),
            },
        }

    return out


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
            "terminal_status": str(record["terminal_status"]),
            "terminal_t": record["terminal_t"],
            "abort_reason": record["abort_reason"],
            "steps": normalized_steps,
        }

    return out


def clone_a_t(a_t: dict) -> dict:
    return {
        "covered_target_ids": list(a_t.get("covered_target_ids", [])),
        "k_br": float(a_t.get("k_br", 0.0)),
        "k_dis": float(a_t.get("k_dis", 0.0)),
        "k_sup": float(a_t.get("k_sup", 0.0)),
    }


def replay_one_raw_step(a_t: dict, positive_unit_id: str, target_info: dict) -> dict:
    a_next = clone_a_t(a_t)

    if is_derived_unit_id(positive_unit_id):
        return a_next

    target_map = target_info["target_map"]
    target_unit_id = parse_chunk_id_from_unit_id(positive_unit_id)
    if target_unit_id not in target_map:
        return a_next

    covered = set(a_next["covered_target_ids"])
    if target_unit_id in covered:
        return a_next

    covered.add(target_unit_id)
    a_next["covered_target_ids"] = list(covered)

    target = target_map[target_unit_id]
    role = canonical_role(target["primary_role"])
    weight = float(target.get("weight", 1.0))
    role_key = get_role_key(role)
    a_next[role_key] = float(a_next.get(role_key, 0.0)) + weight

    return a_next


def compute_raw_deficit(a_t: dict, target_info: dict, alpha: float) -> dict:
    out = {}
    for role, short_name in [
        ("bridge", "d_br_star"),
        ("distinguish", "d_dis_star"),
        ("support", "d_sup_star"),
    ]:
        n_q_r = float(target_info["N_q"].get(role, 0.0))
        if n_q_r <= 0.0:
            out[short_name] = 0.0
            continue

        k_t_r = float(a_t.get(get_role_key(role), 0.0))
        s_t_r = (k_t_r + alpha) / (n_q_r + 2.0 * alpha)
        d_t_r = 1.0 - s_t_r
        out[short_name] = round(float(d_t_r), 6)

    return out


def compute_total_derived_steps(steps: List[dict]) -> int:
    return sum(1 for step in steps if is_derived_unit_id(step["positive_unit_id"]))


def compute_remaining_derived_ratio(steps: List[dict], t: int) -> float:
    total_derived = compute_total_derived_steps(steps)
    if total_derived == 0:
        return 0.0

    remaining = 0
    for step in steps:
        if step["t"] < t:
            continue
        if is_derived_unit_id(step["positive_unit_id"]):
            remaining += 1

    return round(float(remaining / total_derived), 6)


def build_deficit_records_for_qid(qid: str, init_state: dict, full_traj: dict, target_info: dict, alpha: float) -> List[dict]:
    steps = full_traj["steps"]

    # prefix t 对应“执行 steps[t] 之前的状态”
    # 因此先从 init_state 的 A_0 开始，逐步 replay
    current_a_t = clone_a_t(init_state["A_t"])

    output_records = []
    for idx, step in enumerate(steps):
        t = int(step["t"])

        raw_def = compute_raw_deficit(current_a_t, target_info, alpha=alpha)
        d_der_star = compute_remaining_derived_ratio(steps, t)

        output_records.append(
            {
                "qid": qid,
                "t": t,
                "d_t_star": {
                    "d_br_star": raw_def["d_br_star"],
                    "d_dis_star": raw_def["d_dis_star"],
                    "d_sup_star": raw_def["d_sup_star"],
                    "d_der_star": d_der_star,
                },
            }
        )

        # replay 当前 teacher 选中的这一步，供下一个 prefix 使用
        current_a_t = replay_one_raw_step(
            a_t=current_a_t,
            positive_unit_id=step["positive_unit_id"],
            target_info=target_info,
        )

    return output_records


def convert_split(
    full_path: Path,
    init_state_path: Path,
    targets_path: Path,
    output_path: Path,
    alpha: float,
) -> int:
    full_map = load_full_trajectories(full_path)
    init_state_map = load_init_states(init_state_path)
    target_map = load_targets(targets_path)

    def generator():
        for qid in sorted(full_map.keys()):
            if qid not in init_state_map:
                raise ValueError(f"init_state 中找不到 qid: {qid}")
            if qid not in target_map:
                raise ValueError(f"targets 中找不到 qid: {qid}")

            records = build_deficit_records_for_qid(
                qid=qid,
                init_state=init_state_map[qid],
                full_traj=full_map[qid],
                target_info=target_map[qid],
                alpha=alpha,
            )
            for rec in records:
                yield rec

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    trajectories_dir = base_dir / "trajectories"
    targets_dir = base_dir / "targets"
    labels_dir = base_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    out_name_map = {
        "train": "deficit_train.jsonl",
        "val": "deficit_val.jsonl",
        "test": "deficit_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        full_path = trajectories_dir / f"full_{split}.jsonl"
        init_state_path = trajectories_dir / f"init_state_{split}.jsonl"
        targets_path = targets_dir / f"{split}.jsonl"
        output_path = labels_dir / out_name_map[split]

        for path in [full_path, init_state_path, targets_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            full_path=full_path,
            init_state_path=init_state_path,
            targets_path=targets_path,
            output_path=output_path,
            alpha=ALPHA,
        )

    print("deficit labels v2 构建完成：")
    print(f"  alpha={ALPHA}")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {labels_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()
