import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v2"


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


def parse_sent_id_from_unit_id(unit_id: str) -> int:
    parts = unit_id.split("::")
    if len(parts) < 3:
        raise ValueError(f"unit_id 格式错误，无法解析 sent_id: {unit_id}")
    return int(parts[-1])


def load_init_states(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "H_t", "A_t", "S_t", "K_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"init_state 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"init_state 中重复 qid: file={path}, qid={qid}")

        h_t = record["H_t"]
        if not isinstance(h_t, list):
            raise ValueError(f"H_t 必须是 list: qid={qid}")

        s_t = record["S_t"]
        a_t = record["A_t"]
        if not isinstance(s_t, dict):
            raise ValueError(f"S_t 必须是 dict: qid={qid}")
        if not isinstance(a_t, dict):
            raise ValueError(f"A_t 必须是 dict: qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "H_t": h_t,
            "A_t": a_t,
            "S_t": s_t,
            "K_t": str(record["K_t"]),
        }
    return out


def load_teacher_select(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "C_t", "positive_unit_id"]
        for field in required:
            if field not in record:
                raise ValueError(f"teacher_select 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"teacher_select 中重复 qid: file={path}, qid={qid}")

        c_t = record["C_t"]
        if not isinstance(c_t, list) or len(c_t) == 0:
            raise ValueError(f"C_t 必须是非空 list: qid={qid}")

        positive_unit_id = str(record["positive_unit_id"])
        if positive_unit_id not in [str(x) for x in c_t]:
            raise ValueError(f"positive_unit_id 不在 C_t 中: qid={qid}, unit_id={positive_unit_id}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "C_t": [str(x) for x in c_t],
            "positive_unit_id": positive_unit_id,
        }
    return out


def load_targets(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "T_q_raw"]
        for field in required:
            if field not in record:
                raise ValueError(f"targets 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"targets 中重复 qid: file={path}, qid={qid}")

        t_q_raw = record["T_q_raw"]
        if not isinstance(t_q_raw, list) or len(t_q_raw) == 0:
            raise ValueError(f"T_q_raw 必须是非空 list: qid={qid}")

        target_map = {}
        for i, item in enumerate(t_q_raw):
            required_item = ["text", "primary_role", "weight"]
            for field in required_item:
                if field not in item:
                    raise ValueError(f"T_q_raw[{i}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item.get("chunk_id", item.get("unit_id", ""))).strip()
            if not unit_id:
                raise ValueError(f"T_q_raw[{i}] 缺少 chunk_id/unit_id: qid={qid}")
            if unit_id in target_map:
                raise ValueError(f"T_q_raw 中重复 unit_id: qid={qid}, unit_id={unit_id}")

            target_map[unit_id] = {
                "unit_id": unit_id,
                "chunk_id": unit_id,
                "text": str(item["text"]).strip(),
                "primary_role": str(item["primary_role"]).strip(),
                "weight": float(item["weight"]),
            }

        out[qid] = {
            "qid": qid,
            "target_map": target_map,
        }
    return out


def load_raw_unit_map(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = [
            "unit_id",
            "text",
            "doc_id",
            "parent_chunk_id",
            "provenance",
            "candidate_granularity",
        ]
        for field in required:
            if field not in record:
                raise ValueError(f"raw_units 缺少字段: file={path}, row={row_idx}, field={field}")

        unit_id = str(record["unit_id"])
        if unit_id in out:
            raise ValueError(f"raw_units 中重复 unit_id: file={path}, unit_id={unit_id}")

        out[unit_id] = {
            "unit_id": unit_id,
            "text": str(record["text"]).strip(),
            "doc_id": str(record["doc_id"]),
            "parent_chunk_id": str(record["parent_chunk_id"]),
            "span_start": record.get("span_start"),
            "span_end": record.get("span_end"),
            "provenance": str(record["provenance"]),
            "candidate_granularity": str(record["candidate_granularity"]),
        }
    return out


def load_derived_harvest(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "proposal_run", "G_t_harvest"]
        for field in required:
            if field not in record:
                raise ValueError(f"derived_harvest 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"derived_harvest 中重复 qid: file={path}, qid={qid}")

        g_h = record["G_t_harvest"]
        if not isinstance(g_h, list):
            raise ValueError(f"G_t_harvest 必须是 list: qid={qid}")

        derived_map = {}
        for idx, item in enumerate(g_h):
            required_item = ["unit_id", "text", "provenance", "candidate_granularity", "type", "source_unit_ids"]
            for field in required_item:
                if field not in item:
                    raise ValueError(f"G_t_harvest[{idx}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item["unit_id"])
            if unit_id in derived_map:
                raise ValueError(f"G_t_harvest 中重复 unit_id: qid={qid}, unit_id={unit_id}")

            derived_map[unit_id] = {
                "unit_id": unit_id,
                "text": str(item["text"]).strip(),
                "provenance": str(item["provenance"]),
                "candidate_granularity": str(item["candidate_granularity"]),
                "type": str(item["type"]),
                "source_unit_ids": [str(x) for x in item["source_unit_ids"]],
            }

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "proposal_run": bool(record["proposal_run"]),
            "derived_map": derived_map,
        }
    return out


def load_full_trajectories(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "steps"]
        for field in required:
            if field not in record:
                raise ValueError(f"full trajectories 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"full trajectories 中重复 qid: file={path}, qid={qid}")

        steps = record["steps"]
        if not isinstance(steps, list):
            raise ValueError(f"steps 必须是 list: qid={qid}")

        normalized_steps = []
        seen_t = set()
        for step_idx, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(f"steps[{step_idx}] 必须是 dict: qid={qid}")
            if "t" not in step or "positive_unit_id" not in step:
                raise ValueError(f"steps[{step_idx}] 缺少 t 或 positive_unit_id: qid={qid}")

            t = int(step["t"])
            if t in seen_t:
                raise ValueError(f"steps 中重复 t: qid={qid}, t={t}")
            seen_t.add(t)

            normalized_steps.append(
                {
                    "t": t,
                    "positive_unit_id": str(step["positive_unit_id"]),
                }
            )

        out[qid] = {
            "qid": qid,
            "terminal_status": str(record.get("terminal_status", "")),
            "terminal_t": record.get("terminal_t"),
            "steps": normalized_steps,
        }
    return out


def build_unit_registry(raw_unit_map: Dict[str, dict], derived_map: Dict[str, dict]) -> Dict[str, dict]:
    registry = dict(raw_unit_map)
    for unit_id, item in derived_map.items():
        if unit_id in registry:
            raise ValueError(f"UnitRegistry 中 unit_id 冲突: {unit_id}")
        registry[unit_id] = item
    return registry


def normalize_refs_minimal(refs: list) -> List[dict]:
    out = []
    seen = set()
    if not isinstance(refs, list):
        return out

    for item in refs:
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit_id = str(item["unit_id"])
        if unit_id in seen:
            continue
        seen.add(unit_id)
        out.append(
            {
                "unit_id": unit_id,
                "added_step": int(item.get("added_step", 0)),
            }
        )
    return out


def simulate_update(h_t: List[dict], s_t: dict, u_next: dict) -> tuple[list, dict]:
    h_next = copy.deepcopy(h_t)
    s_next = copy.deepcopy(s_t)

    if "raw_refs" not in s_next or not isinstance(s_next["raw_refs"], list):
        s_next["raw_refs"] = []
    if "derived_refs" not in s_next or not isinstance(s_next["derived_refs"], list):
        s_next["derived_refs"] = []

    s_next["raw_refs"] = normalize_refs_minimal(s_next["raw_refs"])
    s_next["derived_refs"] = normalize_refs_minimal(s_next["derived_refs"])

    next_step_id = len(h_next)
    unit_id = u_next["unit_id"]

    existing_h_ids = {str(x["unit_id"]) for x in h_next if isinstance(x, dict) and "unit_id" in x}
    if unit_id not in existing_h_ids:
        h_next.append(
            {
                "step_id": next_step_id,
                "unit_id": unit_id,
            }
        )

    ref_key = "raw_refs" if u_next["provenance"] == "raw" else "derived_refs"
    existing_ref_ids = {str(x["unit_id"]) for x in s_next[ref_key] if isinstance(x, dict) and "unit_id" in x}
    if unit_id not in existing_ref_ids:
        s_next[ref_key].append(
            {
                "unit_id": unit_id,
                "added_step": next_step_id,
            }
        )

    s_next["last_added_unit_id"] = unit_id
    s_next["last_updated_step"] = next_step_id
    return h_next, s_next


def get_role_key(role: str) -> str:
    if role == "bridge":
        return "k_br"
    if role == "distinguish":
        return "k_dis"
    if role == "support":
        return "k_sup"
    raise ValueError(f"非法 role: {role}")


def simulate_ledger(a_t: dict, u_next: dict, target_info: dict) -> dict:
    a_next = copy.deepcopy(a_t)

    covered = [str(x) for x in a_next.get("covered_target_ids", [])]
    covered_set = set(covered)
    target_map = target_info["target_map"]

    if u_next["provenance"] != "raw":
        a_next["covered_target_ids"] = covered
        return a_next

    target_unit_id = str(u_next.get("parent_chunk_id", "")).strip()
    if not target_unit_id:
        target_unit_id = str(u_next["unit_id"]).rsplit("::", 1)[0]

    if target_unit_id not in target_map:
        a_next["covered_target_ids"] = covered
        return a_next

    if target_unit_id in covered_set:
        a_next["covered_target_ids"] = covered
        return a_next

    covered.append(target_unit_id)
    covered_set.add(target_unit_id)

    target = target_map[target_unit_id]
    role = target["primary_role"]
    weight = float(target["weight"])

    role_key = get_role_key(role)
    current = float(a_next.get(role_key, 0.0))
    a_next[role_key] = current + weight
    a_next["covered_target_ids"] = covered
    return a_next


def render_k_t(h_t: List[dict], unit_registry: Dict[str, dict]) -> str:
    lines = []

    for item in h_t:
        if not isinstance(item, dict) or "unit_id" not in item:
            continue

        unit_id = str(item["unit_id"])
        if unit_id not in unit_registry:
            raise ValueError(f"H_t 中的 unit_id 不在 UnitRegistry: unit_id={unit_id}")

        unit = unit_registry[unit_id]
        if unit["provenance"] == "raw":
            sent_id = parse_sent_id_from_unit_id(unit_id)
            lines.append(f"{unit['doc_id']} [{sent_id}] {unit['text']}")
        elif unit["provenance"] == "derived":
            lines.append(f"[{unit['type']}] {unit['text']}")
        else:
            raise ValueError(f"非法 provenance: unit_id={unit_id}, provenance={unit['provenance']}")

    return "\n".join(lines)


def ensure_positive_unit_in_registry(positive_unit_id: str, unit_registry: Dict[str, dict]) -> Dict[str, dict]:
    if positive_unit_id in unit_registry:
        return unit_registry

    if "::derived::" not in positive_unit_id:
        raise ValueError(f"positive_unit_id 不在 UnitRegistry: unit_id={positive_unit_id}")

    patched_registry = dict(unit_registry)
    patched_registry[positive_unit_id] = {
        "unit_id": positive_unit_id,
        "text": f"[missing derived payload] {positive_unit_id}",
        "provenance": "derived",
        "candidate_granularity": "note",
        "type": "derived_note",
        "source_unit_ids": [],
    }
    return patched_registry


def clone_state_record(qid: str, t: int, state: dict) -> dict:
    return {
        "qid": qid,
        "t": t,
        "H_t": copy.deepcopy(state["H_t"]),
        "A_t": copy.deepcopy(state["A_t"]),
        "S_t": copy.deepcopy(state["S_t"]),
        "K_t": str(state["K_t"]),
    }


def replay_one_step(state: dict, positive_unit_id: str, target_info: dict, unit_registry: Dict[str, dict]) -> dict:
    unit_registry = ensure_positive_unit_in_registry(positive_unit_id, unit_registry)
    u_next = unit_registry[positive_unit_id]

    h_next, s_next = simulate_update(
        h_t=state["H_t"],
        s_t=state["S_t"],
        u_next=u_next,
    )
    a_next = simulate_ledger(
        a_t=state["A_t"],
        u_next=u_next,
        target_info=target_info,
    )
    k_next = render_k_t(
        h_t=h_next,
        unit_registry=unit_registry,
    )

    return {
        "H_t": h_next,
        "A_t": a_next,
        "S_t": s_next,
        "K_t": k_next,
    }


def build_state_records_for_qid(qid: str, init_state: dict, full_traj: dict, target_info: dict, unit_registry: Dict[str, dict]) -> List[dict]:
    current_unit_registry = dict(unit_registry)
    current_state = {
        "H_t": copy.deepcopy(init_state["H_t"]),
        "A_t": copy.deepcopy(init_state["A_t"]),
        "S_t": copy.deepcopy(init_state["S_t"]),
        "K_t": str(init_state["K_t"]),
    }

    output_records = []
    for step_idx, step in enumerate(full_traj["steps"]):
        t = int(step["t"])
        if t != step_idx:
            raise ValueError(f"steps t 不连续: qid={qid}, expected={step_idx}, got={t}")

        output_records.append(clone_state_record(qid=qid, t=t, state=current_state))
        current_unit_registry = ensure_positive_unit_in_registry(step["positive_unit_id"], current_unit_registry)
        current_state = replay_one_step(
            state=current_state,
            positive_unit_id=step["positive_unit_id"],
            target_info=target_info,
            unit_registry=current_unit_registry,
        )

    terminal_status = str(full_traj.get("terminal_status", "")).strip().lower()
    terminal_t = full_traj.get("terminal_t")
    if terminal_status == "terminal" and isinstance(terminal_t, int):
        if terminal_t != len(full_traj["steps"]):
            raise ValueError(
                f"terminal_t 与 steps 长度不一致: qid={qid}, terminal_t={terminal_t}, steps={len(full_traj['steps'])}"
            )
        output_records.append(clone_state_record(qid=qid, t=terminal_t, state=current_state))

    return output_records


def convert_split(
    init_state_path: Path,
    full_path: Path,
    targets_path: Path,
    raw_units_path: Path,
    derived_harvest_path: Path,
    output_path: Path,
) -> int:
    init_states = load_init_states(init_state_path)
    full_map = load_full_trajectories(full_path)
    targets = load_targets(targets_path)
    raw_unit_map = load_raw_unit_map(raw_units_path)
    derived_harvest = load_derived_harvest(derived_harvest_path)

    def generator():
        for qid in sorted(full_map.keys()):
            if qid not in init_states:
                raise ValueError(f"init_state 中找不到 qid: {qid}")
            if qid not in targets:
                raise ValueError(f"targets 中找不到 qid: {qid}")
            if qid not in derived_harvest:
                raise ValueError(f"derived_harvest 中找不到 qid: {qid}")

            unit_registry = build_unit_registry(
                raw_unit_map=raw_unit_map,
                derived_map=derived_harvest[qid]["derived_map"],
            )

            records = build_state_records_for_qid(
                qid=qid,
                init_state=init_states[qid],
                full_traj=full_map[qid],
                target_info=targets[qid],
                unit_registry=unit_registry,
            )
            for rec in records:
                yield rec

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    trajectories_dir = base_dir / "trajectories"
    targets_dir = base_dir / "targets"
    unit_registry_dir = base_dir / "unit_registry"

    raw_units_name_map = {
        "train": "raw_units_train.jsonl",
        "val": "raw_units_val.jsonl",
        "test": "raw_units_test.jsonl",
    }
    out_name_map = {
        "train": "states_train.jsonl",
        "val": "states_val.jsonl",
        "test": "states_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        init_state_path = trajectories_dir / f"init_state_{split}.jsonl"
        full_path = trajectories_dir / f"full_{split}.jsonl"
        targets_path = targets_dir / f"{split}.jsonl"
        raw_units_path = unit_registry_dir / raw_units_name_map[split]
        derived_harvest_path = trajectories_dir / f"derived_harvest_{split}.jsonl"
        output_path = trajectories_dir / out_name_map[split]

        for path in [init_state_path, full_path, targets_path, raw_units_path, derived_harvest_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            init_state_path=init_state_path,
            full_path=full_path,
            targets_path=targets_path,
            raw_units_path=raw_units_path,
            derived_harvest_path=derived_harvest_path,
            output_path=output_path,
        )

    print("states v2 构建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {trajectories_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()
