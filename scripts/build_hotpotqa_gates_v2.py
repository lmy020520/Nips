import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v2"

RAW_COMPLETE_THRESHOLD = 0.70
TAU_SEM = 0.50
REDUNDANT_SIM_THRESHOLD = 0.90
COMPOSABLE_SIM_THRESHOLD = 0.85


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


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def load_targets_map(path: Path) -> Dict[str, dict]:
    targets = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "T_q_raw" not in record:
            raise ValueError(f"targets 缺少字段: file={path}, row={row_idx}")

        qid = str(record["qid"])
        if qid in targets:
            raise ValueError(f"targets 中重复 qid: file={path}, qid={qid}")

        t_q_raw = record["T_q_raw"]
        if not isinstance(t_q_raw, list) or len(t_q_raw) == 0:
            raise ValueError(f"T_q_raw 必须是非空 list: qid={qid}")

        target_map = {}
        required_roles = set()

        for i, item in enumerate(t_q_raw):
            for field in ["unit_id", "primary_role"]:
                if field not in item:
                    raise ValueError(f"T_q_raw[{i}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item["unit_id"])
            role = str(item["primary_role"]).strip()
            if role not in {"bridge", "distinguish", "support"}:
                raise ValueError(f"非法 primary_role: qid={qid}, unit_id={unit_id}, role={role}")

            target_map[unit_id] = {"primary_role": role}
            required_roles.add(role)

        targets[qid] = {
            "target_map": target_map,
            "required_roles": sorted(required_roles),
        }

    return targets


def load_raw_unit_map(path: Path) -> Dict[str, dict]:
    units = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["unit_id", "text", "parent_chunk_id", "doc_id"]
        for field in required:
            if field not in record:
                raise ValueError(f"raw_units 缺少字段: file={path}, row={row_idx}, field={field}")

        unit_id = str(record["unit_id"])
        if unit_id in units:
            raise ValueError(f"raw_units 中重复 unit_id: file={path}, unit_id={unit_id}")

        text = str(record["text"]).strip()
        if not text:
            raise ValueError(f"raw unit text 为空: unit_id={unit_id}")

        units[unit_id] = {
            "unit_id": unit_id,
            "text": text,
            "parent_chunk_id": str(record["parent_chunk_id"]),
            "doc_id": str(record["doc_id"]),
        }

    return units


def load_init_states(path: Path) -> Dict[str, dict]:
    states = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "A_t", "S_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"init_state 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in states:
            raise ValueError(f"init_state 中重复 qid: file={path}, qid={qid}")

        t = int(record["t"])
        if t != 0:
            raise ValueError(f"这里只接 init_state(t=0): qid={qid}, t={t}")

        a_t = record["A_t"]
        s_t = record["S_t"]
        if not isinstance(a_t, dict):
            raise ValueError(f"A_t 必须是 dict: qid={qid}")
        if not isinstance(s_t, dict):
            raise ValueError(f"S_t 必须是 dict: qid={qid}")

        states[qid] = {
            "qid": qid,
            "A_t": a_t,
            "S_t": s_t,
        }

    return states


def load_candidates(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "q_t", "R_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"candidates 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"candidates 中重复 qid: file={path}, qid={qid}")

        t = int(record["t"])
        if t != 0:
            raise ValueError(f"这里只接 t=0 candidates: qid={qid}, t={t}")

        r_t = record["R_t"]
        if not isinstance(r_t, list):
            raise ValueError(f"R_t 必须是 list: qid={qid}")

        normalized_r = []
        seen = set()
        for unit_id in r_t:
            unit_id = str(unit_id)
            if unit_id in seen:
                raise ValueError(f"R_t 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen.add(unit_id)
            normalized_r.append(unit_id)

        out[qid] = {
            "qid": qid,
            "t": t,
            "q_t": str(record["q_t"]),
            "R_t": normalized_r,
        }

    return out


def get_rolewise_progress(a_t: dict, required_roles: List[str]) -> Dict[str, float]:
    role_key_map = {
        "bridge": "k_br",
        "distinguish": "k_dis",
        "support": "k_sup",
    }

    covered = a_t.get("covered_target_ids", [])
    denom = max(len(covered), 1)  # fallback; replaced below per role count in caller if needed
    scores = {}

    for role in required_roles:
        key = role_key_map[role]
        val = float(a_t.get(key, 0.0))
        scores[role] = val

    return scores


def get_role_denominators(target_info: dict) -> Dict[str, float]:
    denoms = {"bridge": 0.0, "distinguish": 0.0, "support": 0.0}
    for item in target_info["target_map"].values():
        role = item["primary_role"]
        denoms[role] += 1.0
    return denoms


def normalized_role_scores(a_t: dict, target_info: dict) -> Dict[str, float]:
    raw_scores = get_rolewise_progress(a_t, target_info["required_roles"])
    denoms = get_role_denominators(target_info)
    out = {}
    for role in target_info["required_roles"]:
        denom = max(denoms[role], 1.0)
        out[role] = raw_scores[role] / denom
    return out


def get_recent_raw_texts(s_t: dict, raw_unit_map: Dict[str, dict], n: int = 2) -> List[str]:
    raw_refs = s_t.get("raw_refs", [])
    if not isinstance(raw_refs, list):
        return []

    recent = []
    for item in raw_refs[-n:]:
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit_id = str(item["unit_id"])
        unit = raw_unit_map.get(unit_id)
        if unit is not None:
            recent.append(unit["text"])
    return recent


def has_recent_note(s_t: dict, note_type: str, window: int = 2) -> bool:
    # 当前 v2 初始化状态中 derived_refs 只有引用，没有 type；t=0 时 derived_refs 为空。
    # 这里保留接口，当前极简实现统一返回 False。
    derived_refs = s_t.get("derived_refs", [])
    if not isinstance(derived_refs, list) or len(derived_refs) == 0:
        return False
    return False


def is_raw_pool_redundant(r_t: List[str], s_t: dict, raw_unit_map: Dict[str, dict], sim_threshold: float = REDUNDANT_SIM_THRESHOLD) -> bool:
    recent_raw_texts = get_recent_raw_texts(s_t, raw_unit_map, n=2)
    if not recent_raw_texts:
        return False

    top_raw = r_t[:3]
    redundant_count = 0
    for unit_id in top_raw:
        unit = raw_unit_map.get(unit_id)
        if unit is None:
            continue
        text = unit["text"]
        if any(jaccard_similarity(text, prev) >= sim_threshold for prev in recent_raw_texts):
            redundant_count += 1

    return redundant_count >= 2


def has_composable_raw(r_t: List[str], raw_unit_map: Dict[str, dict], sim_threshold: float = COMPOSABLE_SIM_THRESHOLD) -> bool:
    top_raw = r_t[:3]
    if len(top_raw) < 2:
        return False

    usable = []
    for unit_id in top_raw:
        unit = raw_unit_map.get(unit_id)
        if unit is None:
            continue

        keep = True
        for prev in usable:
            if jaccard_similarity(unit["text"], prev["text"]) >= sim_threshold:
                keep = False
                break

        if keep:
            usable.append(
                {
                    "unit_id": unit_id,
                    "text": unit["text"],
                    "parent_chunk_id": unit["parent_chunk_id"],
                }
            )

    distinct_parents = len({u["parent_chunk_id"] for u in usable})
    return len(usable) >= 2 and distinct_parents >= 2


def cheap_stop_gate(target_info: dict, a_t: dict, s_t: dict, r_t: List[str], raw_unit_map: Dict[str, dict]) -> Dict[str, object]:
    role_scores = normalized_role_scores(a_t, target_info)
    required_roles = target_info["required_roles"]

    raw_complete = all(role_scores[r] >= RAW_COMPLETE_THRESHOLD for r in required_roles)

    has_recent_verif = has_recent_note(s_t, note_type="verification_note", window=2)
    has_recent_bridge = has_recent_note(s_t, note_type="bridge_note", window=2)
    raw_redundant = is_raw_pool_redundant(r_t, s_t, raw_unit_map)

    no_derived_need = has_recent_verif or (raw_redundant and has_recent_bridge)
    stop_candidate = raw_complete and no_derived_need

    return {
        "raw_complete": raw_complete,
        "no_derived_need": no_derived_need,
        "stop_candidate": stop_candidate,
        "role_scores": role_scores,
    }


def need_derived_gate(target_info: dict, a_t: dict, s_t: dict, r_t: List[str], raw_unit_map: Dict[str, dict]) -> Dict[str, object]:
    role_scores = normalized_role_scores(a_t, target_info)
    required_roles = target_info["required_roles"]

    s_sem = sum(role_scores[r] for r in required_roles) / max(1, len(required_roles))
    composable_raw = has_composable_raw(r_t, raw_unit_map)
    has_recent_verification = has_recent_note(s_t, note_type="verification_note", window=2)

    derived_need = composable_raw and (not has_recent_verification)
    trigger_derived = (s_sem >= TAU_SEM) and derived_need

    return {
        "s_sem": s_sem,
        "composable_raw": composable_raw,
        "has_recent_verification": has_recent_verification,
        "derived_need": derived_need,
        "trigger_derived": trigger_derived,
    }


def build_gates_record(qid: str, target_info: dict, state: dict, cand: dict, raw_unit_map: Dict[str, dict]) -> dict:
    stop_info = cheap_stop_gate(
        target_info=target_info,
        a_t=state["A_t"],
        s_t=state["S_t"],
        r_t=cand["R_t"],
        raw_unit_map=raw_unit_map,
    )

    if stop_info["stop_candidate"]:
        s_sem = sum(stop_info["role_scores"].values()) / max(1, len(stop_info["role_scores"]))
        return {
            "qid": qid,
            "t": 0,
            "StopCandidate_t": True,
            "NeedDerived_t": False,
            "s_sem": round(float(s_sem), 6),
        }

    derived_info = need_derived_gate(
        target_info=target_info,
        a_t=state["A_t"],
        s_t=state["S_t"],
        r_t=cand["R_t"],
        raw_unit_map=raw_unit_map,
    )

    return {
        "qid": qid,
        "t": 0,
        "StopCandidate_t": False,
        "NeedDerived_t": bool(derived_info["trigger_derived"]),
        "s_sem": round(float(derived_info["s_sem"]), 6),
    }


def convert_split(
    init_state_path: Path,
    candidates_path: Path,
    targets_path: Path,
    raw_units_path: Path,
    output_path: Path,
) -> int:
    init_states = load_init_states(init_state_path)
    candidates = load_candidates(candidates_path)
    targets = load_targets_map(targets_path)
    raw_unit_map = load_raw_unit_map(raw_units_path)

    def generator():
        for qid in sorted(init_states.keys()):
            if qid not in candidates:
                raise ValueError(f"candidates 中找不到 qid: {qid}")
            if qid not in targets:
                raise ValueError(f"targets 中找不到 qid: {qid}")

            yield build_gates_record(
                qid=qid,
                target_info=targets[qid],
                state=init_states[qid],
                cand=candidates[qid],
                raw_unit_map=raw_unit_map,
            )

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    trajectories_dir = base_dir / "trajectories"
    targets_dir = base_dir / "targets"
    unit_registry_dir = base_dir / "unit_registry"

    init_name_map = {
        "train": "init_state_train.jsonl",
        "val": "init_state_val.jsonl",
        "test": "init_state_test.jsonl",
    }
    cand_name_map = {
        "train": "candidates_train.jsonl",
        "val": "candidates_val.jsonl",
        "test": "candidates_test.jsonl",
    }
    raw_units_name_map = {
        "train": "raw_units_train.jsonl",
        "val": "raw_units_val.jsonl",
        "test": "raw_units_test.jsonl",
    }
    out_name_map = {
        "train": "gates_train.jsonl",
        "val": "gates_val.jsonl",
        "test": "gates_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        init_state_path = trajectories_dir / init_name_map[split]
        candidates_path = trajectories_dir / cand_name_map[split]
        targets_path = targets_dir / f"{split}.jsonl"
        raw_units_path = unit_registry_dir / raw_units_name_map[split]
        output_path = trajectories_dir / out_name_map[split]

        for path in [init_state_path, candidates_path, targets_path, raw_units_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            init_state_path=init_state_path,
            candidates_path=candidates_path,
            targets_path=targets_path,
            raw_units_path=raw_units_path,
            output_path=output_path,
        )

    print("gates v2 构建完成：")
    print(f"  defaults: raw_complete_tau={RAW_COMPLETE_THRESHOLD}, tau_sem={TAU_SEM}")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {trajectories_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()