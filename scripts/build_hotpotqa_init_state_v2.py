import os
import json
from pathlib import Path
from typing import Dict, Iterable, List


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v2")


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


def parse_chunk_id_from_unit_id(unit_id: str) -> str:
    parts = unit_id.rsplit("::", 1)
    if len(parts) != 2:
        raise ValueError(f"unit_id 格式错误，无法解析 chunk_id: {unit_id}")
    return parts[0]


def load_seed_map(path: Path) -> Dict[str, dict]:
    seed_map = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "P_0" not in record:
            raise ValueError(f"seed 缺少字段: file={path}, row={row_idx}")

        qid = str(record["qid"])
        if qid in seed_map:
            raise ValueError(f"seed 中重复 qid: file={path}, qid={qid}")

        p0 = record["P_0"]
        if not isinstance(p0, list) or len(p0) == 0:
            raise ValueError(f"P_0 必须是非空 list: qid={qid}")

        items = []
        seen_unit_ids = set()
        seen_ranks = set()

        for i, item in enumerate(p0):
            for field in ["unit_id", "rank", "source"]:
                if field not in item:
                    raise ValueError(f"P_0[{i}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item["unit_id"])
            rank = int(item["rank"])
            source = str(item["source"])

            if unit_id in seen_unit_ids:
                raise ValueError(f"P_0 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            if rank in seen_ranks:
                raise ValueError(f"P_0 中重复 rank: qid={qid}, rank={rank}")

            seen_unit_ids.add(unit_id)
            seen_ranks.add(rank)

            items.append(
                {
                    "unit_id": unit_id,
                    "rank": rank,
                    "source": source,
                }
            )

        items.sort(key=lambda x: x["rank"])
        expected_ranks = list(range(1, len(items) + 1))
        actual_ranks = [x["rank"] for x in items]
        if actual_ranks != expected_ranks:
            raise ValueError(
                f"P_0 rank 不连续: qid={qid}, actual={actual_ranks}, expected={expected_ranks}"
            )

        seed_map[qid] = {
            "qid": qid,
            "P_0": items,
        }

    return seed_map


def load_targets_map(path: Path) -> Dict[str, dict]:
    targets_map = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "T_q_raw" not in record:
            raise ValueError(f"targets 缺少字段: file={path}, row={row_idx}")

        qid = str(record["qid"])
        if qid in targets_map:
            raise ValueError(f"targets 中重复 qid: file={path}, qid={qid}")

        t_q_raw = record["T_q_raw"]
        if not isinstance(t_q_raw, list):
            raise ValueError(f"T_q_raw 必须是 list: qid={qid}")

        target_map = {}
        for i, item in enumerate(t_q_raw):
            required = ["primary_role"]
            for field in required:
                if field not in item:
                    raise ValueError(f"T_q_raw[{i}] 缺少字段 {field}: qid={qid}")

            unit_id = str(item.get("chunk_id", item.get("unit_id", ""))).strip()
            if not unit_id:
                raise ValueError(f"T_q_raw[{i}] 缺少 chunk_id/unit_id: qid={qid}")
            role = str(item["primary_role"]).strip()

            if role not in {"bridge", "distinguish", "support"}:
                raise ValueError(f"非法 primary_role: qid={qid}, unit_id={unit_id}, role={role}")
            if unit_id in target_map:
                raise ValueError(f"T_q_raw 中重复 unit_id: qid={qid}, unit_id={unit_id}")

            target_map[unit_id] = item

        targets_map[qid] = {
            "qid": qid,
            "target_map": target_map,
        }

    return targets_map


def load_raw_unit_map(path: Path) -> Dict[str, dict]:
    unit_map = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = [
            "unit_id",
            "text",
            "doc_id",
            "parent_chunk_id",
            "span_start",
            "span_end",
            "provenance",
            "candidate_granularity",
        ]
        for field in required:
            if field not in record:
                raise ValueError(f"raw_units 缺少字段: file={path}, row={row_idx}, field={field}")

        unit_id = str(record["unit_id"])
        if unit_id in unit_map:
            raise ValueError(f"raw_units 中重复 unit_id: file={path}, unit_id={unit_id}")

        if str(record["provenance"]) != "raw":
            raise ValueError(f"raw_units 中 provenance 非 raw: unit_id={unit_id}")

        unit_map[unit_id] = {
            "unit_id": unit_id,
            "text": str(record["text"]).strip(),
            "doc_id": str(record["doc_id"]),
            "parent_chunk_id": str(record["parent_chunk_id"]),
            "span_start": record["span_start"],
            "span_end": record["span_end"],
            "provenance": "raw",
            "candidate_granularity": str(record["candidate_granularity"]),
        }

    return unit_map


def build_H0(p0_items: List[dict]) -> List[dict]:
    h0 = []
    for step_id, item in enumerate(p0_items):
        h0.append(
            {
                "step_id": step_id,
                "unit_id": item["unit_id"],
            }
        )
    return h0


def build_A0(qid: str, p0_items: List[dict], targets_map: Dict[str, dict]) -> dict:
    target_map = targets_map[qid]["target_map"]

    covered_target_ids: List[str] = []
    covered_set = set()

    k_br = 0.0
    k_dis = 0.0
    k_sup = 0.0

    for item in p0_items:
        unit_id = item["unit_id"]
        target_chunk_id = parse_chunk_id_from_unit_id(unit_id)

        if target_chunk_id not in target_map:
            continue
        if target_chunk_id in covered_set:
            continue

        covered_set.add(target_chunk_id)
        covered_target_ids.append(target_chunk_id)

        role = str(target_map[target_chunk_id]["primary_role"])
        if role == "bridge":
            k_br += 1.0
        elif role == "distinguish":
            k_dis += 1.0
        elif role == "support":
            k_sup += 1.0
        else:
            raise ValueError(f"非法 role: qid={qid}, unit_id={unit_id}, role={role}")

    return {
        "covered_target_ids": covered_target_ids,
        "k_br": k_br,
        "k_dis": k_dis,
        "k_sup": k_sup,
    }


def build_S0(p0_items: List[dict]) -> dict:
    raw_refs = []
    for step_id, item in enumerate(p0_items):
        raw_refs.append(
            {
                "unit_id": item["unit_id"],
                "added_step": step_id,
            }
        )

    return {
        "raw_refs": raw_refs,
        "derived_refs": [],
        "last_added_unit_id": p0_items[-1]["unit_id"],
        "last_updated_step": len(p0_items) - 1,
    }


def render_K0(h0: List[dict], raw_unit_map: Dict[str, dict]) -> str:
    lines = []
    for step in h0:
        unit_id = step["unit_id"]
        if unit_id not in raw_unit_map:
            raise ValueError(f"H_0 中的 unit_id 不在 UnitRegistry 中: unit_id={unit_id}")

        unit = raw_unit_map[unit_id]
        sent_id = parse_sent_id_from_unit_id(unit_id)
        lines.append(f"{unit['doc_id']} [{sent_id}] {unit['text']}")

    return "\n".join(lines)


def build_init_state_record(qid: str, p0_items: List[dict], targets_map: Dict[str, dict], raw_unit_map: Dict[str, dict]) -> dict:
    for item in p0_items:
        unit_id = item["unit_id"]
        if unit_id not in raw_unit_map:
            raise ValueError(f"P_0 中的 unit_id 不在 raw_units 中: qid={qid}, unit_id={unit_id}")

    h0 = build_H0(p0_items)
    a0 = build_A0(qid=qid, p0_items=p0_items, targets_map=targets_map)
    s0 = build_S0(p0_items)
    k0 = render_K0(h0=h0, raw_unit_map=raw_unit_map)

    return {
        "qid": qid,
        "t": 0,
        "H_t": h0,
        "A_t": a0,
        "S_t": s0,
        "K_t": k0,
    }


def convert_split(seed_path: Path, targets_path: Path, raw_units_path: Path, output_path: Path) -> int:
    seed_map = load_seed_map(seed_path)
    targets_map = load_targets_map(targets_path)
    raw_unit_map = load_raw_unit_map(raw_units_path)

    def generator():
        for qid in sorted(seed_map.keys()):
            if qid not in targets_map:
                raise ValueError(f"targets 中找不到 qid: {qid}")

            yield build_init_state_record(
                qid=qid,
                p0_items=seed_map[qid]["P_0"],
                targets_map=targets_map,
                raw_unit_map=raw_unit_map,
            )

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    seed_dir = base_dir / "seed"
    targets_dir = base_dir / "targets"
    unit_registry_dir = base_dir / "unit_registry"
    trajectories_dir = base_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    output_name_map = {
        "train": "init_state_train.jsonl",
        "val": "init_state_val.jsonl",
        "test": "init_state_test.jsonl",
    }
    raw_units_name_map = {
        "train": "raw_units_train.jsonl",
        "val": "raw_units_val.jsonl",
        "test": "raw_units_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        seed_path = seed_dir / f"{split}.jsonl"
        targets_path = targets_dir / f"{split}.jsonl"
        raw_units_path = unit_registry_dir / raw_units_name_map[split]
        output_path = trajectories_dir / output_name_map[split]

        for path in [seed_path, targets_path, raw_units_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            seed_path=seed_path,
            targets_path=targets_path,
            raw_units_path=raw_units_path,
            output_path=output_path,
        )

    print("init_state v2 构建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {trajectories_dir / output_name_map[split]}")


if __name__ == "__main__":
    main()
