import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v2"

MAX_SOURCE_PER_NOTE = 3
MAX_TOKENS = 45
DUPLICATE_SIM_THRESHOLD = 0.90
MAX_FINAL = 2
ALLOWED_TYPES = {"bridge_note", "verification_note"}


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


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'“”‘’`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    # 极简规则：按句末标点切；若没有标点，则视为 1 句
    pieces = [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]
    return max(1, len(pieces))


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[A-Za-z0-9]+", a.lower()))
    tb = set(re.findall(r"[A-Za-z0-9]+", b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def load_init_states(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "S_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"init_state 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"init_state 中重复 qid: file={path}, qid={qid}")

        s_t = record["S_t"]
        if not isinstance(s_t, dict):
            raise ValueError(f"S_t 必须是 dict: qid={qid}")

        raw_refs = s_t.get("raw_refs", [])
        derived_refs = s_t.get("derived_refs", [])
        if not isinstance(raw_refs, list):
            raise ValueError(f"S_t.raw_refs 必须是 list: qid={qid}")
        if not isinstance(derived_refs, list):
            raise ValueError(f"S_t.derived_refs 必须是 list: qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "S_t": s_t,
        }

    return out


def load_candidates(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "R_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"candidates 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"candidates 中重复 qid: file={path}, qid={qid}")

        r_t = record["R_t"]
        if not isinstance(r_t, list):
            raise ValueError(f"R_t 必须是 list: qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "R_t": [str(x) for x in r_t],
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

        g = record["G_t_harvest"]
        if not isinstance(g, list):
            raise ValueError(f"G_t_harvest 必须是 list: qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "proposal_run": bool(record["proposal_run"]),
            "G_t_harvest": g,
        }

    return out


def visible_raw_ids_from_state_and_candidates(state: dict, cand: dict) -> set:
    s_t = state["S_t"]
    raw_refs = s_t.get("raw_refs", [])
    visible = set()

    for item in raw_refs:
        if isinstance(item, dict) and "unit_id" in item:
            visible.add(str(item["unit_id"]))

    for unit_id in cand["R_t"]:
        visible.add(str(unit_id))

    return visible


def get_existing_note_texts_from_state(state: dict) -> List[str]:
    # 当前 t=0 初始化阶段 derived_refs 为空；这里保留接口
    s_t = state["S_t"]
    derived_refs = s_t.get("derived_refs", [])
    if not isinstance(derived_refs, list) or len(derived_refs) == 0:
        return []
    return []


def legality_filter(state: dict, cand: dict, harvest: List[dict]) -> Tuple[List[dict], List[dict]]:
    visible_raw_ids = visible_raw_ids_from_state_and_candidates(state, cand)
    existing_note_texts = get_existing_note_texts_from_state(state)

    legal = []
    illegal = []
    accepted_texts = []

    for idx, z in enumerate(harvest):
        reasons = []

        if not isinstance(z, dict):
            illegal.append({"candidate": z, "reasons": ["invalid_schema"]})
            continue

        required_fields = ["unit_id", "type", "text", "source_unit_ids"]
        for field in required_fields:
            if field not in z:
                reasons.append("invalid_schema")
                break

        z_type = str(z.get("type", "")).strip()
        z_text = str(z.get("text", "")).strip()
        src_ids = z.get("source_unit_ids", [])
        unit_id = str(z.get("unit_id", "")).strip()

        if z_type not in ALLOWED_TYPES:
            reasons.append("invalid_type")

        if not isinstance(src_ids, list):
            reasons.append("invalid_source_count")
            src_ids = []
        else:
            src_ids = [str(x) for x in src_ids]

        if not (1 <= len(src_ids) <= MAX_SOURCE_PER_NOTE):
            reasons.append("invalid_source_count")
        if len(src_ids) != len(set(src_ids)):
            reasons.append("duplicated_source_ids")
        if not set(src_ids).issubset(visible_raw_ids):
            reasons.append("invisible_source_ids")

        if sentence_count(z_text) != 1:
            reasons.append("not_single_sentence")
        if token_count(z_text) > MAX_TOKENS:
            reasons.append("too_long")

        norm_text = normalize_text(z_text)
        if not norm_text:
            reasons.append("empty_text")

        # duplicate_in_harvest
        for prev_text in accepted_texts:
            if norm_text == prev_text or jaccard_similarity(norm_text, prev_text) > DUPLICATE_SIM_THRESHOLD:
                reasons.append("duplicate_in_harvest")
                break

        # duplicate_with_state
        for prev_text in existing_note_texts:
            prev_norm = normalize_text(prev_text)
            if norm_text == prev_norm or jaccard_similarity(norm_text, prev_norm) > DUPLICATE_SIM_THRESHOLD:
                reasons.append("duplicate_with_state")
                break

        if reasons:
            illegal.append(
                {
                    "candidate": {
                        "unit_id": unit_id,
                        "type": z_type,
                        "text": z_text,
                        "source_unit_ids": src_ids,
                    },
                    "reasons": sorted(set(reasons)),
                }
            )
            continue

        accepted_texts.append(norm_text)
        legal.append(
            {
                "unit_id": unit_id,
                "type": z_type,
                "text": z_text,
                "source_unit_ids": src_ids,
                # 为兼容上一步若没写 coarse_priority，这里退化为输入顺序
                "coarse_priority": int(z.get("coarse_priority", idx)),
            }
        )

    legal.sort(key=lambda x: (x["coarse_priority"], x["unit_id"]))
    return legal, illegal


def final_retain_selection(g_legal: List[dict]) -> Tuple[List[str], List[str]]:
    if not g_legal:
        return [], []

    retained = []
    remaining = list(g_legal)

    # 1. 第一名一定保留
    first = remaining.pop(0)
    retained.append(first)

    # 2. 第二名优先补另一种 type
    if remaining and len(retained) < MAX_FINAL:
        preferred_idx = None
        first_type = first["type"]
        for i, item in enumerate(remaining):
            if item["type"] != first_type:
                preferred_idx = i
                break

        if preferred_idx is None:
            second = remaining.pop(0)
        else:
            second = remaining.pop(preferred_idx)

        retained.append(second)

    final_ids = [x["unit_id"] for x in retained]
    aux_ids = [x["unit_id"] for x in remaining]
    return final_ids, aux_ids


def build_filter_record(qid: str, state: dict, cand: dict, harvest_record: dict) -> dict:
    proposal_run = harvest_record["proposal_run"]
    g_harvest = harvest_record["G_t_harvest"]

    if not proposal_run or not g_harvest:
        return {
            "qid": qid,
            "t": 0,
            "G_t_final": [],
            "G_t_aux": [],
            "G_t_illegal": [],
        }

    g_legal, g_illegal = legality_filter(state, cand, g_harvest)
    g_final, g_aux = final_retain_selection(g_legal)

    # G_t_illegal 只保留 unit_id 列表，保持当前阶段输出轻量
    illegal_ids = []
    for item in g_illegal:
        cand_obj = item.get("candidate", {})
        unit_id = cand_obj.get("unit_id")
        if unit_id is not None:
            illegal_ids.append(str(unit_id))

    return {
        "qid": qid,
        "t": 0,
        "G_t_final": g_final,
        "G_t_aux": g_aux,
        "G_t_illegal": illegal_ids,
    }


def convert_split(
    init_state_path: Path,
    candidates_path: Path,
    harvest_path: Path,
    output_path: Path,
) -> int:
    init_states = load_init_states(init_state_path)
    candidates = load_candidates(candidates_path)
    harvest = load_derived_harvest(harvest_path)

    def generator():
        for qid in sorted(harvest.keys()):
            if qid not in init_states:
                raise ValueError(f"init_state 中找不到 qid: {qid}")
            if qid not in candidates:
                raise ValueError(f"candidates 中找不到 qid: {qid}")

            yield build_filter_record(
                qid=qid,
                state=init_states[qid],
                cand=candidates[qid],
                harvest_record=harvest[qid],
            )

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE
    trajectories_dir = base_dir / "trajectories"

    out_name_map = {
        "train": "derived_filter_train.jsonl",
        "val": "derived_filter_val.jsonl",
        "test": "derived_filter_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        init_state_path = trajectories_dir / f"init_state_{split}.jsonl"
        candidates_path = trajectories_dir / f"candidates_{split}.jsonl"
        harvest_path = trajectories_dir / f"derived_harvest_{split}.jsonl"
        output_path = trajectories_dir / out_name_map[split]

        for path in [init_state_path, candidates_path, harvest_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            init_state_path=init_state_path,
            candidates_path=candidates_path,
            harvest_path=harvest_path,
            output_path=output_path,
        )

    print("derived_filter v2 构建完成：")
    print(
        f"  defaults: max_source_per_note={MAX_SOURCE_PER_NOTE}, "
        f"max_tokens={MAX_TOKENS}, "
        f"duplicate_sim_threshold={DUPLICATE_SIM_THRESHOLD}, "
        f"max_final={MAX_FINAL}"
    )
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {trajectories_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()