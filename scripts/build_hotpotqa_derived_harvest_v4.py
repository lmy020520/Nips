import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v4")

TOP_RAW_J = 3
MAX_CANDIDATES = 4
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 2.0

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
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def shorten(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def extract_goal_from_question(question: str) -> str:
    q = question.strip().rstrip("?").lower()

    patterns = [
        (r"which magazine was started first", "publication founding time"),
        (r"which .* was started first", "founding time comparison"),
        (r"when was .* founded", "founding time"),
        (r"what year was .* founded", "founding time"),
        (r"which university did .* attend", "university attended"),
        (r"who acquired .*", "acquirer"),
        (r"who wrote .*", "author"),
        (r"where was .* born", "birthplace"),
        (r"when did .* die", "death time"),
    ]

    for pattern, goal in patterns:
        if re.search(pattern, q):
            return goal

    return "more evidence for the question"


def build_derived_unit_id(qid: str, idx: int) -> str:
    return f"{qid}::derived::{idx}"


def load_queries(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "question", "answer"]
        for field in required:
            if field not in record:
                raise ValueError(f"queries 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"queries 中重复 qid: file={path}, qid={qid}")

        out[qid] = {
            "qid": qid,
            "question": str(record["question"]).strip(),
            "answer": str(record["answer"]).strip(),
        }
    return out


def load_raw_unit_map(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["unit_id", "text", "doc_id"]
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
        }
    return out


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

        t = int(record["t"])
        if t != 0:
            raise ValueError(f"这里只接 t=0 init_state: qid={qid}, t={t}")

        s_t = record["S_t"]
        if not isinstance(s_t, dict):
            raise ValueError(f"S_t 必须是 dict: qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": t,
            "S_t": s_t,
        }
    return out


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

        out[qid] = {
            "qid": qid,
            "t": t,
            "q_t": str(record["q_t"]),
            "R_t": [str(x) for x in r_t],
        }
    return out


def load_gates(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "StopCandidate_t", "NeedDerived_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"gates 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"gates 中重复 qid: file={path}, qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "StopCandidate_t": bool(record["StopCandidate_t"]),
            "NeedDerived_t": bool(record["NeedDerived_t"]),
        }
    return out


def load_stop_probe(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "TeacherStop_t", "FalseStop_t", "probe_run"]
        for field in required:
            if field not in record:
                raise ValueError(f"stop_probe 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"stop_probe 中重复 qid: file={path}, qid={qid}")

        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "TeacherStop_t": bool(record["TeacherStop_t"]),
            "FalseStop_t": bool(record["FalseStop_t"]),
            "probe_run": bool(record["probe_run"]),
        }
    return out


def get_recent_raw_texts(s_t: dict, raw_unit_map: Dict[str, dict], n: int = 2) -> List[str]:
    raw_refs = s_t.get("raw_refs", [])
    if not isinstance(raw_refs, list):
        return []

    texts = []
    for item in raw_refs[-n:]:
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit_id = str(item["unit_id"])
        unit = raw_unit_map.get(unit_id)
        if unit is not None:
            texts.append(unit["text"])
    return texts


def get_recent_derived_texts(s_t: dict) -> List[str]:
    # 当前 t=0 初始化阶段 derived_refs 为空，这里保留接口
    derived_refs = s_t.get("derived_refs", [])
    if not isinstance(derived_refs, list):
        return []
    return []


def build_state_summary(question: str, s_t: dict, raw_unit_map: Dict[str, dict]) -> str:
    lines = []

    recent_raw = get_recent_raw_texts(s_t, raw_unit_map, n=2)
    for text in recent_raw[:2]:
        lines.append("Evidence: " + shorten(text, 90))

    recent_derived = get_recent_derived_texts(s_t)
    if recent_derived:
        lines.append("Note: " + shorten(recent_derived[-1], 70))

    lines.append("Need: " + extract_goal_from_question(question))
    return "\n".join(lines)


def build_top_raw_candidates(r_t: List[str], raw_unit_map: Dict[str, dict], top_j: int) -> List[dict]:
    out = []
    for unit_id in r_t[:top_j]:
        unit = raw_unit_map.get(unit_id)
        if unit is None:
            continue
        out.append(
            {
                "unit_id": unit_id,
                "text": unit["text"],
                "doc_id": unit["doc_id"],
            }
        )
    return out


def deepseek_chat_json(
    api_key: str,
    model: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 512,
    }

    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_SEC)

    raise RuntimeError(f"DeepSeek 请求失败: {last_err}") from last_err


def build_propose_prompt(
    question: str,
    state_summary: str,
    top_raw_candidates: List[dict],
    gold_answer: Optional[str],
    max_candidates: int,
) -> str:
    raw_lines = []
    for i, item in enumerate(top_raw_candidates, start=1):
        raw_lines.append(
            f"{i}. unit_id={item['unit_id']}\n"
            f"   doc={item['doc_id']}\n"
            f"   text={item['text']}"
        )

    gold_line = gold_answer if gold_answer else "null"

    return (
        f"Question:\n{question}\n\n"
        f"State summary:\n{state_summary}\n\n"
        f"Top raw candidates:\n" + "\n".join(raw_lines) + "\n\n"
        f"Gold answer (offline only, optional hint):\n{gold_line}\n\n"
        f"Please decide whether to derive short grounded notes now.\n"
        f"Return strict JSON with keys:\n"
        f"- should_derive: boolean\n"
        f"- reason: string\n"
        f"- derived_candidates: list\n\n"
        f"Each derived candidate must contain:\n"
        f"- type: one of bridge_note, verification_note\n"
        f"- text: exactly one sentence\n"
        f"- source_unit_ids: list of 1 to 3 raw unit ids from the provided top raw candidates\n"
        f"- coarse_priority: integer\n\n"
        f"Constraints:\n"
        f"- produce at most {max_candidates} candidates\n"
        f"- do not output final answer style\n"
        f"- do not invent unsupported facts\n"
        f"- do not use claimed_role\n"
        f"- source_unit_ids must come only from the provided top raw candidates\n"
    )


def validate_harvest_candidates(
    qid: str,
    parsed: dict,
    visible_source_ids: set,
    max_candidates: int,
) -> List[dict]:
    candidates = parsed.get("derived_candidates", [])
    if not isinstance(candidates, list):
        return []

    validated = []
    seen_unit_ids = set()

    for idx, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            continue

        cand_type = str(cand.get("type", "")).strip()
        text = str(cand.get("text", "")).strip()
        source_unit_ids = cand.get("source_unit_ids", [])
        coarse_priority = cand.get("coarse_priority", len(validated) + 1)

        if cand_type not in ALLOWED_TYPES:
            continue
        if not text:
            continue
        if not isinstance(source_unit_ids, list):
            continue
        source_unit_ids = [str(x) for x in source_unit_ids]
        if not (1 <= len(source_unit_ids) <= 3):
            continue
        if len(source_unit_ids) != len(set(source_unit_ids)):
            continue
        if not set(source_unit_ids).issubset(visible_source_ids):
            continue

        unit_id = str(cand.get("unit_id", "")).strip()
        if not unit_id:
            unit_id = build_derived_unit_id(qid, len(validated))

        if unit_id in seen_unit_ids:
            continue
        seen_unit_ids.add(unit_id)

        validated.append(
            {
                "unit_id": unit_id,
                "text": text,
                "provenance": "derived",
                "candidate_granularity": "note",
                "type": cand_type,
                "source_unit_ids": source_unit_ids,
                "coarse_priority": int(coarse_priority),
            }
        )

    validated.sort(key=lambda x: (x["coarse_priority"], x["unit_id"]))
    validated = validated[:max_candidates]

    # 输出时不保留 coarse_priority
    final_out = []
    for item in validated:
        final_out.append(
            {
                "unit_id": item["unit_id"],
                "text": item["text"],
                "provenance": "derived",
                "candidate_granularity": "note",
                "type": item["type"],
                "source_unit_ids": item["source_unit_ids"],
            }
        )
    return final_out


def propose_derived(
    *,
    api_key: str,
    base_url: str,
    model: str,
    qid: str,
    question: str,
    state_summary: str,
    top_raw_candidates: List[dict],
    gold_answer: Optional[str],
    max_candidates: int,
) -> List[dict]:
    if not top_raw_candidates:
        return []

    system_prompt = (
        "You are proposing grounded derived notes for offline teacher trajectory construction.\n"
        "Only propose short, grounded notes if they help organize currently visible raw evidence.\n"
        "Allowed note types are exactly: bridge_note, verification_note.\n"
        "Do not output claimed_role.\n"
        "Return strict JSON only."
    )

    user_prompt = build_propose_prompt(
        question=question,
        state_summary=state_summary,
        top_raw_candidates=top_raw_candidates,
        gold_answer=gold_answer,
        max_candidates=max_candidates,
    )

    parsed = deepseek_chat_json(
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    should_derive = bool(parsed.get("should_derive", False))
    if not should_derive:
        return []

    visible_source_ids = {item["unit_id"] for item in top_raw_candidates}
    return validate_harvest_candidates(
        qid=qid,
        parsed=parsed,
        visible_source_ids=visible_source_ids,
        max_candidates=max_candidates,
    )


def build_record_for_qid(
    *,
    qid: str,
    queries: Dict[str, dict],
    init_states: Dict[str, dict],
    candidates: Dict[str, dict],
    gates: Dict[str, dict],
    stop_probe: Dict[str, dict],
    raw_unit_map: Dict[str, dict],
    api_key: str,
    base_url: str,
    model: str,
) -> dict:
    teacher_stop = stop_probe[qid]["TeacherStop_t"]
    need_derived = gates[qid]["NeedDerived_t"]

    should_run = (not teacher_stop) and need_derived

    if not should_run:
        return {
            "qid": qid,
            "t": 0,
            "proposal_run": False,
            "G_t_harvest": [],
        }

    question = queries[qid]["question"]
    gold_answer = queries[qid]["answer"]
    s_t = init_states[qid]["S_t"]
    r_t = candidates[qid]["R_t"]

    state_summary = build_state_summary(question, s_t, raw_unit_map)
    top_raw_candidates = build_top_raw_candidates(r_t, raw_unit_map, top_j=TOP_RAW_J)

    g_t_harvest = propose_derived(
        api_key=api_key,
        base_url=base_url,
        model=model,
        qid=qid,
        question=question,
        state_summary=state_summary,
        top_raw_candidates=top_raw_candidates,
        gold_answer=gold_answer,
        max_candidates=MAX_CANDIDATES,
    )

    return {
        "qid": qid,
        "t": 0,
        "proposal_run": True,
        "G_t_harvest": g_t_harvest,
    }


def load_full_trajectories(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record or "steps" not in record:
            raise ValueError(f"full trajectories 缺少字段: file={path}, row={row_idx}")
        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"full trajectories 中重复 qid: file={path}, qid={qid}")
        out[qid] = record
    return out


def normalize_harvest_item(item: dict) -> dict:
    required = ["unit_id", "text", "type", "source_unit_ids"]
    for field in required:
        if field not in item:
            raise ValueError(f"harvest candidate 缺少字段 {field}: item={item}")
    return {
        "unit_id": str(item["unit_id"]),
        "text": str(item["text"]).strip(),
        "provenance": "derived",
        "candidate_granularity": "note",
        "type": str(item["type"]).strip(),
        "source_unit_ids": [str(x) for x in item["source_unit_ids"]],
        "coarse_priority": int(item.get("coarse_priority", 10**9)),
    }


def aggregate_harvest_for_qid(full_traj: dict) -> dict:
    qid = str(full_traj["qid"])
    merged: Dict[str, dict] = {}
    proposal_run = False
    for step in full_traj.get("steps", []):
        proposer_trace = step.get("proposer_trace", {})
        harvest = proposer_trace.get("harvest_candidates", [])
        if not isinstance(harvest, list) or not harvest:
            continue
        proposal_run = True
        for raw_item in harvest:
            if not isinstance(raw_item, dict):
                continue
            item = normalize_harvest_item(raw_item)
            prev = merged.get(item["unit_id"])
            if prev is not None and prev != item:
                raise ValueError(f"同一 derived unit_id payload 不一致: qid={qid}, unit_id={item['unit_id']}")
            merged[item["unit_id"]] = item
    ordered = sorted(merged.values(), key=lambda x: (x["coarse_priority"], x["unit_id"]))
    for item in ordered:
        item.pop("coarse_priority", None)
    return {
        "qid": qid,
        "t": 0,
        "proposal_run": proposal_run,
        "G_t_harvest": ordered,
    }


def convert_split(
    *,
    queries_path: Path,
    full_path: Path,
    output_path: Path,
) -> int:
    queries = load_queries(queries_path)
    full_map = load_full_trajectories(full_path)

    def generator():
        for qid in sorted(full_map.keys()):
            if qid not in queries:
                raise ValueError(f"queries 中找不到 qid: {qid}")
            yield aggregate_harvest_for_qid(full_map[qid])

    return write_jsonl(generator(), output_path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = Path(DEFAULT_BASE)
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir

    trajectories_dir = base_dir / "trajectories"
    queries_dir = base_dir / "queries"
    out_name_map = {
        "train": "derived_harvest_train.jsonl",
        "val": "derived_harvest_val.jsonl",
        "test": "derived_harvest_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        queries_path = queries_dir / f"{split}.jsonl"
        full_path = trajectories_dir / f"full_{split}.jsonl"
        output_path = trajectories_dir / out_name_map[split]

        for path in [queries_path, full_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            queries_path=queries_path,
            full_path=full_path,
            output_path=output_path,
        )

    print("derived_harvest v3 构建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {trajectories_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()
