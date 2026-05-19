import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = "data/hotpotqa_distractor_v2"
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 2.0


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
    # 去掉明显无意义标点，但保留数字和字母
    text = re.sub(r"[\"'“”‘’`.,;:!?()\[\]{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_queries(path: Path) -> Dict[str, dict]:
    queries = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "question", "answer"]
        for field in required:
            if field not in record:
                raise ValueError(f"queries 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in queries:
            raise ValueError(f"queries 中重复 qid: file={path}, qid={qid}")

        answer = str(record["answer"]).strip()
        if not answer:
            raise ValueError(f"answer 为空: qid={qid}")

        queries[qid] = {
            "qid": qid,
            "question": str(record["question"]).strip(),
            "answer": answer,
        }

    return queries


def load_targets(path: Path) -> Dict[str, dict]:
    targets = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "T_q_raw"]
        for field in required:
            if field not in record:
                raise ValueError(f"targets 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in targets:
            raise ValueError(f"targets 中重复 qid: file={path}, qid={qid}")

        t_q_raw = record["T_q_raw"]
        if not isinstance(t_q_raw, list) or len(t_q_raw) == 0:
            raise ValueError(f"T_q_raw 必须是非空 list: qid={qid}")

        unit_ids = []
        seen = set()
        for i, item in enumerate(t_q_raw):
            unit_id = str(item.get("chunk_id", item.get("unit_id", ""))).strip()
            if not unit_id:
                raise ValueError(f"T_q_raw[{i}] 缺少 chunk_id/unit_id: qid={qid}")
            if unit_id in seen:
                raise ValueError(f"T_q_raw 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            seen.add(unit_id)
            unit_ids.append(unit_id)

        targets[qid] = {
            "qid": qid,
            "target_unit_ids": unit_ids,
            "target_unit_id_set": set(unit_ids),
        }

    return targets


def load_init_states(path: Path) -> Dict[str, dict]:
    states = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "A_t", "K_t"]
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
        if not isinstance(a_t, dict):
            raise ValueError(f"A_t 必须是 dict: qid={qid}")

        covered_target_ids = a_t.get("covered_target_ids", [])
        if not isinstance(covered_target_ids, list):
            raise ValueError(f"A_t.covered_target_ids 必须是 list: qid={qid}")

        states[qid] = {
            "qid": qid,
            "t": t,
            "A_t": {
                "covered_target_ids": [str(x) for x in covered_target_ids]
            },
            "K_t": str(record["K_t"]),
        }

    return states


def load_gates(path: Path) -> Dict[str, dict]:
    gates = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "StopCandidate_t", "NeedDerived_t", "s_sem"]
        for field in required:
            if field not in record:
                raise ValueError(f"gates 缺少字段: file={path}, row={row_idx}, field={field}")

        qid = str(record["qid"])
        if qid in gates:
            raise ValueError(f"gates 中重复 qid: file={path}, qid={qid}")

        t = int(record["t"])
        if t != 0:
            raise ValueError(f"这里只接 t=0 gates: qid={qid}, t={t}")

        gates[qid] = {
            "qid": qid,
            "t": t,
            "StopCandidate_t": bool(record["StopCandidate_t"]),
            "NeedDerived_t": bool(record["NeedDerived_t"]),
            "s_sem": float(record["s_sem"]),
        }

    return gates


def deepseek_chat_json(api_key: str, model: str, base_url: str, system_prompt: str, user_prompt: str) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 256,
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
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("模型未返回 JSON object")
                return parsed
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_SEC)

    raise RuntimeError(f"DeepSeek 请求失败: {last_err}") from last_err


def answer_with_deepseek(
    *,
    api_key: str,
    model: str,
    base_url: str,
    question: str,
    k_t: str,
) -> Optional[str]:
    system_prompt = (
        "You answer a question using only the provided evidence context.\n"
        "Return strict JSON with a single key `answer`.\n"
        "If the answer span is explicitly present in the evidence, copy the shortest exact answer span.\n"
        "Return only a short answer phrase, not an explanation.\n"
        "If the context is insufficient, return an empty string."
    )
    user_prompt = (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Evidence Context:\n"
        f"{k_t.strip()}\n\n"
        "Return JSON like {\"answer\": \"...\"}."
    )
    parsed = deepseek_chat_json(
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    answer = parsed.get("answer")
    if answer is None:
        return None
    answer = str(answer).strip()
    return answer or None


def check_answer_correct(gold_answer: str, probe_answer: Optional[str]) -> bool:
    if probe_answer is None:
        return False
    return normalize_text(gold_answer) == normalize_text(probe_answer)


def context_contains_answer(gold_answer: str, k_t: str) -> bool:
    norm_gold = normalize_text(gold_answer)
    norm_ctx = normalize_text(k_t)
    if not norm_gold:
        return False
    return norm_gold in norm_ctx


def check_support_sufficient(covered_target_ids: List[str], target_unit_id_set: set) -> bool:
    covered_set = set(str(x) for x in covered_target_ids)
    return target_unit_id_set.issubset(covered_set)


def build_stop_probe_record(
    qid: str,
    query_info: dict,
    target_info: dict,
    state: dict,
    gate_info: dict,
    api_key: str,
    base_url: str,
    model: str,
) -> dict:
    stop_candidate = gate_info["StopCandidate_t"]

    if not stop_candidate:
        return {
            "qid": qid,
            "t": 0,
            "probe_run": False,
            "answer_source": None,
            "gold_answer": str(query_info["answer"]),
            "gold_answer_normalized": normalize_text(str(query_info["answer"])),
            "probe_answer": None,
            "probe_answer_normalized": None,
            "answer_match_rule": "normalized_exact",
            "AnswerCorrect_t": False,
            "SupportSufficient_t": False,
            "TeacherStop_t": False,
            "FalseStop_t": False,
        }

    probe_answer = answer_with_deepseek(
        api_key=api_key,
        model=model,
        base_url=base_url,
        question=query_info["question"],
        k_t=state["K_t"],
    )
    gold_answer = str(query_info["answer"])
    gold_answer_normalized = normalize_text(gold_answer)
    probe_answer_normalized = normalize_text(probe_answer) if probe_answer is not None else None
    context_exact_match = context_contains_answer(gold_answer, state["K_t"])
    exact_match = check_answer_correct(
        gold_answer=query_info["answer"],
        probe_answer=probe_answer,
    )
    answer_correct = exact_match or context_exact_match
    support_sufficient = check_support_sufficient(
        covered_target_ids=state["A_t"]["covered_target_ids"],
        target_unit_id_set=target_info["target_unit_id_set"],
    )

    teacher_stop = bool(stop_candidate and answer_correct and support_sufficient)
    false_stop = bool(stop_candidate and (not teacher_stop))
    answer_match_rule = "normalized_exact_or_context_exact"
    if exact_match:
        answer_match_rule = "normalized_exact"
    elif context_exact_match:
        answer_match_rule = "context_exact"

    return {
        "qid": qid,
        "t": 0,
        "probe_run": True,
        "answer_source": "llm",
        "gold_answer": gold_answer,
        "gold_answer_normalized": gold_answer_normalized,
        "probe_answer": probe_answer,
        "probe_answer_normalized": probe_answer_normalized,
        "context_contains_gold_answer": context_exact_match,
        "answer_match_rule": answer_match_rule,
        "AnswerCorrect_t": answer_correct,
        "SupportSufficient_t": support_sufficient,
        "TeacherStop_t": teacher_stop,
        "FalseStop_t": false_stop,
    }


def convert_split(
    queries_path: Path,
    targets_path: Path,
    init_state_path: Path,
    gates_path: Path,
    output_path: Path,
    api_key: str,
    base_url: str,
    model: str,
) -> int:
    queries = load_queries(queries_path)
    targets = load_targets(targets_path)
    init_states = load_init_states(init_state_path)
    gates = load_gates(gates_path)

    def generator():
        for qid in sorted(gates.keys()):
            if qid not in queries:
                raise ValueError(f"queries 中找不到 qid: {qid}")
            if qid not in targets:
                raise ValueError(f"targets 中找不到 qid: {qid}")
            if qid not in init_states:
                raise ValueError(f"init_state 中找不到 qid: {qid}")

            yield build_stop_probe_record(
                qid=qid,
                query_info=queries[qid],
                target_info=targets[qid],
                state=init_states[qid],
                gate_info=gates[qid],
                api_key=api_key,
                base_url=base_url,
                model=model,
            )

    return write_jsonl(generator(), output_path)


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"

    if not api_key:
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")

    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    trajectories_dir = base_dir / "trajectories"
    queries_dir = base_dir / "queries"
    targets_dir = base_dir / "targets"

    init_name_map = {
        "train": "init_state_train.jsonl",
        "val": "init_state_val.jsonl",
        "test": "init_state_test.jsonl",
    }
    gates_name_map = {
        "train": "gates_train.jsonl",
        "val": "gates_val.jsonl",
        "test": "gates_test.jsonl",
    }
    out_name_map = {
        "train": "stop_probe_train.jsonl",
        "val": "stop_probe_val.jsonl",
        "test": "stop_probe_test.jsonl",
    }

    stats = {}

    for split in SPLITS:
        queries_path = queries_dir / f"{split}.jsonl"
        targets_path = targets_dir / f"{split}.jsonl"
        init_state_path = trajectories_dir / init_name_map[split]
        gates_path = trajectories_dir / gates_name_map[split]
        output_path = trajectories_dir / out_name_map[split]

        for path in [queries_path, targets_path, init_state_path, gates_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        stats[split] = convert_split(
            queries_path=queries_path,
            targets_path=targets_path,
            init_state_path=init_state_path,
            gates_path=gates_path,
            output_path=output_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    print("stop_probe v2 构建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {trajectories_dir / out_name_map[split]}")


if __name__ == "__main__":
    main()
