import os
import json
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]

DEFAULT_INPUT_BASE = "data/hotpotqa_distractor/processed"
DEFAULT_OUTPUT_BASE = os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v2")
DEFAULT_QUERY_BASE = os.environ.get(
    "HOTPOTQA_QUERY_BASE",
    os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v2") + "/queries",
)

REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 2.0

ALLOWED_ROLES = {"bridge", "distinguish", "support"}
RULE_FALLBACK_SOURCES = {"rule_fallback", "heuristic_fallback"}


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


def load_allowed_qids(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(f"找不到 queries 文件: {path}")

    allowed = set()
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "qid" not in record:
            raise ValueError(f"queries 缺少 qid: file={path}, row={row_idx}")
        allowed.add(str(record["qid"]))

    if not allowed:
        raise ValueError(f"queries 为空: {path}")
    return allowed


def normalize_supporting_facts(supporting_facts: list, qid: str) -> List[Tuple[str, int]]:
    if not isinstance(supporting_facts, list):
        raise ValueError(f"supporting_facts 必须是 list: qid={qid}")

    pairs = []
    seen = set()
    for i, item in enumerate(supporting_facts):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"supporting_facts[{i}] 格式错误: qid={qid}")
        title, sent_id = str(item[0]), int(item[1])
        key = (title, sent_id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def find_sentence_text(context: list, title: str, sent_id: int, qid: str) -> str:
    if not isinstance(context, list):
        raise ValueError(f"context 必须是 list: qid={qid}")

    for block in context:
        if not isinstance(block, dict):
            raise ValueError(f"context block 必须是 dict: qid={qid}")
        if str(block.get("title")) != title:
            continue

        sentences = block.get("sentences", [])
        if not isinstance(sentences, list):
            raise ValueError(f"sentences 必须是 list: qid={qid}, title={title}")

        if sent_id < 0 or sent_id >= len(sentences):
            raise ValueError(
                f"sent_id 越界: qid={qid}, title={title}, sent_id={sent_id}, len={len(sentences)}"
            )

        text = str(sentences[sent_id]).strip()
        if not text:
            raise ValueError(f"target sentence 为空: qid={qid}, title={title}, sent_id={sent_id}")
        return text

    raise ValueError(f"在 context 中找不到 title: qid={qid}, title={title}")


def build_unit_id(qid: str, doc_id: str, sent_id: int) -> str:
    return f"{qid}::{doc_id}::{sent_id}"


def build_parent_chunk_id(qid: str, doc_id: str) -> str:
    return f"{qid}::{doc_id}"


def stable_cache_name(qid: str, unit_id: str) -> str:
    digest = hashlib.sha1(f"{qid}::{unit_id}".encode("utf-8")).hexdigest()
    return f"{digest}.json"


def load_cache(cache_dir: Path, qid: str, unit_id: str) -> Optional[dict]:
    path = cache_dir / stable_cache_name(qid, unit_id)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_dir: Path, qid: str, unit_id: str, payload: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / stable_cache_name(qid, unit_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_for_match(text: str) -> str:
    return " ".join(str(text or "").lower().replace("_", " ").split())


def token_set(text: str) -> set:
    import re

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "both", "by", "did", "do",
        "does", "for", "from", "get", "have", "how", "in", "is", "it", "its",
        "of", "on", "or", "that", "the", "their", "this", "to", "what", "when",
        "where", "which", "who", "whom", "whose", "with", "was", "were",
    }
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", normalize_for_match(text))
        if len(tok) >= 3 and tok not in stopwords
    }


def rule_fallback_role(question: str, answer: str, text: str, doc_id: str) -> str:
    """Cheap deterministic fallback used only when no LLM cache/API is available."""
    q_norm = normalize_for_match(question)
    a_norm = normalize_for_match(answer)
    text_norm = normalize_for_match(text)
    doc_norm = normalize_for_match(doc_id)

    if a_norm in {"yes", "no"}:
        return "support"
    if a_norm and (a_norm in text_norm or a_norm in doc_norm):
        return "support"

    q_tokens = token_set(question)
    doc_tokens = token_set(doc_id)
    text_tokens = token_set(text)
    if doc_tokens and (doc_tokens & q_tokens):
        return "bridge"
    if q_tokens and len(q_tokens & text_tokens) >= 2:
        return "bridge"
    if any(marker in q_norm for marker in ["which", "what", "who", "where"]) and doc_tokens:
        return "bridge"
    return "distinguish"


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
        "max_tokens": 128,
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


def classify_role_with_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    cache_dir: Path,
    qid: str,
    question: str,
    answer: str,
    unit_id: str,
    text: str,
    doc_id: str,
) -> Tuple[str, str]:
    """
    返回:
      (primary_role, role_label_source)
    仅允许:
      bridge / distinguish / support
    """
    require_llm_roles = os.environ.get("HOTPOTQA_REQUIRE_LLM_ROLES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    cached = load_cache(cache_dir, qid, unit_id)
    if cached is not None:
        role = cached.get("primary_role")
        source = cached.get("role_label_source")
        if role in ALLOWED_ROLES and source == "llm":
            return role, source
        if not require_llm_roles and role in ALLOWED_ROLES and source in RULE_FALLBACK_SOURCES:
            return role, source

    label_mode = os.environ.get("HOTPOTQA_ROLE_LABEL_MODE", "auto").strip().lower()
    if label_mode == "rule" or not api_key:
        if require_llm_roles:
            raise RuntimeError(
                "HOTPOTQA_REQUIRE_LLM_ROLES=1 but API key is missing or role mode is rule"
            )
        role = rule_fallback_role(question=question, answer=answer, text=text, doc_id=doc_id)
        cache_payload = {
            "qid": qid,
            "unit_id": unit_id,
            "doc_id": doc_id,
            "primary_role": role,
            "role_label_source": "rule_fallback",
            "raw_response": {"primary_role": role, "fallback": "no_api_or_rule_mode"},
        }
        save_cache(cache_dir, qid, unit_id, cache_payload)
        return role, "rule_fallback"

    system_prompt = (
        "You label the primary raw evidence role for one gold supporting sentence in multi-hop QA.\n"
        "Allowed labels are exactly: bridge, distinguish, support.\n"
        "Definitions:\n"
        "- bridge: mainly links the question to another entity/document/hop.\n"
        "- distinguish: mainly disambiguates which entity/time/event/meaning is intended.\n"
        "- support: mainly provides direct factual support for the answer.\n"
        "Return strict JSON only with one key: primary_role."
    )

    user_prompt = (
        f"Question: {question}\n"
        f"Doc title: {doc_id}\n"
        f"Gold sentence: {text}\n\n"
        "Choose the single best primary_role from: bridge, distinguish, support.\n"
        'Output JSON only, for example: {"primary_role":"support"}'
    )

    resp = deepseek_chat_json(
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    role = str(resp.get("primary_role", "")).strip().lower()
    if role not in ALLOWED_ROLES:
        raise ValueError(f"DeepSeek 返回非法 role: qid={qid}, unit_id={unit_id}, role={role}")

    cache_payload = {
        "qid": qid,
        "unit_id": unit_id,
        "doc_id": doc_id,
        "primary_role": role,
        "role_label_source": "llm",
        "raw_response": resp,
    }
    save_cache(cache_dir, qid, unit_id, cache_payload)
    return role, "llm"


def build_targets_record(
    sample: dict,
    *,
    api_key: str,
    base_url: str,
    model: str,
    cache_dir: Path,
) -> dict:
    required_fields = ["qid", "question", "supporting_facts", "context"]
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"processed 样本缺少字段: {field}")

    qid = str(sample["qid"])
    question = str(sample["question"]).strip()
    answer = str(sample.get("answer", "")).strip()
    if not question:
        raise ValueError(f"question 为空: qid={qid}")

    context = sample["context"]
    supporting_pairs = normalize_supporting_facts(sample["supporting_facts"], qid=qid)

    chunk_supports = {}
    for doc_id, sent_id in supporting_pairs:
        chunk_id = build_parent_chunk_id(qid, doc_id)
        text = find_sentence_text(context=context, title=doc_id, sent_id=sent_id, qid=qid)

        if chunk_id not in chunk_supports:
            chunk_supports[chunk_id] = {
                "doc_id": doc_id,
                "texts": [],
            }
        chunk_supports[chunk_id]["texts"].append(text)

    t_q_raw = []
    for chunk_id, item in chunk_supports.items():
        doc_id = item["doc_id"]
        text = " ".join(x.strip() for x in item["texts"] if str(x).strip())

        primary_role, role_label_source = classify_role_with_deepseek(
            api_key=api_key,
            base_url=base_url,
            model=model,
            cache_dir=cache_dir,
            qid=qid,
            question=question,
            answer=answer,
            unit_id=chunk_id,
            text=text,
            doc_id=doc_id,
        )

        t_q_raw.append(
            {
                "unit_id": chunk_id,
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": doc_id,
                "parent_chunk_id": chunk_id,
                "span_start": None,
                "span_end": None,
                "provenance": "raw",
                "weight": 1.0,
                "primary_role": primary_role,
                "role_label_source": role_label_source,
            }
        )

    if not t_q_raw:
        raise ValueError(f"T_q_raw 为空: qid={qid}")

    return {
        "qid": qid,
        "question": question,
        "T_q_raw": t_q_raw,
    }


def convert_split(
    split: str,
    *,
    input_base: Path,
    query_base: Path,
    output_targets_dir: Path,
    cache_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
) -> int:
    input_path = input_base / f"{split}.jsonl"
    query_path = query_base / f"{split}.jsonl"
    output_path = output_targets_dir / f"{split}.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到 processed 文件: {input_path}")

    allowed_qids = load_allowed_qids(query_path)

    def generator():
        for row_idx, sample in enumerate(read_jsonl(input_path), start=1):
            qid = str(sample.get("qid", ""))
            if qid not in allowed_qids:
                continue

            try:
                yield build_targets_record(
                    sample,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    cache_dir=cache_dir,
                )
            except Exception as e:
                raise ValueError(
                    f"重建 T_q_raw 失败: split={split}, row={row_idx}, qid={sample.get('qid', 'UNKNOWN')}, error={e}"
                ) from e

    count = write_jsonl(generator(), output_path)

    if count != len(allowed_qids):
        raise RuntimeError(
            f"targets 数量与 queries 不一致: split={split}, "
            f"queries={len(allowed_qids)}, written={count}"
        )

    return count


def main():
    project_root = Path(__file__).resolve().parent.parent

    input_base = project_root / DEFAULT_INPUT_BASE
    output_base = project_root / DEFAULT_OUTPUT_BASE
    query_base = project_root / DEFAULT_QUERY_BASE
    output_targets_dir = output_base / "targets"
    cache_dir = output_base / "llm_role_cache"

    output_targets_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"

    role_label_mode = os.environ.get("HOTPOTQA_ROLE_LABEL_MODE", "auto").strip().lower()
    if not api_key and role_label_mode != "rule":
        print("DEEPSEEK_API_KEY 未设置；targets 将使用 rule_fallback role labels。")
        os.environ["HOTPOTQA_ROLE_LABEL_MODE"] = "rule"

    stats = {}
    for split in SPLITS:
        stats[split] = convert_split(
            split,
            input_base=input_base,
            query_base=query_base,
            output_targets_dir=output_targets_dir,
            cache_dir=cache_dir,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    print("T_q_raw v2 重建完成：")
    for split in SPLITS:
        print(f"  {split}: {stats[split]} -> {output_targets_dir / f'{split}.jsonl'}")
    print(f"  role cache: {cache_dir}")


if __name__ == "__main__":
    main()
