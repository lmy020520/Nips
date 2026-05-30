import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from torch.utils.data import Dataset


ROLE_TO_ID = {
    "bridge": 0,
    "support": 1,
    "distinguish": 2,
}


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


def load_memory_map(memory_path: str) -> Dict[str, dict]:
    memory_map = {}
    path = Path(memory_path)

    for row_idx, record in enumerate(read_jsonl(path), start=1):
        if "unit_id" not in record or "text" not in record:
            raise ValueError(
                f"memory 缺少字段: file={path}, row={row_idx}, required=unit_id,text"
            )
        unit_id = str(record["unit_id"])
        if unit_id in memory_map:
            raise ValueError(f"memory 中出现重复 unit_id: file={path}, unit_id={unit_id}")

        text = str(record["text"]).strip()
        if not text:
            raise ValueError(f"memory text 为空: file={path}, unit_id={unit_id}")

        # Support both the old memory schema and the v4/v5 raw unit registry schema.
        title = str(record.get("title") or record.get("doc_id") or "")
        if not title:
            parts = unit_id.split("::")
            title = parts[-2] if len(parts) >= 2 else unit_id

        sent_id_raw = record.get("sent_id")
        if sent_id_raw is None:
            try:
                sent_id_raw = int(unit_id.rsplit("::", 1)[-1])
            except ValueError:
                sent_id_raw = 0

        memory_map[unit_id] = {
            "qid": str(record.get("qid") or unit_id.split("::", 1)[0]),
            "unit_id": unit_id,
            "title": title,
            "sent_id": int(sent_id_raw),
            "text": text,
        }

    return memory_map


def load_role_map(role_targets_path: Optional[str]) -> Dict[str, Dict[str, int]]:
    if not role_targets_path:
        return {}

    role_map: Dict[str, Dict[str, int]] = {}
    path = Path(role_targets_path)
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        qid = str(record.get("qid") or "")
        if not qid:
            raise ValueError(f"role targets 缺少 qid: file={path}, row={row_idx}")

        qid_roles = role_map.setdefault(qid, {})
        for target in record.get("T_q_raw") or []:
            if not isinstance(target, dict):
                continue
            role = str(target.get("primary_role") or "").strip()
            doc_id = str(target.get("doc_id") or "").strip()
            if role not in ROLE_TO_ID or not doc_id:
                continue
            qid_roles[doc_id] = ROLE_TO_ID[role]
    return role_map


def format_candidate_text(memory_item: dict) -> str:
    return f"{memory_item['title']} [{memory_item['sent_id']}] {memory_item['text']}"


def normalize_derived_payloads(payloads) -> Dict[str, dict]:
    if not payloads:
        return {}
    if isinstance(payloads, dict):
        return {str(k): v for k, v in payloads.items() if isinstance(v, dict)}
    if isinstance(payloads, list):
        normalized = {}
        for item in payloads:
            if isinstance(item, dict) and item.get("unit_id"):
                normalized[str(item["unit_id"])] = item
        return normalized
    return {}


def format_derived_candidate_text(unit_id: str, payload: dict) -> str:
    note_type = str(payload.get("type") or "derived_note")
    text = str(payload.get("text") or payload.get("unit_text") or "").strip()
    return f"{note_type} {unit_id}: {text}".strip()


def get_nested(record: dict, path: List[str], default=None):
    cur = record
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def get_candidate_ids(record: dict):
    candidates = record.get("candidates")
    if isinstance(candidates, list):
        return candidates
    if isinstance(candidates, dict):
        return candidates.get("C_t") or candidates.get("candidates")
    return None


def get_positive_unit_id(record: dict) -> Optional[str]:
    positive = record.get("positive_unit_id")
    if positive:
        return str(positive)
    positive = get_nested(record, ["labels", "ranking_label", "positive_unit_id"])
    if positive:
        return str(positive)
    positive = get_nested(record, ["labels", "u_t_plus", "unit_id"])
    if positive:
        return str(positive)
    return None


def get_stop_label(record: dict) -> int:
    stop_label = record.get("stop_label")
    if isinstance(stop_label, bool):
        return int(stop_label)
    if isinstance(stop_label, int):
        return stop_label
    nested_stop = get_nested(record, ["labels", "stop_label"], {})
    if isinstance(nested_stop, dict):
        return int(bool(nested_stop.get("should_stop", False)))
    return 0


def get_k_t(record: dict) -> str:
    if "K_t" in record:
        return str(record["K_t"])
    return str(get_nested(record, ["state", "K_t"], ""))


class PrefixRankingDataset(Dataset):
    def __init__(self, samples_path: str, memory_path: str, role_targets_path: Optional[str] = None):
        self.samples_path = Path(samples_path)
        self.memory_path = Path(memory_path)

        self.memory_map = load_memory_map(str(self.memory_path))
        self.role_map = load_role_map(role_targets_path)
        self.samples: List[dict] = []

        for row_idx, record in enumerate(read_jsonl(self.samples_path), start=1):
            required_fields = ["qid", "t", "question", "candidates"]
            for field in required_fields:
                if field not in record:
                    raise ValueError(
                        f"samples 缺少字段: file={self.samples_path}, row={row_idx}, field={field}"
                    )

            qid = str(record["qid"])
            t = int(record["t"])
            question = str(record["question"]).strip()
            k_t = get_k_t(record)
            candidates = get_candidate_ids(record)
            positive_unit_id = get_positive_unit_id(record)
            stop_label = get_stop_label(record)
            derived_payloads = normalize_derived_payloads(record.get("derived_payloads"))

            if not question:
                raise ValueError(f"question 为空: qid={qid}, row={row_idx}")
            if not isinstance(candidates, list) or len(candidates) == 0:
                raise ValueError(f"candidates 必须是非空 list: qid={qid}, row={row_idx}")
            if not positive_unit_id:
                raise ValueError(f"positive_unit_id 缺失: qid={qid}, row={row_idx}")

            normalized_candidates = []
            candidate_texts = []
            seen = set()
            for unit_id in candidates:
                unit_id = str(unit_id)
                if unit_id in seen:
                    raise ValueError(f"candidates 中重复 unit_id: qid={qid}, unit_id={unit_id}")
                seen.add(unit_id)

                if unit_id in self.memory_map:
                    memory_item = self.memory_map[unit_id]
                    if memory_item["qid"] != qid:
                        raise ValueError(
                            f"candidate qid 不匹配: qid={qid}, unit_id={unit_id}, memory_qid={memory_item['qid']}"
                        )
                    candidate_text = format_candidate_text(memory_item)
                elif unit_id in derived_payloads:
                    candidate_text = format_derived_candidate_text(unit_id, derived_payloads[unit_id])
                    if not candidate_text.strip():
                        raise ValueError(f"derived candidate text 为空: qid={qid}, unit_id={unit_id}")
                else:
                    raise ValueError(
                        f"candidate 不在 memory 或 derived_payloads 中: qid={qid}, unit_id={unit_id}"
                    )

                normalized_candidates.append(unit_id)
                candidate_texts.append(candidate_text)

            if positive_unit_id not in normalized_candidates:
                raise ValueError(
                    f"positive_unit_id 不在 candidates 中: qid={qid}, positive_unit_id={positive_unit_id}"
                )

            label_idx = normalized_candidates.index(positive_unit_id)
            positive_memory = self.memory_map.get(positive_unit_id)
            positive_role_id = -100
            if positive_memory:
                positive_role_id = self.role_map.get(qid, {}).get(positive_memory["title"], -100)

            self.samples.append(
                {
                    "qid": qid,
                    "t": t,
                    "question": question,
                    "K_t": k_t,
                    "candidate_unit_ids": normalized_candidates,
                    "candidate_texts": candidate_texts,
                    "label_idx": label_idx,
                    "positive_unit_id": positive_unit_id,
                    "positive_role_id": positive_role_id,
                    "stop_label": stop_label,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def prefix_ranking_collate_fn(batch: List[dict]) -> dict:
    flat_text_a: List[str] = []
    flat_text_b: List[str] = []
    candidate_counts: List[int] = []
    labels: List[int] = []

    qids: List[str] = []
    ts: List[int] = []
    candidate_unit_ids: List[List[str]] = []
    positive_unit_ids: List[str] = []
    stop_labels: List[int] = []
    positive_role_ids: List[int] = []

    for item in batch:
        qids.append(item["qid"])
        ts.append(item["t"])
        candidate_unit_ids.append(item["candidate_unit_ids"])
        positive_unit_ids.append(item["positive_unit_id"])
        stop_labels.append(item["stop_label"])
        positive_role_ids.append(item["positive_role_id"])

        context_text = f"Question: {item['question']}\nNotebook:\n{item['K_t']}"
        cand_texts = item["candidate_texts"]

        candidate_counts.append(len(cand_texts))
        labels.append(int(item["label_idx"]))

        for cand_text in cand_texts:
            flat_text_a.append(context_text)
            flat_text_b.append(cand_text)

    return {
        "qids": qids,
        "ts": ts,
        "flat_text_a": flat_text_a,
        "flat_text_b": flat_text_b,
        "candidate_counts": candidate_counts,
        "labels": labels,
        "candidate_unit_ids": candidate_unit_ids,
        "positive_unit_ids": positive_unit_ids,
        "stop_labels": stop_labels,
        "positive_role_ids": positive_role_ids,
    }
