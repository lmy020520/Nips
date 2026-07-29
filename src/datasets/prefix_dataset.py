import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from torch.utils.data import Dataset


ROLE_TO_ID = {
    "bridge": 0,
    "support": 1,
    "distinguish": 2,
}

DEFICIT_KEYS = ["d_br", "d_dis", "d_sup", "d_der"]
CONTRIBUTION_KEYS = ["c_br", "c_dis", "c_sup", "c_der"]
CONTEXT_MODES = {
    "direct_evidence_only",
    "full_state",
    "previous_evidence_only",
    "query_only",
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
            if role == "disambiguation":
                role = "distinguish"
            doc_id = str(target.get("doc_id") or "").strip()
            if role not in ROLE_TO_ID or not doc_id:
                continue
            qid_roles[doc_id] = ROLE_TO_ID[role]
    return role_map


def load_role_totals(role_targets_path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not role_targets_path:
        return {}

    totals: Dict[str, Dict[str, float]] = {}
    path = Path(role_targets_path)
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        qid = str(record.get("qid") or "")
        if not qid:
            raise ValueError(f"role targets 缺少 qid: file={path}, row={row_idx}")

        qid_totals = {"bridge": 0.0, "support": 0.0, "distinguish": 0.0}
        for target in record.get("T_q_raw") or []:
            if not isinstance(target, dict):
                continue
            role = str(target.get("primary_role") or "").strip()
            if role == "disambiguation":
                role = "distinguish"
            if role not in qid_totals:
                continue
            qid_totals[role] += float(target.get("weight", 1.0))
        totals[qid] = qid_totals
    return totals


def get_state_a_t(record: dict) -> dict:
    state = record.get("state") or {}
    a_t = state.get("A_t") or record.get("A_t") or {}
    return a_t if isinstance(a_t, dict) else {}


def get_progress_value(a_t: dict, role: str) -> float:
    aliases = {
        "bridge": ("k_bridge", "k_br"),
        "support": ("k_support", "k_sup"),
        "distinguish": ("k_distinguish", "k_dis"),
    }
    for key in aliases[role]:
        if key in a_t:
            return float(a_t.get(key) or 0.0)
    return 0.0


def compute_deficit_label(record: dict, role_totals: Dict[str, Dict[str, float]], alpha: float = 1.0) -> List[float]:
    labels = record.get("labels") or {}
    d_t_star = labels.get("d_t_star") or record.get("d_t_star")
    if isinstance(d_t_star, dict):
        values = []
        for key in DEFICIT_KEYS:
            value = d_t_star.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                break
            values.append(max(0.0, min(1.0, float(value))))
        if len(values) == len(DEFICIT_KEYS):
            return values

    # Backward-compatible fallback for datasets without explicit teacher d_t*.
    qid = str(record.get("qid") or "")
    totals = role_totals.get(qid)
    if not totals:
        return [-100.0, -100.0, -100.0, -100.0]

    a_t = get_state_a_t(record)
    labels = []
    for role in ("bridge", "distinguish", "support"):
        total = float(totals.get(role, 0.0))
        if total <= 0.0:
            labels.append(0.0)
            continue
        progress = get_progress_value(a_t, role)
        sufficiency = (progress + alpha) / (total + 2.0 * alpha)
        labels.append(max(0.0, min(1.0, 1.0 - sufficiency)))

    # Derived evidence is optional in the current HotpotQA setting. Keep the
    # fourth dimension for API compatibility with the KBS formulation.
    labels.append(0.0)
    return labels


def get_positive_contribution_label(record: dict) -> List[float]:
    c_t_star = get_nested(record, ["labels", "c_t_star"], {})
    if not isinstance(c_t_star, dict):
        return [-100.0, -100.0, -100.0, -100.0]
    labels = []
    for key in CONTRIBUTION_KEYS:
        value = c_t_star.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return [-100.0, -100.0, -100.0, -100.0]
        labels.append(max(0.0, min(1.0, float(value))))
    return labels


def format_candidate_text(memory_item: dict) -> str:
    return f"{memory_item['title']} [{memory_item['sent_id']}] {memory_item['text']}"


def format_notebook_evidence(memory_item: dict, index: int = 1) -> str:
    return f"[{index}] {memory_item['title']}: {memory_item['text']}"


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


def get_state_h_ids(record: dict) -> List[str]:
    h_t = get_nested(record, ["state", "H_t"], [])
    if not isinstance(h_t, list):
        return []

    unit_ids = []
    for item in h_t:
        if isinstance(item, dict) and item.get("unit_id"):
            unit_ids.append(str(item["unit_id"]))
        elif isinstance(item, str):
            unit_ids.append(item)
    return unit_ids


def build_context_text(
    question: str,
    k_t: str,
    record: dict,
    memory_map: Dict[str, dict],
    context_mode: str,
) -> tuple[str, Optional[str]]:
    if context_mode == "query_only":
        return f"Question: {question}", None

    if context_mode == "previous_evidence_only":
        h_ids = get_state_h_ids(record)
        anchor_unit_id = h_ids[-1] if h_ids else None
        anchor_item = memory_map.get(anchor_unit_id or "")
        if int(record.get("t", 0)) > 0 and anchor_item is None:
            raise ValueError(
                "previous_evidence_only requires the last H_t unit in memory: "
                f"qid={record.get('qid')}, t={record.get('t')}, unit_id={anchor_unit_id}"
            )
        notebook = format_notebook_evidence(anchor_item) if anchor_item else ""
        return f"Question: {question}\nNotebook:\n{notebook}", anchor_unit_id

    if context_mode == "direct_evidence_only":
        h_ids = get_state_h_ids(record)
        anchor_unit_id = h_ids[0] if h_ids else None
        anchor_item = memory_map.get(anchor_unit_id or "")
        if int(record.get("t", 0)) > 0 and anchor_item is None:
            raise ValueError(
                "direct_evidence_only requires the first H_t unit in memory: "
                f"qid={record.get('qid')}, t={record.get('t')}, unit_id={anchor_unit_id}"
            )
        notebook = format_notebook_evidence(anchor_item) if anchor_item else ""
        return f"Question: {question}\nNotebook:\n{notebook}", anchor_unit_id

    return f"Question: {question}\nNotebook:\n{k_t}", None


class PrefixRankingDataset(Dataset):
    def __init__(
        self,
        samples_path: str,
        memory_path: str,
        role_targets_path: Optional[str] = None,
        require_labeled_positive: bool = False,
        context_mode: str = "full_state",
    ):
        self.samples_path = Path(samples_path)
        self.memory_path = Path(memory_path)
        self.context_mode = str(context_mode)
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(
                f"unsupported context_mode={self.context_mode}; "
                f"expected one of {sorted(CONTEXT_MODES)}"
            )

        self.memory_map = load_memory_map(str(self.memory_path))
        self.role_map = load_role_map(role_targets_path)
        self.role_totals = load_role_totals(role_targets_path)
        self.samples: List[dict] = []
        self.skipped_unlabeled_positive = 0

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
            deficit_label = compute_deficit_label(record, self.role_totals)
            positive_contribution_label = get_positive_contribution_label(record)
            if bool((record.get("build_meta") or {}).get("mask_auxiliary_labels", False)):
                deficit_label = [-100.0, -100.0, -100.0, -100.0]
                positive_contribution_label = [-100.0, -100.0, -100.0, -100.0]
            derived_payloads = normalize_derived_payloads(record.get("derived_payloads"))

            if not question:
                raise ValueError(f"question 为空: qid={qid}, row={row_idx}")
            if not isinstance(candidates, list) or len(candidates) == 0:
                raise ValueError(f"candidates 必须是非空 list: qid={qid}, row={row_idx}")
            if not positive_unit_id:
                raise ValueError(f"positive_unit_id 缺失: qid={qid}, row={row_idx}")
            context_text, context_anchor_unit_id = build_context_text(
                question=question,
                k_t=k_t,
                record=record,
                memory_map=self.memory_map,
                context_mode=self.context_mode,
            )

            normalized_candidates = []
            candidate_texts = []
            candidate_role_ids = []
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
                    candidate_role_id = self.role_map.get(qid, {}).get(memory_item["title"], -100)
                elif unit_id in derived_payloads:
                    candidate_text = format_derived_candidate_text(unit_id, derived_payloads[unit_id])
                    if not candidate_text.strip():
                        raise ValueError(f"derived candidate text 为空: qid={qid}, unit_id={unit_id}")
                    candidate_role_id = -100
                else:
                    raise ValueError(
                        f"candidate 不在 memory 或 derived_payloads 中: qid={qid}, unit_id={unit_id}"
                    )

                normalized_candidates.append(unit_id)
                candidate_texts.append(candidate_text)
                candidate_role_ids.append(candidate_role_id)

            if positive_unit_id not in normalized_candidates:
                raise ValueError(
                    f"positive_unit_id 不在 candidates 中: qid={qid}, positive_unit_id={positive_unit_id}"
                )

            label_idx = normalized_candidates.index(positive_unit_id)
            positive_memory = self.memory_map.get(positive_unit_id)
            positive_role_id = -100
            if positive_memory:
                positive_role_id = self.role_map.get(qid, {}).get(positive_memory["title"], -100)
            if require_labeled_positive and positive_role_id == -100:
                self.skipped_unlabeled_positive += 1
                continue

            self.samples.append(
                {
                    "qid": qid,
                    "t": t,
                    "question": question,
                    "K_t": k_t,
                    "context_text": context_text,
                    "context_mode": self.context_mode,
                    "context_anchor_unit_id": context_anchor_unit_id,
                    "candidate_unit_ids": normalized_candidates,
                    "candidate_texts": candidate_texts,
                    "candidate_role_ids": candidate_role_ids,
                    "label_idx": label_idx,
                    "positive_unit_id": positive_unit_id,
                    "positive_role_id": positive_role_id,
                    "stop_label": stop_label,
                    "deficit_label": deficit_label,
                    "positive_contribution_label": positive_contribution_label,
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
    deficit_labels: List[List[float]] = []
    positive_contribution_labels: List[List[float]] = []
    flat_candidate_role_ids: List[int] = []

    for item in batch:
        qids.append(item["qid"])
        ts.append(item["t"])
        candidate_unit_ids.append(item["candidate_unit_ids"])
        positive_unit_ids.append(item["positive_unit_id"])
        stop_labels.append(item["stop_label"])
        positive_role_ids.append(item["positive_role_id"])
        deficit_labels.append(item["deficit_label"])
        positive_contribution_labels.append(item["positive_contribution_label"])

        context_text = item["context_text"]
        cand_texts = item["candidate_texts"]

        candidate_counts.append(len(cand_texts))
        labels.append(int(item["label_idx"]))

        for cand_text in cand_texts:
            flat_text_a.append(context_text)
            flat_text_b.append(cand_text)
        flat_candidate_role_ids.extend(item["candidate_role_ids"])

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
        "deficit_labels": deficit_labels,
        "positive_contribution_labels": positive_contribution_labels,
        "flat_candidate_role_ids": flat_candidate_role_ids,
    }
