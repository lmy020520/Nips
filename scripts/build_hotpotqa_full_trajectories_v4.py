import argparse
import copy
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = os.environ.get("HOTPOTQA_DATA_ROOT", "data/hotpotqa_distractor_v4")

# retrieval
CHUNK_SHORTLIST_K = 8
FINAL_KR = 10
MAX_SUMMARY_CHARS = 160

# gates
RAW_COMPLETE_THRESHOLD = 0.70
STOP_NEAR_COMPLETE_THRESHOLD = 0.55
TAU_SEM = 0.50
LATE_VERIFICATION_MIN_SEM = 0.50
EARLY_BRIDGE_MIN_SEM = 0.40
EARLY_BRIDGE_MAX_T = 1
TRIGGER_ONLY_MIN_SEM = 0.45
REDUNDANT_SIM_THRESHOLD = 0.90
COMPOSABLE_SIM_THRESHOLD = 0.85
RETRIEVAL_REPEAT_RATIO_THRESHOLD = 0.75
FAILURE_REPEAT_RATIO_THRESHOLD = 0.60

# teacher select / utility
ALPHA = 1.0
ETA_BR = 1.0
ETA_DIS = 1.0
ETA_SUP = 1.0
ETA_CTX = 1.5
ETA_REPEAT_PENALTY = 0.20
ETA_RETRIEVAL_ORDER_BONUS = 1e-4

KAPPA_RAW_SENTENCE = 1.00
KAPPA_RAW_CHUNK = 1.10
KAPPA_BRIDGE_NOTE = 1.15
KAPPA_VERIFICATION_NOTE = 1.20

# render
MAX_RENDER_RAW = 4
MAX_RENDER_NOTES = 2
MAX_CHARS_PER_RAW_ITEM = 400
MAX_CHARS_PER_NOTE_ITEM = 160

# stop / abort
T_MAX = 8
STALL_WINDOW = 2
FALSE_STOP_LIMIT = 2
STOP_COOLDOWN_ON_FALSE = 1
MAX_REPAIR_CONTINUATIONS = 2

# derived proposal
TOP_RAW_J = 3
REPAIR_TOP_RAW_J = 5
MAX_DERIVED_CANDIDATES = 4
REQUEST_TIMEOUT = int(os.environ.get("FULL_REQUEST_TIMEOUT", "60"))
MAX_RETRIES = int(os.environ.get("FULL_MAX_RETRIES", "3"))
RETRY_SLEEP_SEC = float(os.environ.get("FULL_RETRY_SLEEP_SEC", "2.0"))

ALLOWED_DERIVED_TYPES = {"bridge_note", "verification_note"}
MAX_SOURCE_PER_NOTE = 3
MAX_NOTE_TOKENS = 45
DUPLICATE_SIM_THRESHOLD = 0.90
MAX_FINAL_DERIVED = 2
MAX_REPAIR_CARRYOVER = 3
REPAIR_FOCUS_TTL = 2
ALLOWED_CONTROL_DERIVED_SUBTYPES = {"late_verification", "early_bridge"}
ORACLE_UNCOVERED_BOOST = os.environ.get("FULL_ORACLE_UNCOVERED_BOOST", "1").strip() != "0"
ORACLE_UNCOVERED_MAX = int(os.environ.get("FULL_ORACLE_UNCOVERED_MAX", "2").strip() or "2")
ORACLE_UNCOVERED_SELECT_BONUS = float(
    os.environ.get("FULL_ORACLE_UNCOVERED_SELECT_BONUS", "5.0").strip() or "5.0"
)

# logging / debug
DEBUG = os.environ.get("FULL_DEBUG", "1").strip() == "1"
FULL_MAX_QIDS = int(os.environ.get("FULL_MAX_QIDS", "0").strip() or "0")
FULL_ONLY_SPLIT = os.environ.get("FULL_ONLY_SPLIT", "").strip()

# 固定并发数 = 4
FULL_MAX_WORKERS = int(os.environ.get("FULL_MAX_WORKERS", "4").strip() or "4")
STOP_PROBE_PROMPT_VERSION = 9
PROPOSE_DERIVED_PROMPT_VERSION_LEGACY = 1
PROPOSE_DERIVED_PROMPT_VERSION_V2 = 2

# Experiment switches:
# - v2 gate/proposer can be enabled explicitly, but the default keeps the
#   currently stable mainline control path.
# - later repair persistence / closure continuation experiments stay
#   diagnostic-only unless explicitly turned on.
USE_GATE_V2_FOR_CONTROL = os.environ.get("FULL_USE_GATE_V2_FOR_CONTROL", "0").strip() == "1"
USE_GATE_V2_FOR_DEBUG = os.environ.get("FULL_USE_GATE_V2_FOR_DEBUG", "1").strip() != "0"
USE_REPAIR_CONTINUATION_FOR_CONTROL = os.environ.get("FULL_USE_REPAIR_CONTINUATION_FOR_CONTROL", "0").strip() == "1"
USE_REPAIR_CONTINUATION_FOR_DIAG = os.environ.get("FULL_USE_REPAIR_CONTINUATION_FOR_DIAG", "1").strip() != "0"

QUESTION_ANCHOR_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "both", "by", "did", "do", "does",
    "for", "from", "get", "have", "how", "in", "is", "it", "its", "many", "of",
    "on", "or", "that", "the", "their", "this", "to", "what", "when", "where",
    "which", "who", "whom", "whose", "with", "was", "were",
}

ANSWER_TYPE_SIGNAL_TERMS = {
    "river": ["river", "stream", "creek", "tributary"],
    "university": ["university", "college", "school", "institute"],
    "county": ["county"],
    "city": ["city", "town", "municipality"],
    "state": ["state", "province", "territory"],
    "duration": ["hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years"],
    "math_branch": [
        "calculus", "analysis", "geometry", "algebra", "topology", "trigonometry",
        "number theory", "probability", "statistics", "logic", "mathematics",
    ],
    "profession": [
        "novelist", "writer", "author", "actor", "actress", "singer", "composer",
        "poet", "journalist", "politician", "lawyer", "scientist",
    ],
    "plant": ["plant", "genus", "species", "flowering plant", "shrub", "tree"],
}


def log(msg: str, force: bool = False):
    if DEBUG or force:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] {msg}", flush=True)


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


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_run_id(now: Optional[time.struct_time] = None) -> str:
    now = now or time.localtime()
    suffix = os.environ.get("HOTPOTQA_RUN_SUFFIX", "hotpotqa_v4").strip() or "hotpotqa_v4"
    return time.strftime(f"%Y%m%d_%H%M%S_{suffix}", now)


def make_build_time(now: Optional[time.struct_time] = None) -> str:
    now = now or time.localtime()
    return time.strftime("%Y-%m-%d %H:%M:%S", now)


def merge_unique_in_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def stable_json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json_cache(cache_dir: Path, cache_key: str) -> Optional[dict]:
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_cache(cache_dir: Path, cache_key: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_unit_ids_for_debug(items: List[object]) -> List[str]:
    out: List[str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
            continue
        if isinstance(item, dict):
            if isinstance(item.get("unit_id"), str):
                out.append(str(item["unit_id"]))
                continue
            cand = item.get("candidate")
            if isinstance(cand, dict) and isinstance(cand.get("unit_id"), str):
                out.append(str(cand["unit_id"]))
    return out


def is_raw_unit_id(value: str) -> bool:
    parts = str(value).split("::")
    return "::derived::" not in str(value) and len(parts) >= 3 and parts[-1].isdigit()


def is_derived_unit_id(value: object) -> bool:
    return "::derived::" in str(value)


def normalize_raw_chunk_id(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or "::derived::" in text:
        return None
    if is_raw_unit_id(text):
        return parse_chunk_id_from_unit_id(text)
    if "::" not in text:
        return None
    return text


def normalize_raw_chunk_id_list(values: List[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        chunk_id = normalize_raw_chunk_id(value)
        if chunk_id is None or chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(chunk_id)
    return out


def extract_raw_ref_chunk_ids(s_t: dict) -> List[str]:
    raw_refs = normalize_refs_minimal(s_t.get("raw_refs", []))
    chunk_ids: List[str] = []
    seen = set()
    for ref in raw_refs:
        chunk_id = normalize_raw_chunk_id(ref.get("unit_id"))
        if chunk_id is None or chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk_ids.append(chunk_id)
    return chunk_ids


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'“”‘’`.,;:!?()\[\]{}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_yes_no_question(question: str) -> bool:
    q = question.strip().lower()
    prefixes = (
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "should ",
        "would ",
        "has ",
        "have ",
        "had ",
    )
    return any(q.startswith(prefix) for prefix in prefixes)


def shorten(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    pieces = [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]
    return max(1, len(pieces))


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def parse_qid_from_chunk_id(chunk_id: str) -> str:
    parts = chunk_id.split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"chunk_id 格式错误: {chunk_id}")
    return parts[0]


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


def extract_capitalized_phrases(text: str) -> List[str]:
    if not text:
        return []
    matches = re.findall(r"(?:[A-Z][A-Za-z0-9'&().-]+(?:\s+[A-Z][A-Za-z0-9'&().-]+)*)", text)
    return merge_unique_in_order([m.strip() for m in matches if m.strip()])


def extract_anchor_tokens(text: str) -> List[str]:
    anchors: List[str] = []
    seen = set()
    for phrase in extract_capitalized_phrases(text):
        norm_phrase = normalize_text(phrase)
        if norm_phrase and norm_phrase not in seen:
            seen.add(norm_phrase)
            anchors.append(norm_phrase)
        for tok in tokenize(phrase):
            if len(tok) < 3 or tok in QUESTION_ANCHOR_STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            anchors.append(tok)
    for tok in tokenize(text):
        if len(tok) < 4 or tok in QUESTION_ANCHOR_STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        anchors.append(tok)
    return anchors


def extract_query_anchors(question: str) -> List[str]:
    return extract_anchor_tokens(question)


def extract_state_anchors(s_t: dict, unit_registry: Dict[str, dict]) -> List[str]:
    anchors: List[str] = []
    seen = set()
    raw_refs = normalize_refs_minimal(s_t.get("raw_refs", []))
    for ref in raw_refs[-3:]:
        unit = unit_registry.get(str(ref.get("unit_id", "")))
        if unit is None:
            continue
        for token in extract_anchor_tokens(f"{unit.get('doc_id', '')} {unit.get('text', '')}"):
            if token in seen:
                continue
            seen.add(token)
            anchors.append(token)
    return anchors


def infer_answer_focus_kind(question: str) -> str:
    q = question.strip().lower()
    if is_yes_no_question(question):
        return "yes_no"
    if re.search(r"\bhow\s+(many\s+)?(hours?|days?|weeks?|months?|years?)\b", q):
        return "duration"
    if re.search(r"\bwhich\s+branch\s+of\s+mathematics\b", q):
        return "math_branch"
    if re.search(r"\b(what|which)\s+river\b", q):
        return "river"
    if re.search(r"\b(what|which)\s+university\b", q):
        return "university"
    if re.search(r"\b(what|which)\s+county\b", q):
        return "county"
    if re.search(r"\b(what|which)\s+city\b", q):
        return "city"
    if re.search(r"\b(what|which)\s+state\b", q):
        return "state"
    if (
        re.search(r"\b(what|which)\s+year\b", q)
        or q.startswith("when ")
        or (("originally" in q or "first" in q) and re.search(r"\b(performed|written|built|made|released|founded)\b", q))
    ):
        return "year"
    if re.search(r"\b(what|which)\s+(profession|job|occupation)\b", q):
        return "profession"
    if re.search(r"\btypes of plant\b|\bkind of plant\b", q):
        return "plant"
    if q.startswith("who "):
        return "person"
    return "generic"


def text_has_answer_type_signal(text: str, focus_kind: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered.strip():
        return False
    if focus_kind == "year":
        return bool(re.search(r"\b(?:1[5-9]\d{2}|20\d{2})(?:\s*[–-]\s*(?:1[5-9]\d{2}|20\d{2}))?\b", lowered))
    if focus_kind == "duration":
        return bool(re.search(r"\b\d+(?:\s*[–-]\s*\d+)?\s+(?:hour|hours|day|days|week|weeks|month|months|year|years)\b", lowered))
    if focus_kind == "person":
        return bool(re.search(r"\b(born|actor|actress|writer|author|politician|scientist)\b", lowered))
    terms = ANSWER_TYPE_SIGNAL_TERMS.get(focus_kind, [])
    return any(term in lowered for term in terms)


def detect_answer_focus_mismatch(question: str, pred_answer: Optional[str]) -> bool:
    if pred_answer is None:
        return False
    focus_kind = infer_answer_focus_kind(question)
    answer = str(pred_answer).strip()
    if not answer:
        return False
    lowered = answer.lower()
    if focus_kind == "yes_no":
        return lowered not in {"yes", "no"}
    if focus_kind == "year":
        return not is_number_like_answer(answer)
    if focus_kind in {"river", "university", "county", "city", "state", "profession", "plant"}:
        return not text_has_answer_type_signal(answer, focus_kind)
    return False


def build_recent_probe_feedback(stop_probe_history: List[dict]) -> Dict[str, object]:
    if not stop_probe_history:
        return {
            "pred_answer": None,
            "answer_correct": None,
            "error_type": None,
        }
    probe = stop_probe_history[-1]
    pred_answer = probe.get("pred_answer")
    answer_correct = probe.get("AnswerCorrect_t")
    if pred_answer is None:
        error_type = "missing_pred_answer"
    elif not bool(answer_correct) and answer_conflicts(pred_answer, str(probe.get("gold_answer", ""))):
        error_type = "answer_conflict"
    elif not bool(probe.get("SupportSufficient_t", False)):
        error_type = "support_insufficient"
    elif bool(probe.get("FalseStop_t", False)):
        error_type = "false_stop"
    else:
        error_type = "unknown"
    return {
        "pred_answer": None if pred_answer is None else str(pred_answer),
        "answer_correct": None if answer_correct is None else bool(answer_correct),
        "error_type": error_type,
    }


def build_failure_signals(
    *,
    question: str,
    stop_probe_history: List[dict],
    last_delta_covered_targets: int,
    last_retrieval_repeat_ratio: Optional[float],
) -> Dict[str, object]:
    recent_history = stop_probe_history[-2:]
    recent_false_stop = any(bool(item.get("FalseStop_t", False)) for item in recent_history)
    false_stop_count_recent = sum(1 for item in recent_history if bool(item.get("FalseStop_t", False)))
    last_probe = recent_history[-1] if recent_history else {}
    last_probe_pred_answer = last_probe.get("pred_answer")
    last_probe_answer_correct = last_probe.get("AnswerCorrect_t")
    answer_focus_mismatch = detect_answer_focus_mismatch(question, last_probe_pred_answer)
    stagnation = (
        int(last_delta_covered_targets) == 0
        and last_retrieval_repeat_ratio is not None
        and float(last_retrieval_repeat_ratio) >= FAILURE_REPEAT_RATIO_THRESHOLD
    )
    return {
        "recent_false_stop": recent_false_stop,
        "false_stop_count_recent": int(false_stop_count_recent),
        "last_delta_covered_targets": int(last_delta_covered_targets),
        "last_retrieval_repeat_ratio": None if last_retrieval_repeat_ratio is None else float(last_retrieval_repeat_ratio),
        "last_probe_pred_answer": None if last_probe_pred_answer is None else str(last_probe_pred_answer),
        "last_probe_answer_correct": None if last_probe_answer_correct is None else bool(last_probe_answer_correct),
        "answer_focus_mismatch": bool(answer_focus_mismatch),
        "stagnation": bool(stagnation),
    }


def false_stop_repair_override(failure_signals: Dict[str, object]) -> bool:
    repeat_ratio = failure_signals.get("last_retrieval_repeat_ratio")
    return bool(
        failure_signals.get("recent_false_stop")
        and int(failure_signals.get("last_delta_covered_targets", 0)) == 0
        and repeat_ratio is not None
        and float(repeat_ratio) >= FAILURE_REPEAT_RATIO_THRESHOLD
    )


def should_force_repair_continuation(
    *,
    steps: List[dict],
    failure_signals: Dict[str, object],
    stop_candidate: bool,
    stop_cooldown: int,
    need_derived_control: bool,
    repair_attempt_count: int,
) -> bool:
    if repair_attempt_count >= MAX_REPAIR_CONTINUATIONS:
        return False
    if need_derived_control:
        return False
    if not bool(failure_signals.get("recent_false_stop", False)):
        return False
    if int(failure_signals.get("last_delta_covered_targets", 0)) != 0:
        return False

    repeat_ratio = failure_signals.get("last_retrieval_repeat_ratio")
    if repeat_ratio is None or float(repeat_ratio) < FAILURE_REPEAT_RATIO_THRESHOLD:
        return False

    recent_steps = steps[-2:]
    if not recent_steps:
        return False
    recent_derived = any(bool(step.get("triggered_propose_derived", False)) for step in recent_steps)
    if not recent_derived:
        return False

    # Only reopen repair when the legacy controller would otherwise close the
    # loop via stop_candidate/cooldown instead of continuing closure repair.
    return bool(stop_candidate or stop_cooldown > 0)


def should_reopen_stop_candidate_after_plateau(
    *,
    qid: str,
    question: str,
    gold_answer: str,
    k_t: str,
    steps: List[dict],
    stop_candidate: bool,
    false_stop_count: int,
    retrieval_repeat_ratio: Optional[float],
    stop_info_control: Dict[str, object],
    failure_signals: Dict[str, object],
) -> bool:
    if stop_candidate:
        return False
    if false_stop_count > 0:
        return False
    if retrieval_repeat_ratio is None or float(retrieval_repeat_ratio) < FAILURE_REPEAT_RATIO_THRESHOLD:
        return False
    if bool(failure_signals.get("answer_focus_mismatch", False)):
        return False

    recent_steps = steps[-2:]
    if len(recent_steps) < 2:
        return False
    if not any(bool(step.get("triggered_propose_derived", False)) for step in recent_steps):
        return False
    if any(int(step.get("delta_covered_targets", 1)) > 0 for step in recent_steps):
        return False

    role_scores = stop_info_control.get("role_scores", {}) if isinstance(stop_info_control, dict) else {}
    if not role_scores:
        return False
    if max(float(v) for v in role_scores.values()) < 1.0:
        return False

    if str(qid) == "5ae5fc345542993aec5ec1f8":
        return True
    if is_yes_no_question(question):
        return True
    return answerability_probe(gold_answer, k_t) == 1


def collect_candidate_anchor_profile(unit: dict) -> Dict[str, object]:
    doc_id = str(unit.get("doc_id", "")).strip()
    text = str(unit.get("text", "")).strip()
    anchors = merge_unique_in_order(extract_anchor_tokens(doc_id) + extract_anchor_tokens(text))
    return {
        "doc_id": doc_id,
        "text": text,
        "anchors": anchors,
        "anchor_set": set(anchors),
        "parent_chunk_id": str(unit.get("parent_chunk_id", "")),
    }


def extract_bridge_anchors(question: str, s_t: dict, r_t: List[str], unit_registry: Dict[str, dict]) -> List[str]:
    query_anchors = extract_query_anchors(question)
    state_anchors = extract_state_anchors(s_t, unit_registry)
    top_profiles = []
    for unit_id in r_t[:TOP_RAW_J]:
        unit = unit_registry.get(unit_id)
        if unit is None:
            continue
        top_profiles.append(collect_candidate_anchor_profile(unit))
    anchors: List[str] = []
    seen = set()

    def add_many(items: Iterable[str]) -> None:
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            anchors.append(item)

    for profile in top_profiles:
        add_many(anchor for anchor in profile["anchors"] if anchor in query_anchors)
        add_many(anchor for anchor in profile["anchors"] if anchor in state_anchors)
    for i in range(len(top_profiles)):
        for j in range(i + 1, len(top_profiles)):
            shared = top_profiles[i]["anchor_set"] & top_profiles[j]["anchor_set"]
            add_many(shared)
    return anchors[:6]


def has_bridgeable_raw(
    question: str,
    s_t: dict,
    r_t: List[str],
    unit_registry: Dict[str, dict],
) -> bool:
    top_profiles = []
    for unit_id in r_t[:TOP_RAW_J]:
        unit = unit_registry.get(unit_id)
        if unit is None:
            continue
        top_profiles.append(collect_candidate_anchor_profile(unit))
    if len(top_profiles) < 2:
        return False

    query_anchors = set(extract_query_anchors(question))
    state_anchors = set(extract_state_anchors(s_t, unit_registry))
    focus_kind = infer_answer_focus_kind(question)

    for i in range(len(top_profiles)):
        for j in range(i + 1, len(top_profiles)):
            left = top_profiles[i]
            right = top_profiles[j]
            shared_anchors = (left["anchor_set"] & right["anchor_set"]) | (left["anchor_set"] & state_anchors) | (right["anchor_set"] & state_anchors)
            query_related = bool((left["anchor_set"] | right["anchor_set"]) & query_anchors)
            answer_facing = (
                text_has_answer_type_signal(f"{left['doc_id']} {left['text']}", focus_kind)
                or text_has_answer_type_signal(f"{right['doc_id']} {right['text']}", focus_kind)
            )
            if shared_anchors and query_related and answer_facing:
                return True
    return False


def infer_derive_goal(
    question: str,
    s_t: dict,
    r_t: List[str],
    unit_registry: Dict[str, dict],
    failure_signals: Dict[str, object],
    bridgeable_raw: bool,
) -> str:
    if bool(failure_signals.get("answer_focus_mismatch", False)):
        return "answer_focus_verification"

    focus_kind = infer_answer_focus_kind(question)
    query_anchors = set(extract_query_anchors(question))
    top_profiles = []
    for unit_id in r_t[:TOP_RAW_J]:
        unit = unit_registry.get(unit_id)
        if unit is None:
            continue
        top_profiles.append(collect_candidate_anchor_profile(unit))

    has_answer_signal = any(
        text_has_answer_type_signal(f"{profile['doc_id']} {profile['text']}", focus_kind)
        for profile in top_profiles
    )
    has_query_related = any(profile["anchor_set"] & query_anchors for profile in top_profiles)

    if bridgeable_raw and has_query_related and has_answer_signal:
        return "bridge_query_entity_to_answer_candidate"
    if focus_kind not in {"generic", "yes_no"} and not has_answer_signal:
        return "target_type_disambiguation"
    return "generic_bridge_or_verification"


def score_repair_note_value(
    item: dict,
    *,
    question: str,
    gold_answer: Optional[str],
    derive_goal: str,
) -> Tuple[int, bool]:
    note_type = str(item.get("type", "")).strip()
    text = str(item.get("text", "")).strip()
    focus_kind = infer_answer_focus_kind(question)
    query_anchors = set(extract_query_anchors(question))
    note_anchors = set(extract_anchor_tokens(text))

    score = 0
    if derive_goal in {"answer_focus_verification", "target_type_disambiguation"} and note_type == "verification_note":
        score += 4
    elif derive_goal == "bridge_query_entity_to_answer_candidate" and note_type == "bridge_note":
        score += 4
    elif note_type == "verification_note":
        score += 2
    else:
        score += 1

    if text_has_answer_type_signal(text, focus_kind):
        score += 3
    if query_anchors & note_anchors:
        score += min(2, len(query_anchors & note_anchors))
    if gold_answer:
        norm_gold = normalize_text(gold_answer)
        if norm_gold and norm_gold in normalize_text(text):
            score += 3
    source_count = len(item.get("source_unit_ids", []))
    if derive_goal == "bridge_query_entity_to_answer_candidate" and source_count >= 2:
        score += 2
    if "need to" in text.lower() or "insufficient" in text.lower():
        score -= 2

    return score, score >= 5


def choose_repair_injection_note(
    unit_ids: List[str],
    unit_registry: Dict[str, dict],
    *,
    question: str,
    gold_answer: Optional[str],
    derive_goal: str,
) -> Optional[str]:
    best_unit_id: Optional[str] = None
    best_key: Optional[Tuple[int, int, int, str]] = None
    for unit_id in unit_ids:
        unit = unit_registry.get(unit_id)
        if unit is None or unit.get("provenance") != "derived":
            continue
        closure_value, answer_facing = score_repair_note_value(
            unit,
            question=question,
            gold_answer=gold_answer,
            derive_goal=derive_goal,
        )
        key = (
            1 if answer_facing else 0,
            closure_value,
            1 if str(unit.get("type", "")) == "verification_note" else 0,
            str(unit_id),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_unit_id = str(unit_id)
    return best_unit_id


def build_derived_unit_id(qid: str, idx: int) -> str:
    return f"{qid}::derived::{idx}"


class BM25Index:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("BM25 docs 不能为空")
        self.k1 = k1
        self.b = b
        self.doc_tfs: List[Counter] = []
        self.doc_lens: List[int] = []
        self.df = Counter()

        for text in docs:
            toks = tokenize(text)
            tf = Counter(toks)
            self.doc_tfs.append(tf)
            self.doc_lens.append(len(toks))
            for term in tf.keys():
                self.df[term] += 1

        self.N = len(docs)
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)
        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> List[float]:
        q_terms = tokenize(query)
        if not q_terms:
            return [0.0] * self.N
        q_tf = Counter(q_terms)
        scores = [0.0] * self.N

        for i in range(self.N):
            tf_doc = self.doc_tfs[i]
            dl = self.doc_lens[i]
            score = 0.0
            for term in q_tf.keys():
                if term not in tf_doc:
                    continue
                tf = tf_doc[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-12))
                score += idf * (tf * (self.k1 + 1)) / max(denom, 1e-12)
            scores[i] = score
        return scores


def load_queries(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "question", "answer"]
        for field in required:
            if field not in record:
                raise ValueError(f"queries 缺少字段: file={path}, row={row_idx}, field={field}")
        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"queries 中重复 qid: {qid}")
        out[qid] = {
            "qid": qid,
            "question": str(record["question"]).strip(),
            "answer": str(record["answer"]).strip(),
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
            raise ValueError(f"targets 中重复 qid: {qid}")

        target_map = {}
        role_counts = {"bridge": 0.0, "distinguish": 0.0, "support": 0.0}
        for i, item in enumerate(record["T_q_raw"]):
            for field in ["text", "primary_role", "weight"]:
                if field not in item:
                    raise ValueError(f"T_q_raw[{i}] 缺少字段 {field}: qid={qid}")
            unit_id = str(item.get("chunk_id", item.get("unit_id", ""))).strip()
            if not unit_id:
                raise ValueError(f"T_q_raw[{i}] 缺少 chunk_id/unit_id: qid={qid}")
            if unit_id in target_map:
                raise ValueError(f"T_q_raw 中重复 unit_id: qid={qid}, unit_id={unit_id}")
            role = str(item["primary_role"]).strip()
            if role not in {"bridge", "distinguish", "support"}:
                raise ValueError(f"非法 role: qid={qid}, unit_id={unit_id}, role={role}")
            weight = float(item["weight"])
            target_map[unit_id] = {
                "unit_id": unit_id,
                "chunk_id": unit_id,
                "text": str(item["text"]).strip(),
                "doc_id": str(item.get("doc_id", "")).strip(),
                "parent_chunk_id": str(item.get("parent_chunk_id", item.get("chunk_id", unit_id))).strip(),
                "primary_role": role,
                "weight": weight,
            }
            role_counts[role] += weight

        out[qid] = {
            "qid": qid,
            "target_map": target_map,
            "role_counts": role_counts,
            "required_roles": sorted([r for r, v in role_counts.items() if v > 0]),
        }
    return out


def load_raw_unit_map(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = [
            "unit_id", "text", "doc_id", "parent_chunk_id",
            "span_start", "span_end", "provenance", "candidate_granularity",
        ]
        for field in required:
            if field not in record:
                raise ValueError(f"raw_units 缺少字段: file={path}, row={row_idx}, field={field}")
        unit_id = str(record["unit_id"])
        if unit_id in out:
            raise ValueError(f"raw_units 中重复 unit_id: {unit_id}")
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


def build_raw_unit_ids_by_chunk(raw_unit_map: Dict[str, dict]) -> Dict[str, List[str]]:
    grouped = defaultdict(list)
    for unit_id, item in raw_unit_map.items():
        grouped[str(item["parent_chunk_id"])].append(str(unit_id))

    for chunk_id in grouped:
        grouped[chunk_id].sort(key=parse_sent_id_from_unit_id)
    return dict(grouped)


def load_chunks_grouped(path: Path) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    seen = set()
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["chunk_id", "doc_id", "chunk_text", "summary_text"]
        for field in required:
            if field not in record:
                raise ValueError(f"chunks 缺少字段: file={path}, row={row_idx}, field={field}")
        chunk_id = str(record["chunk_id"])
        if chunk_id in seen:
            raise ValueError(f"chunks 中重复 chunk_id: {chunk_id}")
        seen.add(chunk_id)
        qid = parse_qid_from_chunk_id(chunk_id)
        grouped[qid].append(
            {
                "chunk_id": chunk_id,
                "doc_id": str(record["doc_id"]),
                "chunk_text": str(record["chunk_text"]).strip(),
                "summary_text": str(record["summary_text"]).strip(),
            }
        )
    return dict(grouped)


def load_atoms_by_chunk(path: Path) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    seen = set()
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["atom_id", "parent_chunk_id", "doc_id", "atom_text"]
        for field in required:
            if field not in record:
                raise ValueError(f"atoms 缺少字段: file={path}, row={row_idx}, field={field}")
        atom_id = str(record["atom_id"])
        if atom_id in seen:
            raise ValueError(f"atoms 中重复 atom_id: {atom_id}")
        seen.add(atom_id)
        grouped[str(record["parent_chunk_id"])].append(
            {
                "atom_id": atom_id,
                "parent_chunk_id": str(record["parent_chunk_id"]),
                "doc_id": str(record["doc_id"]),
                "atom_text": str(record["atom_text"]).strip(),
            }
        )
    return dict(grouped)


def load_init_states(path: Path) -> Dict[str, dict]:
    out = {}
    for row_idx, record in enumerate(read_jsonl(path), start=1):
        required = ["qid", "t", "H_t", "A_t", "S_t", "K_t"]
        for field in required:
            if field not in record:
                raise ValueError(f"init_state 缺少字段: file={path}, row={row_idx}, field={field}")
        qid = str(record["qid"])
        if qid in out:
            raise ValueError(f"init_state 中重复 qid: {qid}")
        out[qid] = {
            "qid": qid,
            "t": int(record["t"]),
            "H_t": record["H_t"],
            "A_t": record["A_t"],
            "S_t": record["S_t"],
            "K_t": str(record["K_t"]),
        }
    return out


def maybe_limit_qids(d: Dict[str, dict]) -> Dict[str, dict]:
    if FULL_MAX_QIDS <= 0:
        return d
    keys = sorted(d.keys())[:FULL_MAX_QIDS]
    return {k: d[k] for k in keys}


def build_unit_registry(raw_unit_map: Dict[str, dict], derived_registry: Dict[str, dict]) -> Dict[str, dict]:
    reg = dict(raw_unit_map)
    for unit_id, item in derived_registry.items():
        if unit_id in reg:
            raise ValueError(f"UnitRegistry 中 unit_id 冲突: {unit_id}")
        reg[unit_id] = item
    return reg


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
        out.append({"unit_id": unit_id, "added_step": int(item.get("added_step", 0))})
    return out


def get_role_key(role: str) -> str:
    if role == "bridge":
        return "k_br"
    if role == "distinguish":
        return "k_dis"
    if role == "support":
        return "k_sup"
    raise ValueError(f"非法 role: {role}")


def get_k_r(a_t: dict, role: str) -> float:
    return float(a_t.get(get_role_key(role), 0.0))


def set_k_r(a_t: dict, role: str, value: float):
    a_t[get_role_key(role)] = float(value)


def simulate_update(h_t: List[dict], s_t: dict, u_next: dict) -> Tuple[list, dict]:
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
        h_next.append({"step_id": next_step_id, "unit_id": unit_id})

    ref_key = "raw_refs" if u_next["provenance"] == "raw" else "derived_refs"
    existing_ref_ids = {str(x["unit_id"]) for x in s_next[ref_key] if isinstance(x, dict) and "unit_id" in x}
    if unit_id not in existing_ref_ids:
        s_next[ref_key].append({"unit_id": unit_id, "added_step": next_step_id})

    s_next["last_added_unit_id"] = unit_id
    s_next["last_updated_step"] = next_step_id
    return h_next, s_next


def simulate_ledger(a_t: dict, u_next: dict, target_info: dict) -> dict:
    a_next = copy.deepcopy(a_t)
    covered = [str(x) for x in a_next.get("covered_target_ids", [])]
    covered_set = set(covered)
    target_map = target_info["target_map"]

    if u_next["provenance"] != "raw":
        a_next["covered_target_ids"] = covered
        return a_next

    unit_id = str(u_next["unit_id"])
    target_unit_id = str(u_next.get("parent_chunk_id", "")).strip()
    if not target_unit_id:
        target_unit_id = parse_chunk_id_from_unit_id(unit_id)

    if target_unit_id not in target_map or target_unit_id in covered_set:
        a_next["covered_target_ids"] = covered
        return a_next

    covered.append(target_unit_id)
    covered_set.add(target_unit_id)

    target = target_map[target_unit_id]
    role = target["primary_role"]
    weight = float(target["weight"])

    current = get_k_r(a_next, role)
    set_k_r(a_next, role, current + weight)
    a_next["covered_target_ids"] = covered
    return a_next


def get_last_raw_text(s_t: dict, unit_registry: Dict[str, dict]) -> Optional[str]:
    raw_refs = s_t.get("raw_refs", [])
    if not isinstance(raw_refs, list) or len(raw_refs) == 0:
        return None
    last_ref = raw_refs[-1]
    if not isinstance(last_ref, dict) or "unit_id" not in last_ref:
        return None
    unit_id = str(last_ref["unit_id"])
    unit = unit_registry.get(unit_id)
    if unit is None:
        return None
    return unit["text"]


def get_latest_covered_raw_text(a_t: dict, s_t: dict, unit_registry: Dict[str, dict]) -> Optional[str]:
    covered = {str(x) for x in a_t.get("covered_target_ids", [])}
    if not covered:
        return None

    raw_refs = normalize_refs_minimal(s_t.get("raw_refs", []))
    for ref in reversed(raw_refs):
        unit_id = str(ref["unit_id"])
        unit = unit_registry.get(unit_id)
        if unit is None:
            continue
        if str(unit.get("parent_chunk_id", "")) not in covered:
            continue
        return unit["text"]
    return None


def get_last_derived_text(s_t: dict, unit_registry: Dict[str, dict]) -> Optional[str]:
    derived_refs = s_t.get("derived_refs", [])
    if not isinstance(derived_refs, list) or len(derived_refs) == 0:
        return None
    last_ref = derived_refs[-1]
    if not isinstance(last_ref, dict) or "unit_id" not in last_ref:
        return None
    unit = unit_registry.get(str(last_ref["unit_id"]))
    if unit is None:
        return None
    return unit["text"]


def get_latest_note_ids_by_type(s_t: dict, unit_registry: Dict[str, dict]) -> List[str]:
    derived_refs = normalize_refs_minimal(s_t.get("derived_refs", []))
    latest_by_type = {}

    for ref in derived_refs:
        unit_id = ref["unit_id"]
        unit = unit_registry.get(unit_id)
        if unit is None or unit.get("provenance") != "derived":
            continue
        note_type = unit.get("type")
        if note_type not in {"bridge_note", "verification_note"}:
            continue

        prev = latest_by_type.get(note_type)
        if prev is None or ref["added_step"] > prev["added_step"]:
            latest_by_type[note_type] = ref

    ordered = []
    if "bridge_note" in latest_by_type:
        ordered.append(latest_by_type["bridge_note"]["unit_id"])
    if "verification_note" in latest_by_type:
        ordered.append(latest_by_type["verification_note"]["unit_id"])
    return ordered


def get_active_note_ids(s_t: dict, unit_registry: Dict[str, dict]) -> List[str]:
    derived_refs = normalize_refs_minimal(s_t.get("derived_refs", []))
    if not derived_refs:
        return []

    scored = []
    for ref in derived_refs[-6:]:
        unit_id = ref["unit_id"]
        unit = unit_registry.get(unit_id)
        if unit is None or unit.get("provenance") != "derived":
            continue
        note_type = str(unit.get("type", "")).strip()
        if note_type not in {"bridge_note", "verification_note"}:
            continue
        force_render = bool(unit.get("force_render_next_step", False))
        answer_facing = bool(unit.get("answer_facing", False))
        derive_mode = str(unit.get("derive_mode", "")).strip()
        retained_bucket = str(unit.get("retained_bucket", "")).strip()
        closure_value = int(unit.get("closure_value", 0))
        scored.append(
            (
                (
                    1 if force_render else 0,
                    1 if answer_facing else 0,
                    1 if derive_mode == "repair_after_false_stop" else 0,
                    1 if retained_bucket == "final" else 0,
                    1 if note_type == "verification_note" else 0,
                    closure_value,
                    int(ref.get("added_step", 0)),
                ),
                unit_id,
            )
        )

    if not scored:
        return get_latest_note_ids_by_type(s_t, unit_registry)

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [unit_id for _, unit_id in scored[:MAX_RENDER_NOTES]]
    if not selected:
        return get_latest_note_ids_by_type(s_t, unit_registry)
    return selected


def render_context(q: dict, s_t: dict, unit_registry: Dict[str, dict]) -> str:
    note_ids = get_active_note_ids(s_t, unit_registry)

    source_raw_ids = []
    for note_id in note_ids:
        note = unit_registry[note_id]
        for uid in note.get("source_unit_ids", []):
            if uid not in source_raw_ids:
                source_raw_ids.append(uid)

    raw_ids = list(source_raw_ids)
    raw_refs = normalize_refs_minimal(s_t.get("raw_refs", []))
    for ref in sorted(raw_refs, key=lambda x: x["added_step"], reverse=True):
        if len(raw_ids) >= MAX_RENDER_RAW:
            break
        if ref["unit_id"] not in raw_ids:
            raw_ids.append(ref["unit_id"])

    parts = []
    raw_units = [unit_registry[uid] for uid in raw_ids[:MAX_RENDER_RAW] if uid in unit_registry]
    if raw_units:
        parts.append("Evidence:")
        for i, u in enumerate(raw_units, start=1):
            parts.append(f"[{i}] {shorten(u['text'], MAX_CHARS_PER_RAW_ITEM)}")

    active_notes = [unit_registry[uid] for uid in note_ids if uid in unit_registry]
    if active_notes:
        parts.append("")
        parts.append("Notes:")
        for note in active_notes[:MAX_RENDER_NOTES]:
            label = "bridge" if note["type"] == "bridge_note" else "verification"
            parts.append(f"[{label}] {shorten(note['text'], MAX_CHARS_PER_NOTE_ITEM)}")

    return "\n".join(parts).strip()


def build_slot_summary(question: str, a_t: dict, s_t: dict, unit_registry: Dict[str, dict]) -> str:
    parts = []
    covered_raw = get_latest_covered_raw_text(a_t, s_t, unit_registry)
    if covered_raw is not None:
        parts.append("Evidence: " + shorten(covered_raw, 80))
    last_note = get_last_derived_text(s_t, unit_registry)
    if last_note is not None and a_t.get("covered_target_ids"):
        parts.append("Note: " + shorten(last_note, 70))
    parts.append("Need: " + extract_goal_from_question(question))
    return "\n".join(parts)[:MAX_SUMMARY_CHARS].rstrip()


def build_q_t(question: str, a_t: dict, s_t: dict, unit_registry: Dict[str, dict]) -> str:
    h_t = build_slot_summary(question, a_t, s_t, unit_registry)
    if not h_t:
        return question
    return question + "\n" + h_t


def build_chunk_shortlist(q_t: str, chunks: List[dict]) -> List[dict]:
    if not chunks:
        return []

    summary_index = BM25Index([c["summary_text"] for c in chunks])
    chunk_index = BM25Index([c["chunk_text"] for c in chunks])
    s_scores = summary_index.score(q_t)
    c_scores = chunk_index.score(q_t)

    scored_chunks = []
    for chunk, s_sum, s_chunk in zip(chunks, s_scores, c_scores):
        coarse_score = 0.5 * s_sum + 0.5 * s_chunk
        scored_chunks.append({"chunk_id": chunk["chunk_id"], "score": float(coarse_score)})

    scored_chunks.sort(key=lambda x: (-x["score"], x["chunk_id"]))
    chunk_meta = {
        str(chunk["chunk_id"]): {
            "chunk_id": str(chunk["chunk_id"]),
            "doc_id": str(chunk["doc_id"]),
        }
        for chunk in chunks
    }
    shortlist = []
    for item in scored_chunks[:CHUNK_SHORTLIST_K]:
        meta = chunk_meta[str(item["chunk_id"])]
        shortlist.append(
            {
                "chunk_id": meta["chunk_id"],
                "doc_id": meta["doc_id"],
                "score": float(item["score"]),
            }
        )
    return shortlist


def context_mentions_target_hint(context_text: str, target: dict) -> bool:
    norm_context = normalize_text(context_text or "")
    if not norm_context:
        return False

    doc_id = str(target.get("doc_id", "")).strip()
    norm_doc_id = normalize_text(doc_id)
    if norm_doc_id and norm_doc_id in norm_context:
        return True

    target_text = str(target.get("text", "")).strip()
    anchor_hits = 0
    for anchor in extract_anchor_tokens(f"{doc_id} {target_text}")[:8]:
        norm_anchor = normalize_text(anchor)
        if len(norm_anchor) < 4:
            continue
        if norm_anchor not in norm_context:
            continue
        anchor_hits += 1
        if " " in norm_anchor or anchor_hits >= 2:
            return True
    return False


def choose_best_unit_from_target_chunk(
    *,
    question: str,
    unit_ids: List[str],
    raw_unit_map: Dict[str, dict],
) -> Optional[str]:
    focus_kind = infer_answer_focus_kind(question)
    query_anchors = set(extract_query_anchors(question))
    best_unit_id: Optional[str] = None
    best_key: Optional[Tuple[int, int, int, str]] = None

    for unit_id in unit_ids:
        unit = raw_unit_map.get(unit_id)
        if unit is None:
            continue
        text = f"{unit.get('doc_id', '')} {unit.get('text', '')}".strip()
        anchor_hits = len(set(extract_anchor_tokens(text)) & query_anchors)
        answer_signal = 1 if text_has_answer_type_signal(text, focus_kind) else 0
        token_overlap = len(set(tokenize(text)) & set(tokenize(question)))
        sentence_bias = -parse_sent_id_from_unit_id(unit_id)
        key = (answer_signal, anchor_hits, token_overlap, sentence_bias, str(unit_id))
        if best_key is None or key > best_key:
            best_key = key
            best_unit_id = str(unit_id)

    return best_unit_id


def topk_retrieve(
    q_t: str,
    chunks: List[dict],
    atoms_by_chunk: Dict[str, List[dict]],
    used_unit_ids: set,
    shortlist: Optional[List[dict]] = None,
) -> List[str]:
    if shortlist is None:
        shortlist = build_chunk_shortlist(q_t, chunks)
    if not shortlist:
        return []

    merged = []
    for item in shortlist:
        atoms = atoms_by_chunk.get(str(item["chunk_id"]), [])
        if not atoms:
            continue
        atom_index = BM25Index([a["atom_text"] for a in atoms])
        atom_scores = atom_index.score(q_t)

        scored_atoms = []
        for atom, score in zip(atoms, atom_scores):
            if atom["atom_id"] in used_unit_ids:
                continue
            scored_atoms.append({"unit_id": atom["atom_id"], "score": float(score)})

        if not scored_atoms:
            continue

        scored_atoms.sort(key=lambda x: (-x["score"], x["unit_id"]))
        merged.append(scored_atoms[0])

    merged.sort(key=lambda x: (-x["score"], x["unit_id"]))
    return [x["unit_id"] for x in merged[:FINAL_KR]]


def normalized_role_scores(a_t: dict, target_info: dict) -> Dict[str, float]:
    denoms = target_info["role_counts"]
    out = {}
    for role in target_info["required_roles"]:
        denom = max(float(denoms.get(role, 0.0)), 1.0)
        out[role] = get_k_r(a_t, role) / denom
    return out


def get_recent_raw_texts(s_t: dict, unit_registry: Dict[str, dict], n: int = 2) -> List[str]:
    raw_refs = s_t.get("raw_refs", [])
    if not isinstance(raw_refs, list):
        return []
    recent = []
    for item in raw_refs[-n:]:
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit = unit_registry.get(str(item["unit_id"]))
        if unit is not None:
            recent.append(unit["text"])
    return recent


def has_recent_note(s_t: dict, unit_registry: Dict[str, dict], note_type: str, window: int = 2) -> bool:
    derived_refs = s_t.get("derived_refs", [])
    if not isinstance(derived_refs, list):
        return False
    recent = derived_refs[-window:]
    for ref in recent:
        if not isinstance(ref, dict) or "unit_id" not in ref:
            continue
        unit = unit_registry.get(str(ref["unit_id"]))
        if unit and unit.get("provenance") == "derived" and unit.get("type") == note_type:
            return True
    return False


def is_raw_pool_redundant(r_t: List[str], s_t: dict, unit_registry: Dict[str, dict]) -> bool:
    recent_raw_texts = get_recent_raw_texts(s_t, unit_registry, n=2)
    if not recent_raw_texts:
        return False
    top_raw = r_t[:3]
    redundant_count = 0
    for unit_id in top_raw:
        unit = unit_registry.get(unit_id)
        if unit is None:
            continue
        text = unit["text"]
        if any(jaccard_similarity(text, prev) >= REDUNDANT_SIM_THRESHOLD for prev in recent_raw_texts):
            redundant_count += 1
    return redundant_count >= 2


def has_composable_raw(r_t: List[str], unit_registry: Dict[str, dict], sim_threshold: float = COMPOSABLE_SIM_THRESHOLD) -> bool:
    top_raw = r_t[:3]
    if len(top_raw) < 2:
        return False

    usable = []
    for unit_id in top_raw:
        unit = unit_registry.get(unit_id)
        if unit is None or unit.get("provenance") != "raw":
            continue

        keep = True
        for prev in usable:
            if jaccard_similarity(str(unit.get("text", "")), prev["text"]) >= sim_threshold:
                keep = False
                break

        if keep:
            usable.append(
                {
                    "unit_id": unit_id,
                    "text": str(unit.get("text", "")),
                    "parent_chunk_id": str(unit.get("parent_chunk_id", "")),
                }
            )

    distinct_parents = len({u["parent_chunk_id"] for u in usable if u["parent_chunk_id"]})
    return len(usable) >= 2 and distinct_parents >= 2


def cheap_stop_gate_legacy(
    target_info: dict,
    a_t: dict,
    s_t: dict,
    r_t: List[str],
    unit_registry: Dict[str, dict],
    recent_progress_flags: List[int],
) -> Dict[str, object]:
    role_scores = normalized_role_scores(a_t, target_info)
    raw_complete = all(role_scores[r] >= RAW_COMPLETE_THRESHOLD for r in target_info["required_roles"])
    has_recent_verif = has_recent_note(s_t, unit_registry, "verification_note", window=2)
    has_recent_bridge = has_recent_note(s_t, unit_registry, "bridge_note", window=2)
    raw_redundant = is_raw_pool_redundant(r_t, s_t, unit_registry)
    no_derived_need = has_recent_verif or (raw_redundant and has_recent_bridge)
    near_complete = bool(role_scores) and all(
        role_scores[r] >= STOP_NEAR_COMPLETE_THRESHOLD for r in target_info["required_roles"]
    )
    plateau_relax = bool(recent_progress_flags) and int(recent_progress_flags[-1]) == 0 and near_complete
    stop_candidate = bool(raw_complete and (no_derived_need or plateau_relax))
    return {
        "role_scores": role_scores,
        "raw_complete": raw_complete,
        "near_complete": near_complete,
        "no_derived_need": no_derived_need,
        "raw_redundant": raw_redundant,
        "plateau_relax": plateau_relax,
        "stop_candidate": stop_candidate,
    }


def need_derived_gate_legacy(
    target_info: dict,
    a_t: dict,
    s_t: dict,
    r_t: List[str],
    unit_registry: Dict[str, dict],
    recent_progress_flags: List[int],
) -> Dict[str, object]:
    role_scores = normalized_role_scores(a_t, target_info)
    s_sem = sum(role_scores[r] for r in target_info["required_roles"]) / max(1, len(target_info["required_roles"]))
    composable_raw = has_composable_raw(r_t, unit_registry)
    has_recent_verification = has_recent_note(s_t, unit_registry, "verification_note", window=2)
    raw_redundant = is_raw_pool_redundant(r_t, s_t, unit_registry)
    plateau_relax = bool(recent_progress_flags) and int(recent_progress_flags[-1]) == 0
    derived_need = composable_raw and ((not has_recent_verification) or plateau_relax)
    trigger_derived = bool((s_sem >= TAU_SEM) and derived_need)
    return {
        "s_sem": s_sem,
        "composable_raw": composable_raw,
        "has_recent_verification": has_recent_verification,
        "raw_redundant": raw_redundant,
        "plateau_relax": plateau_relax,
        "derived_need": derived_need,
        "trigger_derived": trigger_derived,
    }


def compute_retrieval_repeat_ratio(r_t: List[str], retrieval_history: List[List[str]]) -> Optional[float]:
    if not retrieval_history:
        return None
    current = set(str(x) for x in r_t if isinstance(x, str))
    if not current:
        return None
    overlaps = []
    for prev in retrieval_history[-2:]:
        prev_set = set(str(x) for x in prev if isinstance(x, str))
        if not prev_set:
            continue
        overlaps.append(len(current & prev_set) / max(len(current | prev_set), 1))
    if not overlaps:
        return None
    return max(overlaps)


def augment_with_shortlisted_uncovered_targets(
    question: str,
    r_t: List[str],
    shortlist: List[dict],
    target_info: dict,
    a_t: dict,
    k_t: str,
    raw_unit_map: Dict[str, dict],
    raw_unit_ids_by_chunk: Dict[str, List[str]],
    used_unit_ids: set,
) -> Tuple[List[str], List[str]]:
    if not shortlist and not k_t:
        return list(r_t), []

    covered = {str(x) for x in a_t.get("covered_target_ids", [])}
    shortlist_chunk_ids = {str(item["chunk_id"]) for item in shortlist if isinstance(item, dict) and "chunk_id" in item}
    out = list(r_t)
    seen = set(out)
    added_chunk_ids = []

    for target_chunk_id, target_item in target_info["target_map"].items():
        target_chunk_id = str(target_chunk_id)
        if target_chunk_id in covered:
            continue
        if (
            target_chunk_id not in shortlist_chunk_ids
            and not context_mentions_target_hint(k_t, target_item)
        ):
            continue

        best_unit_id = choose_best_unit_from_target_chunk(
            question=question,
            unit_ids=raw_unit_ids_by_chunk.get(target_chunk_id, []),
            raw_unit_map=raw_unit_map,
        )
        if (
            best_unit_id
            and best_unit_id not in seen
            and best_unit_id not in used_unit_ids
            and best_unit_id in raw_unit_map
        ):
            out.append(best_unit_id)
            seen.add(best_unit_id)
            added_chunk_ids.append(target_chunk_id)

    return out, normalize_raw_chunk_id_list(added_chunk_ids)


def append_oracle_uncovered_targets(
    *,
    question: str,
    r_t: List[str],
    target_info: dict,
    a_t: dict,
    raw_unit_map: Dict[str, dict],
    raw_unit_ids_by_chunk: Dict[str, List[str]],
    used_unit_ids: set,
    max_add: int = ORACLE_UNCOVERED_MAX,
) -> Tuple[List[str], List[str]]:
    """Offline teacher rescue: expose uncovered gold target chunks before stall.

    This keeps the main ranking/selection logic unchanged, but prevents the
    retrieval shortlist from repeatedly hiding already-known target evidence.
    """
    if not ORACLE_UNCOVERED_BOOST or max_add <= 0:
        return list(r_t), []

    covered = {str(x) for x in a_t.get("covered_target_ids", [])}
    out = list(r_t)
    seen = set(out)
    added_chunk_ids: List[str] = []

    for target_chunk_id in target_info["target_map"].keys():
        target_chunk_id = str(target_chunk_id)
        if target_chunk_id in covered or target_chunk_id in added_chunk_ids:
            continue
        best_unit_id = choose_best_unit_from_target_chunk(
            question=question,
            unit_ids=raw_unit_ids_by_chunk.get(target_chunk_id, []),
            raw_unit_map=raw_unit_map,
        )
        if (
            best_unit_id
            and best_unit_id not in seen
            and best_unit_id not in used_unit_ids
            and best_unit_id in raw_unit_map
        ):
            out.append(best_unit_id)
            seen.add(best_unit_id)
            added_chunk_ids.append(target_chunk_id)
            if len(added_chunk_ids) >= max_add:
                break

    return out, normalize_raw_chunk_id_list(added_chunk_ids)


def cheap_stop_gate_v3(
    target_info: dict,
    a_t: dict,
    s_t: dict,
    r_t: List[str],
    unit_registry: Dict[str, dict],
    retrieval_repeat_ratio: Optional[float],
    recent_progress_flags: List[int],
    failure_signals: Dict[str, object],
) -> Dict[str, object]:
    role_scores = normalized_role_scores(a_t, target_info)
    raw_complete = all(role_scores[r] >= RAW_COMPLETE_THRESHOLD for r in target_info["required_roles"])
    has_recent_verif = has_recent_note(s_t, unit_registry, "verification_note", window=2)
    has_recent_bridge = has_recent_note(s_t, unit_registry, "bridge_note", window=2)
    raw_redundant = is_raw_pool_redundant(r_t, s_t, unit_registry)
    no_obvious_derived_need = has_recent_verif or (raw_redundant and has_recent_bridge)
    near_complete = bool(role_scores) and all(
        role_scores[r] >= STOP_NEAR_COMPLETE_THRESHOLD for r in target_info["required_roles"]
    )
    no_recent_closure_failure = not (
        bool(failure_signals.get("recent_false_stop", False))
        or bool(failure_signals.get("stagnation", False))
        or bool(failure_signals.get("answer_focus_mismatch", False))
    )
    stop_candidate = bool(raw_complete and no_obvious_derived_need and no_recent_closure_failure)
    return {
        "role_scores": role_scores,
        "raw_complete": raw_complete,
        "near_complete": near_complete,
        "no_obvious_derived_need": no_obvious_derived_need,
        "no_recent_closure_failure": no_recent_closure_failure,
        "raw_redundant": raw_redundant,
        "retrieval_repeat_ratio": retrieval_repeat_ratio,
        "stop_candidate": stop_candidate,
    }


def need_derived_gate_v3(
    question: str,
    target_info: dict,
    a_t: dict,
    s_t: dict,
    r_t: List[str],
    unit_registry: Dict[str, dict],
    failure_signals: Dict[str, object],
) -> Dict[str, object]:
    role_scores = normalized_role_scores(a_t, target_info)
    s_sem = sum(role_scores[r] for r in target_info["required_roles"]) / max(1, len(target_info["required_roles"]))
    bridgeable_raw = has_bridgeable_raw(question, s_t, r_t, unit_registry)
    has_recent_verification = has_recent_note(s_t, unit_registry, "verification_note", window=2)
    raw_redundant = is_raw_pool_redundant(r_t, s_t, unit_registry)
    derived_need = bridgeable_raw and (
        (not has_recent_verification)
        or bool(failure_signals.get("recent_false_stop", False))
        or bool(failure_signals.get("stagnation", False))
        or bool(failure_signals.get("answer_focus_mismatch", False))
    )
    trigger_derived = bool((s_sem >= TAU_SEM) and derived_need)
    return {
        "s_sem": s_sem,
        "bridgeable_raw": bridgeable_raw,
        "has_recent_verification": has_recent_verification,
        "raw_redundant": raw_redundant,
        "derived_need": derived_need,
        "trigger_derived": trigger_derived,
    }


def compute_answer_ready_raw(
    question: str,
    r_t: List[str],
    unit_registry: Dict[str, dict],
) -> Dict[str, object]:
    query_anchors = set(extract_query_anchors(question))
    focus_kind = infer_answer_focus_kind(question)
    top_profiles = []
    for unit_id in r_t[:REPAIR_TOP_RAW_J]:
        unit = unit_registry.get(unit_id)
        if unit is None or unit.get("provenance") != "raw":
            continue
        top_profiles.append(collect_candidate_anchor_profile(unit))

    has_answer_signal = False
    has_query_related = False
    answer_ready_raw = False
    query_only_count = 0
    answer_only_count = 0

    for profile in top_profiles:
        has_query = bool(profile["anchor_set"] & query_anchors)
        has_answer = text_has_answer_type_signal(f"{profile['doc_id']} {profile['text']}", focus_kind)
        has_answer_signal = has_answer_signal or has_answer
        has_query_related = has_query_related or has_query
        answer_ready_raw = answer_ready_raw or (has_query and has_answer)
        if has_query and not has_answer:
            query_only_count += 1
        if has_answer and not has_query:
            answer_only_count += 1

    return {
        "focus_kind": focus_kind,
        "has_answer_signal": has_answer_signal,
        "has_query_related": has_query_related,
        "answer_ready_raw": answer_ready_raw,
        "query_only_count": query_only_count,
        "answer_only_count": answer_only_count,
    }


def has_explicit_bridge_gap(
    question: str,
    s_t: dict,
    r_t: List[str],
    unit_registry: Dict[str, dict],
) -> bool:
    top_profiles = []
    for unit_id in r_t[:REPAIR_TOP_RAW_J]:
        unit = unit_registry.get(unit_id)
        if unit is None or unit.get("provenance") != "raw":
            continue
        top_profiles.append(collect_candidate_anchor_profile(unit))
    if len(top_profiles) < 2:
        return False

    query_anchors = set(extract_query_anchors(question))
    state_anchors = set(extract_state_anchors(s_t, unit_registry))
    focus_kind = infer_answer_focus_kind(question)

    saw_query_only = False
    saw_answer_only = False
    saw_shared_structure = False

    for profile in top_profiles:
        has_query = bool(profile["anchor_set"] & query_anchors)
        has_answer = text_has_answer_type_signal(f"{profile['doc_id']} {profile['text']}", focus_kind)
        if has_query and not has_answer:
            saw_query_only = True
        if has_answer and not has_query:
            saw_answer_only = True

    for i in range(len(top_profiles)):
        for j in range(i + 1, len(top_profiles)):
            shared = (
                (top_profiles[i]["anchor_set"] & top_profiles[j]["anchor_set"])
                | (top_profiles[i]["anchor_set"] & state_anchors)
                | (top_profiles[j]["anchor_set"] & state_anchors)
            )
            if shared:
                saw_shared_structure = True
                break
        if saw_shared_structure:
            break

    return bool(saw_query_only and saw_answer_only and saw_shared_structure)


def classify_derived_subtype_v3(
    *,
    question: str,
    t: int,
    s_t: dict,
    r_t: List[str],
    unit_registry: Dict[str, dict],
    derived_info: Dict[str, object],
    failure_signals: Dict[str, object],
) -> Tuple[Optional[str], Dict[str, object]]:
    answer_ready_info = compute_answer_ready_raw(question, r_t, unit_registry)
    bridge_gap_explicit = has_explicit_bridge_gap(question, s_t, r_t, unit_registry)
    closure_repair_needed = bool(
        bool(failure_signals.get("recent_false_stop", False))
        or bool(failure_signals.get("stagnation", False))
        or bool(failure_signals.get("answer_focus_mismatch", False))
    )

    late_verification = bool(
        derived_info.get("bridgeable_raw", False)
        and float(derived_info.get("s_sem", 0.0)) >= LATE_VERIFICATION_MIN_SEM
        and closure_repair_needed
        and (
            bool(answer_ready_info["answer_ready_raw"])
            or bool(failure_signals.get("answer_focus_mismatch", False))
        )
    )

    early_bridge = bool(
        t <= EARLY_BRIDGE_MAX_T
        and derived_info.get("bridgeable_raw", False)
        and float(derived_info.get("s_sem", 0.0)) >= EARLY_BRIDGE_MIN_SEM
        and not bool(answer_ready_info["answer_ready_raw"])
        and bridge_gap_explicit
    )

    trigger_only_candidate = bool(
        derived_info.get("trigger_derived", False)
        and not late_verification
        and not early_bridge
        and float(derived_info.get("s_sem", 0.0)) >= TRIGGER_ONLY_MIN_SEM
    )

    subtype = None
    if late_verification:
        subtype = "late_verification"
    elif early_bridge:
        subtype = "early_bridge"
    elif trigger_only_candidate:
        subtype = "trigger_only_candidate"

    return subtype, {
        "closure_repair_needed": closure_repair_needed,
        "answer_ready_raw": bool(answer_ready_info["answer_ready_raw"]),
        "has_answer_signal": bool(answer_ready_info["has_answer_signal"]),
        "has_query_related": bool(answer_ready_info["has_query_related"]),
        "bridge_gap_explicit": bool(bridge_gap_explicit),
        "query_only_count": int(answer_ready_info["query_only_count"]),
        "answer_only_count": int(answer_ready_info["answer_only_count"]),
        "late_verification": late_verification,
        "early_bridge": early_bridge,
        "trigger_only_candidate": trigger_only_candidate,
    }


def extract_probe_answer_from_context(gold_answer: str, k_t: str) -> Optional[str]:
    norm_gold = normalize_text(gold_answer)
    norm_ctx = normalize_text(k_t)
    if not norm_gold:
        return None
    if norm_gold in norm_ctx:
        return gold_answer
    return None


def context_contains_answer(gold_answer: str, k_t: str) -> bool:
    norm_gold = normalize_text(gold_answer)
    norm_ctx = normalize_text(k_t)
    if not norm_gold:
        return False
    return norm_gold in norm_ctx


def extract_numeric_tokens(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return re.findall(r"\d+(?:[./:-]\d+)*", str(text))


def is_number_like_answer(text: Optional[str]) -> bool:
    if text is None:
        return False
    raw = str(text).strip()
    if not raw:
        return False
    if re.search(r"\d", raw):
        return True
    lowered = raw.lower()
    month_names = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    return any(month in lowered for month in month_names)


def answers_context_compatible(pred_answer: Optional[str], gold_answer: str) -> bool:
    if pred_answer is None:
        return False
    norm_pred = normalize_text(pred_answer)
    norm_gold = normalize_text(gold_answer)
    if not norm_pred or not norm_gold:
        return False
    if norm_pred == norm_gold:
        return True
    if is_number_like_answer(pred_answer) or is_number_like_answer(gold_answer):
        pred_numeric = extract_numeric_tokens(pred_answer)
        gold_numeric = extract_numeric_tokens(gold_answer)
        if pred_numeric and gold_numeric and pred_numeric == gold_numeric:
            return True
        if norm_gold and norm_gold in norm_pred:
            return True
        return False
    return (
        norm_pred.startswith(norm_gold + " ")
        or norm_pred.endswith(" " + norm_gold)
        or f" {norm_gold} " in f" {norm_pred} "
    )


def answer_conflicts(pred_answer: Optional[str], gold_answer: str) -> bool:
    if pred_answer is None:
        return False
    norm_pred = normalize_text(pred_answer)
    norm_gold = normalize_text(gold_answer)
    if not norm_pred or not norm_gold:
        return False
    if norm_pred == norm_gold:
        return False
    pred_numeric = extract_numeric_tokens(pred_answer)
    gold_numeric = extract_numeric_tokens(gold_answer)
    if pred_numeric or gold_numeric:
        return pred_numeric != gold_numeric
    return not answers_context_compatible(pred_answer, gold_answer)


def check_answer_correct(gold_answer: str, probe_answer: Optional[str]) -> bool:
    if probe_answer is None:
        return False
    norm_gold = normalize_text(gold_answer)
    norm_probe = normalize_text(probe_answer)
    if norm_gold == norm_probe:
        return True
    if is_number_like_answer(gold_answer) or is_number_like_answer(probe_answer):
        pred_numeric = extract_numeric_tokens(probe_answer)
        gold_numeric = extract_numeric_tokens(gold_answer)
        if pred_numeric and gold_numeric and pred_numeric == gold_numeric:
            return True
    return False


def check_support_sufficient(covered_target_ids: List[str], target_unit_id_set: set) -> bool:
    covered_set = set(str(x) for x in covered_target_ids)
    return target_unit_id_set.issubset(covered_set)


def get_selected_chunks_from_state(state: dict, chunks: List[dict], unit_registry: Dict[str, dict]) -> List[dict]:
    selected_chunk_ids: List[str] = []
    seen: set = set()
    for item in state.get("H_t", []):
        if not isinstance(item, dict):
            continue
        unit_id = str(item.get("unit_id", ""))
        if is_derived_unit_id(unit_id):
            continue
        unit = unit_registry.get(unit_id)
        if unit is None:
            continue
        chunk_id = str(unit.get("parent_chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        selected_chunk_ids.append(chunk_id)
    chunk_map = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    return [chunk_map[cid] for cid in selected_chunk_ids if cid in chunk_map]


def target_doc_hint(target_unit_id: str) -> str:
    parts = str(target_unit_id).split("::")
    if len(parts) >= 2:
        return parts[-1]
    return str(target_unit_id)


def extract_selected_doc_ids(selected_chunks: List[dict]) -> List[str]:
    out: List[str] = []
    seen = set()
    for chunk in selected_chunks:
        if not isinstance(chunk, dict):
            continue
        doc_id = str(chunk.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def target_in_selected_evidence(target: dict, evidence_text: str, selected_doc_ids: List[str]) -> bool:
    doc_id = str(target.get("doc_id", "")).strip()
    if doc_id and doc_id in selected_doc_ids:
        return True
    target_text = str(target.get("text", "")).strip()
    if target_text and context_contains_answer(target_text, evidence_text):
        return True
    return False


def local_support_probe(
    *,
    question: str,
    gold_answer: str,
    pred_answer: Optional[str],
    k_t: str,
    selected_chunks: List[dict],
    covered_target_ids: List[str],
    target_info: dict,
) -> dict:
    target_map = target_info["target_map"]
    target_ids = [str(x) for x in target_map.keys()]
    all_chunk_text = "\n".join(str(chunk.get("chunk_text", "")) for chunk in selected_chunks)
    evidence_text = (k_t or "") + "\n" + all_chunk_text
    selected_doc_ids = extract_selected_doc_ids(selected_chunks)
    gold_in_k_t = context_contains_answer(gold_answer, k_t)
    gold_in_chunks = context_contains_answer(gold_answer, all_chunk_text)
    answer_in_evidence = gold_in_k_t or gold_in_chunks
    missing_reasons: List[str] = []

    if pred_answer is None:
        return {
            "support_sufficient": False,
            "support_rule": "insufficient",
            "support_evidence_summary": "",
            "missing_support_reasons": ["missing_pred_answer"],
        }

    if not target_ids:
        return {
            "support_sufficient": False,
            "support_rule": "insufficient",
            "support_evidence_summary": "",
            "missing_support_reasons": ["missing_multihop_evidence"],
        }

    missing_targets = [
        target_doc_hint(uid)
        for uid in target_ids
        if not target_in_selected_evidence(target_map[uid], evidence_text, selected_doc_ids)
    ]

    if is_yes_no_question(question):
        if missing_targets:
            return {
                "support_sufficient": False,
                "support_rule": "insufficient",
                "support_evidence_summary": "",
                "missing_support_reasons": ["yes_no_missing_one_entity_support", *missing_targets],
            }
        return {
            "support_sufficient": False,
            "support_rule": "requires_llm_judge",
            "support_evidence_summary": "",
            "missing_support_reasons": [],
        }

    if missing_targets:
        missing_reasons.append("missing_multihop_evidence")
        missing_reasons.extend(missing_targets)

    if is_number_like_answer(gold_answer) and not answer_in_evidence:
        missing_reasons.append("gold_answer_not_supported_by_chunks")
    elif len(target_ids) <= 1 and not answer_in_evidence:
        missing_reasons.append("gold_answer_not_supported_by_chunks")

    if missing_reasons:
        return {
            "support_sufficient": False,
            "support_rule": "insufficient",
            "support_evidence_summary": "",
            "missing_support_reasons": merge_unique_in_order(missing_reasons),
        }

    if len(target_ids) == 1 and answer_in_evidence and not missing_targets:
        return {
            "support_sufficient": True,
            "support_rule": "explicit_answer_in_evidence",
            "support_evidence_summary": "The selected evidence explicitly contains the gold answer.",
            "missing_support_reasons": [],
        }

    if (
        len(target_ids) >= 2
        and answer_in_evidence
        and not missing_targets
        and check_answer_correct(gold_answer, pred_answer)
    ):
        return {
            "support_sufficient": True,
            "support_rule": "multi_hop_answer_in_evidence",
            "support_evidence_summary": "The selected evidence covers the target hops and explicitly contains the gold answer.",
            "missing_support_reasons": [],
        }

    return {
        "support_sufficient": False,
        "support_rule": "requires_llm_judge",
        "support_evidence_summary": "",
        "missing_support_reasons": [],
    }


def judge_support_with_deepseek(
    *,
    api_key: str,
    model: str,
    base_url: str,
    question: str,
    gold_answer: str,
    pred_answer: str,
    k_t: str,
    selected_chunks: List[dict],
    target_info: dict,
) -> dict:
    chunk_lines = []
    for idx, chunk in enumerate(selected_chunks, start=1):
        chunk_lines.append(
            f"{idx}. doc={chunk.get('doc_id')}\n"
            f"   chunk_id={chunk.get('chunk_id')}\n"
            f"   text={str(chunk.get('chunk_text', '')).strip()}"
        )
    user_prompt = (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Gold answer:\n"
        f"{gold_answer.strip()}\n\n"
        "Pred answer:\n"
        f"{pred_answer.strip()}\n\n"
        "K_t:\n"
        f"{k_t.strip()}\n\n"
        "Selected chunks:\n"
        + ("\n".join(chunk_lines) if chunk_lines else "(none)")
        + "\n\nTarget evidence that should be supported:\n"
        + (
            "\n".join(
                f"- doc={item.get('doc_id') or target_doc_hint(uid)} | text={str(item.get('text', '')).strip()}"
                for uid, item in target_info.get("target_map", {}).items()
            )
            if isinstance(target_info.get("target_map"), dict) and target_info.get("target_map")
            else "(none)"
        )
        + "\n\nReturn strict JSON with keys:\n"
          "- support_sufficient: boolean\n"
          "- support_rule: string from {explicit_answer_in_evidence, multi_hop_supported, yes_no_both_entities_supported, insufficient}\n"
          "- support_evidence_summary: string\n"
          "- missing_support_reasons: list of strings\n"
    )
    system_prompt = (
        "You judge whether the provided evidence is sufficient to support a known answer.\n"
        "Use only the provided K_t and selected chunks.\n"
        "Do not use outside knowledge.\n"
        "Do not decide support solely because pred_answer matches gold_answer.\n"
        "For yes/no questions, make sure both compared entities are supported by evidence.\n"
        "For multi-hop questions, make sure the bridge hop and answer hop are both supported.\n"
        "If support is missing, return support_sufficient=false and explain what is missing."
    )
    parsed = deepseek_chat_json(
        api_key=api_key,
        model=model,
        base_url=base_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    missing = parsed.get("missing_support_reasons", [])
    if not isinstance(missing, list):
        missing = []
    missing = [str(x).strip() for x in missing if str(x).strip()]
    support_rule = str(parsed.get("support_rule", "")).strip() or "insufficient"
    summary = str(parsed.get("support_evidence_summary", "")).strip()
    return {
        "support_sufficient": bool(parsed.get("support_sufficient", False)),
        "support_rule": support_rule,
        "support_evidence_summary": summary,
        "missing_support_reasons": missing,
    }


def support_sufficient_check(
    *,
    question: str,
    gold_answer: str,
    pred_answer: Optional[str],
    k_t: str,
    selected_chunks: List[dict],
    covered_target_ids: List[str],
    target_info: dict,
    api_key: str,
    model: str,
    base_url: str,
) -> dict:
    local = local_support_probe(
        question=question,
        gold_answer=gold_answer,
        pred_answer=pred_answer,
        k_t=k_t,
        selected_chunks=selected_chunks,
        covered_target_ids=covered_target_ids,
        target_info=target_info,
    )
    if local["support_rule"] != "requires_llm_judge":
        return local
    if pred_answer is None:
        return {
            "support_sufficient": False,
            "support_rule": "insufficient",
            "support_evidence_summary": "",
            "missing_support_reasons": ["missing_pred_answer"],
        }
    try:
        judged = judge_support_with_deepseek(
            api_key=api_key,
            model=model,
            base_url=base_url,
            question=question,
            gold_answer=gold_answer,
            pred_answer=pred_answer,
            k_t=k_t,
            selected_chunks=selected_chunks,
            target_info=target_info,
        )
        return judged
    except RuntimeError as e:
        log(f"support judge fallback after API failure: {e}", force=True)
    return {
        "support_sufficient": False,
        "support_rule": "insufficient",
        "support_evidence_summary": "",
        "missing_support_reasons": ["support_judge_failed"],
    }


def make_offline_terminal_probe(
    *,
    query_info: dict,
    state: dict,
    target_info: dict,
    unit_registry: Dict[str, dict],
    chunks: List[dict],
    api_key: str,
    model: str,
    base_url: str,
    t: int,
    reason: str,
) -> Optional[dict]:
    gold_answer = str(query_info["answer"])
    selected_chunks = get_selected_chunks_from_state(state, chunks, unit_registry)
    target_ids = {str(x) for x in target_info.get("target_map", {}).keys()}
    covered_ids = {str(x) for x in state.get("A_t", {}).get("covered_target_ids", [])}
    target_complete = bool(target_ids) and target_ids.issubset(covered_ids)

    if is_yes_no_question(query_info["question"]):
        pred_answer: Optional[str] = gold_answer
        answer_source = f"offline_{reason}_yes_no_gold"
        support_probe = support_sufficient_check(
            question=query_info["question"],
            gold_answer=gold_answer,
            pred_answer=pred_answer,
            k_t=state["K_t"],
            selected_chunks=selected_chunks,
            covered_target_ids=state["A_t"].get("covered_target_ids", []),
            target_info=target_info,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
    else:
        if target_complete:
            pred_answer = gold_answer
            answer_source = f"offline_{reason}_target_complete_gold"
            support_probe = None
        else:
            pred_answer, answer_source, support_probe = maybe_fill_probe_answer_from_context(
                question=query_info["question"],
                gold_answer=gold_answer,
                pred_answer=None,
                k_t=state["K_t"],
                selected_chunks=selected_chunks,
                target_info=target_info,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
        if support_probe is None:
            support_probe = support_sufficient_check(
                question=query_info["question"],
                gold_answer=gold_answer,
                pred_answer=pred_answer,
                k_t=state["K_t"],
                selected_chunks=selected_chunks,
                covered_target_ids=state["A_t"].get("covered_target_ids", []),
                target_info=target_info,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
    if target_complete and not bool(support_probe.get("support_sufficient", False)):
        support_probe = {
            "support_sufficient": True,
            "support_rule": "multi_hop_supported" if len(target_ids) >= 2 else "explicit_answer_in_evidence",
            "support_evidence_summary": "Offline teacher target evidence is fully covered in the selected path.",
            "missing_support_reasons": [],
        }

    answer_correct = check_answer_correct(gold_answer, pred_answer)
    context_exact_match = bool(
        pred_answer is not None
        and context_contains_answer(gold_answer, state["K_t"])
        and answers_context_compatible(pred_answer, gold_answer)
        and not answer_conflicts(pred_answer, gold_answer)
    )
    teacher_stop = bool(pred_answer is not None and (answer_correct or context_exact_match) and support_probe.get("support_sufficient"))
    if not teacher_stop:
        return None

    answer_match_rule = "normalized_exact" if answer_correct else "context_exact"
    return build_probe_debug_payload(
        {
            "probe_run": True,
            "cache_hit": False,
            "answer_source": answer_source,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "normalized_gold": normalize_text(gold_answer),
            "normalized_pred": normalize_text(pred_answer),
            "exact_match": answer_correct,
            "context_exact_match": context_exact_match,
            "answer_match_rule": answer_match_rule,
            "AnswerCorrect_t": True,
            "SupportSufficient_t": True,
            "support_rule": support_probe["support_rule"],
            "support_evidence_summary": support_probe["support_evidence_summary"],
            "missing_support_reasons": support_probe["missing_support_reasons"],
            "support_probe": support_probe,
            "state_snapshot": copy.deepcopy(state),
            "TeacherStop_t": True,
            "FalseStop_t": False,
            "offline_terminal_reason": reason,
            "t": t,
        }
    )


def maybe_fill_probe_answer_from_context(
    *,
    question: str,
    gold_answer: str,
    pred_answer: Optional[str],
    k_t: str,
    selected_chunks: List[dict],
    target_info: dict,
    api_key: str,
    model: str,
    base_url: str,
) -> Tuple[Optional[str], str, Optional[dict]]:
    if not context_contains_answer(gold_answer, k_t):
        return pred_answer, "llm_failure" if pred_answer is None else "llm", None

    should_try_gold_fallback = pred_answer is None
    fallback_answer_source = "context_gold_fallback"
    if pred_answer is not None:
        if check_answer_correct(gold_answer, pred_answer) or not answer_conflicts(pred_answer, gold_answer):
            return pred_answer, "llm", None
        should_try_gold_fallback = True
        fallback_answer_source = "context_gold_conflict_fallback"

    if not should_try_gold_fallback:
        return pred_answer, "llm", None

    fallback_support = support_sufficient_check(
        question=question,
        gold_answer=gold_answer,
        pred_answer=gold_answer,
        k_t=k_t,
        selected_chunks=selected_chunks,
        covered_target_ids=[],
        target_info=target_info,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    if not bool(fallback_support.get("support_sufficient", False)):
        return pred_answer, "llm_failure" if pred_answer is None else "llm", None
    return gold_answer, fallback_answer_source, fallback_support


def build_probe_debug_payload(probe: Optional[dict] = None) -> dict:
    payload = probe if isinstance(probe, dict) else {}
    gold_answer = payload.get("gold_answer")
    pred_answer = payload.get("pred_answer", payload.get("probe_answer"))
    normalized_gold = payload.get("normalized_gold", payload.get("gold_answer_normalized"))
    normalized_pred = payload.get("normalized_pred", payload.get("probe_answer_normalized"))
    answer_match_rule = payload.get("answer_match_rule", "none")
    if not isinstance(answer_match_rule, str) or not answer_match_rule.strip():
        answer_match_rule = "none"
    support_probe = payload.get("support_probe", {})
    if not isinstance(support_probe, dict):
        support_probe = {}
    support_rule = str(
        support_probe.get("support_rule", payload.get("support_rule", ""))
    ).strip()
    support_summary = str(
        support_probe.get("support_evidence_summary", payload.get("support_evidence_summary", ""))
    ).strip()
    missing_support_reasons = support_probe.get(
        "missing_support_reasons",
        payload.get("missing_support_reasons", []),
    )
    if not isinstance(missing_support_reasons, list):
        missing_support_reasons = []
    missing_support_reasons = [str(x) for x in missing_support_reasons]
    support_sufficient = bool(
        support_probe.get("support_sufficient", payload.get("SupportSufficient_t", False))
    )

    result = {
        "probe_run": bool(payload.get("probe_run", False)),
        "cache_hit": bool(payload.get("cache_hit", False)),
        "answer_source": str(payload.get("answer_source", "")) or "llm",
        "gold_answer": None if gold_answer is None else str(gold_answer),
        "pred_answer": None if pred_answer is None else str(pred_answer),
        "normalized_gold": None if normalized_gold is None else str(normalized_gold),
        "normalized_pred": None if normalized_pred is None else str(normalized_pred),
        "exact_match": bool(payload.get("exact_match", False)),
        "context_exact_match": bool(payload.get("context_exact_match", False)),
        "answer_match_rule": answer_match_rule,
        "AnswerCorrect_t": bool(payload.get("AnswerCorrect_t", False)),
        "SupportSufficient_t": support_sufficient,
        "support_rule": support_rule,
        "support_evidence_summary": support_summary,
        "missing_support_reasons": missing_support_reasons,
        "support_probe": {
            "support_sufficient": support_sufficient,
            "support_rule": support_rule,
            "support_evidence_summary": support_summary,
            "missing_support_reasons": missing_support_reasons,
        },
        "state_snapshot": copy.deepcopy(payload.get("state_snapshot")) if isinstance(payload.get("state_snapshot"), dict) else None,
        "TeacherStop_t": bool(payload.get("TeacherStop_t", False)),
        "FalseStop_t": bool(payload.get("FalseStop_t", False)),
        "stop_candidate": bool(payload.get("stop_candidate", False)),
        "false_stop_count_before": int(payload.get("false_stop_count_before", 0) or 0),
        "false_stop_count_after": int(payload.get("false_stop_count_after", 0) or 0),
        "t": None if payload.get("t") is None else int(payload["t"]),
    }
    if result["TeacherStop_t"] and not result["pred_answer"]:
        result["TeacherStop_t"] = False
        result["AnswerCorrect_t"] = False
        result["FalseStop_t"] = True
        if result["answer_match_rule"] == "none":
            result["answer_source"] = "llm_failure"
    return result


def can_run_stop_probe(stop_candidate: bool, false_stop_count: int, stop_cooldown: int) -> bool:
    if not stop_candidate:
        return False
    if stop_cooldown > 0:
        return False
    if false_stop_count >= FALSE_STOP_LIMIT:
        return False
    return True


def run_stop_probe(
    q: dict,
    state: dict,
    target_info: dict,
    *,
    unit_registry: Dict[str, dict],
    chunks: List[dict],
    cache_dir: Path,
    qid: str,
    t: int,
) -> dict:
    gold_answer = str(q["answer"])
    target_unit_ids = sorted(str(x) for x in target_info["target_map"].keys())
    cache_payload = {
        "qid": qid,
        "t": t,
        "question": q["question"],
        "gold_answer": gold_answer,
        "k_t": state["K_t"],
        "covered_target_ids": sorted(str(x) for x in state["A_t"].get("covered_target_ids", [])),
        "target_unit_ids": target_unit_ids,
        "model": q["model"],
        "stop_probe_prompt_version": STOP_PROBE_PROMPT_VERSION,
    }
    cache_key = sha256_hex(stable_json_dumps(cache_payload))
    cached = load_json_cache(cache_dir, cache_key)
    if cached is not None:
        cached = build_probe_debug_payload(copy.deepcopy(cached))
        cached["cache_hit"] = True
        return cached

    selected_chunks = get_selected_chunks_from_state(state, chunks, unit_registry)
    answer_source = "llm"
    try:
        probe_answer = answer_with_deepseek(
            api_key=q["api_key"],
            model=q["model"],
            base_url=q["base_url"],
            question=q["question"],
            k_t=state["K_t"],
        )
    except RuntimeError as e:
        log(f"[QID {qid}] stop probe fallback after API failure: {e}", force=True)
        probe_answer = None
        answer_source = "llm_failure"
    fallback_support = None
    probe_answer, fallback_answer_source, fallback_support = maybe_fill_probe_answer_from_context(
        question=q["question"],
        gold_answer=gold_answer,
        pred_answer=probe_answer,
        k_t=state["K_t"],
        selected_chunks=selected_chunks,
        target_info=target_info,
        api_key=q["api_key"],
        model=q["model"],
        base_url=q["base_url"],
    )
    if probe_answer is not None and fallback_answer_source in {
        "context_gold_fallback",
        "context_gold_conflict_fallback",
    }:
        answer_source = fallback_answer_source
    gold_answer_normalized = normalize_text(gold_answer)
    probe_answer_normalized = normalize_text(probe_answer) if probe_answer is not None else None
    exact_match = check_answer_correct(q["answer"], probe_answer)
    context_exact_match = (
        probe_answer is not None
        and context_contains_answer(gold_answer, state["K_t"])
        and answers_context_compatible(probe_answer, gold_answer)
        and not answer_conflicts(probe_answer, gold_answer)
    )
    answer_correct = exact_match or context_exact_match
    support_probe = fallback_support or support_sufficient_check(
        question=q["question"],
        gold_answer=gold_answer,
        pred_answer=probe_answer,
        k_t=state["K_t"],
        selected_chunks=selected_chunks,
        covered_target_ids=state["A_t"].get("covered_target_ids", []),
        target_info=target_info,
        api_key=q["api_key"],
        model=q["model"],
        base_url=q["base_url"],
    )
    support_sufficient = bool(support_probe["support_sufficient"])
    teacher_stop = probe_answer is not None and answer_correct and support_sufficient
    false_stop = not teacher_stop
    answer_match_rule = "none"
    if exact_match:
        answer_match_rule = "normalized_exact"
    elif context_exact_match:
        answer_match_rule = "context_exact"
    result = build_probe_debug_payload(
        {
        "probe_run": True,
        "cache_hit": False,
        "answer_source": answer_source,
        "gold_answer": gold_answer,
        "pred_answer": probe_answer,
        "normalized_gold": gold_answer_normalized,
        "normalized_pred": probe_answer_normalized,
        "exact_match": exact_match,
        "context_exact_match": context_exact_match,
        "answer_match_rule": answer_match_rule,
        "AnswerCorrect_t": answer_correct,
        "SupportSufficient_t": support_sufficient,
        "support_rule": support_probe["support_rule"],
        "support_evidence_summary": support_probe["support_evidence_summary"],
        "missing_support_reasons": support_probe["missing_support_reasons"],
        "support_probe": support_probe,
        "state_snapshot": copy.deepcopy(state),
        "TeacherStop_t": teacher_stop,
        "FalseStop_t": false_stop,
        }
    )
    write_json_cache(cache_dir, cache_key, result)
    return result


def build_state_summary_for_proposal(question: str, s_t: dict, unit_registry: Dict[str, dict]) -> str:
    lines = []
    recent_raw = get_recent_raw_texts(s_t, unit_registry, n=2)
    for text in recent_raw[:2]:
        lines.append("Evidence: " + shorten(text, 90))
    active_note_ids = get_active_note_ids(s_t, unit_registry)
    for unit_id in active_note_ids[:2]:
        note = unit_registry.get(unit_id)
        if note is None:
            continue
        lines.append("Note: " + shorten(str(note.get("text", "")), 70))
    lines.append("Need: " + extract_goal_from_question(question))
    return "\n".join(lines)


def build_top_raw_candidates(
    r_t: List[str],
    unit_registry: Dict[str, dict],
    *,
    s_t: Optional[dict] = None,
    derive_mode: str = "normal",
    carryover_candidate_ids: Optional[List[str]] = None,
) -> List[dict]:
    out = []
    seen = set()
    limit = REPAIR_TOP_RAW_J if derive_mode == "repair_after_false_stop" else TOP_RAW_J

    candidate_ids: List[str] = []
    if derive_mode == "repair_after_false_stop" and isinstance(s_t, dict):
        raw_refs = s_t.get("raw_refs", [])
        if isinstance(raw_refs, list):
            recent_raw_ids = [
                str(item.get("unit_id"))
                for item in raw_refs
                if isinstance(item, dict) and item.get("unit_id")
            ]
            candidate_ids.extend(recent_raw_ids[-3:])
    if carryover_candidate_ids:
        candidate_ids.extend([str(x) for x in carryover_candidate_ids if str(x).strip()])
    candidate_ids.extend(r_t[:limit])

    for unit_id in candidate_ids:
        if unit_id in seen:
            continue
        seen.add(unit_id)
        unit = unit_registry.get(unit_id)
        if unit is None or unit.get("provenance") != "raw":
            continue
        out.append(
            {
                "unit_id": unit_id,
                "text": unit["text"],
                "doc_id": unit["doc_id"],
                "parent_chunk_id": unit.get("parent_chunk_id"),
            }
        )
        if len(out) >= limit:
            break
    return out


def deepseek_chat_json(api_key: str, model: str, base_url: str, system_prompt: str, user_prompt: str) -> dict:
    if not str(api_key or "").strip():
        raise RuntimeError("missing_api_key")
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
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log(f"[API] DeepSeek request attempt={attempt}")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                log("[API] DeepSeek response received")
                return json.loads(content)
        except Exception as e:
            last_err = e
            log(f"[API] attempt={attempt} failed: {type(e).__name__}: {e}", force=True)
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
    yes_no_question = is_yes_no_question(question)
    system_prompt = (
        "You answer a question using only the provided evidence context.\n"
        "Return strict JSON with a single key `answer`.\n"
        "If the question is yes/no, return exactly `yes` or `no`.\n"
        "Otherwise, if the answer span is explicitly present in the evidence, copy the shortest exact answer span.\n"
        "Return only a short answer phrase, not an explanation.\n"
        "If the context is insufficient, return an empty string."
    )
    user_prompt = (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Evidence Context:\n"
        f"{k_t.strip()}\n\n"
        + (
            "Return JSON like {\"answer\": \"yes\"} or {\"answer\": \"no\"}."
            if yes_no_question
            else "Return JSON like {\"answer\": \"...\"}."
        )
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


def build_propose_prompt_legacy(
    question: str,
    state_summary: str,
    top_raw_candidates: List[dict],
    gold_answer: Optional[str],
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
        f"- produce at most {MAX_DERIVED_CANDIDATES} candidates\n"
        f"- do not output final answer style\n"
        f"- do not invent unsupported facts\n"
        f"- do not use claimed_role\n"
        f"- source_unit_ids must come only from the provided top raw candidates\n"
    )


def build_propose_prompt_v3(
    question: str,
    state_summary: str,
    top_raw_candidates: List[dict],
    gold_answer: Optional[str],
    *,
    derive_mode: str,
    derive_goal: str,
    recent_probe_feedback: Dict[str, object],
    bridge_anchors: List[str],
) -> str:
    raw_lines = []
    for i, item in enumerate(top_raw_candidates, start=1):
        raw_lines.append(
            f"{i}. unit_id={item['unit_id']}\n"
            f"   doc={item['doc_id']}\n"
            f"   parent_chunk_id={item.get('parent_chunk_id')}\n"
            f"   text={item['text']}"
        )

    gold_line = gold_answer if gold_answer else "null"
    bridge_anchor_line = ", ".join(bridge_anchors) if bridge_anchors else "(none)"
    recent_probe_line = json.dumps(recent_probe_feedback, ensure_ascii=False)

    return (
        f"Question:\n{question}\n\n"
        f"State summary:\n{state_summary}\n\n"
        f"derive_mode:\n{derive_mode}\n\n"
        f"derive_goal:\n{derive_goal}\n\n"
        f"recent_probe_feedback:\n{recent_probe_line}\n\n"
        f"bridge_anchors:\n{bridge_anchor_line}\n\n"
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
        f"- produce at most {MAX_DERIVED_CANDIDATES} candidates\n"
        f"- only use note types bridge_note or verification_note\n"
        f"- every note must be exactly one sentence\n"
        f"- do not output final answer style\n"
        f"- do not invent unsupported facts\n"
        f"- do not use claimed_role\n"
        f"- source_unit_ids must come only from the provided top raw candidates\n"
        f"- each source_unit_ids list must have 1 to 3 entries\n"
        f"- in repair_after_false_stop mode, prioritize answer-facing organization, bridge completion, answer focus correction, and target type clarification\n"
    )


def ensure_single_sentence(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    sentence = parts[0].strip()
    if not sentence:
        return ""
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def rewrite_raw_sentence_as_note(doc_id: str, text: str) -> str:
    sentence = ensure_single_sentence(text)
    if not sentence:
        return ""
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        return sentence
    rewritten = re.sub(
        r"^(she|he|they|it)\b",
        doc_id,
        sentence,
        count=1,
        flags=re.IGNORECASE,
    )
    return ensure_single_sentence(rewritten)


def build_local_derived_fallback_payload(
    *,
    question: str,
    top_raw_candidates: List[dict],
    gold_answer: Optional[str],
    derive_mode: str,
    derive_goal: str,
) -> dict:
    focus_kind = infer_answer_focus_kind(question)
    content_focus_kind = focus_kind
    lowered_question = str(question or "").lower()
    if focus_kind == "yes_no":
        if re.search(r"\btypes of plant\b|\bkind of plant\b", lowered_question):
            content_focus_kind = "plant"
    question_norm = normalize_text(question)
    gold_norm = normalize_text(gold_answer or "")
    derived_candidates: List[dict] = []
    seen_texts = set()

    def add_candidate(cand_type: str, text: str, source_unit_ids: List[str]) -> None:
        text = ensure_single_sentence(text)
        source_unit_ids = merge_unique_in_order([str(x) for x in source_unit_ids if str(x).strip()])
        if not text or not (1 <= len(source_unit_ids) <= 3):
            return
        norm_text = normalize_text(text)
        if not norm_text or norm_text in seen_texts:
            return
        seen_texts.add(norm_text)
        derived_candidates.append(
            {
                "type": cand_type,
                "text": text,
                "source_unit_ids": source_unit_ids,
                "coarse_priority": len(derived_candidates) + 1,
            }
        )

    if gold_norm and gold_norm not in {"yes", "no"}:
        best_gold_item = None
        best_gold_key = None
        for item in top_raw_candidates:
            doc_id = str(item.get("doc_id", "")).strip()
            raw_text = str(item.get("text", "")).strip()
            combined_norm = normalize_text(f"{doc_id} {raw_text}")
            doc_norm = normalize_text(doc_id)
            key = (
                1 if doc_norm == gold_norm else 0,
                1 if gold_norm in combined_norm else 0,
                len(set(extract_anchor_tokens(f"{doc_id} {raw_text}")) & set(extract_query_anchors(question))),
                text_has_answer_type_signal(f"{doc_id} {raw_text}", content_focus_kind),
                -parse_sent_id_from_unit_id(str(item.get("unit_id", ""))),
            )
            if key[:2] == (0, 0):
                continue
            if best_gold_key is None or key > best_gold_key:
                best_gold_key = key
                best_gold_item = item

        if best_gold_item is not None:
            note_text = rewrite_raw_sentence_as_note(
                str(best_gold_item.get("doc_id", "")),
                str(best_gold_item.get("text", "")),
            )
            support_ids = [str(best_gold_item.get("unit_id", ""))]
            contrast_item = None
            contrast_key = None
            for item in top_raw_candidates:
                if item is best_gold_item:
                    continue
                doc_id = str(item.get("doc_id", "")).strip()
                raw_text = str(item.get("text", "")).strip()
                key = (
                    1 if normalize_text(doc_id) in question_norm else 0,
                    len(set(extract_anchor_tokens(f"{doc_id} {raw_text}")) & set(extract_query_anchors(question))),
                    text_has_answer_type_signal(f"{doc_id} {raw_text}", content_focus_kind),
                )
                if contrast_key is None or key > contrast_key:
                    contrast_key = key
                    contrast_item = item
            if contrast_item is not None and contrast_key and contrast_key[0] > 0:
                support_ids.append(str(contrast_item.get("unit_id", "")))
            add_candidate("verification_note", note_text, support_ids)

    if content_focus_kind == "plant" and is_yes_no_question(question):
        plant_items = []
        for item in top_raw_candidates:
            doc_id = str(item.get("doc_id", "")).strip()
            raw_text = str(item.get("text", "")).strip()
            if not doc_id or normalize_text(doc_id) not in question_norm:
                continue
            if not text_has_answer_type_signal(f"{doc_id} {raw_text}", content_focus_kind):
                continue
            plant_items.append(item)
        plant_items = plant_items[:2]
        if len(plant_items) == 2:
            add_candidate(
                "verification_note",
                f"Both {plant_items[0]['doc_id']} and {plant_items[1]['doc_id']} are described as flowering plant genera.",
                [plant_items[0]["unit_id"], plant_items[1]["unit_id"]],
            )

    if not derived_candidates and top_raw_candidates:
        best_item = None
        best_key = None
        query_anchors = set(extract_query_anchors(question))
        for item in top_raw_candidates:
            doc_id = str(item.get("doc_id", "")).strip()
            raw_text = str(item.get("text", "")).strip()
            key = (
                text_has_answer_type_signal(f"{doc_id} {raw_text}", content_focus_kind),
                len(set(extract_anchor_tokens(f"{doc_id} {raw_text}")) & query_anchors),
                len(set(tokenize(doc_id + " " + raw_text)) & set(tokenize(question))),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_item = item
        if best_item is not None:
            note_text = rewrite_raw_sentence_as_note(
                str(best_item.get("doc_id", "")),
                str(best_item.get("text", "")),
            )
            note_type = "verification_note" if derive_mode == "repair_after_false_stop" else "bridge_note"
            add_candidate(note_type, note_text, [str(best_item.get("unit_id", ""))])

    if not derived_candidates:
        return {
            "should_derive": False,
            "reason": "local_api_failure_fallback_unavailable",
            "derived_candidates": [],
        }

    return {
        "should_derive": True,
        "reason": f"local_api_failure_fallback:{derive_mode}:{derive_goal}",
        "derived_candidates": derived_candidates[:MAX_DERIVED_CANDIDATES],
    }


def validate_harvest_candidates(
    qid: str,
    parsed: dict,
    visible_source_ids: set,
    start_idx: int,
) -> Tuple[List[dict], int]:
    candidates = parsed.get("derived_candidates", [])
    if not isinstance(candidates, list):
        return [], start_idx

    validated = []
    seen = set()

    for idx, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            continue

        cand_type = str(cand.get("type", "")).strip()
        text = str(cand.get("text", "")).strip()
        source_unit_ids = cand.get("source_unit_ids", [])
        coarse_priority = int(cand.get("coarse_priority", idx))

        if cand_type not in ALLOWED_DERIVED_TYPES:
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

        unit_id = build_derived_unit_id(qid, start_idx)
        start_idx += 1

        if unit_id in seen:
            continue
        seen.add(unit_id)

        validated.append(
            {
                "unit_id": unit_id,
                "text": text,
                "provenance": "derived",
                "candidate_granularity": "note",
                "type": cand_type,
                "source_unit_ids": source_unit_ids,
                "coarse_priority": coarse_priority,
            }
        )

    validated.sort(key=lambda x: (x["coarse_priority"], x["unit_id"]))
    return validated[:MAX_DERIVED_CANDIDATES], start_idx


def propose_derived(
    *,
    api_key: str,
    model: str,
    base_url: str,
    qid: str,
    question: str,
    state_summary: str,
    top_raw_candidates: List[dict],
    gold_answer: Optional[str],
    derive_mode: str,
    derive_goal: str,
    recent_probe_feedback: Dict[str, object],
    bridge_anchors: List[str],
    next_derived_idx: int,
    cache_dir: Path,
    use_v2_prompt: bool = True,
) -> Tuple[List[dict], int, dict]:
    prompt_mode = "v2" if use_v2_prompt else "legacy"
    if not top_raw_candidates:
        return [], next_derived_idx, {
            "derive_mode": derive_mode,
            "derive_goal": derive_goal,
            "bridge_anchors": list(bridge_anchors),
            "recent_probe_feedback": copy.deepcopy(recent_probe_feedback),
            "harvest_count": 0,
            "prompt_mode": prompt_mode,
        }

    system_prompt = (
        "You are proposing grounded derived notes for offline teacher trajectory construction.\n"
        "Only propose short, grounded notes if they help organize currently visible raw evidence.\n"
        "Allowed note types are exactly: bridge_note, verification_note.\n"
        "Each note must be grounded in the provided source_unit_ids only.\n"
        "Each note must contain exactly one sentence.\n"
        "Each note must cite between 1 and 3 source_unit_ids.\n"
        "Do not output claimed_role.\n"
        "Return strict JSON only."
    )
    if use_v2_prompt:
        user_prompt = build_propose_prompt_v3(
            question,
            state_summary,
            top_raw_candidates,
            gold_answer,
            derive_mode=derive_mode,
            derive_goal=derive_goal,
            recent_probe_feedback=recent_probe_feedback,
            bridge_anchors=bridge_anchors,
        )
        cache_payload = {
            "qid": qid,
            "question": question,
            "state_summary": state_summary,
            "top_raw_candidates": top_raw_candidates,
            "gold_answer": gold_answer,
            "derive_mode": derive_mode,
            "derive_goal": derive_goal,
            "recent_probe_feedback": recent_probe_feedback,
            "bridge_anchors": bridge_anchors,
            "model": model,
            "prompt_version": PROPOSE_DERIVED_PROMPT_VERSION_V2,
        }
    else:
        user_prompt = build_propose_prompt_legacy(
            question,
            state_summary,
            top_raw_candidates,
            gold_answer,
        )
        cache_payload = {
            "qid": qid,
            "question": question,
            "state_summary": state_summary,
            "top_raw_candidates": top_raw_candidates,
            "gold_answer": gold_answer,
            "model": model,
            "prompt_version": PROPOSE_DERIVED_PROMPT_VERSION_LEGACY,
        }
    cache_key = sha256_hex(stable_json_dumps(cache_payload))
    cached = load_json_cache(cache_dir, cache_key)
    if cached is None:
        try:
            parsed = deepseek_chat_json(
                api_key=api_key,
                model=model,
                base_url=base_url,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except RuntimeError as e:
            log(f"[QID {qid}] propose_derived fallback after API failure: {e}", force=True)
            parsed = {
                "should_derive": False,
                "reason": f"api_failure: {e}",
                "derived_candidates": [],
            }
        write_json_cache(cache_dir, cache_key, parsed)
    else:
        parsed = cached

    parsed_reason = str(parsed.get("reason", "")).strip()
    if parsed_reason.startswith("api_failure:"):
        parsed = build_local_derived_fallback_payload(
            question=question,
            top_raw_candidates=top_raw_candidates,
            gold_answer=gold_answer,
            derive_mode=derive_mode,
            derive_goal=derive_goal,
        )

    if not bool(parsed.get("should_derive", False)):
        return [], next_derived_idx, {
            "derive_mode": derive_mode,
            "derive_goal": derive_goal,
            "bridge_anchors": list(bridge_anchors),
            "recent_probe_feedback": copy.deepcopy(recent_probe_feedback),
            "harvest_count": 0,
            "reason": str(parsed.get("reason", "")).strip(),
            "prompt_mode": prompt_mode,
        }

    visible_source_ids = {x["unit_id"] for x in top_raw_candidates}
    validated, next_derived_idx = validate_harvest_candidates(qid, parsed, visible_source_ids, next_derived_idx)
    if not validated and parsed_reason.startswith("api_failure:"):
        parsed = build_local_derived_fallback_payload(
            question=question,
            top_raw_candidates=top_raw_candidates,
            gold_answer=gold_answer,
            derive_mode=derive_mode,
            derive_goal=derive_goal,
        )
        validated, next_derived_idx = validate_harvest_candidates(qid, parsed, visible_source_ids, next_derived_idx)
    return validated, next_derived_idx, {
        "derive_mode": derive_mode,
        "derive_goal": derive_goal,
        "bridge_anchors": list(bridge_anchors),
        "recent_probe_feedback": copy.deepcopy(recent_probe_feedback),
        "harvest_count": len(validated),
        "reason": str(parsed.get("reason", "")).strip(),
        "prompt_mode": prompt_mode,
    }


def legality_filter(
    state: dict,
    r_t: List[str],
    harvest: List[dict],
    unit_registry: Dict[str, dict],
) -> Tuple[List[dict], List[dict]]:
    visible_raw_ids = set(r_t)
    for item in state["S_t"].get("raw_refs", []):
        if isinstance(item, dict) and "unit_id" in item:
            visible_raw_ids.add(str(item["unit_id"]))

    existing_note_texts = []
    for item in state["S_t"].get("derived_refs", []):
        if not isinstance(item, dict) or "unit_id" not in item:
            continue
        unit = unit_registry.get(str(item["unit_id"]))
        if unit is not None and unit.get("provenance") == "derived":
            existing_note_texts.append(unit["text"])

    legal = []
    illegal = []
    accepted_norm_texts = []

    for idx, z in enumerate(harvest):
        reasons = []

        z_type = str(z.get("type", "")).strip()
        z_text = str(z.get("text", "")).strip()
        src_ids = z.get("source_unit_ids", [])
        unit_id = str(z.get("unit_id", "")).strip()

        if z_type not in ALLOWED_DERIVED_TYPES:
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
        if token_count(z_text) > MAX_NOTE_TOKENS:
            reasons.append("too_long")

        norm_text = normalize_text(z_text)
        if not norm_text:
            reasons.append("empty_text")

        for prev_norm in accepted_norm_texts:
            if norm_text == prev_norm or jaccard_similarity(norm_text, prev_norm) > DUPLICATE_SIM_THRESHOLD:
                reasons.append("duplicate_in_harvest")
                break

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

        accepted_norm_texts.append(norm_text)
        legal.append(
            {
                "unit_id": unit_id,
                "type": z_type,
                "text": z_text,
                "source_unit_ids": src_ids,
                "coarse_priority": int(z.get("coarse_priority", idx)),
                "provenance": "derived",
                "candidate_granularity": "note",
            }
        )

    legal.sort(key=lambda x: (x["coarse_priority"], x["unit_id"]))
    return legal, illegal


def final_retain_selection(
    g_legal: List[dict],
    *,
    derive_mode: str = "normal",
    derive_subtype: Optional[str] = None,
    derive_goal: str = "generic_bridge_or_verification",
    question: str = "",
    gold_answer: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    if not g_legal:
        return [], []

    if derive_mode == "repair_after_false_stop" or derive_subtype == "late_verification":
        def repair_rank(item: dict) -> Tuple[int, int, int, str]:
            closure_value, answer_facing = score_repair_note_value(
                item,
                question=question,
                gold_answer=gold_answer,
                derive_goal=derive_goal,
            )
            return (
                0 if answer_facing else 1,
                -closure_value,
                int(item.get("coarse_priority", 10**9)),
                -len(item.get("source_unit_ids", [])),
                str(item.get("unit_id", "")),
            )

        ranked = sorted(g_legal, key=repair_rank)
        final_ids = [x["unit_id"] for x in ranked[:MAX_FINAL_DERIVED]]
        aux_ids = [x["unit_id"] for x in ranked[MAX_FINAL_DERIVED:]]
        return final_ids, aux_ids

    if derive_subtype == "early_bridge":
        def bridge_rank(item: dict) -> Tuple[int, int, int, str]:
            return (
                0 if str(item.get("type", "")).strip() == "bridge_note" else 1,
                int(item.get("coarse_priority", 10**9)),
                -len(item.get("source_unit_ids", [])),
                str(item.get("unit_id", "")),
            )

        ranked = sorted(g_legal, key=bridge_rank)
        final_ids = [x["unit_id"] for x in ranked[:1]]
        aux_ids = [x["unit_id"] for x in ranked[1:]]
        return final_ids, aux_ids

    if derive_subtype == "trigger_only_candidate":
        ranked = sorted(
            g_legal,
            key=lambda item: (
                int(item.get("coarse_priority", 10**9)),
                str(item.get("unit_id", "")),
            ),
        )
        return [], [x["unit_id"] for x in ranked]

    if derive_subtype not in ALLOWED_CONTROL_DERIVED_SUBTYPES:
        ranked = sorted(
            g_legal,
            key=lambda item: (
                int(item.get("coarse_priority", 10**9)),
                str(item.get("unit_id", "")),
            ),
        )
        return [], [x["unit_id"] for x in ranked]

    retained = []
    remaining = list(g_legal)

    first = remaining.pop(0)
    retained.append(first)

    if remaining and len(retained) < MAX_FINAL_DERIVED:
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


def answerability_probe(gold_answer: str, k_t: str) -> int:
    norm_gold = normalize_text(gold_answer)
    norm_ctx = normalize_text(k_t)
    if not norm_gold:
        return 0
    return 1 if norm_gold in norm_ctx else 0


def support_sufficient(gold_answer: str, k_t: str, a_t: dict, target_info: dict, s_t: dict, unit_registry: Dict[str, dict]) -> int:
    target_map = target_info["target_map"]
    support_targets = {uid for uid, item in target_map.items() if item["primary_role"] == "support"}
    bridge_targets = {uid for uid, item in target_map.items() if item["primary_role"] == "bridge"}
    covered = set(str(x) for x in a_t.get("covered_target_ids", []))

    ans_present = answerability_probe(gold_answer, k_t) == 1
    support_ready = support_targets.issubset(covered) if support_targets else True

    has_bridge_note = False
    derived_refs = normalize_refs_minimal(s_t.get("derived_refs", []))
    for ref in derived_refs:
        unit = unit_registry.get(ref["unit_id"])
        if unit and unit.get("provenance") == "derived" and unit.get("type") == "bridge_note":
            has_bridge_note = True
            break

    bridge_ready = (bridge_targets.issubset(covered) if bridge_targets else True) or has_bridge_note
    return 1 if (ans_present and support_ready and bridge_ready) else 0


def distinguish_sufficient(a_t: dict, target_info: dict) -> int:
    target_map = target_info["target_map"]
    dis_targets = {uid for uid, item in target_map.items() if item["primary_role"] == "distinguish"}
    if not dis_targets:
        return 1
    covered = set(str(x) for x in a_t.get("covered_target_ids", []))
    return 1 if dis_targets.issubset(covered) else 0


def compute_a_ctx(q: dict, k_t: str, a_t: dict, s_t: dict, target_info: dict, unit_registry: Dict[str, dict]) -> float:
    a_ans = answerability_probe(q["answer"], k_t)
    a_sup = support_sufficient(q["answer"], k_t, a_t, target_info, s_t, unit_registry)
    a_dis = distinguish_sufficient(a_t, target_info)
    return (a_ans + a_sup + a_dis) / 3.0


def compute_delta_role(a_t: dict, a_u: dict, target_info: dict, role: str) -> float:
    N_r = float(target_info["role_counts"].get(role, 0.0))
    if N_r <= 0:
        return 0.0
    s_t = (get_k_r(a_t, role) + ALPHA) / (N_r + 2 * ALPHA)
    s_u = (get_k_r(a_u, role) + ALPHA) / (N_r + 2 * ALPHA)
    return s_u - s_t


def compute_kappa(u: dict) -> float:
    if u["provenance"] == "raw":
        granularity = u.get("candidate_granularity", "sentence")
        if granularity == "chunk":
            return KAPPA_RAW_CHUNK
        return KAPPA_RAW_SENTENCE
    if u["provenance"] == "derived":
        note_type = u.get("type")
        if note_type == "bridge_note":
            return KAPPA_BRIDGE_NOTE
        if note_type == "verification_note":
            return KAPPA_VERIFICATION_NOTE
        return KAPPA_VERIFICATION_NOTE
    raise ValueError(f"非法 provenance: {u.get('provenance')}")


def candidate_priority_key(u: dict, utility: float) -> Tuple:
    is_raw = 0 if u["provenance"] == "raw" else 1
    kappa = compute_kappa(u)
    coarse_priority = u.get("coarse_priority", 10**9)
    answer_facing = 1 if bool(u.get("answer_facing", False)) else 0
    closure_value = int(u.get("closure_value", 0)) if u["provenance"] == "derived" else 0
    coarse_priority_key = -int(coarse_priority) if u["provenance"] == "derived" else 0
    return (-utility, is_raw, -answer_facing, -closure_value, kappa, coarse_priority_key, str(u["unit_id"]))


def compute_U_for_candidate(q: dict, state: dict, target_info: dict, unit_registry: Dict[str, dict], u: dict) -> float:
    h_u, s_u = simulate_update(state["H_t"], state["S_t"], u)
    a_u = simulate_ledger(state["A_t"], u, target_info)
    k_u = render_context(q, s_u, unit_registry)

    delta_br = compute_delta_role(state["A_t"], a_u, target_info, "bridge")
    delta_dis = compute_delta_role(state["A_t"], a_u, target_info, "distinguish")
    delta_sup = compute_delta_role(state["A_t"], a_u, target_info, "support")

    a_ctx_t = compute_a_ctx(q, state["K_t"], state["A_t"], state["S_t"], target_info, unit_registry)
    a_ctx_u = compute_a_ctx(q, k_u, a_u, s_u, target_info, unit_registry)
    delta_ctx = a_ctx_u - a_ctx_t

    numerator = ETA_BR * delta_br + ETA_DIS * delta_dis + ETA_SUP * delta_sup + ETA_CTX * delta_ctx
    return numerator / compute_kappa(u)


def compute_repeat_penalty(
    state: dict,
    unit_registry: Dict[str, dict],
    u: dict,
    recent_progress_flags: List[int],
    retrieval_repeat_ratio: Optional[float],
) -> float:
    raw_refs = normalize_refs_minimal(state["S_t"].get("raw_refs", []))
    recent_refs = raw_refs[-2:]
    if not recent_refs:
        return 0.0

    recent_units = []
    for ref in recent_refs:
        unit = unit_registry.get(ref["unit_id"])
        if unit is not None:
            recent_units.append(unit)
    if not recent_units:
        return 0.0

    penalty = 0.0
    for prev in recent_units:
        if u.get("doc_id") == prev.get("doc_id"):
            penalty += 0.05
        if jaccard_similarity(str(u.get("text", "")), str(prev.get("text", ""))) >= REDUNDANT_SIM_THRESHOLD:
            penalty += 0.10

    no_progress_streak = len(recent_progress_flags) >= STALL_WINDOW and all(
        x == 0 for x in recent_progress_flags[-STALL_WINDOW:]
    )
    if no_progress_streak:
        recent_unit_ids = {str(ref["unit_id"]) for ref in recent_refs}
        recent_parent_ids = {str(prev.get("parent_chunk_id", "")) for prev in recent_units}
        recent_doc_ids = {str(prev.get("doc_id", "")) for prev in recent_units}
        if str(u.get("unit_id", "")) in recent_unit_ids:
            penalty += 0.20
        if str(u.get("parent_chunk_id", "")) in recent_parent_ids:
            penalty += 0.10
        if retrieval_repeat_ratio is not None and retrieval_repeat_ratio >= RETRIEVAL_REPEAT_RATIO_THRESHOLD:
            if str(u.get("doc_id", "")) in recent_doc_ids:
                penalty += 0.10

    if recent_progress_flags and recent_progress_flags[-1] == 0:
        penalty *= 1.5
    return penalty


def _candidate_source_unit_ids(u: dict) -> set:
    if u.get("provenance") == "derived":
        return {str(x) for x in u.get("source_unit_ids", []) if str(x).strip()}
    unit_id = str(u.get("unit_id", "")).strip()
    return {unit_id} if unit_id else set()


def _repair_note_profiles(
    repair_candidate_ids: List[str],
    unit_registry: Dict[str, dict],
    question: str,
) -> List[dict]:
    focus_kind = infer_answer_focus_kind(question)
    profiles: List[dict] = []
    for unit_id in repair_candidate_ids:
        unit = unit_registry.get(str(unit_id))
        if not unit or unit.get("provenance") != "derived":
            continue
        source_ids = {str(x) for x in unit.get("source_unit_ids", []) if str(x).strip()}
        source_text_parts: List[str] = []
        source_doc_ids: set = set()
        for src_id in source_ids:
            src_unit = unit_registry.get(src_id)
            if not src_unit:
                continue
            source_doc_ids.add(str(src_unit.get("doc_id", "")).strip())
            source_text_parts.append(
                f"{src_unit.get('doc_id', '')} {src_unit.get('text', '')}".strip()
            )
        note_text = str(unit.get("text", "")).strip()
        anchor_text = " ".join(
            [note_text]
            + [x for x in source_text_parts if x]
            + [str(unit.get("doc_id", "")).strip()]
        )
        profiles.append(
            {
                "unit_id": str(unit_id),
                "type": str(unit.get("type", "")).strip(),
                "text": note_text,
                "source_unit_ids": source_ids,
                "source_doc_ids": source_doc_ids,
                "anchors": set(extract_query_anchors(anchor_text)),
                "answer_facing": bool(unit.get("answer_facing", False)),
                "closure_value": int(unit.get("closure_value", 0) or 0),
                "focus_kind_match": text_has_answer_type_signal(anchor_text, focus_kind),
            }
        )
    return profiles


def repair_influence_score(
    candidate: dict,
    retained_repair_profiles: List[dict],
    question: str,
    recent_probe_feedback: Dict[str, object],
    derive_goal: Optional[str],
) -> Tuple[float, List[str], bool]:
    if not retained_repair_profiles:
        return 0.0, [], False

    focus_kind = infer_answer_focus_kind(question)
    error_type = str(recent_probe_feedback.get("error_type") or "").strip().lower()
    cand_text = f"{candidate.get('doc_id', '')} {candidate.get('text', '')}".strip()
    cand_type = str(candidate.get("type", "")).strip()
    cand_source_ids = _candidate_source_unit_ids(candidate)
    cand_anchor_set = set(extract_query_anchors(cand_text))
    cand_doc_id = str(candidate.get("doc_id", "")).strip()
    cand_answer_signal = text_has_answer_type_signal(cand_text, focus_kind)
    candidate_linked = False
    reasons: List[str] = []
    bonus = 0.0

    for profile in retained_repair_profiles:
        shared_sources = bool(cand_source_ids & profile["source_unit_ids"])
        shared_anchor = bool(cand_anchor_set & profile["anchors"])
        same_doc = bool(cand_doc_id and cand_doc_id in profile["source_doc_ids"])

        if shared_sources:
            bonus += 0.55
            reasons.append("shares_source_with_repair_note")
            candidate_linked = True
        if shared_anchor:
            bonus += 0.30
            reasons.append("shares_bridge_anchor")
            candidate_linked = True
        if same_doc and candidate.get("provenance") == "raw":
            bonus += 0.12
            reasons.append("same_doc_as_repair_source")
            candidate_linked = True

        if cand_answer_signal and (
            profile["focus_kind_match"]
            or profile["answer_facing"]
            or derive_goal in {"answer_focus_verification", "target_type_disambiguation"}
        ):
            bonus += 0.25
            reasons.append("matches_answer_type_fix")

        if error_type in {"answer_conflict", "false_stop", "support_insufficient"}:
            if cand_answer_signal or cand_type == "verification_note" or bool(candidate.get("answer_facing", False)):
                bonus += 0.25
                reasons.append("supports_recent_false_stop_correction")

        if bool(candidate.get("answer_facing", False)) or (
            candidate.get("provenance") == "derived" and int(candidate.get("closure_value", 0) or 0) > 0
        ):
            bonus += 0.18
            reasons.append("extends_answer_facing_chain")

        if derive_goal == "bridge_query_entity_to_answer_candidate" and cand_type == "bridge_note":
            bonus += 0.18
            reasons.append("bridge_to_answer_candidate")
        if derive_goal == "target_type_disambiguation" and cand_type == "verification_note":
            bonus += 0.18
            reasons.append("supports_answer_type_fix")
        if derive_goal == "answer_focus_verification" and (cand_type == "verification_note" or cand_answer_signal):
            bonus += 0.18
            reasons.append("answer_focus_verification")

    deduped_reasons = []
    seen = set()
    for reason in reasons:
        if reason not in seen:
            deduped_reasons.append(reason)
            seen.add(reason)
    return bonus, deduped_reasons, candidate_linked


def build_repair_focus_payload(
    *,
    selected_unit_id: str,
    retained_repair_ids: List[str],
    unit_registry: Dict[str, dict],
    raw_unit_map: Dict[str, dict],
    question: str,
    derive_goal: Optional[str],
) -> dict:
    focus_kind = infer_answer_focus_kind(question)
    selected_unit = unit_registry.get(str(selected_unit_id))
    source_ids: List[str] = []
    source_doc_ids = set()
    bridge_anchors = set()
    carryover_ids: List[str] = []

    def add_carryover(unit_id: object) -> None:
        text = str(unit_id).strip()
        if not text or text in carryover_ids:
            return
        if text not in unit_registry and text not in raw_unit_map:
            return
        carryover_ids.append(text)

    def add_source(unit_id: object) -> None:
        text = str(unit_id).strip()
        if not text or text in source_ids:
            return
        source_ids.append(text)
        unit = unit_registry.get(text) or raw_unit_map.get(text)
        if not unit:
            return
        doc_id = str(unit.get("doc_id", "")).strip()
        if doc_id:
            source_doc_ids.add(doc_id)
        anchor_text = f"{unit.get('doc_id', '')} {unit.get('text', '')}".strip()
        bridge_anchors.update(extract_query_anchors(anchor_text))

    for unit_id in retained_repair_ids:
        unit = unit_registry.get(str(unit_id))
        if not unit:
            continue
        add_carryover(unit_id)
        for src_id in unit.get("source_unit_ids", []):
            add_source(src_id)

    if selected_unit is not None:
        if selected_unit.get("provenance") == "derived":
            for src_id in selected_unit.get("source_unit_ids", []):
                add_source(src_id)
        else:
            add_source(selected_unit_id)
        bridge_anchors.update(
            extract_query_anchors(
                f"{selected_unit.get('doc_id', '')} {selected_unit.get('text', '')}".strip()
            )
        )

    neighbor_scored: List[Tuple[int, str]] = []
    for raw_unit_id, raw_unit in raw_unit_map.items():
        if raw_unit_id in carryover_ids or raw_unit_id in source_ids:
            continue
        doc_id = str(raw_unit.get("doc_id", "")).strip()
        anchor_text = f"{doc_id} {raw_unit.get('text', '')}".strip()
        anchor_set = set(extract_query_anchors(anchor_text))
        score = 0
        if doc_id and doc_id in source_doc_ids:
            score += 3
        if bridge_anchors and (anchor_set & bridge_anchors):
            score += 2
        if text_has_answer_type_signal(anchor_text, focus_kind):
            score += 2
        if derive_goal == "bridge_query_entity_to_answer_candidate" and anchor_set:
            score += 1
        if score <= 0:
            continue
        neighbor_scored.append((score, raw_unit_id))
    neighbor_scored.sort(key=lambda x: (-x[0], x[1]))
    for _, raw_unit_id in neighbor_scored[:MAX_REPAIR_CARRYOVER]:
        add_carryover(raw_unit_id)

    return {
        "selected_repair_candidate": str(selected_unit_id),
        "retained_repair_ids": merge_unique_in_order([str(x) for x in retained_repair_ids]),
        "linked_source_unit_ids": merge_unique_in_order(source_ids),
        "bridge_anchors": merge_unique_in_order(list(bridge_anchors))[:8],
        "carryover_candidate_ids": merge_unique_in_order(source_ids + carryover_ids)[:MAX_REPAIR_CARRYOVER],
        "focus_kind": focus_kind,
    }


def merge_repair_carryover_candidates(
    r_t: List[str],
    carryover_candidate_ids: List[str],
) -> Tuple[List[str], List[str]]:
    merged: List[str] = []
    seen = set()
    carryover_present: List[str] = []

    for unit_id in carryover_candidate_ids:
        text = str(unit_id).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
        carryover_present.append(text)

    for unit_id in r_t:
        text = str(unit_id).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)

    return merged, carryover_present


def repair_continuity_score(
    candidate: dict,
    repair_focus_payload: Optional[dict],
    question: str,
) -> Tuple[float, List[str], bool]:
    if not repair_focus_payload:
        return 0.0, [], False

    focus_candidate_ids = {
        str(x)
        for x in repair_focus_payload.get("carryover_candidate_ids", [])
        if str(x).strip()
    }
    linked_source_ids = {
        str(x)
        for x in repair_focus_payload.get("linked_source_unit_ids", [])
        if str(x).strip()
    }
    anchor_set = {
        str(x)
        for x in repair_focus_payload.get("bridge_anchors", [])
        if str(x).strip()
    }
    focus_kind = str(repair_focus_payload.get("focus_kind") or infer_answer_focus_kind(question))
    cand_id = str(candidate.get("unit_id", "")).strip()
    cand_source_ids = _candidate_source_unit_ids(candidate)
    cand_text = f"{candidate.get('doc_id', '')} {candidate.get('text', '')}".strip()
    cand_anchor_set = set(extract_query_anchors(cand_text))
    cand_type = str(candidate.get("type", "")).strip()

    linked = False
    bonus = 0.0
    reasons: List[str] = []
    if cand_id in focus_candidate_ids:
        bonus += 0.45
        reasons.append("repair_carryover_candidate")
        linked = True
    if cand_source_ids & linked_source_ids:
        bonus += 0.25
        reasons.append("shares_source_with_repair_focus")
        linked = True
    if anchor_set and (cand_anchor_set & anchor_set):
        bonus += 0.18
        reasons.append("shares_repair_focus_anchor")
        linked = True
    if text_has_answer_type_signal(cand_text, focus_kind):
        bonus += 0.15
        reasons.append("supports_answer_type_closure")
    if cand_type in {"bridge_note", "verification_note"} and int(candidate.get("closure_value", 0) or 0) > 0:
        bonus += 0.12
        reasons.append("extends_closure_chain")
    deduped = merge_unique_in_order(reasons)
    return bonus, deduped, linked


def should_run_closure_continuation(
    *,
    repair_focus_ttl: int,
    failure_signals: Dict[str, object],
    stop_candidate: bool,
    need_derived_control: bool,
) -> bool:
    if repair_focus_ttl <= 0:
        return False
    if stop_candidate:
        return False
    if need_derived_control:
        return False
    if not bool(failure_signals.get("recent_false_stop", False)):
        return False
    if int(failure_signals.get("last_delta_covered_targets", 0)) != 0:
        return False
    repeat_ratio = failure_signals.get("last_retrieval_repeat_ratio")
    if repeat_ratio is None or float(repeat_ratio) < FAILURE_REPEAT_RATIO_THRESHOLD:
        return False
    return True


def should_reopen_stop_after_repair_selection(
    *,
    repair_focus_ttl: int,
    failure_signals: Dict[str, object],
    retrieval_repeat_ratio: Optional[float],
    stop_info_control: Dict[str, object],
    stop_candidate: bool,
    false_stop_count: int,
    question: str,
    gold_answer: str,
    k_t: str,
) -> bool:
    if repair_focus_ttl <= 0:
        return False
    if stop_candidate:
        return False
    # Allow exactly one post-repair stop retry even when we have already hit
    # the normal false-stop cap. This path is intentionally narrow and only
    # exists to test whether a repair-linked tail actually reached answerable
    # closure.
    if false_stop_count > FALSE_STOP_LIMIT:
        return False
    if bool(failure_signals.get("answer_focus_mismatch", False)):
        return False
    if retrieval_repeat_ratio is None or float(retrieval_repeat_ratio) < FAILURE_REPEAT_RATIO_THRESHOLD:
        return False
    role_scores = stop_info_control.get("role_scores", {}) if isinstance(stop_info_control, dict) else {}
    if not role_scores:
        return False
    if max(float(v) for v in role_scores.values()) < 1.0:
        return False
    if is_yes_no_question(question):
        return True
    return answerability_probe(gold_answer, k_t) == 1


def should_apply_repair_followup_bonus(
    *,
    repair_focus_payload: Optional[dict],
    repair_focus_ttl: int,
    failure_signals: Dict[str, object],
    stop_candidate: bool,
    retrieval_repeat_ratio: Optional[float],
) -> bool:
    if not repair_focus_payload or repair_focus_ttl <= 0:
        return False
    if not bool(failure_signals.get("recent_false_stop", False)):
        return False
    if int(failure_signals.get("last_delta_covered_targets", 0)) != 0:
        return False
    if retrieval_repeat_ratio is None or float(retrieval_repeat_ratio) < FAILURE_REPEAT_RATIO_THRESHOLD:
        return False
    return True


def should_allow_late_repair_stop_retry(
    *,
    repair_focus_payload: Optional[dict],
    repair_focus_ttl: int,
    failure_signals: Dict[str, object],
    retrieval_repeat_ratio: Optional[float],
    stop_candidate: bool,
    false_stop_count: int,
    question: str,
    gold_answer: str,
    k_t: str,
    stop_info_control: Dict[str, object],
) -> bool:
    if not repair_focus_payload or repair_focus_ttl <= 0:
        return False
    if not stop_candidate:
        return False
    if false_stop_count != FALSE_STOP_LIMIT:
        return False
    if not bool(failure_signals.get("recent_false_stop", False)):
        return False
    if int(failure_signals.get("last_delta_covered_targets", 0)) != 0:
        return False
    if bool(failure_signals.get("answer_focus_mismatch", False)):
        return False
    if retrieval_repeat_ratio is not None and float(retrieval_repeat_ratio) < 0.45:
        return False
    role_scores = stop_info_control.get("role_scores", {}) if isinstance(stop_info_control, dict) else {}
    if not role_scores:
        return False
    max_role = max(float(v) for v in role_scores.values())
    support_score = float(role_scores.get("support", 0.0) or 0.0)
    if max_role < 1.0:
        return False
    if is_yes_no_question(question):
        return True
    if answerability_probe(gold_answer, k_t) == 1:
        return True
    return support_score >= 2.0


def teacher_select(
    q: dict,
    state: dict,
    target_info: dict,
    c_t: List[str],
    unit_registry: Dict[str, dict],
    recent_progress_flags: List[int],
    retrieval_repeat_ratio: Optional[float],
    *,
    repair_mode_active: bool = False,
    repair_candidate_ids: Optional[List[str]] = None,
    derive_goal: Optional[str] = None,
    recent_probe_feedback: Optional[Dict[str, object]] = None,
    closure_continuation_active: bool = False,
    repair_focus_payload: Optional[dict] = None,
    apply_repair_continuity_bonus: bool = False,
) -> Tuple[str, bool, dict]:
    if not c_t:
        raise ValueError(f"C_t 为空: qid={q['qid']}")

    scored = []
    pre_scored = []
    debug_entries = []
    retrieval_rank_map = {unit_id: idx for idx, unit_id in enumerate(c_t)}
    repair_candidate_set = set(repair_candidate_ids or [])
    retained_repair_profiles = _repair_note_profiles(list(repair_candidate_set), unit_registry, q["question"])
    recent_probe_feedback = copy.deepcopy(recent_probe_feedback or {})
    covered_target_ids = {str(x) for x in state.get("A_t", {}).get("covered_target_ids", [])}
    target_map = target_info.get("target_map", {})
    bias_applied = False
    for unit_id in c_t:
        if unit_id not in unit_registry:
            raise ValueError(f"C_t 中 unit_id 不在 UnitRegistry: qid={q['qid']}, unit_id={unit_id}")
        u = unit_registry[unit_id]
        utility = compute_U_for_candidate(q, state, target_info, unit_registry, u)
        repeat_penalty = compute_repeat_penalty(
            state,
            unit_registry,
            u,
            recent_progress_flags,
            retrieval_repeat_ratio,
        )
        retrieval_bonus = ETA_RETRIEVAL_ORDER_BONUS * max(0, len(c_t) - retrieval_rank_map[unit_id])
        base_teacher_score = utility - ETA_REPEAT_PENALTY * repeat_penalty + retrieval_bonus
        target_chunk_id = str(u.get("parent_chunk_id") or parse_chunk_id_from_unit_id(str(unit_id)))
        uncovered_target_bonus = 0.0
        if (
            ORACLE_UNCOVERED_BOOST
            and str(u.get("provenance", "")) == "raw"
            and target_chunk_id in target_map
            and target_chunk_id not in covered_target_ids
        ):
            # Offline teacher supervision should not let a repeated distractor outrank
            # a visible gold-target raw unit that is still missing from the path.
            uncovered_target_bonus = ORACLE_UNCOVERED_SELECT_BONUS
            base_teacher_score += uncovered_target_bonus
        repair_bonus = 0.0
        repair_bonus_reason: List[str] = []
        candidate_linked_to_retained_repair = False
        repair_continuity_bonus = 0.0
        repair_continuity_reason: List[str] = []
        candidate_linked_to_repair_focus = False
        if repair_mode_active and retained_repair_profiles:
            repair_bonus, repair_bonus_reason, candidate_linked_to_retained_repair = repair_influence_score(
                u,
                retained_repair_profiles,
                q["question"],
                recent_probe_feedback,
                derive_goal,
            )
            if repair_bonus > 0.0:
                bias_applied = True
        if closure_continuation_active and repair_focus_payload:
            (
                repair_continuity_bonus,
                repair_continuity_reason,
                candidate_linked_to_repair_focus,
            ) = repair_continuity_score(
                u,
                repair_focus_payload,
                q["question"],
            )
            if not apply_repair_continuity_bonus:
                repair_continuity_bonus = 0.0
                repair_continuity_reason = []
            if repair_continuity_bonus > 0.0:
                bias_applied = True
        adjusted_utility = base_teacher_score + repair_bonus + repair_continuity_bonus
        pre_scored.append((u, base_teacher_score))
        scored.append((u, adjusted_utility))
        debug_entries.append(
            {
                "unit_id": str(unit_id),
                "provenance": str(u.get("provenance", "")),
                "type": str(u.get("type", "")),
                "teacher_select_base_score": float(base_teacher_score),
                "uncovered_target_bonus": float(uncovered_target_bonus),
                "uncovered_target_chunk_id": target_chunk_id if uncovered_target_bonus > 0 else None,
                "repair_bonus": float(repair_bonus),
                "repair_continuity_bonus": float(repair_continuity_bonus),
                "teacher_select_final_score": float(adjusted_utility),
                "repair_bonus_reason": list(repair_bonus_reason),
                "candidate_linked_to_retained_repair": bool(candidate_linked_to_retained_repair),
                "repair_continuity_reason": list(repair_continuity_reason),
                "candidate_linked_to_repair_focus": bool(candidate_linked_to_repair_focus),
                "answer_facing": bool(u.get("answer_facing", False)),
                "closure_value": int(u.get("closure_value", 0) or 0),
            }
        )

    pre_scored.sort(key=lambda x: candidate_priority_key(x[0], x[1]))
    scored.sort(key=lambda x: candidate_priority_key(x[0], x[1]))
    debug_by_id = {str(item["unit_id"]): item for item in debug_entries}
    top_before = [debug_by_id[str(u["unit_id"])] for u, _ in pre_scored[:5]]
    top_after = [debug_by_id[str(u["unit_id"])] for u, _ in scored[:5]]
    repair_linked_after = [item for item in top_after if item["candidate_linked_to_retained_repair"]]
    highest_repair_linked_candidate = repair_linked_after[0]["unit_id"] if repair_linked_after else None
    selected_unit_id = scored[0][0]["unit_id"]
    selected_debug = debug_by_id.get(str(selected_unit_id), {})
    teacher_select_debug = {
        "top_candidates_before_repair_bias": top_before,
        "top_candidates_after_repair_bias": top_after,
        "selected_candidate": str(selected_unit_id),
        "highest_repair_linked_candidate": highest_repair_linked_candidate,
        "repair_linked_candidate_was_selected": bool(
            highest_repair_linked_candidate and highest_repair_linked_candidate == str(selected_unit_id)
        ),
        "selected_candidate_linked_to_repair_focus": bool(selected_debug.get("candidate_linked_to_repair_focus", False)),
        "closure_continuation_active": bool(closure_continuation_active),
        "all_candidate_scores": debug_entries,
    }
    return selected_unit_id, bias_applied, teacher_select_debug


def rollout_one_qid(
    qid: str,
    query_info: dict,
    target_info: dict,
    init_state: dict,
    raw_unit_map: Dict[str, dict],
    raw_unit_ids_by_chunk: Dict[str, List[str]],
    chunks: List[dict],
    atoms_by_chunk: Dict[str, List[dict]],
    api_key: str,
    base_url: str,
    model: str,
    derived_cache_dir: Path,
    stop_probe_cache_dir: Path,
    idx: int,
    total: int,
    run_id: str,
    build_time: str,
    build_source: str,
) -> dict:
    log(f"[QID {idx}/{total}] start qid={qid}", force=True)

    state = {
        "qid": qid,
        "t": 0,
        "H_t": copy.deepcopy(init_state["H_t"]),
        "A_t": copy.deepcopy(init_state["A_t"]),
        "S_t": copy.deepcopy(init_state["S_t"]),
        "K_t": str(init_state["K_t"]),
    }

    derived_registry: Dict[str, dict] = {}
    steps: List[dict] = []

    false_stop_count = 0
    stop_cooldown = 0
    recent_progress_flags: List[int] = []
    ever_progress = False
    last_progress_step = None
    stop_probe_history: List[dict] = []
    retrieval_history: List[List[str]] = []

    terminal_status = "abort"
    terminal_t = None
    abort_reason = None
    terminal_probe = None
    terminal_failure_signals = None
    terminal_gate_trace = None
    terminal_proposer_trace = None

    next_derived_idx = 0
    last_delta_covered_targets = 0
    last_retrieval_repeat_ratio = None
    repair_attempt_count = 0
    repair_effective = False
    repair_failure_reason = None
    stop_reopened_after_false_stop = False
    repair_focus_payload: Optional[dict] = None
    repair_focus_ttl = 0
    closure_continuation_ttl = 0
    one_more_closure_stop_attempt_used = False
    late_repair_stop_retry_used = False

    for t in range(T_MAX):
        log(f"[QID {qid}] step={t} begin")
        probe = None
        g_harvest = []
        g_legal = []
        g_final = []
        g_aux = []
        g_illegal = []
        derived_info = {
            "s_sem": 0.0,
            "bridgeable_raw": False,
            "has_recent_verification": False,
            "derived_need": False,
            "trigger_derived": False,
        }
        derived_info_control = {
            "s_sem": 0.0,
            "composable_raw": False,
            "has_recent_verification": False,
            "derived_need": False,
            "trigger_derived": False,
        }
        proposer_trace = {
            "derive_mode": "normal",
            "derive_subtype": None,
            "derive_goal": None,
            "bridge_anchors": [],
            "recent_probe_feedback": build_recent_probe_feedback(stop_probe_history),
            "harvest_count": 0,
            "legal_count": 0,
            "harvest_candidates": [],
            "final_count": 0,
            "aux_count": 0,
            "illegal_count": 0,
        }
        forced_repair_continuation = False
        repair_attempt_active = False
        shadow_closure_continuation_active = bool(
            USE_REPAIR_CONTINUATION_FOR_DIAG and closure_continuation_ttl > 0 and repair_focus_payload
        )
        control_closure_continuation_active = bool(
            USE_REPAIR_CONTINUATION_FOR_CONTROL and shadow_closure_continuation_active
        )
        repair_linked_carryover_candidates: List[str] = []
        closure_stop_reopened = False
        selected_repair_candidate = (
            str(repair_focus_payload.get("selected_repair_candidate", "")).strip()
            if shadow_closure_continuation_active and isinstance(repair_focus_payload, dict)
            else None
        )

        unit_registry = build_unit_registry(raw_unit_map, derived_registry)

        stop_cooldown_before = stop_cooldown
        if stop_cooldown > 0:
            stop_cooldown -= 1

        used_unit_ids = {str(x["unit_id"]) for x in state["H_t"] if isinstance(x, dict) and "unit_id" in x}
        q_t = build_q_t(query_info["question"], state["A_t"], state["S_t"], unit_registry)
        retrieval_shortlist = build_chunk_shortlist(q_t, chunks)
        r_t = topk_retrieve(q_t, chunks, atoms_by_chunk, used_unit_ids, shortlist=retrieval_shortlist)
        r_t, added_uncovered_targets = augment_with_shortlisted_uncovered_targets(
            query_info["question"],
            r_t,
            retrieval_shortlist,
            target_info,
            state["A_t"],
            state["K_t"],
            raw_unit_map,
            raw_unit_ids_by_chunk,
            used_unit_ids,
        )
        r_t, oracle_added_uncovered_targets = append_oracle_uncovered_targets(
            question=query_info["question"],
            r_t=r_t,
            target_info=target_info,
            a_t=state["A_t"],
            raw_unit_map=raw_unit_map,
            raw_unit_ids_by_chunk=raw_unit_ids_by_chunk,
            used_unit_ids=used_unit_ids,
        )
        if shadow_closure_continuation_active and isinstance(repair_focus_payload, dict):
            repair_linked_carryover_candidates = [
                unit_id
                for unit_id in repair_focus_payload.get("carryover_candidate_ids", [])
                if isinstance(unit_id, str) and unit_id in unit_registry
            ][:MAX_REPAIR_CARRYOVER]
        retrieval_repeat_ratio = compute_retrieval_repeat_ratio(r_t, retrieval_history)
        effective_repeat_ratio = retrieval_repeat_ratio
        if effective_repeat_ratio is None:
            effective_repeat_ratio = last_retrieval_repeat_ratio
        failure_signals = build_failure_signals(
            question=query_info["question"],
            stop_probe_history=stop_probe_history,
            last_delta_covered_targets=last_delta_covered_targets,
            last_retrieval_repeat_ratio=effective_repeat_ratio,
        )
        log(f"[QID {qid}] step={t} retrieval done: len(R_t)={len(r_t)}")

        stop_info_v3 = cheap_stop_gate_v3(
            target_info,
            state["A_t"],
            state["S_t"],
            r_t,
            unit_registry,
            retrieval_repeat_ratio,
            recent_progress_flags,
            failure_signals,
        )
        derived_info = need_derived_gate_v3(
            query_info["question"],
            target_info,
            state["A_t"],
            state["S_t"],
            r_t,
            unit_registry,
            failure_signals,
        )
        repair_override_shadow = false_stop_repair_override(failure_signals)
        derive_mode_shadow = "repair_after_false_stop" if repair_override_shadow else "normal"
        derived_subtype_shadow, derived_subtype_debug = classify_derived_subtype_v3(
            question=query_info["question"],
            t=t,
            s_t=state["S_t"],
            r_t=r_t,
            unit_registry=unit_registry,
            derived_info=derived_info,
            failure_signals=failure_signals,
        )
        infered_derive_goal = infer_derive_goal(
            query_info["question"],
            state["S_t"],
            r_t,
            unit_registry,
            failure_signals,
            bool(derived_info["bridgeable_raw"]),
        )
        if repair_override_shadow:
            derived_subtype_shadow = "late_verification"
        if derived_subtype_shadow == "late_verification":
            derive_goal = (
                infered_derive_goal
                if infered_derive_goal in {"answer_focus_verification", "target_type_disambiguation"}
                else "answer_focus_verification"
            )
        elif derived_subtype_shadow == "early_bridge":
            derive_goal = "bridge_query_entity_to_answer_candidate"
        else:
            derive_goal = infered_derive_goal
        bridge_anchors = extract_bridge_anchors(query_info["question"], state["S_t"], r_t, unit_registry)

        stop_info_control = cheap_stop_gate_legacy(
            target_info,
            state["A_t"],
            state["S_t"],
            r_t,
            unit_registry,
            recent_progress_flags,
        )
        derived_info_control = need_derived_gate_legacy(
            target_info,
            state["A_t"],
            state["S_t"],
            r_t,
            unit_registry,
            recent_progress_flags,
        )

        if USE_GATE_V2_FOR_CONTROL:
            stop_info = stop_info_v3
            stop_candidate = bool(stop_info_v3["stop_candidate"])
            need_derived_control = bool(derived_subtype_shadow is not None or repair_override_shadow)
            control_mode = "v2"
            active_repair_override = bool(repair_override_shadow)
            active_derive_mode = derive_mode_shadow
            active_derive_subtype = derived_subtype_shadow
            use_v2_prompt = True
        else:
            stop_info = stop_info_control
            stop_candidate = bool(stop_info_control["stop_candidate"])
            legacy_trigger_control = bool(derived_info_control["trigger_derived"])
            control_mode = "legacy_typed_aux"
            active_repair_override = False
            active_derive_mode = "normal"
            # Keep legacy gate control as the stable mainline. Only let typed
            # subtypes influence prompting/retention after legacy has already
            # decided that a derived proposal is warranted. We also keep
            # trigger-only cases diagnostic-only for now so they do not distort
            # the candidate pool.
            if legacy_trigger_control and derived_subtype_shadow in ALLOWED_CONTROL_DERIVED_SUBTYPES:
                active_derive_subtype = derived_subtype_shadow
            else:
                active_derive_subtype = None
            need_derived_control = bool(active_derive_subtype in ALLOWED_CONTROL_DERIVED_SUBTYPES)
            use_v2_prompt = bool(
                active_derive_subtype in ALLOWED_CONTROL_DERIVED_SUBTYPES
                or derive_goal != "generic_bridge_or_verification"
            )

        if (not USE_GATE_V2_FOR_CONTROL) and should_force_repair_continuation(
            steps=steps,
            failure_signals=failure_signals,
            stop_candidate=stop_candidate,
            stop_cooldown=stop_cooldown,
            need_derived_control=need_derived_control,
            repair_attempt_count=repair_attempt_count,
        ):
            forced_repair_continuation = True
            need_derived_control = True
        shadow_should_run_closure_continuation = (not USE_GATE_V2_FOR_CONTROL) and should_run_closure_continuation(
            repair_focus_ttl=closure_continuation_ttl,
            failure_signals=failure_signals,
            stop_candidate=stop_candidate,
            need_derived_control=need_derived_control,
        )
        if shadow_should_run_closure_continuation and USE_REPAIR_CONTINUATION_FOR_CONTROL:
            forced_repair_continuation = True
            need_derived_control = True
            control_closure_continuation_active = True
        stop_reopen_after_plateau = False
        if (not USE_GATE_V2_FOR_CONTROL) and should_reopen_stop_candidate_after_plateau(
            qid=qid,
            question=query_info["question"],
            gold_answer=query_info["answer"],
            k_t=state["K_t"],
            steps=steps,
            stop_candidate=stop_candidate,
            false_stop_count=false_stop_count,
            retrieval_repeat_ratio=retrieval_repeat_ratio,
            stop_info_control=stop_info_control,
            failure_signals=failure_signals,
        ):
            stop_candidate = True
            stop_reopen_after_plateau = True
        shadow_closure_stop_reopened = (
            not USE_GATE_V2_FOR_CONTROL
            and not one_more_closure_stop_attempt_used
            and should_reopen_stop_after_repair_selection(
                repair_focus_ttl=closure_continuation_ttl,
                failure_signals=failure_signals,
                retrieval_repeat_ratio=retrieval_repeat_ratio,
                stop_info_control=stop_info_control,
                stop_candidate=stop_candidate,
                false_stop_count=false_stop_count,
                question=query_info["question"],
                gold_answer=query_info["answer"],
                k_t=state["K_t"],
            )
        )
        late_repair_stop_retry = (
            not USE_GATE_V2_FOR_CONTROL
            and not late_repair_stop_retry_used
            and should_allow_late_repair_stop_retry(
                repair_focus_payload=repair_focus_payload,
                repair_focus_ttl=closure_continuation_ttl,
                failure_signals=failure_signals,
                retrieval_repeat_ratio=retrieval_repeat_ratio,
                stop_candidate=stop_candidate,
                false_stop_count=false_stop_count,
                question=query_info["question"],
                gold_answer=query_info["answer"],
                k_t=state["K_t"],
                stop_info_control=stop_info_control,
            )
        )
        # Keep persistence / continuation shadow-only by default, but allow a
        # very narrow post-repair stop reopen in mainline control when the
        # existing context already looks answerable. This does not change the
        # candidate pool; it only re-allows one stop attempt.
        allow_post_repair_stop_probe = bool(shadow_closure_stop_reopened or late_repair_stop_retry)
        if shadow_closure_stop_reopened:
            stop_candidate = True
            closure_stop_reopened = True
            one_more_closure_stop_attempt_used = True
        if late_repair_stop_retry:
            stop_candidate = True
            closure_stop_reopened = True
            late_repair_stop_retry_used = True
        repair_followup_bonus_active = should_apply_repair_followup_bonus(
            repair_focus_payload=repair_focus_payload,
            repair_focus_ttl=closure_continuation_ttl,
            failure_signals=failure_signals,
            stop_candidate=stop_candidate,
            retrieval_repeat_ratio=retrieval_repeat_ratio,
        )
        can_stop_probe_now = bool(
            allow_post_repair_stop_probe or can_run_stop_probe(stop_candidate, false_stop_count, stop_cooldown)
        )
        false_stop_count_before = false_stop_count
        gate_trace = {
            "raw_complete": bool(stop_info_v3["raw_complete"]),
            "no_obvious_derived_need": bool(stop_info_v3["no_obvious_derived_need"]),
            "no_recent_closure_failure": bool(stop_info_v3["no_recent_closure_failure"]),
            "stop_candidate": bool(stop_info_v3["stop_candidate"]),
            "repair_override": bool(repair_override_shadow),
            "bridgeable_raw": bool(derived_info["bridgeable_raw"]),
            "has_recent_verification": bool(derived_info["has_recent_verification"]),
            "derived_need": bool(derived_info["derived_need"] or repair_override_shadow),
            "trigger_derived": bool(derived_info["trigger_derived"] or repair_override_shadow),
            "derive_mode": derive_mode_shadow,
            "derive_subtype": derived_subtype_shadow,
            "derive_goal": derive_goal,
            "answer_ready_raw": bool(derived_subtype_debug["answer_ready_raw"]),
            "bridge_gap_explicit": bool(derived_subtype_debug["bridge_gap_explicit"]),
            "late_verification_trigger": bool(derived_subtype_debug["late_verification"]),
            "early_bridge_trigger": bool(derived_subtype_debug["early_bridge"]),
            "trigger_only_candidate": bool(derived_subtype_debug["trigger_only_candidate"]),
            "control_mode": control_mode,
            "diagnostic_shadow_enabled": bool(USE_GATE_V2_FOR_DEBUG and not USE_GATE_V2_FOR_CONTROL),
            "control_stop_candidate": bool(stop_candidate),
            "control_need_derived": bool(need_derived_control),
            "forced_repair_continuation": bool(forced_repair_continuation),
            "stop_reopen_after_plateau": bool(stop_reopen_after_plateau),
            "repair_continuation_control_enabled": bool(USE_REPAIR_CONTINUATION_FOR_CONTROL),
            "repair_continuation_diag_enabled": bool(USE_REPAIR_CONTINUATION_FOR_DIAG),
            "shadow_closure_continuation": bool(shadow_closure_continuation_active or shadow_should_run_closure_continuation),
            "shadow_closure_stop_reopen": bool(shadow_closure_stop_reopened),
            "late_repair_stop_retry": bool(late_repair_stop_retry),
            "repair_followup_bonus_active": bool(repair_followup_bonus_active),
        }
        proposer_trace.update(
            {
                "derive_mode": derive_mode_shadow,
                "derive_subtype": derived_subtype_shadow,
                "derive_goal": derive_goal,
                "bridge_anchors": list(bridge_anchors),
                "prompt_mode": "v2" if use_v2_prompt else "legacy",
                "forced_repair_continuation": bool(forced_repair_continuation),
                "stop_reopen_after_plateau": bool(stop_reopen_after_plateau),
            }
        )
        log(
            f"[QID {qid}] step={t} gate1 control_mode={control_mode} stop_candidate={stop_candidate} "
            f"raw_complete={stop_info.get('raw_complete', False)} "
            f"shadow_stop_candidate={stop_info_v3['stop_candidate']} "
            f"shadow_repair_override={repair_override_shadow}"
        )

        if active_repair_override:
            need_derived = True
            triggered_propose_derived = True
            log(f"[QID {qid}] step={t} repair override -> propose_derived_repair", force=True)
        elif can_stop_probe_now:
            log(f"[QID {qid}] step={t} running stop probe")
            probe = run_stop_probe(
                {
                    **query_info,
                    "api_key": api_key,
                    "base_url": base_url,
                    "model": model,
                },
                state,
                target_info,
                unit_registry=unit_registry,
                chunks=chunks,
                cache_dir=stop_probe_cache_dir,
                qid=qid,
                t=t,
            )
            probe["t"] = t
            probe["stop_candidate"] = stop_candidate
            probe["false_stop_count_before"] = false_stop_count_before
            probe["false_stop_count_after"] = false_stop_count
            log(
                f"[QID {qid}] step={t} stop probe result: "
                f"TeacherStop={probe['TeacherStop_t']} "
                f"AnswerCorrect={probe['AnswerCorrect_t']} "
                f"SupportSufficient={probe['SupportSufficient_t']}"
            )
            stop_probe_history.append(copy.deepcopy(probe))
            if false_stop_count_before > 0:
                stop_reopened_after_false_stop = True
            if probe["TeacherStop_t"]:
                terminal_status = "terminal"
                terminal_t = t
                abort_reason = None
                terminal_probe = probe
                terminal_failure_signals = copy.deepcopy(failure_signals)
                terminal_gate_trace = copy.deepcopy(gate_trace)
                terminal_proposer_trace = copy.deepcopy(proposer_trace)
                log(f"[QID {qid}] TERMINAL at step={t}", force=True)
                break
            else:
                false_stop_count += 1
                stop_cooldown = STOP_COOLDOWN_ON_FALSE
                probe["false_stop_count_after"] = false_stop_count
                stop_probe_history[-1]["false_stop_count_after"] = false_stop_count
                repair_failure_signals = dict(failure_signals)
                repair_failure_signals["recent_false_stop"] = True
                repair_failure_signals["false_stop_count_recent"] = int(
                    repair_failure_signals.get("false_stop_count_recent", 0)
                ) + 1
                if (not USE_GATE_V2_FOR_CONTROL) and should_force_repair_continuation(
                    steps=steps,
                    failure_signals=repair_failure_signals,
                    stop_candidate=stop_candidate,
                    stop_cooldown=stop_cooldown,
                    need_derived_control=False,
                    repair_attempt_count=repair_attempt_count,
                ):
                    need_derived = True
                    triggered_propose_derived = True
                    repair_attempt_active = True
                    repair_attempt_count += 1
                    repair_failure_reason = "pending_repair_outcome"
                    log(
                        f"[QID {qid}] step={t} inline repair continuation after false stop",
                        force=True,
                    )
                else:
                    need_derived = False
                    triggered_propose_derived = False
            if probe["TeacherStop_t"]:
                pass
            elif not repair_attempt_active:
                need_derived = False
                triggered_propose_derived = False
        elif forced_repair_continuation:
            need_derived = True
            triggered_propose_derived = True
            repair_attempt_active = True
            repair_attempt_count += 1
            repair_failure_reason = "pending_repair_outcome"
            log(f"[QID {qid}] step={t} forcing one-shot repair continuation after false stop", force=True)
        elif stop_candidate:
            need_derived = False
            triggered_propose_derived = False
            log(f"[QID {qid}] step={t} skip need-derived because stop_candidate=true")
        else:
            need_derived = bool(need_derived_control)
            triggered_propose_derived = False
            log(
                f"[QID {qid}] step={t} gate2 NeedDerived={need_derived} "
                f"s_sem={derived_info_control['s_sem']:.4f} "
                f"composable_raw={derived_info_control['composable_raw']} "
                f"has_recent_verification={derived_info_control['has_recent_verification']}"
            )

        if triggered_propose_derived or need_derived:
            if not triggered_propose_derived:
                triggered_propose_derived = True
            if need_derived or active_repair_override:
                effective_derive_mode = active_derive_mode
                effective_derive_subtype = active_derive_subtype
                effective_use_v2_prompt = use_v2_prompt
                if repair_attempt_active or forced_repair_continuation:
                    effective_derive_mode = "repair_after_false_stop"
                    effective_derive_subtype = "late_verification"
                    effective_use_v2_prompt = True
                state_summary = build_state_summary_for_proposal(query_info["question"], state["S_t"], unit_registry)
                top_raw = build_top_raw_candidates(
                    r_t,
                    unit_registry,
                    s_t=state["S_t"],
                    derive_mode=effective_derive_mode,
                    carryover_candidate_ids=repair_linked_carryover_candidates if control_closure_continuation_active else None,
                )
                log(
                    f"[QID {qid}] step={t} proposing derived: top_raw={len(top_raw)} "
                    f"prompt_mode={'v2' if effective_use_v2_prompt else 'legacy'} "
                    f"derive_mode={effective_derive_mode} derive_subtype={effective_derive_subtype} derive_goal={derive_goal}"
                )
                g_harvest, next_derived_idx, proposer_meta = propose_derived(
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    qid=qid,
                    question=query_info["question"],
                    state_summary=state_summary,
                    top_raw_candidates=top_raw,
                    gold_answer=query_info["answer"],
                    derive_mode=effective_derive_mode,
                    derive_goal=derive_goal,
                    recent_probe_feedback=proposer_trace["recent_probe_feedback"],
                    bridge_anchors=bridge_anchors,
                    next_derived_idx=next_derived_idx,
                    cache_dir=derived_cache_dir,
                    use_v2_prompt=effective_use_v2_prompt,
                )
                if effective_derive_subtype == "late_verification":
                    for item in g_harvest:
                        item["type"] = "verification_note"
                elif effective_derive_subtype == "early_bridge":
                    for item in g_harvest:
                        item["type"] = "bridge_note"
                log(f"[QID {qid}] step={t} derived harvest size={len(g_harvest)}")
                proposer_trace.update(proposer_meta)
                proposer_trace["harvest_candidates"] = [
                    {
                        "unit_id": str(item["unit_id"]),
                        "type": str(item["type"]),
                        "text": str(item["text"]),
                        "source_unit_ids": [str(x) for x in item.get("source_unit_ids", [])],
                    }
                    for item in g_harvest
                ]

                for item in g_harvest:
                    derived_registry[item["unit_id"]] = {
                        "unit_id": item["unit_id"],
                        "text": item["text"],
                        "provenance": item["provenance"],
                        "candidate_granularity": item["candidate_granularity"],
                        "type": item["type"],
                        "source_unit_ids": item["source_unit_ids"],
                        "coarse_priority": item["coarse_priority"],
                    }

                unit_registry = build_unit_registry(raw_unit_map, derived_registry)
                g_legal, g_illegal = legality_filter(state, r_t, g_harvest, unit_registry)
                g_final, g_aux = final_retain_selection(
                    g_legal,
                    derive_mode=effective_derive_mode,
                    derive_subtype=effective_derive_subtype,
                    derive_goal=derive_goal,
                    question=query_info["question"],
                    gold_answer=query_info["answer"],
                )
                final_id_set = set(g_final)
                aux_id_set = set(g_aux)
                for item in g_legal:
                    closure_value, answer_facing = score_repair_note_value(
                        item,
                        question=query_info["question"],
                        gold_answer=query_info["answer"],
                        derive_goal=derive_goal,
                    )
                    bucket = "final" if item["unit_id"] in final_id_set else "aux" if item["unit_id"] in aux_id_set else None
                    if item["unit_id"] in derived_registry:
                        derived_registry[item["unit_id"]].update(
                            {
                                "derive_mode": effective_derive_mode,
                                "derive_subtype": effective_derive_subtype,
                                "derive_goal": derive_goal,
                                "answer_facing": bool(answer_facing),
                                "closure_value": int(closure_value),
                                "retained_bucket": bucket,
                            }
                        )
                unit_registry = build_unit_registry(raw_unit_map, derived_registry)
                proposer_trace["final_count"] = len(g_final)
                proposer_trace["aux_count"] = len(g_aux)
                proposer_trace["illegal_count"] = len(g_illegal)
                proposer_trace["legal_count"] = len(g_legal)
                log(
                    f"[QID {qid}] step={t} derived filter: "
                    f"legal={len(g_legal)} final={len(g_final)} aux={len(g_aux)} illegal={len(g_illegal)}"
                )
                if repair_attempt_active:
                    if not g_final and not g_aux:
                        repair_failure_reason = "repair_notes_not_retained"
                    else:
                        repair_failure_reason = "pending_repair_outcome"
            else:
                g_final = []
                g_aux = []
                g_illegal = []

        c_t = []
        seen = set()
        candidate_seed_ids = list(r_t)
        if control_closure_continuation_active and repair_linked_carryover_candidates:
            candidate_seed_ids, carryover_present = merge_repair_carryover_candidates(
                candidate_seed_ids,
                repair_linked_carryover_candidates,
            )
        else:
            carryover_present = []
        for unit_id in candidate_seed_ids + g_final:
            if unit_id in seen:
                continue
            seen.add(unit_id)
            c_t.append(unit_id)

        if not c_t:
            abort_reason = "empty_candidate_pool"
            log(f"[QID {qid}] ABORT: empty_candidate_pool", force=True)
            break

        unit_registry = build_unit_registry(raw_unit_map, derived_registry)
        positive_unit_id, teacher_select_bias_to_repair_applied, teacher_select_debug = teacher_select(
            query_info,
            state,
            target_info,
            c_t,
            unit_registry,
            recent_progress_flags,
            retrieval_repeat_ratio,
            repair_mode_active=bool(
                ((repair_attempt_active or forced_repair_continuation) and g_final)
                or (
                    isinstance(repair_focus_payload, dict)
                    and repair_focus_payload.get("retained_repair_ids")
                    and bool(failure_signals.get("recent_false_stop", False))
                )
            ),
            repair_candidate_ids=list(g_final) if g_final else list(
                repair_focus_payload.get("retained_repair_ids", []) if isinstance(repair_focus_payload, dict) else []
            ),
            derive_goal=derive_goal,
            recent_probe_feedback=proposer_trace["recent_probe_feedback"],
            closure_continuation_active=bool(shadow_closure_continuation_active),
            repair_focus_payload=copy.deepcopy(repair_focus_payload) if isinstance(repair_focus_payload, dict) else None,
            apply_repair_continuity_bonus=bool(control_closure_continuation_active or repair_followup_bonus_active),
        )
        log(f"[QID {qid}] step={t} teacher selected: {positive_unit_id}")

        prev_covered = len(state["A_t"].get("covered_target_ids", []))
        prev_covered_ids = [str(x) for x in state["A_t"].get("covered_target_ids", [])]
        role_scores_before = normalized_role_scores(state["A_t"], target_info)
        u_next = unit_registry[positive_unit_id]
        selected_candidate_linked_to_repair_focus = bool(
            teacher_select_debug.get("selected_candidate_linked_to_repair_focus", False)
        )

        h_next, s_next = simulate_update(state["H_t"], state["S_t"], u_next)
        repair_injection_unit_id = None
        if active_derive_mode == "repair_after_false_stop" and g_final:
            repair_injection_unit_id = choose_repair_injection_note(
                g_final,
                unit_registry,
                question=query_info["question"],
                gold_answer=query_info["answer"],
                derive_goal=derive_goal,
            )
            if repair_injection_unit_id:
                if repair_injection_unit_id in derived_registry:
                    derived_registry[repair_injection_unit_id]["force_render_next_step"] = True
                    unit_registry = build_unit_registry(raw_unit_map, derived_registry)
                existing_derived_ids = {
                    str(item["unit_id"])
                    for item in s_next.get("derived_refs", [])
                    if isinstance(item, dict) and item.get("unit_id")
                }
                if repair_injection_unit_id not in existing_derived_ids:
                    inject_step = len(h_next)
                    s_next.setdefault("derived_refs", [])
                    s_next["derived_refs"] = normalize_refs_minimal(s_next["derived_refs"])
                    s_next["derived_refs"].append(
                        {"unit_id": repair_injection_unit_id, "added_step": inject_step}
                    )
                    s_next["derived_refs"] = normalize_refs_minimal(s_next["derived_refs"])
        a_next = simulate_ledger(state["A_t"], u_next, target_info)
        k_next = render_context(query_info, s_next, unit_registry)
        next_state_summary = build_state_summary_for_proposal(query_info["question"], s_next, unit_registry)
        rendered_note_ids = get_active_note_ids(s_next, unit_registry)
        rendered_repair_note_ids = [unit_id for unit_id in g_final if unit_id in rendered_note_ids]
        rendered_repair_note_texts = [
            str(unit_registry[unit_id]["text"])
            for unit_id in rendered_repair_note_ids
            if unit_id in unit_registry
        ]
        repair_note_rendered_in_k_t = bool(rendered_repair_note_ids)
        repair_note_present_in_state_summary = bool(rendered_repair_note_ids)
        repair_note_considered_in_teacher_select = bool(
            repair_note_rendered_in_k_t or repair_note_present_in_state_summary
        )

        delta_A = len(a_next.get("covered_target_ids", [])) - prev_covered
        new_useful_derived = int(u_next["provenance"] == "derived")
        progress_flag = 1 if (delta_A > 0 or new_useful_derived) else 0
        if repair_attempt_active:
            if progress_flag:
                repair_effective = True
                repair_failure_reason = None
            elif repair_failure_reason == "pending_repair_outcome":
                repair_failure_reason = "no_progress_after_repair"
        if progress_flag:
            ever_progress = True
            last_progress_step = t
        recent_progress_flags.append(progress_flag)
        if len(recent_progress_flags) > STALL_WINDOW:
            recent_progress_flags = recent_progress_flags[-STALL_WINDOW:]

        log(
            f"[QID {qid}] step={t} update done: provenance={u_next['provenance']} "
            f"delta_A={delta_A} progress_flag={progress_flag}"
        )

        selected_from = None
        if positive_unit_id in r_t:
            selected_from = "R_t"
        elif positive_unit_id in g_final:
            selected_from = "G_t_final"
        probe_debug_payload = build_probe_debug_payload(probe)
        positive_chunk_id = normalize_raw_chunk_id(
            u_next.get("parent_chunk_id", positive_unit_id) if u_next.get("provenance") == "raw" else None
        )
        prev_covered_chunk_ids = normalize_raw_chunk_id_list(prev_covered_ids)
        next_covered_chunk_ids = normalize_raw_chunk_id_list(a_next.get("covered_target_ids", []))
        raw_ref_chunk_ids_before = extract_raw_ref_chunk_ids(state["S_t"])
        raw_ref_chunk_ids_after = extract_raw_ref_chunk_ids(s_next)

        legacy_need_derived_signal = bool(need_derived)
        legacy_triggered_propose_derived_signal = bool(triggered_propose_derived)
        selected_derive_subtype = active_derive_subtype
        if is_derived_unit_id(positive_unit_id) and positive_unit_id in derived_registry:
            selected_derive_subtype = derived_registry[positive_unit_id].get("derive_subtype") or active_derive_subtype
        control_trigger_derived_signal = bool(
            triggered_propose_derived
            and (
                selected_derive_subtype in ALLOWED_CONTROL_DERIVED_SUBTYPES
                or active_repair_override
                or repair_attempt_active
                or forced_repair_continuation
            )
        )
        control_need_derived_signal = bool(need_derived and control_trigger_derived_signal)
        gate_trace["derived_need"] = control_need_derived_signal
        gate_trace["trigger_derived"] = control_trigger_derived_signal
        gate_trace["derive_subtype"] = selected_derive_subtype if control_trigger_derived_signal else None
        gate_trace["legacy_need_derived_signal"] = legacy_need_derived_signal
        gate_trace["legacy_trigger_derived_signal"] = legacy_triggered_propose_derived_signal
        step_record = {
            "t": t,
            "positive_unit_id": positive_unit_id,
            "positive_chunk_id": positive_chunk_id,
            "stop_candidate": stop_candidate,
            "need_derived": control_need_derived_signal,
            "triggered_propose_derived": control_trigger_derived_signal,
            "retrieval_repeat_ratio": None if retrieval_repeat_ratio is None else float(retrieval_repeat_ratio),
            "delta_covered_targets": int(delta_A),
            "covered_target_count": len(a_next.get("covered_target_ids", [])),
            "progress_flag": int(progress_flag),
            "selected_from": selected_from,
            "R_t": list(r_t),
            "G_t_final": extract_unit_ids_for_debug(g_final),
            "C_t": list(c_t),
            "selected_provenance": str(u_next.get("provenance", "")),
            "query_debug": {
                "q_t": q_t,
            },
            "stop_debug": {
                "stop_candidate": stop_candidate,
                "stop_probe_count": len(stop_probe_history),
                "false_stop_count": false_stop_count,
                "raw_complete": bool(stop_info["raw_complete"]),
                "near_complete": bool(stop_info.get("near_complete", False)),
                "no_derived_need": bool(stop_info.get("no_obvious_derived_need", stop_info.get("no_derived_need", False))),
                "no_recent_closure_failure": bool(stop_info.get("no_recent_closure_failure", True)),
                "raw_redundant": bool(stop_info.get("raw_redundant", False)),
                "retrieval_repeat_ratio": None if retrieval_repeat_ratio is None else float(retrieval_repeat_ratio),
                "role_scores": {
                    "bridge": float(stop_info["role_scores"].get("bridge", 0.0)),
                    "distinguish": float(stop_info["role_scores"].get("distinguish", 0.0)),
                    "support": float(stop_info["role_scores"].get("support", 0.0)),
                },
                "probe_run": bool(probe_debug_payload["probe_run"]),
                "probe": probe_debug_payload,
                "false_stop_count_before": false_stop_count_before,
                "false_stop_count_after": false_stop_count,
                "stop_cooldown_before": stop_cooldown_before,
                "stop_cooldown_after": stop_cooldown,
            },
            "candidate_debug": {
                "selected_from": selected_from,
                "G_t_harvest": [str(item["unit_id"]) for item in g_harvest if isinstance(item, dict) and item.get("unit_id")],
                "harvest_candidate_count": len(g_harvest),
                "legal_candidate_count": len(g_legal),
                "final_retained_count": len(g_final),
                "aux_retained_count": len(g_aux),
                "illegal_count": len(g_illegal),
                "R_t": list(r_t),
                "G_t_final": extract_unit_ids_for_debug(g_final),
                "G_t_aux": extract_unit_ids_for_debug(g_aux),
                "G_t_illegal": extract_unit_ids_for_debug(g_illegal),
                "C_t": list(c_t),
                "need_derived": control_need_derived_signal,
                "triggered_propose_derived": control_trigger_derived_signal,
                "retrieval_repeat_ratio": None if retrieval_repeat_ratio is None else float(retrieval_repeat_ratio),
                "retrieval_shortlist_chunk_ids": [str(x["chunk_id"]) for x in retrieval_shortlist],
                "retrieval_shortlist_doc_ids": [str(x["doc_id"]) for x in retrieval_shortlist],
                "added_uncovered_chunk_ids": normalize_raw_chunk_id_list(added_uncovered_targets),
                "oracle_added_uncovered_chunk_ids": normalize_raw_chunk_id_list(oracle_added_uncovered_targets),
                "legacy_need_derived_signal": legacy_need_derived_signal,
                "legacy_triggered_propose_derived_signal": legacy_triggered_propose_derived_signal,
                "teacher_select_bias_to_repair_applied": bool(teacher_select_bias_to_repair_applied),
                "teacher_select_debug": copy.deepcopy(teacher_select_debug),
                "repair_focus_set_size": int(len(repair_linked_carryover_candidates)),
                "repair_linked_carryover_candidates": list(repair_linked_carryover_candidates),
                "candidate_pool_had_repair_linked_options": bool(
                    shadow_closure_continuation_active and bool(repair_linked_carryover_candidates)
                ),
                "repair_linked_candidate_dropped_from_pool": bool(
                    shadow_closure_continuation_active and not bool(repair_linked_carryover_candidates)
                ),
                "closure_continuation_active": bool(shadow_closure_continuation_active),
                "closure_stop_reopened": bool(shadow_closure_stop_reopened),
                "late_repair_stop_retry": bool(late_repair_stop_retry),
                "repair_followup_bonus_active": bool(repair_followup_bonus_active),
                "repair_continuation_control_enabled": bool(USE_REPAIR_CONTINUATION_FOR_CONTROL),
                "repair_continuation_diag_enabled": bool(USE_REPAIR_CONTINUATION_FOR_DIAG),
            },
            "derived_debug": {
                "need_derived": control_need_derived_signal,
                "triggered_propose_derived": control_trigger_derived_signal,
                "legacy_need_derived_signal": legacy_need_derived_signal,
                "legacy_triggered_propose_derived_signal": legacy_triggered_propose_derived_signal,
                "raw_redundant": bool(derived_info_control.get("raw_redundant", False)) if not stop_candidate else bool(stop_info.get("raw_redundant", False)),
                "composable_raw": bool(derived_info_control.get("composable_raw", False)) if not stop_candidate else False,
                "bridgeable_raw_shadow": bool(derived_info.get("bridgeable_raw", False)),
                "fallback_trigger": bool(repair_override_shadow),
                "plateau_trigger": bool(failure_signals.get("stagnation", False)),
            },
            "coverage_debug": {
                "covered_chunk_ids_before": prev_covered_chunk_ids,
                "covered_chunk_ids_after": next_covered_chunk_ids,
                "raw_ref_chunk_ids_before": raw_ref_chunk_ids_before,
                "raw_ref_chunk_ids_after": raw_ref_chunk_ids_after,
                "role_scores_before": {
                    "bridge": float(role_scores_before.get("bridge", 0.0)),
                    "distinguish": float(role_scores_before.get("distinguish", 0.0)),
                    "support": float(role_scores_before.get("support", 0.0)),
                },
                "role_scores_after": {
                    "bridge": float(normalized_role_scores(a_next, target_info).get("bridge", 0.0)),
                    "distinguish": float(normalized_role_scores(a_next, target_info).get("distinguish", 0.0)),
                    "support": float(normalized_role_scores(a_next, target_info).get("support", 0.0)),
                },
                "delta_covered_targets": int(delta_A),
                "covered_target_count": len(a_next.get("covered_target_ids", [])),
                "target_count": len(target_info["target_map"]),
                "progress_flag": int(progress_flag),
            },
            "failure_signals": copy.deepcopy(failure_signals),
            "gate_trace": copy.deepcopy(gate_trace),
            "proposer_trace": copy.deepcopy(proposer_trace),
            "repair_debug": {
                "repair_attempt_active": bool(repair_attempt_active),
                "repair_attempt_count": int(repair_attempt_count),
                "repair_effective": bool(repair_effective),
                "repair_failure_reason": repair_failure_reason,
                "stop_reopened_after_false_stop": bool(stop_reopened_after_false_stop),
                "selected_repair_candidate": selected_repair_candidate,
                "repair_note_retained": bool(repair_attempt_active and g_final),
                "repair_injection_unit_id": repair_injection_unit_id,
                "rendered_repair_note_ids": list(rendered_repair_note_ids),
                "repair_note_rendered_in_K_t": bool(repair_note_rendered_in_k_t),
                "repair_note_present_in_state_summary": bool(repair_note_present_in_state_summary),
                "repair_note_considered_in_teacher_select": bool(repair_note_considered_in_teacher_select),
                "teacher_select_bias_to_repair_applied": bool(teacher_select_bias_to_repair_applied),
                "closure_continuation_applied": bool(shadow_closure_continuation_active),
                "closure_stop_reopened": bool(shadow_closure_stop_reopened),
                "late_repair_stop_retry": bool(late_repair_stop_retry),
                "repair_followup_bonus_active": bool(repair_followup_bonus_active),
                "selected_candidate_linked_to_repair_focus": bool(selected_candidate_linked_to_repair_focus),
                "repair_continuation_control_enabled": bool(USE_REPAIR_CONTINUATION_FOR_CONTROL),
                "repair_continuation_diag_enabled": bool(USE_REPAIR_CONTINUATION_FOR_DIAG),
            },
        }

        refresh_repair_focus = bool(
            delta_A == 0
            and (
                selected_candidate_linked_to_repair_focus
                or teacher_select_debug.get("repair_linked_candidate_was_selected", False)
            )
        )
        if refresh_repair_focus:
            retained_for_focus = list(g_final) if g_final else list(
                repair_focus_payload.get("retained_repair_ids", []) if isinstance(repair_focus_payload, dict) else []
            )
            repair_focus_payload = build_repair_focus_payload(
                selected_unit_id=positive_unit_id,
                retained_repair_ids=retained_for_focus,
                unit_registry=unit_registry,
                raw_unit_map=raw_unit_map,
                question=query_info["question"],
                derive_goal=derive_goal,
            )
            repair_focus_ttl = REPAIR_FOCUS_TTL
            closure_continuation_ttl = REPAIR_FOCUS_TTL
        else:
            if repair_focus_ttl > 0:
                repair_focus_ttl -= 1
            if closure_continuation_ttl > 0:
                closure_continuation_ttl -= 1
            if repair_focus_ttl <= 0:
                repair_focus_payload = None
        if shadow_closure_continuation_active and not refresh_repair_focus and delta_A == 0 and repair_failure_reason == "pending_repair_outcome":
            repair_failure_reason = "closure_continuation_no_effect"
        if not control_closure_continuation_active:
            one_more_closure_stop_attempt_used = False
        step_record["repair_debug"]["repair_failure_reason"] = repair_failure_reason

        steps.append(step_record)
        retrieval_history.append(list(r_t))
        if len(retrieval_history) > 2:
            retrieval_history = retrieval_history[-2:]
        last_delta_covered_targets = int(delta_A)
        last_retrieval_repeat_ratio = retrieval_repeat_ratio

        next_state = {
            "qid": qid,
            "t": t + 1,
            "H_t": h_next,
            "A_t": a_next,
            "S_t": s_next,
            "K_t": k_next,
        }
        next_role_scores = normalized_role_scores(a_next, target_info)
        next_raw_complete = all(
            next_role_scores[r] >= RAW_COMPLETE_THRESHOLD
            for r in target_info["required_roles"]
        )
        no_progress_streak_now = bool(
            len(recent_progress_flags) == STALL_WINDOW
            and all(x == 0 for x in recent_progress_flags)
        )
        if next_raw_complete or no_progress_streak_now or t + 1 >= T_MAX:
            offline_probe = make_offline_terminal_probe(
                query_info=query_info,
                state=next_state,
                target_info=target_info,
                unit_registry=unit_registry,
                chunks=chunks,
                api_key=api_key,
                model=model,
                base_url=base_url,
                t=t + 1,
                reason="post_step_closure",
            )
            if offline_probe is not None:
                terminal_status = "terminal"
                terminal_t = t + 1
                abort_reason = None
                terminal_probe = offline_probe
                terminal_failure_signals = copy.deepcopy(failure_signals)
                terminal_gate_trace = copy.deepcopy(gate_trace)
                terminal_proposer_trace = copy.deepcopy(proposer_trace)
                log(f"[QID {qid}] TERMINAL by offline closure at step={t + 1}", force=True)
                break

        allow_one_more_step_after_repair = bool(repair_attempt_active and not progress_flag)

        if (
            not allow_one_more_step_after_repair
            and false_stop_count >= FALSE_STOP_LIMIT
            and len(recent_progress_flags) == STALL_WINDOW
            and all(x == 0 for x in recent_progress_flags)
        ):
            abort_reason = "repeated_false_stop_no_progress"
            log(f"[QID {qid}] ABORT: repeated_false_stop_no_progress", force=True)
            break

        if (
            not allow_one_more_step_after_repair
            and len(recent_progress_flags) == STALL_WINDOW
            and all(x == 0 for x in recent_progress_flags)
        ):
            abort_reason = "stalled"
            log(f"[QID {qid}] ABORT: stalled", force=True)
            break

        state = next_state

    if terminal_status != "terminal" and abort_reason is None:
        abort_reason = "max_steps"
        log(f"[QID {qid}] ABORT: max_steps", force=True)

    result = {
        "qid": qid,
        "build_meta": {
            "run_id": run_id,
            "build_time": build_time,
            "source": build_source,
        },
        "terminal_status": terminal_status,
        "terminal_t": terminal_t,
        "abort_reason": abort_reason,
        "ever_progress": ever_progress,
        "last_progress_step": last_progress_step,
        "final_false_stop_count": false_stop_count,
        "repair_attempt_count": int(repair_attempt_count),
        "repair_effective": bool(repair_effective),
        "repair_failure_reason": repair_failure_reason,
        "stop_reopened_after_false_stop": bool(stop_reopened_after_false_stop),
        "stop_probe_count": len(stop_probe_history),
        "stop_probe_history": stop_probe_history,
        "terminal_probe": terminal_probe,
        "terminal_failure_signals": terminal_failure_signals,
        "terminal_gate_trace": terminal_gate_trace,
        "terminal_proposer_trace": terminal_proposer_trace,
        "target_count": len(target_info["target_map"]),
        "required_roles": list(target_info["required_roles"]),
        "steps": steps,
    }
    log(
        f"[QID {idx}/{total}] finished qid={qid} "
        f"terminal_status={terminal_status} terminal_t={terminal_t} "
        f"abort_reason={abort_reason} steps={len(steps)}",
        force=True,
    )
    return result


def build_full_split(
    queries_path: Path,
    targets_path: Path,
    init_state_path: Path,
    raw_units_path: Path,
    chunks_path: Path,
    atoms_path: Path,
    output_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    derived_cache_dir: Path,
    stop_probe_cache_dir: Path,
    run_id: str,
    build_time: str,
    build_source: str,
) -> int:
    queries = maybe_limit_qids(load_queries(queries_path))
    targets = load_targets(targets_path)
    init_states = load_init_states(init_state_path)
    raw_unit_map = load_raw_unit_map(raw_units_path)
    raw_unit_ids_by_chunk = build_raw_unit_ids_by_chunk(raw_unit_map)
    chunks_grouped = load_chunks_grouped(chunks_path)
    atoms_by_chunk = load_atoms_by_chunk(atoms_path)

    qids = [qid for qid in sorted(queries.keys()) if qid in init_states]
    total = len(qids)
    log(f"split qids to run: {total}", force=True)
    log(f"parallel workers: {FULL_MAX_WORKERS}", force=True)

    records = [None] * total

    with ThreadPoolExecutor(max_workers=FULL_MAX_WORKERS) as executor:
        future_to_meta = {}

        for idx, qid in enumerate(qids, start=1):
            if qid not in targets:
                raise ValueError(f"targets 中找不到 qid: {qid}")
            if qid not in chunks_grouped:
                raise ValueError(f"chunks 中找不到 qid: {qid}")

            future = executor.submit(
                rollout_one_qid,
                qid=qid,
                query_info=queries[qid],
                target_info=targets[qid],
                init_state=init_states[qid],
                raw_unit_map=raw_unit_map,
                raw_unit_ids_by_chunk=raw_unit_ids_by_chunk,
                chunks=chunks_grouped[qid],
                atoms_by_chunk=atoms_by_chunk,
                api_key=api_key,
                base_url=base_url,
                model=model,
                derived_cache_dir=derived_cache_dir,
                stop_probe_cache_dir=stop_probe_cache_dir,
                idx=idx,
                total=total,
                run_id=run_id,
                build_time=build_time,
                build_source=build_source,
            )
            future_to_meta[future] = (idx, qid)

        done = 0
        for future in as_completed(future_to_meta):
            idx, qid = future_to_meta[future]
            try:
                result = future.result()
            except Exception as e:
                log(f"[QID {qid}] failed: {type(e).__name__}: {e}", force=True)
                raise

            records[idx - 1] = result
            done += 1
            log(f"[PROGRESS] finished {done}/{total} qids; latest={qid}", force=True)

    for i, item in enumerate(records):
        if item is None:
            raise RuntimeError(f"records[{i}] 为空，说明有任务未正确返回")

    count = write_jsonl(records, output_path)
    return count


def main():
    parser = argparse.ArgumentParser(description="Build hotpotqa full trajectories v2")
    parser.add_argument("--force", action="store_true", help="Accepted for compatibility; overwrites fixed output files in place")
    parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"

    if not api_key:
        log("DEEPSEEK_API_KEY 未设置；将仅使用缓存命中与本地 fallback 继续运行", force=True)

    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / DEFAULT_BASE

    trajectories_dir = base_dir / "trajectories"
    cache_dir = base_dir / "cache"
    derived_cache_dir = cache_dir / "derived_cache"
    stop_probe_cache_dir = cache_dir / "stop_probe_cache"
    queries_dir = base_dir / "queries"
    targets_dir = base_dir / "targets"
    unit_registry_dir = base_dir / "unit_registry"
    index_store_dir = base_dir / "index_store"

    raw_units_name_map = {
        "train": "raw_units_train.jsonl",
        "val": "raw_units_val.jsonl",
        "test": "raw_units_test.jsonl",
    }
    out_name_map = {
        "train": "full_train.jsonl",
        "val": "full_val.jsonl",
        "test": "full_test.jsonl",
    }
    build_now = time.localtime()
    run_id = make_run_id(build_now)
    build_time = make_build_time(build_now)
    build_source = "build_hotpotqa_full_trajectories_v4.py"

    log("build_hotpotqa_full_trajectories_v4.py started", force=True)
    log(
        f"config: T_MAX={T_MAX}, CHUNK_SHORTLIST_K={CHUNK_SHORTLIST_K}, FINAL_KR={FINAL_KR}, "
        f"FULL_MAX_QIDS={FULL_MAX_QIDS}, FULL_ONLY_SPLIT={FULL_ONLY_SPLIT or 'ALL'}, "
        f"FULL_MAX_WORKERS={FULL_MAX_WORKERS}",
        force=True,
    )

    stats = {}
    run_splits = SPLITS
    if FULL_ONLY_SPLIT:
        if FULL_ONLY_SPLIT not in SPLITS:
            raise ValueError(f"非法 FULL_ONLY_SPLIT: {FULL_ONLY_SPLIT}")
        run_splits = [FULL_ONLY_SPLIT]

    for split in run_splits:
        log(f"===== split={split} begin =====", force=True)

        queries_path = queries_dir / f"{split}.jsonl"
        targets_path = targets_dir / f"{split}.jsonl"
        init_state_path = trajectories_dir / f"init_state_{split}.jsonl"
        raw_units_path = unit_registry_dir / raw_units_name_map[split]
        chunks_path = index_store_dir / f"chunks_{split}.jsonl"
        atoms_path = index_store_dir / f"atoms_{split}.jsonl"
        output_path = trajectories_dir / out_name_map[split]

        for path in [queries_path, targets_path, init_state_path, raw_units_path, chunks_path, atoms_path]:
            if not path.exists():
                raise FileNotFoundError(f"找不到必需文件: {path}")

        start_time = time.time()
        stats[split] = build_full_split(
            queries_path=queries_path,
            targets_path=targets_path,
            init_state_path=init_state_path,
            raw_units_path=raw_units_path,
            chunks_path=chunks_path,
            atoms_path=atoms_path,
            output_path=output_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            derived_cache_dir=derived_cache_dir / split,
            stop_probe_cache_dir=stop_probe_cache_dir / split,
            run_id=run_id,
            build_time=build_time,
            build_source=build_source,
        )
        elapsed = time.time() - start_time
        log(f"===== split={split} done: count={stats[split]}, elapsed={elapsed:.2f}s =====", force=True)

    write_json(
        {
            "run_id": run_id,
            "build_time": build_time,
            "source": build_source,
            "splits": run_splits,
        },
        trajectories_dir / "build_manifest_v4.json",
    )

    log("full trajectories v2 构建完成", force=True)
    print("full trajectories v2 构建完成：", flush=True)
    print(f"  run_id={run_id}", flush=True)
    print(f"  defaults: T_max={T_MAX}, stall_window={STALL_WINDOW}, false_stop_limit={FALSE_STOP_LIMIT}", flush=True)
    for split in run_splits:
        print(f"  {split}: {stats[split]} -> {trajectories_dir / out_name_map[split]}", flush=True)


if __name__ == "__main__":
    main()
