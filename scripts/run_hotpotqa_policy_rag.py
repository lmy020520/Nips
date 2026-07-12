#!/usr/bin/env python3
"""Trajectory-aware policy RAG evaluation for HotpotQA samples.

This script treats the trained model as an evidence-selection policy, not as a
conventional one-shot reranker. For each HotpotQA trajectory state, it scores the
candidate evidence units conditioned on Question + Notebook(K_t), selects the
next evidence unit, and aggregates step-level and trajectory-level metrics.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.models.ranker import CrossEncoderRanker


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_memory(path: Path) -> dict[str, dict]:
    memory = {}
    for row in read_jsonl(path):
        unit_id = str(row.get("unit_id") or "")
        if not unit_id:
            continue
        title = str(row.get("title") or row.get("doc_id") or "")
        if not title:
            parts = unit_id.split("::")
            title = parts[-2] if len(parts) >= 2 else unit_id
        sent_id = row.get("sent_id")
        if sent_id is None:
            try:
                sent_id = int(unit_id.rsplit("::", 1)[-1])
            except ValueError:
                sent_id = 0
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        memory[unit_id] = {
            "unit_id": unit_id,
            "title": title,
            "sent_id": int(sent_id),
            "text": text,
            "doc_id": str(row.get("doc_id") or title),
        }
    return memory


def format_candidate_text(memory_item: dict) -> str:
    return f"{memory_item['title']} [{memory_item['sent_id']}] {memory_item['text']}"


def format_notebook_evidence(memory_item: dict, index: int) -> str:
    return f"[{index}] {memory_item['title']}: {memory_item['text']}"


def tokenize_for_retrieval(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def sample_candidate_ids(row: dict) -> list[str]:
    candidates = row.get("candidates") or {}
    for key in ("C_t", "R_t"):
        value = candidates.get(key)
        if isinstance(value, list) and value:
            return [str(unit_id) for unit_id in value]
    return []


def sample_positive_id(row: dict) -> str:
    labels = row.get("labels") or {}
    ranking = labels.get("ranking_label") or {}
    return str(ranking.get("positive_unit_id") or "")


def sample_k_t(row: dict) -> str:
    if "K_t" in row:
        return str(row["K_t"])
    state = row.get("state") or {}
    return str(state.get("K_t") or "")


def init_online_state() -> dict:
    return {
        "H_t": [],
        "A_t": {
            "raw_unit_ids": [],
            "doc_ids": [],
            "unit_doc": {},
        },
        "S_t": {
            "raw_refs": [],
            "derived_refs": [],
            "last_added_unit_id": None,
            "last_updated_step": -1,
        },
        "K_t": "",
    }


def render_online_k_t(state: dict, memory: dict[str, dict], *, max_raw: int = 8, max_chars_per_item: int = 260) -> str:
    """Render the current lightweight notebook state as answer-facing context."""
    s_t = state.get("S_t") or {}
    raw_refs = s_t.get("raw_refs") if isinstance(s_t.get("raw_refs"), list) else []
    ordered_refs = sorted(
        raw_refs,
        key=lambda ref: (int(ref.get("added_step", 0)), int(ref.get("selected_count", 0))),
        reverse=True,
    )

    parts = []
    if ordered_refs:
        parts.append("Evidence:")
        for index, ref in enumerate(ordered_refs[:max_raw], start=1):
            unit_id = str(ref.get("unit_id") or "")
            item = memory.get(unit_id)
            if not item:
                continue
            text = str(item.get("text") or "").strip()
            if len(text) > max_chars_per_item:
                text = text[: max_chars_per_item - 3].rstrip() + "..."
            title = str(item.get("title") or item.get("doc_id") or "")
            parts.append(f"[{index}] {title}: {text}")
    return "\n".join(parts).strip()


def update_online_state(
    state: dict,
    unit_id: str,
    memory_item: dict,
    memory: dict[str, dict],
    *,
    step_id: int,
    max_raw: int = 8,
    max_chars_per_item: int = 260,
) -> dict:
    """Deterministically apply Update -> Ledger -> Render for online policy mode."""
    next_state = {
        "H_t": list(state.get("H_t") or []),
        "A_t": {
            "raw_unit_ids": list((state.get("A_t") or {}).get("raw_unit_ids") or []),
            "doc_ids": list((state.get("A_t") or {}).get("doc_ids") or []),
            "unit_doc": dict((state.get("A_t") or {}).get("unit_doc") or {}),
        },
        "S_t": {
            "raw_refs": [dict(ref) for ref in (state.get("S_t") or {}).get("raw_refs") or []],
            "derived_refs": [dict(ref) for ref in (state.get("S_t") or {}).get("derived_refs") or []],
            "last_added_unit_id": (state.get("S_t") or {}).get("last_added_unit_id"),
            "last_updated_step": (state.get("S_t") or {}).get("last_updated_step", -1),
        },
        "K_t": str(state.get("K_t") or ""),
    }

    # Update: keep a prefix trajectory and lightweight notebook refs.
    if unit_id not in next_state["H_t"]:
        next_state["H_t"].append(unit_id)
    raw_refs = next_state["S_t"]["raw_refs"]
    existing_ref = next((ref for ref in raw_refs if ref.get("unit_id") == unit_id), None)
    if existing_ref is None:
        raw_refs.append(
            {
                "unit_id": unit_id,
                "added_step": step_id,
                "used_in_summary_count": 0,
                "selected_count": 1,
            }
        )
    else:
        existing_ref["selected_count"] = int(existing_ref.get("selected_count") or 0) + 1
    next_state["S_t"]["last_added_unit_id"] = unit_id
    next_state["S_t"]["last_updated_step"] = step_id

    # Ledger: record raw coverage in a deterministic, inspectable form.
    doc_id = str(memory_item.get("doc_id") or memory_item.get("title") or "")
    if unit_id not in next_state["A_t"]["raw_unit_ids"]:
        next_state["A_t"]["raw_unit_ids"].append(unit_id)
    if doc_id and doc_id not in next_state["A_t"]["doc_ids"]:
        next_state["A_t"]["doc_ids"].append(doc_id)
    if doc_id:
        next_state["A_t"]["unit_doc"][unit_id] = doc_id

    # Render: expose the compiled notebook context for the next selection step.
    next_state["K_t"] = render_online_k_t(
        next_state,
        memory,
        max_raw=max_raw,
        max_chars_per_item=max_chars_per_item,
    )
    return next_state


def load_queries(path: str) -> dict[str, dict]:
    if not path:
        return {}
    query_path = Path(path)
    if not query_path.exists():
        return {}
    return {str(row.get("qid")): row for row in read_jsonl(query_path) if row.get("qid")}


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def exact_match_score(prediction: str, gold_answer: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold_answer))


def answer_contains_score(prediction: str, gold_answer: str) -> int:
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(gold_answer)
    return int(bool(norm_gold) and norm_gold in norm_pred)


def f1_score(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def deepseek_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_sleep: float = 2.0,
) -> tuple[str, int]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    last_error = None
    for attempt in range(1, max(1, max_retries) + 1):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"], int(data.get("usage", {}).get("total_tokens") or 0)
        except (
            TimeoutError,
            ConnectionError,
            ConnectionResetError,
            http.client.RemoteDisconnected,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * attempt)
    raise RuntimeError(f"DeepSeek request failed after {max_retries} retries: {last_error}") from last_error


def extract_answer_from_json(raw_answer: str) -> str:
    text = str(raw_answer or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("answer") is not None:
            return str(data["answer"]).strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and data.get("answer") is not None:
                return str(data["answer"]).strip()
        except json.JSONDecodeError:
            pass
    return text.splitlines()[0].strip() if text else ""


def extract_ranked_indices_from_json(raw_answer: str, candidate_count: int) -> list[int]:
    text = str(raw_answer or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
            except json.JSONDecodeError:
                payload = None
    if not isinstance(payload, dict):
        return []

    raw_indices = payload.get("selected_indices") or payload.get("ranked_indices") or payload.get("indices") or []
    if isinstance(raw_indices, (int, float, str)):
        raw_indices = [raw_indices]
    if not isinstance(raw_indices, list):
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for value in raw_indices:
        try:
            idx = int(value) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < candidate_count and idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def select_with_agentic_llm(
    question: str,
    context: str,
    candidate_texts: list[str],
    *,
    select_top_k: int,
    max_candidates: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[list[int], str, int, float]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for --selector agentic_llm")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    max_candidates = min(max(1, max_candidates), len(candidate_texts))
    visible = candidate_texts[:max_candidates]
    candidates_text = "\n".join(f"[{idx}] {text}" for idx, text in enumerate(visible, start=1))
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-selection agent for multi-hop QA. "
                "Given a question, the current knowledge state, and candidate evidence sentences, "
                "select evidence that best complements the current state. "
                "Prefer evidence that adds missing bridge/support information over redundant or merely lexical matches. "
                "Output valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Current knowledge state:\n{context}\n\n"
                f"Candidate evidence:\n{candidates_text}\n\n"
                f"Return exactly: {{\"selected_indices\":[i1,i2,...]}}\n"
                f"Use 1-based candidate indices. Select up to {max(1, select_top_k)} indices, ranked best first."
            ),
        },
    ]
    started = time.time()
    raw_answer, tokens = deepseek_chat(
        api_key,
        base_url,
        model,
        messages,
        temperature=0.0,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    indices = extract_ranked_indices_from_json(raw_answer, max_candidates)
    return indices, raw_answer, tokens, time.time() - started


def answer_with_llm(
    question: str,
    evidence: list[dict],
    answer_mode: str,
    max_retries: int,
    retry_sleep: float,
) -> tuple[str, str, int, float]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return "", "", 0, 0.0
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    context = "\n".join(format_notebook_evidence(item, idx + 1) for idx, item in enumerate(evidence))
    if answer_mode == "json":
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an extractive HotpotQA answer module. "
                    "Use only the evidence. Output valid JSON only. "
                    "The answer must be the shortest exact answer phrase, usually an entity, date, number, yes, or no. "
                    "Do not include explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    'Return exactly: {"answer":"..."}'
                ),
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided evidence. "
                    "Return only the short answer, not an explanation. "
                    "If the evidence is insufficient, say unknown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Short answer:"
                ),
            },
        ]
    started = time.time()
    raw_answer, tokens = deepseek_chat(
        api_key,
        base_url,
        model,
        messages,
        temperature=0.0,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )
    answer = extract_answer_from_json(raw_answer) if answer_mode == "json" else raw_answer.strip()
    return answer, raw_answer, tokens, time.time() - started


def cache_file_for_qid(cache_dir: Path, qid: str) -> Path:
    safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
    return cache_dir / f"{safe_qid}.json"


def profile_start(profile: dict) -> float | None:
    if not profile.get("active"):
        return None
    if profile.get("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def profile_stop(profile: dict, stage: str, started: float | None) -> None:
    if started is None:
        return
    if profile.get("cuda"):
        torch.cuda.synchronize()
    profile["seconds"][stage] += time.perf_counter() - started
    profile["calls"][stage] += 1


class PolicyModel:
    def __init__(self, model_dir: Path, checkpoint: Path, device: str, max_length: int, batch_size: int):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = CrossEncoderRanker(pretrained_name=str(model_dir), dropout=0.1).to(self.device)
        checkpoint_obj = torch.load(str(checkpoint), map_location=self.device)
        self.model.load_state_dict(checkpoint_obj.get("model_state_dict", checkpoint_obj), strict=False)
        self.model.eval()

    def score(self, context: str, candidate_texts: list[str]) -> np.ndarray:
        scores, _, _ = self.score_with_aux(context, candidate_texts, return_aux=False)
        return scores

    def score_with_aux(
        self,
        context: str,
        candidate_texts: list[str],
        return_aux: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        scores = []
        role_probs = []
        deficit_preds = []
        with torch.no_grad():
            for start in range(0, len(candidate_texts), self.batch_size):
                batch_texts = candidate_texts[start : start + self.batch_size]
                tokens = self.tokenizer(
                    [context] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                model_outputs = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    token_type_ids=tokens.get("token_type_ids"),
                    return_deficit=return_aux,
                )
                if return_aux:
                    output, role_logits, deficit_output = model_outputs
                    role_probs.extend(torch.softmax(role_logits, dim=-1).detach().cpu().tolist())
                    deficit_preds.extend(deficit_output.detach().cpu().tolist())
                else:
                    output, _ = model_outputs
                scores.extend(output.detach().cpu().tolist())
        return (
            np.array(scores),
            np.array(role_probs) if return_aux else None,
            np.array(deficit_preds) if return_aux else None,
        )

    def deficit_aware_score(self, context: str, candidate_texts: list[str], role_weight: float) -> np.ndarray:
        scores, role_probs, deficit_preds = self.score_with_aux(context, candidate_texts, return_aux=True)
        if role_probs is None or deficit_preds is None or len(scores) == 0:
            return scores

        # Role logits use [bridge, support, distinguish]. Deficit labels use
        # [bridge, distinguish, support, derived]. Align them before matching.
        deficit_state = np.mean(deficit_preds, axis=0)
        role_deficit = np.array([deficit_state[0], deficit_state[2], deficit_state[1]])
        contribution_match = np.dot(role_probs[:, :3], role_deficit)

        return minmax_normalize(scores) + float(role_weight) * contribution_match

    def estimate_deficit(self, context: str, candidate_texts: list[str]) -> dict:
        _, _, deficit_preds = self.score_with_aux(context, candidate_texts, return_aux=True)
        if deficit_preds is None or len(deficit_preds) == 0:
            values = np.zeros(4, dtype=float)
        else:
            values = np.mean(deficit_preds, axis=0)
        return {
            "d_br": float(values[0]),
            "d_dis": float(values[1]),
            "d_sup": float(values[2]),
            "d_der": float(values[3]),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }

    def contribution_aware_score(
        self,
        context: str,
        candidate_texts: list[str],
        contribution_weight: float,
    ) -> np.ndarray:
        scores = []
        deficit_preds = []
        contribution_preds = []
        with torch.no_grad():
            for start in range(0, len(candidate_texts), self.batch_size):
                batch_texts = candidate_texts[start : start + self.batch_size]
                tokens = self.tokenizer(
                    [context] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                output, _, deficit_output, contribution_output = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    token_type_ids=tokens.get("token_type_ids"),
                    return_deficit=True,
                    return_contribution=True,
                )
                scores.extend(output.detach().cpu().tolist())
                deficit_preds.extend(deficit_output.detach().cpu().tolist())
                contribution_preds.extend(contribution_output.detach().cpu().tolist())
        scores = np.array(scores)
        if len(scores) == 0:
            return scores
        deficit_state = np.mean(np.array(deficit_preds), axis=0)
        contribution_match = np.dot(np.array(contribution_preds), deficit_state)
        return minmax_normalize(scores) + float(contribution_weight) * minmax_normalize(contribution_match)


class DenseScorer:
    def __init__(self, model_name_or_path: str, device: str, batch_size: int):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.batch_size = max(1, batch_size)

    def score(self, query: str, candidate_texts: list[str]) -> np.ndarray:
        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        candidate_embeddings = self.model.encode(
            candidate_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.dot(candidate_embeddings, query_embedding)


class GenericReranker:
    def __init__(self, model_name_or_path: str, device: str, batch_size: int):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name_or_path, device=device)
        self.batch_size = max(1, batch_size)

    def score(self, query: str, candidate_texts: list[str]) -> np.ndarray:
        pairs = [(query, text) for text in candidate_texts]
        return np.array(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False))


class T5Reranker:
    """monoT5/RankT5-style generative reranker.

    The model is prompted to judge whether a document is relevant to a query.
    We score each pair by the first-step logit margin: logit("true") -
    logit("false"). This keeps the baseline deterministic and avoids actual
    text generation.
    """

    def __init__(self, model_name_or_path: str, device: str, batch_size: int, max_length: int):
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()
        self.true_token_id = self._single_token_id("true")
        self.false_token_id = self._single_token_id("false")

    def _single_token_id(self, text: str) -> int:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(f"Unable to encode reranker label token: {text!r}")
        return int(token_ids[0])

    @staticmethod
    def format_pair(query: str, candidate_text: str) -> str:
        return f"Query: {query} Document: {candidate_text} Relevant:"

    def score(self, query: str, candidate_texts: list[str]) -> np.ndarray:
        scores: list[float] = []
        decoder_start = self.model.config.decoder_start_token_id
        if decoder_start is None:
            decoder_start = self.tokenizer.pad_token_id
        if decoder_start is None:
            raise RuntimeError("T5 reranker requires decoder_start_token_id or pad_token_id.")

        with torch.no_grad():
            for start in range(0, len(candidate_texts), self.batch_size):
                batch_texts = candidate_texts[start : start + self.batch_size]
                inputs = [self.format_pair(query, text) for text in batch_texts]
                tokens = self.tokenizer(
                    inputs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                decoder_input_ids = torch.full(
                    (len(batch_texts), 1),
                    int(decoder_start),
                    dtype=torch.long,
                    device=self.device,
                )
                outputs = self.model(**tokens, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, 0, :]
                margins = logits[:, self.true_token_id] - logits[:, self.false_token_id]
                scores.extend(margins.detach().cpu().float().tolist())
        return np.array(scores, dtype=float)


def bm25_scores(query: str, candidate_texts: list[str]) -> np.ndarray:
    from rank_bm25 import BM25Okapi

    tokenized_candidates = [tokenize_for_retrieval(text) for text in candidate_texts]
    return np.array(BM25Okapi(tokenized_candidates).get_scores(tokenize_for_retrieval(query)))


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score - min_score < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)


def reciprocal_rank_fusion(orders: list[list[int]], size: int, k: int = 60) -> np.ndarray:
    scores = np.zeros(size, dtype=float)
    for order in orders:
        for rank, index in enumerate(order, start=1):
            scores[index] += 1.0 / (k + rank)
    return scores


def local_expanded_order(
    order: list[int],
    candidate_ids: list[str],
    memory: dict[str, dict],
    *,
    window: int,
    limit: int,
) -> list[int]:
    """Keep front-end winners while pulling in nearby sentences from the same document."""
    if limit <= 0:
        limit = len(order)
    if window <= 0:
        return order[:limit]

    by_doc: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for idx, unit_id in enumerate(candidate_ids):
        item = memory.get(unit_id)
        if not item:
            continue
        qid = unit_id.split("::", 1)[0]
        by_doc[(qid, str(item.get("doc_id") or item.get("title") or ""))].append((int(item.get("sent_id") or 0), idx))
    for doc_key in by_doc:
        by_doc[doc_key].sort()

    expanded: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> None:
        if index not in seen and len(expanded) < limit:
            seen.add(index)
            expanded.append(index)

    for index in order:
        add(index)
        item = memory.get(candidate_ids[index])
        if item:
            qid = candidate_ids[index].split("::", 1)[0]
            doc_key = (qid, str(item.get("doc_id") or item.get("title") or ""))
            sent_id = int(item.get("sent_id") or 0)
            for neighbor_sent_id, neighbor_index in by_doc.get(doc_key, []):
                if abs(neighbor_sent_id - sent_id) <= window:
                    add(neighbor_index)
                if len(expanded) >= limit:
                    break
        if len(expanded) >= limit:
            break

    for index in order:
        add(index)
        if len(expanded) >= limit:
            break
    return expanded


def local_expanded_pool(
    seed_indices: list[int],
    candidate_ids: list[str],
    memory: dict[str, dict],
    *,
    window: int,
) -> list[int]:
    """Expand a seed pool with nearby same-document sentences, preserving seed order."""
    by_doc: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for idx, unit_id in enumerate(candidate_ids):
        item = memory.get(unit_id)
        if not item:
            continue
        qid = unit_id.split("::", 1)[0]
        by_doc[(qid, str(item.get("doc_id") or item.get("title") or ""))].append((int(item.get("sent_id") or 0), idx))
    for doc_key in by_doc:
        by_doc[doc_key].sort()

    expanded: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> None:
        if index not in seen:
            seen.add(index)
            expanded.append(index)

    for index in seed_indices:
        add(index)
        if window <= 0:
            continue
        item = memory.get(candidate_ids[index])
        if not item:
            continue
        qid = candidate_ids[index].split("::", 1)[0]
        doc_key = (qid, str(item.get("doc_id") or item.get("title") or ""))
        sent_id = int(item.get("sent_id") or 0)
        for neighbor_sent_id, neighbor_index in by_doc.get(doc_key, []):
            if abs(neighbor_sent_id - sent_id) <= window:
                add(neighbor_index)
    return expanded


def text_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize_for_retrieval(left))
    right_tokens = set(tokenize_for_retrieval(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def candidate_similarity(
    left_index: int,
    right_index: int,
    candidate_texts: list[str],
    candidate_ids: list[str],
    memory: dict[str, dict],
    *,
    same_doc_similarity: float,
) -> float:
    lexical_similarity = text_jaccard(candidate_texts[left_index], candidate_texts[right_index])
    left = memory.get(candidate_ids[left_index]) or {}
    right = memory.get(candidate_ids[right_index]) or {}
    same_doc = bool(left.get("doc_id") and left.get("doc_id") == right.get("doc_id"))
    if same_doc:
        lexical_similarity = max(lexical_similarity, same_doc_similarity)
    return lexical_similarity


def mmr_select(
    pool_indices: list[int],
    relevance_scores: np.ndarray,
    candidate_texts: list[str],
    candidate_ids: list[str],
    memory: dict[str, dict],
    *,
    limit: int,
    lambda_: float,
    same_doc_similarity: float,
) -> list[int]:
    """Select a relevant but non-redundant subset from a front-end candidate pool."""
    if limit <= 0 or len(pool_indices) <= limit:
        return pool_indices[:limit] if limit > 0 else pool_indices

    lambda_ = min(1.0, max(0.0, lambda_))
    relevance = minmax_normalize(relevance_scores)
    selected: list[int] = []
    remaining = list(dict.fromkeys(pool_indices))

    while remaining and len(selected) < limit:
        best_index = None
        best_score = None
        for index in remaining:
            redundancy = 0.0
            if selected:
                redundancy = max(
                    candidate_similarity(
                        index,
                        selected_index,
                        candidate_texts,
                        candidate_ids,
                        memory,
                        same_doc_similarity=same_doc_similarity,
                    )
                    for selected_index in selected
                )
            score = lambda_ * float(relevance[index]) - (1.0 - lambda_) * redundancy
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        selected.append(int(best_index))
        remaining.remove(int(best_index))
    return selected


def multi_query_variants(question: str, context: str) -> list[str]:
    variants = [
        question,
        f"Find evidence that answers: {question}",
        f"Find the bridge entity and final answer evidence for: {question}",
    ]
    context = str(context or "").strip()
    if context:
        variants.append(f"{question}\nKnown evidence:\n{context}")
    deduped = []
    seen = set()
    for query in variants:
        key = " ".join(query.lower().split())
        if key and key not in seen:
            seen.add(key)
            deduped.append(query)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trajectory-aware policy RAG over HotpotQA sample states.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--queries", default="")
    parser.add_argument("--model-dir", default="models/deberta-v3-large")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ks", default="1,2,3,5")
    parser.add_argument("--max-items", type=int, default=0, help="Limit number of sample states, not qids.")
    parser.add_argument("--max-qids", type=int, default=0)
    parser.add_argument("--max-policy-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--state-mode", choices=["dataset", "policy"], default="dataset")
    parser.add_argument(
        "--policy-context-source",
        choices=["legacy", "online_state", "query_only"],
        default="legacy",
        help=(
            "For --state-mode policy, use legacy selected-evidence notebook, "
            "rendered online K_t, or question-only context without knowledge state."
        ),
    )
    parser.add_argument(
        "--selector",
        choices=[
            "policy",
            "bm25",
            "dense",
            "hybrid",
            "hybrid_policy",
            "multi_query_dense",
            "iterative_dense",
            "iterative_hybrid",
            "dense_policy",
            "generic_reranker",
            "t5_reranker",
            "agentic_llm",
            "first",
            "random",
            "gold_oracle",
        ],
        default="policy",
    )
    parser.add_argument("--dense-model", default="")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--dense-query-mode", choices=["question", "state"], default="question")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5, help="Dense score weight for hybrid selector.")
    parser.add_argument(
        "--front-pool-k",
        type=int,
        default=30,
        help="For hybrid_policy, merge BM25 topK and dense topK before local expansion/MMR.",
    )
    parser.add_argument(
        "--front-fusion",
        choices=["rrf", "score"],
        default="rrf",
        help="For hybrid_policy, rank merged BM25/Dense candidates with RRF or normalized score fusion.",
    )
    parser.add_argument(
        "--local-expansion-window",
        type=int,
        default=0,
        help="For hybrid_policy, include +/-N same-document neighbor sentences before policy compression.",
    )
    parser.add_argument("--mmr-lambda", type=float, default=0.7, help="Relevance weight for hybrid_policy MMR compression.")
    parser.add_argument(
        "--mmr-same-doc-similarity",
        type=float,
        default=0.35,
        help="Minimum redundancy similarity assigned to candidates from the same document during MMR.",
    )
    parser.add_argument("--reranker-model", default="")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
        help="Input max length for seq2seq rerankers such as monoT5/RankT5.",
    )
    parser.add_argument("--candidate-top-k", type=int, default=8)
    parser.add_argument("--select-top-k", type=int, default=1)
    parser.add_argument(
        "--policy-score-mode",
        choices=["rank", "deficit_role", "deficit_contribution", "front_policy_blend"],
        default="rank",
        help=(
            "Use plain ranking score, deficit-role compatibility, or "
            "deficit-true-contribution compatibility. For hybrid_policy, "
            "front_policy_blend safely combines front-end and policy scores."
        ),
    )
    parser.add_argument(
        "--policy-blend-weight",
        type=float,
        default=0.35,
        help=(
            "For --policy-score-mode front_policy_blend, weight assigned to the "
            "student policy score. Final score = (1-w)*front_score + w*policy_score."
        ),
    )
    parser.add_argument(
        "--deficit-role-weight",
        type=float,
        default=0.5,
        help="Weight for contribution-role and predicted-deficit compatibility.",
    )
    parser.add_argument(
        "--deficit-contribution-weight",
        type=float,
        default=0.5,
        help="Weight for true contribution head and predicted-deficit compatibility.",
    )
    parser.add_argument("--answer-mode", choices=["short", "json"], default="json")
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--save-online-states", action="store_true")
    parser.add_argument(
        "--refresh-answer-cache",
        action="store_true",
        help="Ignore existing cached answers and call the LLM again; still writes fresh cache files.",
    )
    parser.add_argument("--online-state-max-raw", type=int, default=8)
    parser.add_argument("--online-state-max-chars", type=int, default=260)
    parser.add_argument(
        "--stop-control",
        choices=["none", "deficit"],
        default="none",
        help="Optional online stopping controller. Default keeps old fixed-step behavior.",
    )
    parser.add_argument("--stop-min-steps", type=int, default=1)
    parser.add_argument("--stop-deficit-threshold", type=float, default=0.12)
    parser.add_argument(
        "--stop-deficit-mode",
        choices=["mean", "max"],
        default="mean",
        help="Use mean or max predicted typed deficit for deficit-driven stopping.",
    )
    parser.add_argument("--answer-cache-dir", default="")
    parser.add_argument("--agentic-cache-dir", default="")
    parser.add_argument(
        "--agentic-max-candidates",
        type=int,
        default=50,
        help="Maximum visible candidates per step for --selector agentic_llm.",
    )
    parser.add_argument("--llm-max-retries", type=int, default=8)
    parser.add_argument("--llm-retry-sleep", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--output", default="outputs/rag/hotpotqa_policy_rag_report.json")
    parser.add_argument(
        "--profile-runtime",
        action="store_true",
        help="Measure local selection stages and peak PyTorch GPU memory; excludes answer API time.",
    )
    parser.add_argument(
        "--profile-warmup-qids",
        type=int,
        default=20,
        help="Leading qids excluded from runtime and peak-memory measurements.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = list(read_jsonl(Path(args.samples)))
    if args.max_items > 0:
        samples = samples[: args.max_items]
    memory = load_memory(Path(args.memory))
    queries = load_queries(args.queries)
    ks = sorted({int(k) for k in args.ks.split(",") if k.strip()})
    rng = random.Random(args.seed)
    answer_cache_dir = Path(args.answer_cache_dir) if args.answer_cache_dir else Path(f"{args.output}.cache")
    if args.generate_answers:
        answer_cache_dir.mkdir(parents=True, exist_ok=True)
    agentic_cache_dir = Path(args.agentic_cache_dir) if args.agentic_cache_dir else Path(f"{args.output}.agentic_cache")
    if args.selector == "agentic_llm":
        agentic_cache_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in samples:
        qid = str(row.get("qid") or "")
        if qid:
            grouped[qid].append(row)
    qids = sorted(grouped)
    if args.max_qids > 0:
        qids = qids[: args.max_qids]

    policy = None
    if args.selector in {"policy", "dense_policy", "hybrid_policy"}:
        policy = PolicyModel(
            model_dir=Path(args.model_dir),
            checkpoint=Path(args.checkpoint),
            device=args.device,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    if args.stop_control == "deficit" and policy is None:
        raise RuntimeError("--stop-control deficit requires a policy-based selector.")
    dense = None
    if args.selector in {
        "dense",
        "hybrid",
        "hybrid_policy",
        "multi_query_dense",
        "iterative_dense",
        "iterative_hybrid",
        "dense_policy",
    }:
        if not args.dense_model:
            raise RuntimeError(
                "--dense-model is required for --selector dense, hybrid, hybrid_policy, multi_query_dense, "
                "iterative_dense, iterative_hybrid, or dense_policy"
            )
        dense = DenseScorer(args.dense_model, device=args.device, batch_size=args.dense_batch_size)
    reranker = None
    if args.selector == "generic_reranker":
        if not args.reranker_model:
            raise RuntimeError("--reranker-model is required for --selector generic_reranker")
        reranker = GenericReranker(args.reranker_model, device=args.device, batch_size=args.reranker_batch_size)
    if args.selector == "t5_reranker":
        if not args.reranker_model:
            raise RuntimeError("--reranker-model is required for --selector t5_reranker")
        reranker = T5Reranker(
            args.reranker_model,
            device=args.device,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
        )

    totals = Counter()
    qid_success = Counter()
    answer_metrics = Counter()
    records = []
    profile = {
        "enabled": bool(args.profile_runtime),
        "active": False,
        "cuda": bool(torch.cuda.is_available() and str(args.device).startswith("cuda")),
        "seconds": Counter(),
        "calls": Counter(),
        "measured_qids": 0,
        "measured_steps": 0,
    }

    for qid_index, qid in enumerate(tqdm(qids, desc="policy-rag")):
        if profile["enabled"] and not profile["active"] and qid_index >= max(0, args.profile_warmup_qids):
            if profile["cuda"]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            profile["active"] = True
        qid_profile_started = profile_start(profile)
        rows = sorted(grouped[qid], key=lambda item: int(item.get("t", 0)))
        if args.max_policy_steps > 0:
            rows = rows[: args.max_policy_steps]
        selected_units: list[str] = []
        selected_evidence: list[dict] = []
        selected_doc_ids: set[str] = set()
        online_state = init_online_state()
        gold_units: list[str] = []
        gold_doc_ids: set[str] = set()
        target_gold_units: list[str] = []
        target_gold_doc_ids: set[str] = set()
        for target_row in rows:
            target_positive_id = sample_positive_id(target_row)
            target_item = memory.get(target_positive_id)
            if target_positive_id and target_item:
                target_gold_units.append(target_positive_id)
                target_gold_doc_ids.add(target_item["doc_id"])
        step_records = []
        stop_record = None
        all_steps_correct = True

        for row in rows:
            question = str(row.get("question") or "")
            candidate_ids = sample_candidate_ids(row)
            positive_id = sample_positive_id(row)
            if not question or not candidate_ids or positive_id not in candidate_ids:
                totals["skipped"] += 1
                all_steps_correct = False
                continue

            candidate_texts = []
            usable_candidate_ids = []
            for unit_id in candidate_ids:
                item = memory.get(unit_id)
                if not item:
                    continue
                usable_candidate_ids.append(unit_id)
                candidate_texts.append(format_candidate_text(item))
            if positive_id not in usable_candidate_ids or not candidate_texts:
                totals["skipped"] += 1
                all_steps_correct = False
                continue

            if args.state_mode == "policy":
                if args.policy_context_source == "query_only":
                    context = f"Question: {question}"
                elif args.policy_context_source == "online_state":
                    notebook = online_state["K_t"]
                    context = f"Question: {question}\nNotebook:\n{notebook}"
                else:
                    notebook = "\n".join(
                        format_notebook_evidence(item, index + 1)
                        for index, item in enumerate(selected_evidence)
                    )
                    context = f"Question: {question}\nNotebook:\n{notebook}"
            else:
                context = f"Question: {question}\nNotebook:\n{sample_k_t(row)}"

            label = usable_candidate_ids.index(positive_id)
            display_scores = np.zeros(len(usable_candidate_ids), dtype=float)
            agentic_decision = None
            selector_profile_started = profile_start(profile)
            if args.selector == "policy":
                if args.policy_score_mode == "front_policy_blend":
                    raise RuntimeError("--policy-score-mode front_policy_blend requires --selector hybrid_policy.")
                if args.policy_score_mode == "deficit_role":
                    scores = policy.deficit_aware_score(context, candidate_texts, args.deficit_role_weight)
                elif args.policy_score_mode == "deficit_contribution":
                    scores = policy.contribution_aware_score(
                        context, candidate_texts, args.deficit_contribution_weight
                    )
                else:
                    scores = policy.score(context, candidate_texts)
                display_scores = scores
                order = np.argsort(scores)[::-1].tolist()
            elif args.selector == "bm25":
                stage_started = profile_start(profile)
                scores = bm25_scores(question, candidate_texts)
                profile_stop(profile, "bm25_retrieval", stage_started)
                display_scores = scores
                order = np.argsort(scores)[::-1].tolist()
            elif args.selector == "dense":
                dense_query = context if args.dense_query_mode == "state" else question
                stage_started = profile_start(profile)
                scores = dense.score(dense_query, candidate_texts)
                profile_stop(profile, "dense_retrieval", stage_started)
                display_scores = scores
                order = np.argsort(scores)[::-1].tolist()
            elif args.selector == "hybrid":
                dense_query = context if args.dense_query_mode == "state" else question
                stage_started = profile_start(profile)
                dense_scores = dense.score(dense_query, candidate_texts)
                profile_stop(profile, "dense_retrieval", stage_started)
                stage_started = profile_start(profile)
                lexical_scores = bm25_scores(question, candidate_texts)
                profile_stop(profile, "bm25_retrieval", stage_started)
                stage_started = profile_start(profile)
                alpha = min(1.0, max(0.0, args.hybrid_alpha))
                scores = alpha * minmax_normalize(dense_scores) + (1.0 - alpha) * minmax_normalize(lexical_scores)
                display_scores = scores
                order = np.argsort(scores)[::-1].tolist()
                profile_stop(profile, "score_fusion", stage_started)
            elif args.selector == "hybrid_policy":
                front_query = context
                dense_query = front_query if args.dense_query_mode == "state" else question
                stage_started = profile_start(profile)
                dense_scores = dense.score(dense_query, candidate_texts)
                profile_stop(profile, "dense_retrieval", stage_started)
                stage_started = profile_start(profile)
                lexical_scores = bm25_scores(front_query, candidate_texts)
                profile_stop(profile, "bm25_retrieval", stage_started)
                stage_started = profile_start(profile)
                alpha = min(1.0, max(0.0, args.hybrid_alpha))
                hybrid_scores = alpha * minmax_normalize(dense_scores) + (1.0 - alpha) * minmax_normalize(lexical_scores)
                dense_order = np.argsort(dense_scores)[::-1].tolist()
                lexical_order = np.argsort(lexical_scores)[::-1].tolist()
                front_scores = (
                    reciprocal_rank_fusion([lexical_order, dense_order], len(usable_candidate_ids))
                    if args.front_fusion == "rrf"
                    else hybrid_scores
                )
                front_order = np.argsort(front_scores)[::-1].tolist()
                front_pool_k = min(max(1, args.front_pool_k), len(front_order))
                seed_indices = list(dict.fromkeys(lexical_order[:front_pool_k] + dense_order[:front_pool_k]))
                profile_stop(profile, "score_fusion", stage_started)
                stage_started = profile_start(profile)
                expanded_indices = local_expanded_pool(
                    seed_indices,
                    usable_candidate_ids,
                    memory,
                    window=max(0, args.local_expansion_window),
                )
                expanded_indices.sort(key=lambda index: float(front_scores[index]), reverse=True)
                candidate_top_k = min(max(1, args.candidate_top_k), len(front_order))
                compressed_indices = mmr_select(
                    expanded_indices,
                    front_scores,
                    candidate_texts,
                    usable_candidate_ids,
                    memory,
                    limit=candidate_top_k,
                    lambda_=args.mmr_lambda,
                    same_doc_similarity=args.mmr_same_doc_similarity,
                )
                profile_stop(profile, "local_expansion_mmr", stage_started)
                compressed_texts = [candidate_texts[index] for index in compressed_indices]
                stage_started = profile_start(profile)
                if args.policy_score_mode == "front_policy_blend":
                    policy_scores = policy.score(context, compressed_texts)
                    front_local_scores = np.array([front_scores[index] for index in compressed_indices], dtype=float)
                    policy_weight = min(1.0, max(0.0, args.policy_blend_weight))
                    rerank_scores = (
                        (1.0 - policy_weight) * minmax_normalize(front_local_scores)
                        + policy_weight * minmax_normalize(policy_scores)
                    )
                elif args.policy_score_mode == "deficit_role":
                    policy_scores = policy.deficit_aware_score(context, compressed_texts, args.deficit_role_weight)
                    rerank_scores = policy_scores
                elif args.policy_score_mode == "deficit_contribution":
                    policy_scores = policy.contribution_aware_score(
                        context, compressed_texts, args.deficit_contribution_weight
                    )
                    rerank_scores = policy_scores
                else:
                    policy_scores = policy.score(context, compressed_texts)
                    rerank_scores = policy_scores
                reranked_local = np.argsort(rerank_scores)[::-1].tolist()
                order = [compressed_indices[index] for index in reranked_local]
                order += [index for index in front_order if index not in set(order)]
                display_scores = front_scores
                for local_index, original_index in enumerate(compressed_indices):
                    display_scores[original_index] = float(rerank_scores[local_index])
                profile_stop(profile, "policy_scoring", stage_started)
            elif args.selector == "multi_query_dense":
                query_scores = [
                    dense.score(query, candidate_texts)
                    for query in multi_query_variants(question, context)
                ]
                scores = np.max(np.vstack(query_scores), axis=0)
                display_scores = scores
                order = np.argsort(scores)[::-1].tolist()
            elif args.selector == "iterative_dense":
                stage_started = profile_start(profile)
                scores = dense.score(context, candidate_texts)
                profile_stop(profile, "dense_retrieval", stage_started)
                display_scores = scores
                order = np.argsort(scores)[::-1].tolist()
            elif args.selector == "iterative_hybrid":
                # Both retrieval channels consume the updated online state at every round.
                stage_started = profile_start(profile)
                dense_scores = dense.score(context, candidate_texts)
                profile_stop(profile, "dense_retrieval", stage_started)
                stage_started = profile_start(profile)
                lexical_scores = bm25_scores(context, candidate_texts)
                profile_stop(profile, "bm25_retrieval", stage_started)
                stage_started = profile_start(profile)
                alpha = min(1.0, max(0.0, args.hybrid_alpha))
                scores = (
                    alpha * minmax_normalize(dense_scores)
                    + (1.0 - alpha) * minmax_normalize(lexical_scores)
                )
                display_scores = scores
                order = np.argsort(scores)[::-1].tolist()
                profile_stop(profile, "score_fusion", stage_started)
            elif args.selector == "dense_policy":
                dense_query = context if args.dense_query_mode == "state" else question
                dense_scores = dense.score(dense_query, candidate_texts)
                dense_order = np.argsort(dense_scores)[::-1].tolist()
                candidate_top_k = min(max(1, args.candidate_top_k), len(dense_order))
                rerank_indices = dense_order[:candidate_top_k]
                rerank_texts = [candidate_texts[index] for index in rerank_indices]
                if args.policy_score_mode == "front_policy_blend":
                    raise RuntimeError("--policy-score-mode front_policy_blend requires --selector hybrid_policy.")
                if args.policy_score_mode == "deficit_role":
                    policy_scores = policy.deficit_aware_score(context, rerank_texts, args.deficit_role_weight)
                elif args.policy_score_mode == "deficit_contribution":
                    policy_scores = policy.contribution_aware_score(
                        context, rerank_texts, args.deficit_contribution_weight
                    )
                else:
                    policy_scores = policy.score(context, rerank_texts)
                reranked_local = np.argsort(policy_scores)[::-1].tolist()
                order = [rerank_indices[index] for index in reranked_local]
                order += [index for index in dense_order if index not in set(order)]
                display_scores = dense_scores
                for local_index, original_index in enumerate(rerank_indices):
                    display_scores[original_index] = float(policy_scores[local_index])
            elif args.selector == "generic_reranker":
                stage_started = profile_start(profile)
                scores = reranker.score(context, candidate_texts)
                profile_stop(profile, "reranker_scoring", stage_started)
                display_scores = scores
                order = np.argsort(scores)[::-1].tolist()
            elif args.selector == "t5_reranker":
                scores = reranker.score(context, candidate_texts)
                display_scores = scores
                order = np.argsort(scores)[::-1].tolist()
            elif args.selector == "agentic_llm":
                cache_path = agentic_cache_dir / f"{qid}_t{int(row.get('t', 0))}.json"
                if cache_path.exists() and not args.refresh_answer_cache:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    raw_agentic_answer = str(cached.get("raw_answer") or "")
                    agentic_tokens = int(cached.get("tokens") or 0)
                    agentic_latency = float(cached.get("latency") or 0.0)
                    selected_local = [
                        int(idx)
                        for idx in cached.get("selected_indices", [])
                        if isinstance(idx, int) and 0 <= int(idx) < len(usable_candidate_ids)
                    ]
                else:
                    selected_local, raw_agentic_answer, agentic_tokens, agentic_latency = select_with_agentic_llm(
                        question,
                        context,
                        candidate_texts,
                        select_top_k=args.select_top_k,
                        max_candidates=args.agentic_max_candidates,
                        max_retries=args.llm_max_retries,
                        retry_sleep=args.llm_retry_sleep,
                    )
                    cache_path.write_text(
                        json.dumps(
                            {
                                "qid": qid,
                                "t": int(row.get("t", 0)),
                                "selected_indices": selected_local,
                                "raw_answer": raw_agentic_answer,
                                "tokens": agentic_tokens,
                                "latency": agentic_latency,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                order = selected_local + [idx for idx in range(len(usable_candidate_ids)) if idx not in set(selected_local)]
                display_scores = np.zeros(len(usable_candidate_ids), dtype=float)
                for rank, idx in enumerate(order):
                    display_scores[idx] = float(len(order) - rank)
                agentic_decision = {
                    "raw_answer": raw_agentic_answer,
                    "tokens": agentic_tokens,
                    "latency": round(agentic_latency, 3),
                    "selected_indices": selected_local,
                }
            elif args.selector == "first":
                order = list(range(len(usable_candidate_ids)))
            elif args.selector == "random":
                order = list(range(len(usable_candidate_ids)))
                rng.shuffle(order)
            else:
                order = [label] + [index for index in range(len(usable_candidate_ids)) if index != label]
            profile_stop(profile, "selector_total", selector_profile_started)
            if profile["active"]:
                profile["measured_steps"] += 1

            deficit_estimate = None
            stop_decision = {
                "should_stop": False,
                "reason": "disabled",
            }
            if args.stop_control == "deficit":
                deficit_estimate = policy.estimate_deficit(context, candidate_texts)
                stop_value = float(deficit_estimate[args.stop_deficit_mode])
                enough_steps = len(selected_units) >= max(0, args.stop_min_steps)
                should_stop = enough_steps and stop_value <= args.stop_deficit_threshold
                stop_decision = {
                    "should_stop": bool(should_stop),
                    "reason": "deficit_below_threshold" if should_stop else "continue",
                    "mode": args.stop_deficit_mode,
                    "value": round(stop_value, 6),
                    "threshold": args.stop_deficit_threshold,
                    "selected_units": len(selected_units),
                    "min_steps": args.stop_min_steps,
                    "deficit": {key: round(float(value), 6) for key, value in deficit_estimate.items()},
                }
                if should_stop:
                    totals["stop_triggered"] += 1
                    stop_record = {
                        "t": int(row.get("t", 0)),
                        "question": question,
                        **stop_decision,
                    }
                    all_steps_correct = False
                    break
            pred_index = order[0]
            pred_id = usable_candidate_ids[pred_index]
            positive_memory = memory[positive_id]
            selected_indices = order[: max(1, args.select_top_k)]
            selected_step_ids = [usable_candidate_ids[index] for index in selected_indices]
            online_state_before = json.loads(json.dumps(online_state)) if args.save_online_states else None

            totals["steps"] += 1
            step_correct = pred_id == positive_id
            all_steps_correct = all_steps_correct and step_correct
            totals["step_selected_contains_gold"] += int(positive_id in selected_step_ids)
            for k in ks:
                totals[f"step_acc@{k}"] += int(label in order[:k])

            for selected_id in selected_step_ids:
                if selected_id in selected_units:
                    continue
                selected_memory = memory[selected_id]
                selected_units.append(selected_id)
                selected_evidence.append(selected_memory)
                selected_doc_ids.add(selected_memory["doc_id"])
                online_state = update_online_state(
                    online_state,
                    selected_id,
                    selected_memory,
                    memory,
                    step_id=int(row.get("t", 0)),
                    max_raw=args.online_state_max_raw,
                    max_chars_per_item=args.online_state_max_chars,
                )
            gold_units.append(positive_id)
            gold_doc_ids.add(positive_memory["doc_id"])
            step_record = {
                "t": int(row.get("t", 0)),
                "question": question,
                "positive_unit_id": positive_id,
                "predicted_unit_id": pred_id,
                "selected_unit_ids": selected_step_ids,
                "correct": step_correct,
                "selected_contains_gold": positive_id in selected_step_ids,
                "positive_rank": order.index(label) + 1,
                "top5": [
                    {
                        "unit_id": usable_candidate_ids[index],
                        "doc_id": memory[usable_candidate_ids[index]]["doc_id"],
                        "score": round(float(display_scores[index]), 6),
                    }
                    for index in order[:5]
                ],
            }
            if deficit_estimate is not None:
                step_record["deficit_estimate"] = {
                    key: round(float(value), 6) for key, value in deficit_estimate.items()
                }
                step_record["stop_decision"] = stop_decision
            if agentic_decision is not None:
                step_record["agentic_decision"] = agentic_decision
            if args.save_online_states:
                step_record["online_state_before"] = online_state_before
                step_record["online_state_after"] = online_state
            step_records.append(step_record)

        if not step_records:
            continue

        profile_stop(profile, "qid_selection_pipeline", qid_profile_started)
        if profile["active"]:
            profile["measured_qids"] += 1

        qid_success["total"] += 1
        qid_success["stopped"] += int(stop_record is not None)
        qid_success["all_steps_correct"] += int(all_steps_correct)
        qid_success["any_gold_doc_selected"] += int(bool(selected_doc_ids & target_gold_doc_ids))
        qid_success["full_gold_doc_coverage"] += int(target_gold_doc_ids.issubset(selected_doc_ids))
        qid_success["full_gold_unit_coverage"] += int(set(target_gold_units).issubset(set(selected_units)))

        answer = ""
        raw_answer = ""
        answer_tokens = 0
        answer_latency = 0.0
        question = str(rows[0].get("question") or "")
        if args.generate_answers:
            cache_path = cache_file_for_qid(answer_cache_dir, qid)
            if cache_path.exists() and not args.refresh_answer_cache:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                answer = str(cached.get("answer") or "")
                raw_answer = str(cached.get("raw_answer") or "")
                answer_tokens = int(cached.get("answer_tokens") or 0)
                answer_latency = float(cached.get("answer_latency") or 0.0)
            else:
                try:
                    answer, raw_answer, answer_tokens, answer_latency = answer_with_llm(
                        question,
                        selected_evidence,
                        answer_mode=args.answer_mode,
                        max_retries=args.llm_max_retries,
                        retry_sleep=args.llm_retry_sleep,
                    )
                except Exception as exc:
                    answer = ""
                    raw_answer = f"ERROR: {exc}"
                    answer_tokens = 0
                    answer_latency = 0.0
                    answer_metrics["answer_errors"] += 1
                cache_path.write_text(
                    json.dumps(
                        {
                            "qid": qid,
                            "answer": answer,
                            "raw_answer": raw_answer,
                            "answer_tokens": answer_tokens,
                            "answer_latency": answer_latency,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        gold_answer = str((queries.get(qid) or {}).get("answer") or "")
        if args.generate_answers and gold_answer:
            answer_metrics["answer_judged"] += 1
            answer_metrics["answer_em"] += exact_match_score(answer, gold_answer)
            answer_metrics["answer_contains"] += answer_contains_score(answer, gold_answer)
            answer_metrics["answer_f1"] += f1_score(answer, gold_answer)
            answer_metrics["answer_tokens"] += answer_tokens
            answer_metrics["answer_latency"] += answer_latency

        records.append(
            {
                "qid": qid,
                "question": question,
                "answer": answer,
                "raw_answer": raw_answer,
                "answer_tokens": answer_tokens,
                "answer_latency": round(answer_latency, 3),
                "gold_answer": gold_answer,
                "answer_em": exact_match_score(answer, gold_answer) if gold_answer and answer else None,
                "answer_contains": answer_contains_score(answer, gold_answer) if gold_answer and answer else None,
                "answer_f1": round(f1_score(answer, gold_answer), 6) if gold_answer and answer else None,
                "selected_unit_ids": selected_units,
                "gold_unit_ids": target_gold_units,
                "selected_doc_ids": sorted(selected_doc_ids),
                "gold_doc_ids": sorted(target_gold_doc_ids),
                "all_steps_correct": all_steps_correct,
                "stopped_early": stop_record is not None,
                "stop_record": stop_record,
                "steps": step_records,
                "final_online_state": online_state if args.save_online_states else None,
            }
        )

    step_total = max(totals["steps"], 1)
    qid_total = max(qid_success["total"], 1)
    summary = {
        "samples": args.samples,
        "memory": args.memory,
        "queries": args.queries,
        "checkpoint": args.checkpoint,
        "state_mode": args.state_mode,
        "policy_context_source": args.policy_context_source,
        "selector": args.selector,
        "dense_model": args.dense_model,
        "dense_query_mode": args.dense_query_mode,
        "hybrid_alpha": args.hybrid_alpha,
        "front_pool_k": args.front_pool_k,
        "front_fusion": args.front_fusion,
        "local_expansion_window": args.local_expansion_window,
        "mmr_lambda": args.mmr_lambda,
        "mmr_same_doc_similarity": args.mmr_same_doc_similarity,
        "reranker_model": args.reranker_model,
        "reranker_max_length": args.reranker_max_length,
        "candidate_top_k": args.candidate_top_k,
        "select_top_k": args.select_top_k,
        "policy_score_mode": args.policy_score_mode,
        "policy_blend_weight": args.policy_blend_weight,
        "deficit_role_weight": args.deficit_role_weight,
        "deficit_contribution_weight": args.deficit_contribution_weight,
        "answer_mode": args.answer_mode,
        "refresh_answer_cache": args.refresh_answer_cache,
        "save_online_states": args.save_online_states,
        "online_state_max_raw": args.online_state_max_raw,
        "online_state_max_chars": args.online_state_max_chars,
        "stop_control": args.stop_control,
        "stop_min_steps": args.stop_min_steps,
        "stop_deficit_threshold": args.stop_deficit_threshold,
        "stop_deficit_mode": args.stop_deficit_mode,
        "answer_cache_dir": str(answer_cache_dir) if args.generate_answers else "",
        "agentic_cache_dir": str(agentic_cache_dir) if args.selector == "agentic_llm" else "",
        "agentic_max_candidates": args.agentic_max_candidates,
        "seed": args.seed,
        "sample_states": len(samples),
        "qids": qid_success["total"],
        "steps": totals["steps"],
        "skipped": totals["skipped"],
        "stopped_qids": qid_success["stopped"],
        "stop_triggered": totals["stop_triggered"],
        "trajectory_all_steps_correct": round(qid_success["all_steps_correct"] / qid_total, 6),
        "any_gold_doc_selected": round(qid_success["any_gold_doc_selected"] / qid_total, 6),
        "full_gold_doc_coverage": round(qid_success["full_gold_doc_coverage"] / qid_total, 6),
        "full_gold_unit_coverage": round(qid_success["full_gold_unit_coverage"] / qid_total, 6),
        "step_selected_contains_gold": round(totals["step_selected_contains_gold"] / step_total, 6),
    }
    for k in ks:
        summary[f"step_acc@{k}"] = round(totals[f"step_acc@{k}"] / step_total, 6)

    judged_answers = max(answer_metrics["answer_judged"], 1)
    summary.update(
        {
            "answer_judged": answer_metrics["answer_judged"],
            "answer_em": round(answer_metrics["answer_em"] / judged_answers, 6)
            if answer_metrics["answer_judged"]
            else None,
            "answer_contains": round(answer_metrics["answer_contains"] / judged_answers, 6)
            if answer_metrics["answer_judged"]
            else None,
            "answer_f1": round(answer_metrics["answer_f1"] / judged_answers, 6)
            if answer_metrics["answer_judged"]
            else None,
            "avg_answer_tokens": round(answer_metrics["answer_tokens"] / judged_answers, 2)
            if answer_metrics["answer_judged"]
            else None,
            "avg_answer_latency": round(answer_metrics["answer_latency"] / judged_answers, 3)
            if answer_metrics["answer_judged"]
            else None,
            "answer_errors": answer_metrics["answer_errors"],
        }
    )

    if args.profile_runtime:
        measured_qids = int(profile["measured_qids"])
        pipeline_seconds = float(profile["seconds"].get("qid_selection_pipeline", 0.0))
        summary["runtime_profile"] = {
            "warmup_qids": min(max(0, args.profile_warmup_qids), len(qids)),
            "measured_qids": measured_qids,
            "measured_steps": int(profile["measured_steps"]),
            "stage_seconds": {
                stage: round(float(seconds), 6)
                for stage, seconds in sorted(profile["seconds"].items())
            },
            "stage_calls": dict(sorted(profile["calls"].items())),
            "stage_avg_ms_per_call": {
                stage: round(1000.0 * float(profile["seconds"][stage]) / profile["calls"][stage], 4)
                for stage in sorted(profile["seconds"])
                if profile["calls"][stage]
            },
            "selection_avg_ms_per_qid": (
                round(1000.0 * pipeline_seconds / measured_qids, 4) if measured_qids else None
            ),
            "selection_throughput_qids_per_second": (
                round(measured_qids / pipeline_seconds, 4) if pipeline_seconds > 0 else None
            ),
            "peak_gpu_allocated_mb": (
                round(torch.cuda.max_memory_allocated() / (1024**2), 2) if profile["cuda"] else 0.0
            ),
            "peak_gpu_reserved_mb": (
                round(torch.cuda.max_memory_reserved() / (1024**2), 2) if profile["cuda"] else 0.0
            ),
            "includes_answer_api": False,
        }

    report = {"summary": summary, "results": records}
    write_json(report, Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
