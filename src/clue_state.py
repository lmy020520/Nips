"""Deterministic question-derived clue state for the Stage 3.3 baseline."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
QUOTED_RE = re.compile(r"""["']([^"']{2,80})["']""")
CAPITALIZED_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’-]*)(?:\s+(?:[A-Z][A-Za-z0-9'’-]*|of|the|and|de)){0,5}\b"
)
CLUE_STATE_VERSION = "fiske_inspired_textual_clues_v1"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "between",
    "both",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


def normalize_tokens(text: object) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


def stable_digest(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def format_clue_evidence(memory_item: dict) -> str:
    """Format acquired evidence identically for training and online updates."""
    title = str(memory_item.get("title") or memory_item.get("doc_id") or "").strip()
    text = str(memory_item.get("text") or "").strip()
    return f"{title}: {text}" if title else text


def _dedupe_phrases(phrases: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for phrase in phrases:
        normalized = " ".join(normalize_tokens(phrase))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(" ".join(str(phrase).strip().split()))
    return result


def infer_answer_type(question: str) -> str:
    normalized = " ".join(normalize_tokens(question))
    if normalized.startswith(("who ", "whom ", "whose ")):
        return "person"
    if normalized.startswith("when "):
        return "date or time"
    if normalized.startswith("where "):
        return "location"
    if normalized.startswith(("how many ", "how much ")):
        return "number"
    if normalized.startswith(
        ("what city ", "what country ", "what location ", "what place ")
    ):
        return "location"
    if normalized.startswith(("what year ", "what date ", "what time ")):
        return "date or time"
    if normalized.startswith("which "):
        return "named choice"
    return "answer entity or value"


def extract_question_clues(
    question: str,
    *,
    max_entities: int = 4,
    max_relations: int = 8,
) -> list[dict]:
    """Extract a frozen clue list using the question text only."""
    question = " ".join(str(question).strip().split())
    if not question:
        raise ValueError("question must not be empty")

    quoted = [match.group(1) for match in QUOTED_RE.finditer(question)]
    capitalized = [match.group(0) for match in CAPITALIZED_RE.finditer(question)]
    entities = [
        phrase
        for phrase in _dedupe_phrases([*quoted, *capitalized])
        if any(token not in STOPWORDS for token in normalize_tokens(phrase))
    ][:max_entities]

    entity_tokens = {
        token
        for phrase in entities
        for token in normalize_tokens(phrase)
    }
    relation_terms = []
    for token in normalize_tokens(question):
        if len(token) < 3 or token in STOPWORDS or token in entity_tokens:
            continue
        if token not in relation_terms:
            relation_terms.append(token)
    relation_terms = relation_terms[:max_relations]

    clues = []
    for phrase in entities:
        clues.append(
            {
                "clue_id": f"entity_{len(clues) + 1}",
                "kind": "entity",
                "text": phrase,
                "tokens": normalize_tokens(phrase),
                "coverage_rule": "lexical_all",
            }
        )
    for term in relation_terms:
        clues.append(
            {
                "clue_id": f"relation_{len(clues) + 1}",
                "kind": "relation",
                "text": term,
                "tokens": [term],
                "coverage_rule": "lexical_all",
            }
        )
    clues.append(
        {
            "clue_id": f"answer_type_{len(clues) + 1}",
            "kind": "answer_type",
            "text": infer_answer_type(question),
            "tokens": [],
            "coverage_rule": "answer_type_heuristic",
        }
    )
    return clues


def _answer_type_covered(answer_type: str, evidence_text: str) -> bool:
    text = str(evidence_text)
    normalized = " ".join(normalize_tokens(text))
    if not normalized:
        return False
    if answer_type == "number":
        return bool(re.search(r"\b\d+(?:[.,]\d+)?\b", text))
    if answer_type == "date or time":
        return bool(
            re.search(
                r"\b(?:1[0-9]{3}|20[0-9]{2}|"
                r"january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\b",
                normalized,
            )
        )
    if answer_type == "location":
        return bool(
            re.search(
                r"\b(?:in|at|near|from|city|country|state|province|district|"
                r"county|island|river|mountain)\b",
                normalized,
            )
        )
    if answer_type == "person":
        return bool(
            re.search(
                r"\b(?:born|died|actor|actress|author|director|president|"
                r"king|queen|professor|scientist|he|she|his|her)\b",
                normalized,
            )
        )
    # Generic and choice questions do not have a reliable answer-type test
    # without a learned model, so they remain unresolved.
    return False


def build_clue_state(question: str, evidence_texts: Sequence[str]) -> dict:
    """Mark frozen question clues using only evidence already in the prefix."""
    clues = extract_question_clues(question)
    evidence_text = "\n".join(str(text) for text in evidence_texts if str(text).strip())
    evidence_tokens = set(normalize_tokens(evidence_text))

    resolved = []
    for clue in clues:
        item = dict(clue)
        tokens = list(item["tokens"])
        if item["coverage_rule"] == "lexical_all":
            covered = bool(tokens) and all(token in evidence_tokens for token in tokens)
            score = (
                sum(token in evidence_tokens for token in tokens) / len(tokens)
                if tokens
                else 0.0
            )
        else:
            covered = _answer_type_covered(item["text"], evidence_text)
            score = float(covered)
        item["covered"] = bool(covered)
        item["coverage_score"] = round(float(score), 6)
        resolved.append(item)

    return {
        "version": CLUE_STATE_VERSION,
        "question_sha256": stable_digest(question),
        "generator_inputs": ["question"],
        "coverage_inputs": ["current_prefix_evidence"],
        "clues": resolved,
        "coverage_vector": [int(item["covered"]) for item in resolved],
    }


def render_clue_state(clue_state: dict) -> str:
    clues = list(clue_state.get("clues") or [])
    lines = ["Textual clue state:"]
    for item in clues:
        status = "covered" if item.get("covered") else "unresolved"
        lines.append(f"[{status}] {item.get('kind')}: {item.get('text')}")
    unresolved = [str(item.get("text")) for item in clues if not item.get("covered")]
    lines.append(
        "Unresolved clues: " + (", ".join(unresolved) if unresolved else "none")
    )
    return "\n".join(lines)
