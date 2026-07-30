"""Shared deterministic knowledge-state update and rendering utilities."""

from __future__ import annotations


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


def render_online_k_t(
    state: dict,
    memory: dict[str, dict],
    *,
    max_raw: int = 8,
    max_chars_per_item: int = 260,
) -> str:
    """Render the notebook exactly as used by the online policy."""
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
    """Apply the shared Update -> Ledger -> Render transition."""
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

    doc_id = str(memory_item.get("doc_id") or memory_item.get("title") or "")
    if unit_id not in next_state["A_t"]["raw_unit_ids"]:
        next_state["A_t"]["raw_unit_ids"].append(unit_id)
    if doc_id and doc_id not in next_state["A_t"]["doc_ids"]:
        next_state["A_t"]["doc_ids"].append(doc_id)
    if doc_id:
        next_state["A_t"]["unit_doc"][unit_id] = doc_id

    next_state["K_t"] = render_online_k_t(
        next_state,
        memory,
        max_raw=max_raw,
        max_chars_per_item=max_chars_per_item,
    )
    return next_state
