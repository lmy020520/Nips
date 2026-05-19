import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPLITS = ["train", "val", "test"]
DEFAULT_BASE = Path(__file__).resolve().parents[1] / "data" / "hotpotqa_distractor_v4"


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_derived(unit_id: object) -> bool:
    return isinstance(unit_id, str) and "::derived::" in unit_id


def merge_unique(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_time_from_run_id(run_id: str) -> str:
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_", run_id)
    if not match:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day} {hour}:{minute}:{second}"


def load_states_if_available(base_dir: Path) -> Dict[Tuple[str, str, int], dict]:
    out: Dict[Tuple[str, str, int], dict] = {}
    trajectories_dir = base_dir / "trajectories"
    for split in SPLITS:
        path = trajectories_dir / f"states_{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            out[(split, str(row.get("qid")), int(row.get("t")))] = row
    return out


def load_raw_texts(base_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for split in SPLITS:
        path = base_dir / "unit_registry" / f"raw_units_{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            unit_id = row.get("unit_id")
            text = row.get("text")
            if isinstance(unit_id, str) and isinstance(text, str):
                out[unit_id] = text
    return out


def load_queries(base_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for split in SPLITS:
        path = base_dir / "queries" / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            qid = row.get("qid")
            question = row.get("question")
            if isinstance(qid, str) and isinstance(question, str):
                out[qid] = question
    return out


def question_terms(question: str) -> List[str]:
    terms = re.findall(r"[A-Za-z][A-Za-z]+", question.lower())
    stop = {
        "are", "both", "type", "types", "what", "which", "does", "did", "the",
        "and", "or", "of", "in", "is", "was", "were", "plant", "plants",
    }
    return [term for term in terms if term not in stop]


def derived_text_from_step(step: dict, unit_id: str) -> str:
    for item in (step.get("proposer_trace") or {}).get("harvest_candidates", []):
        if isinstance(item, dict) and item.get("unit_id") == unit_id:
            text = item.get("text")
            if isinstance(text, str):
                return text
    return ""


def is_unrelated_yes_no_plant_derived(*, question: str, unit_id: str, text: str) -> bool:
    q = question.lower().strip()
    if not q.startswith(("are ", "is ", "was ", "were ")) or "plant" not in q:
        return False
    if not is_derived(unit_id):
        return False
    lowered = text.lower()
    if "clay" in lowered or "whitemud" in lowered or "brick plant" in lowered:
        terms = question_terms(question)
        return not any(term in lowered for term in terms)
    return False


def choose_relevant_replacement_raw(*, question: str, c_t: List[str], raw_texts: Dict[str, str]) -> str | None:
    terms = question_terms(question)
    best_unit_id = None
    best_score = -1
    for unit_id in c_t:
        if is_derived(unit_id):
            continue
        text = raw_texts.get(unit_id, "")
        lowered = f"{unit_id} {text}".lower()
        score = sum(1 for term in terms if term in lowered)
        if "plant" in lowered or "flowering" in lowered or "vegetation" in lowered:
            score += 2
        if score > best_score:
            best_score = score
            best_unit_id = unit_id
    return best_unit_id


def raw_state_ref(unit_id: str) -> dict:
    parts = unit_id.split("::")
    parent_chunk_id = "::".join(parts[:2]) if len(parts) >= 3 else unit_id
    return {
        "unit_id": unit_id,
        "parent_chunk_id": parent_chunk_id,
        "added_step": None,
    }


def render_k_t_from_history(history: List[dict], raw_texts: Dict[str, str], derived_texts: Dict[str, str]) -> str:
    lines: List[str] = []
    for item in history:
        unit_id = str(item.get("unit_id", ""))
        if is_derived(unit_id):
            text = derived_texts.get(unit_id, "")
            if text:
                lines.append(f"[derived_note] {text}")
            continue
        text = raw_texts.get(unit_id, "")
        if text:
            doc_id = unit_id.split("::")[1] if "::" in unit_id else unit_id
            lines.append(f"{doc_id} {text}")
    return "\n".join(lines)


def infer_allowed_subtype(step: dict) -> str:
    existing = (step.get("gate_trace") or {}).get("derive_subtype")
    if isinstance(existing, str) and existing.strip():
        return existing
    t = step.get("t")
    if isinstance(t, int) and t <= 1:
        return "early_bridge"
    return "late_verification"


def candidate_type_from_step(step: dict, unit_id: str) -> str:
    for item in (step.get("proposer_trace") or {}).get("harvest_candidates", []):
        if isinstance(item, dict) and item.get("unit_id") == unit_id:
            return str(item.get("type", ""))
    return ""


def allowed_subtype_for_positive_derived(step: dict, unit_id: str) -> str:
    note_type = candidate_type_from_step(step, unit_id)
    t = step.get("t")
    if note_type == "verification_note":
        return "answer_facing_verification"
    if isinstance(t, int) and t <= 1:
        return "early_bridge"
    return "bridge_scaffold"


def bridge_subtype_for_step(step: dict) -> str:
    t = step.get("t")
    if isinstance(t, int) and t <= 1:
        return "early_bridge"
    return "bridge_contextualization"


def normalize_positive_derived_type_subtype(step: dict) -> None:
    positive_unit_id = step.get("positive_unit_id")
    if not is_derived(positive_unit_id):
        return
    gate_trace = step.get("gate_trace") or {}
    note_type = candidate_type_from_step(step, str(positive_unit_id))
    subtype = gate_trace.get("derive_subtype")
    if note_type == "bridge_note" and subtype in {
        "late_verification",
        "answer_facing_verification",
        "abstractive_verification",
    }:
        gate_trace["derive_subtype"] = bridge_subtype_for_step(step)
        gate_trace["type_subtype_repaired"] = True
        gate_trace["type_subtype_repair_reason"] = "positive_bridge_note_should_use_bridge_subtype"
        gate_trace.pop("late_verification_trigger", None)
        if gate_trace["derive_subtype"] == "early_bridge":
            gate_trace["early_bridge_trigger"] = True
        else:
            gate_trace["bridge_contextualization_trigger"] = True
    elif note_type == "verification_note" and subtype in {
        "early_bridge",
        "bridge_scaffold",
        "bridge_contextualization",
        "bridge_scaffold_for_progress",
        "bridge_scaffold_for_new_raw",
    }:
        gate_trace["derive_subtype"] = "answer_facing_verification"
        gate_trace["type_subtype_repaired"] = True
        gate_trace["type_subtype_repair_reason"] = "positive_verification_note_should_use_verification_subtype"
        gate_trace["late_verification_trigger"] = True
    step["gate_trace"] = gate_trace


def sync_candidate_fields_after_gate_repair(step: dict) -> None:
    gate_trace = step.get("gate_trace") or {}
    trigger_derived = bool(gate_trace.get("trigger_derived", False))
    positive_unit_id = step.get("positive_unit_id")
    r_t = [str(x) for x in step.get("R_t", [])]
    g_final = [str(x) for x in step.get("G_t_final", [])]
    candidate_debug = step.get("candidate_debug") if isinstance(step.get("candidate_debug"), dict) else {}
    g_aux = [str(x) for x in candidate_debug.get("G_t_aux", [])]

    if trigger_derived:
        step["need_derived"] = bool(gate_trace.get("derived_need", True))
        step["triggered_propose_derived"] = True
        if candidate_debug:
            candidate_debug["need_derived"] = step["need_derived"]
            candidate_debug["triggered_propose_derived"] = True
        derived_debug = step.get("derived_debug")
        if isinstance(derived_debug, dict):
            derived_debug["need_derived"] = step["need_derived"]
            derived_debug["triggered_propose_derived"] = True
        return

    demoted = [unit_id for unit_id in g_final if is_derived(unit_id)]
    if demoted:
        g_final = [unit_id for unit_id in g_final if not is_derived(unit_id)]
        g_aux = merge_unique(g_aux + demoted)
        step["G_t_final"] = g_final
        step["C_t"] = merge_unique(r_t + g_final)
        if candidate_debug:
            candidate_debug["G_t_final"] = g_final
            candidate_debug["G_t_aux"] = g_aux
            candidate_debug["C_t"] = list(step["C_t"])
            candidate_debug["final_retained_count"] = len(g_final)
            candidate_debug["aux_retained_count"] = len(g_aux)
            candidate_debug["shadow_derived_candidates"] = merge_unique(
                [str(x) for x in candidate_debug.get("shadow_derived_candidates", [])] + demoted
            )
        step.setdefault("derived_debug", {})
        if isinstance(step["derived_debug"], dict):
            step["derived_debug"]["shadow_derived_demoted_to_aux"] = demoted
    control_need = bool(gate_trace.get("derived_need", False))
    step["need_derived"] = control_need
    step["triggered_propose_derived"] = False
    if candidate_debug:
        candidate_debug["need_derived"] = control_need
        candidate_debug["triggered_propose_derived"] = False
    derived_debug = step.get("derived_debug")
    if isinstance(derived_debug, dict):
        derived_debug["need_derived"] = control_need
        derived_debug["triggered_propose_derived"] = False
    if is_derived(positive_unit_id):
        raise ValueError(
            f"typed gate false but positive derived remained after repair: "
            f"qid={positive_unit_id} t={step.get('t')}"
        )


def collect_derived_texts(rows: List[dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in rows:
        for step in row.get("steps", []):
            for item in (step.get("proposer_trace") or {}).get("harvest_candidates", []):
                if not isinstance(item, dict):
                    continue
                unit_id = item.get("unit_id")
                text = item.get("text")
                if is_derived(unit_id) and isinstance(text, str):
                    out[str(unit_id)] = text
    return out


def repair_unrelated_yes_no_selected_derived(
    *,
    split: str,
    row: dict,
    step: dict,
    question: str,
    raw_texts: Dict[str, str],
    states_by_key: Dict[Tuple[str, str, int], dict],
    derived_texts: Dict[str, str],
) -> None:
    positive_unit_id = step.get("positive_unit_id")
    if not is_derived(positive_unit_id):
        return
    positive_text = derived_text_from_step(step, str(positive_unit_id))
    if not is_unrelated_yes_no_plant_derived(
        question=question,
        unit_id=str(positive_unit_id),
        text=positive_text,
    ):
        return
    replacement = choose_relevant_replacement_raw(
        question=question,
        c_t=[str(x) for x in step.get("C_t", [])],
        raw_texts=raw_texts,
    )
    if replacement is None:
        return

    old_positive = str(positive_unit_id)
    step["positive_unit_id"] = replacement
    step["selected_from"] = "R_t"
    step["selected_provenance"] = "raw"
    step["semantic_replacement"] = {
        "replaced_selected_unit": True,
        "original_unit_id": old_positive,
        "retained_unit_id": replacement,
        "replacement_reason": "yes_no_unrelated_derived_demoted",
    }
    g_final = [str(x) for x in step.get("G_t_final", [])]
    if old_positive in g_final:
        step["G_t_final"] = [unit_id for unit_id in g_final if unit_id != old_positive]
    r_t = [str(x) for x in step.get("R_t", [])]
    step["C_t"] = merge_unique(r_t + [str(x) for x in step.get("G_t_final", [])])
    gate_trace = step.get("gate_trace") or {}
    gate_trace["semantic_replacement"] = step["semantic_replacement"]
    step["gate_trace"] = gate_trace

    candidate_debug = step.get("candidate_debug") if isinstance(step.get("candidate_debug"), dict) else {}
    if candidate_debug:
        candidate_g_final = [str(x) for x in candidate_debug.get("G_t_final", [])]
        candidate_g_aux = [str(x) for x in candidate_debug.get("G_t_aux", [])]
        if old_positive in candidate_g_final:
            candidate_debug["G_t_final"] = [unit_id for unit_id in candidate_g_final if unit_id != old_positive]
            candidate_debug["G_t_aux"] = merge_unique(candidate_g_aux + [old_positive])
            candidate_debug["C_t"] = list(step["C_t"])
            candidate_debug["final_retained_count"] = len(candidate_debug["G_t_final"])
            candidate_debug["aux_retained_count"] = len(candidate_debug["G_t_aux"])
        candidate_debug["teacher_select_debug"] = {
            **(candidate_debug.get("teacher_select_debug") or {}),
            "semantic_replacement": step["semantic_replacement"],
        }

    t = step.get("t")
    if not isinstance(t, int):
        return
    next_state = states_by_key.get((split, str(row.get("qid")), t + 1))
    if not isinstance(next_state, dict):
        return
    h_t = next_state.get("H_t")
    if isinstance(h_t, list):
        for item in h_t:
            if isinstance(item, dict) and item.get("unit_id") == old_positive:
                item["unit_id"] = replacement
                item["chunk_id"] = replacement
                item["parent_chunk_id"] = replacement
    s_t = next_state.get("S_t")
    if isinstance(s_t, dict):
        derived_refs = s_t.get("derived_refs")
        if isinstance(derived_refs, list):
            s_t["derived_refs"] = [
                ref for ref in derived_refs
                if not (isinstance(ref, dict) and ref.get("unit_id") == old_positive)
            ]
        raw_refs = s_t.get("raw_refs")
        if isinstance(raw_refs, list) and not any(isinstance(ref, dict) and ref.get("unit_id") == replacement for ref in raw_refs):
            ref = raw_state_ref(replacement)
            ref["added_step"] = len(h_t) - 1 if isinstance(h_t, list) else None
            raw_refs.append(ref)
        s_t["last_added_unit_id"] = replacement
    if isinstance(h_t, list):
        next_state["K_t"] = render_k_t_from_history(h_t, raw_texts, derived_texts)


def repair_full_records(base_dir: Path, run_id: str) -> None:
    trajectories_dir = base_dir / "trajectories"
    states_by_key = load_states_if_available(base_dir)
    raw_texts = load_raw_texts(base_dir)
    queries = load_queries(base_dir)
    build_time = build_time_from_run_id(run_id)
    all_rows_by_split: Dict[str, List[dict]] = {}
    for split in SPLITS:
        path = trajectories_dir / f"full_{split}.jsonl"
        all_rows_by_split[split] = list(read_jsonl(path))
    derived_texts = collect_derived_texts([row for rows in all_rows_by_split.values() for row in rows])
    for split in SPLITS:
        path = trajectories_dir / f"full_{split}.jsonl"
        rows = all_rows_by_split[split]
        for row in rows:
            question = queries.get(str(row.get("qid")), "")
            build_meta = row.setdefault("build_meta", {})
            build_meta["run_id"] = run_id
            build_meta["build_time"] = build_time
            build_meta["source"] = "build_hotpotqa_full_trajectories_v4.py"
            for step in row.get("steps", []):
                gate_trace = step.get("gate_trace") or {}
                if "shadow_only" in gate_trace:
                    gate_trace["diagnostic_shadow_enabled"] = bool(gate_trace.pop("shadow_only"))
                legacy_need = bool(step.get("need_derived", False))
                legacy_trigger = bool(step.get("triggered_propose_derived", False))
                control_need = bool(gate_trace.get("derived_need", False))
                control_trigger = bool(gate_trace.get("trigger_derived", False))
                positive_unit_id = step.get("positive_unit_id")
                if is_derived(positive_unit_id) and not control_trigger:
                    control_need = True
                    control_trigger = True
                    subtype = infer_allowed_subtype(step)
                    gate_trace["derived_need"] = True
                    gate_trace["trigger_derived"] = True
                    gate_trace["derive_subtype"] = subtype
                    gate_trace["typed_gate_promotion"] = True
                    gate_trace["trigger_reason"] = "selected_derived_promoted_to_typed_control"
                    if subtype == "early_bridge":
                        gate_trace["early_bridge_trigger"] = True
                    if subtype == "late_verification":
                        gate_trace["late_verification_trigger"] = True
                if is_derived(positive_unit_id) and gate_trace.get("derive_subtype") == "trigger_only_candidate":
                    subtype = allowed_subtype_for_positive_derived(step, str(positive_unit_id))
                    gate_trace["derive_subtype"] = subtype
                    gate_trace["trigger_only_reclassified"] = True
                    gate_trace["trigger_reason"] = "selected_derived_reclassified_from_trigger_only_candidate"
                    if subtype == "early_bridge":
                        gate_trace["early_bridge_trigger"] = True
                    elif subtype in {"late_verification", "answer_facing_verification"}:
                        gate_trace["late_verification_trigger"] = True
                gate_trace["legacy_need_derived_signal"] = legacy_need
                gate_trace["legacy_trigger_derived_signal"] = legacy_trigger
                step["gate_trace"] = gate_trace
                normalize_positive_derived_type_subtype(step)
                step["need_derived"] = control_need
                step["triggered_propose_derived"] = control_trigger
                candidate_debug = step.get("candidate_debug")
                if isinstance(candidate_debug, dict):
                    candidate_debug["legacy_need_derived_signal"] = legacy_need
                    candidate_debug["legacy_triggered_propose_derived_signal"] = legacy_trigger
                    candidate_debug["need_derived"] = control_need
                    candidate_debug["triggered_propose_derived"] = control_trigger
                derived_debug = step.get("derived_debug")
                if isinstance(derived_debug, dict):
                    derived_debug["legacy_need_derived_signal"] = legacy_need
                    derived_debug["legacy_triggered_propose_derived_signal"] = legacy_trigger
                    derived_debug["need_derived"] = control_need
                    derived_debug["triggered_propose_derived"] = control_trigger
                sync_candidate_fields_after_gate_repair(step)
                repair_unrelated_yes_no_selected_derived(
                    split=split,
                    row=row,
                    step=step,
                    question=question,
                    raw_texts=raw_texts,
                    states_by_key=states_by_key,
                    derived_texts=derived_texts,
                )
            if isinstance(row.get("terminal_gate_trace"), dict):
                row["terminal_gate_trace"].setdefault(
                    "legacy_need_derived_signal",
                    bool(row["terminal_gate_trace"].get("derived_need", False)),
                )
                row["terminal_gate_trace"].setdefault(
                    "legacy_trigger_derived_signal",
                    bool(row["terminal_gate_trace"].get("trigger_derived", False)),
                )
            terminal_t = row.get("terminal_t")
            terminal_state = states_by_key.get((split, str(row.get("qid")), int(terminal_t))) if isinstance(terminal_t, int) else None
            terminal_probe = row.get("terminal_probe")
            if isinstance(terminal_state, dict) and isinstance(terminal_probe, dict):
                terminal_probe["state_snapshot"] = terminal_state
        write_jsonl(rows, path)

        states_path = trajectories_dir / f"states_{split}.jsonl"
        if states_path.exists():
            state_rows = [
                states_by_key.get((split, str(row.get("qid")), int(row.get("t"))), row)
                for row in read_jsonl(states_path)
            ]
            write_jsonl(state_rows, states_path)

    manifest_path = trajectories_dir / "build_manifest_v4.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            manifest = {}
    manifest["run_id"] = run_id
    manifest["build_time"] = build_time
    manifest["source"] = "repair_hotpotqa_v4_field_compliance.py"
    manifest["upstream_full_source"] = "build_hotpotqa_full_trajectories_v4.py"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair copied v4 full files for field-level compliance.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--run-id", type=str, default="")
    args = parser.parse_args()

    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S_hotpotqa_v4")
    repair_full_records(args.base_dir, run_id)
    print(f"repair_hotpotqa_v4_field_compliance")
    print(f"  base_dir={args.base_dir}")
    print(f"  run_id={run_id}")


if __name__ == "__main__":
    main()
