import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_BASE = PROJECT_DIR / "data" / "hotpotqa_distractor_v4"
OUTPUT_JSON = "derived_policy_analysis_v3.json"
OUTPUT_MD = "derived_policy_analysis_v3.md"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def summarize_case(case: dict, counterfactual_rows: Dict[str, List[dict]]) -> dict:
    qid = str(case["qid"])
    rows = counterfactual_rows.get(qid, [])
    mechanism = str(case.get("case_mechanism", "unknown"))
    final_answer_source = str(case.get("final_answer_source", "unknown"))
    derived_positive_events = list(case.get("derived_positive_events", []))
    trigger_count = int(case.get("derived_trigger_step_count", 0))

    if mechanism == "late_verification_repair":
        policy_bucket = "keep_late_verification"
        recommendation = (
            "Preserve late verification notes after raw semantic readiness; these notes are answer-facing closure aids."
        )
    elif mechanism == "early_bridge_scaffold":
        policy_bucket = "allow_narrow_early_bridge"
        recommendation = (
            "Allow early bridge only under narrow conditions, because it appears to scaffold later raw progress rather than replace raw retrieval."
        )
    elif trigger_count > 0 and not derived_positive_events:
        policy_bucket = "downweight_trigger_only_derived"
        recommendation = (
            "Do not let trigger-only derived dominate supervision; raw/stop closure was doing the real work."
        )
    else:
        policy_bucket = "raw_stop_dominant"
        recommendation = (
            "Keep raw-first behavior; derived was not part of the winning path."
        )

    evidence = {
        "mechanism": mechanism,
        "derived_positive_count": len(derived_positive_events),
        "derived_trigger_step_count": trigger_count,
        "final_answer_source": final_answer_source,
        "final_k_t_has_derived": bool(case.get("final_k_t_has_derived", False)),
        "counterfactual_roles": [str(row.get("counterfactual_role", "unknown")) for row in rows],
    }
    return {
        "qid": qid,
        "split": str(case["split"]),
        "question": str(case["question"]),
        "policy_bucket": policy_bucket,
        "recommendation": recommendation,
        "evidence": evidence,
    }


def aggregate_policy_recommendations(
    *,
    strategy_obj: dict,
    counterfactual_obj: dict,
) -> dict:
    historical_cases = strategy_obj["historical_failure_to_success"]["case_records"]
    counterfactual_rows = {}
    for row in counterfactual_obj["historical_failure_to_success_records"]:
        qid = str(row["qid"])
        counterfactual_rows.setdefault(qid, []).append(row)

    case_recommendations = [
        summarize_case(case, counterfactual_rows)
        for case in historical_cases
    ]

    bucket_counts: Dict[str, int] = {}
    for row in case_recommendations:
        bucket = row["policy_bucket"]
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    teacher_policy = {
        "default_policy": {
            "name": "raw_first_then_late_verification",
            "description": (
                "Default to raw-first supervision. Derived should usually appear after raw semantic readiness, "
                "mainly as late verification repair for answer-facing closure."
            ),
        },
        "narrow_early_bridge_exception": {
            "name": "allow_early_bridge_scaffold_only_when_strictly_needed",
            "description": (
                "Allow early bridge only when initialization already provides bridgeable raw, the derive goal is "
                "target-type clarification or bridge scaffolding, and the next step is expected to continue into raw."
            ),
            "evidence_cases": [
                row["qid"] for row in case_recommendations if row["policy_bucket"] == "allow_narrow_early_bridge"
            ],
        },
        "suppress_trigger_only_derived": {
            "name": "do_not_reward_triggered_but_unselected_derived",
            "description": (
                "If derived repeatedly triggers but never enters the winning path, keep it from dominating teacher labels. "
                "These cases are better explained by raw retrieval plus stop-side closure."
            ),
            "evidence_cases": [
                row["qid"] for row in case_recommendations if row["policy_bucket"] == "downweight_trigger_only_derived"
            ],
        },
    }

    hypothesis = {
        "mainstream_mode": "late_verification_repair",
        "minority_mode": "early_bridge_scaffold",
        "training_risk": (
            "If t=0 positive derived is treated as a mainstream pattern, it may weaken the narrative value of initial deep raw retrieval."
        ),
    }

    return {
        "case_recommendations": case_recommendations,
        "bucket_counts": bucket_counts,
        "teacher_policy": teacher_policy,
        "hypothesis_summary": hypothesis,
    }


def build_markdown(*, run_id: str, payload: dict) -> str:
    lines: List[str] = []
    lines.append("# Derived Policy Analysis v2")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- bucket_counts: `{json.dumps(payload['bucket_counts'], ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Teacher Policy")
    lines.append("")
    teacher_policy = payload["teacher_policy"]
    for key in ["default_policy", "narrow_early_bridge_exception", "suppress_trigger_only_derived"]:
        block = teacher_policy[key]
        lines.append(f"### {block['name']}")
        lines.append(f"- description: {block['description']}")
        if "evidence_cases" in block:
            lines.append(f"- evidence_cases: `{block['evidence_cases']}`")
        lines.append("")

    lines.append("## Hypothesis Summary")
    lines.append("")
    lines.append(f"- summary: `{json.dumps(payload['hypothesis_summary'], ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Case Recommendations")
    lines.append("")
    for row in payload["case_recommendations"]:
        lines.append(f"### {row['split']} / {row['qid']}")
        lines.append(f"- question: {row['question']}")
        lines.append(f"- policy_bucket: `{row['policy_bucket']}`")
        lines.append(f"- recommendation: {row['recommendation']}")
        lines.append(f"- evidence: `{json.dumps(row['evidence'], ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    strategy_obj = read_json(base_dir / "debug" / "derived_strategy_analysis_v3.json")
    counterfactual_obj = read_json(base_dir / "debug" / "derived_counterfactual_analysis_v3.json")
    run_id = str(strategy_obj.get("build_meta", {}).get("run_id", "")).strip()

    analysis = aggregate_policy_recommendations(
        strategy_obj=strategy_obj,
        counterfactual_obj=counterfactual_obj,
    )
    output = {
        "build_meta": {
            "run_id": run_id,
            "source": "analyze_derived_policy_v4.py",
        },
        **analysis,
    }

    debug_dir = base_dir / "debug"
    json_path = debug_dir / OUTPUT_JSON
    md_path = debug_dir / OUTPUT_MD
    write_json(json_path, output)
    write_text(md_path, build_markdown(run_id=run_id, payload=analysis))
    print(f"derived policy analysis written: {json_path}")
    print(f"derived policy analysis written: {md_path}")


if __name__ == "__main__":
    main()
