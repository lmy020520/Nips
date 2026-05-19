# Derived Policy Analysis v2

- run_id: `20260421_120911_hotpotqa_v2`
- bucket_counts: `{"downweight_trigger_only_derived": 7, "keep_late_verification": 2, "allow_narrow_early_bridge": 1}`

## Teacher Policy

### raw_first_then_late_verification
- description: Default to raw-first supervision. Derived should usually appear after raw semantic readiness, mainly as late verification repair for answer-facing closure.

### allow_early_bridge_scaffold_only_when_strictly_needed
- description: Allow early bridge only when initialization already provides bridgeable raw, the derive goal is target-type clarification or bridge scaffolding, and the next step is expected to continue into raw.
- evidence_cases: `['5ab678f455429954757d32d3']`

### do_not_reward_triggered_but_unselected_derived
- description: If derived repeatedly triggers but never enters the winning path, keep it from dominating teacher labels. These cases are better explained by raw retrieval plus stop-side closure.
- evidence_cases: `['5a7630cb554299109176e6af', '5a82a36a55429954d2e2eb8d', '5ac2e9c45542990b17b154a0', '5adbcbca5542996e6852523b', '5ae0c9f355429906c02dab51', '5ae536065542990ba0bbb227', '5ae5fc345542993aec5ec1f8']`

## Hypothesis Summary

- summary: `{"mainstream_mode": "late_verification_repair", "minority_mode": "early_bridge_scaffold", "training_risk": "If t=0 positive derived is treated as a mainstream pattern, it may weaken the narrative value of initial deep raw retrieval."}`

## Case Recommendations

### train / 5a7630cb554299109176e6af
- question: Ulf Merbold and Lodewijk van den Berg, both flew in ?
- policy_bucket: `downweight_trigger_only_derived`
- recommendation: Do not let trigger-only derived dominate supervision; raw/stop closure was doing the real work.
- evidence: `{"mechanism": "raw_or_stop_closure", "derived_positive_count": 0, "derived_trigger_step_count": 3, "final_answer_source": "context_gold_fallback", "final_k_t_has_derived": false, "counterfactual_roles": []}`

### val / 5a7bbe76554299042af8f7d3
- question: Did Manny Coto, a Cuban American writer or Lina Wertmüller, an Italian screenwriter get nominated for an Academy Award for Directing "Seven Beauties"?
- policy_bucket: `keep_late_verification`
- recommendation: Preserve late verification notes after raw semantic readiness; these notes are answer-facing closure aids.
- evidence: `{"mechanism": "late_verification_repair", "derived_positive_count": 2, "derived_trigger_step_count": 5, "final_answer_source": "context_gold_fallback", "final_k_t_has_derived": true, "counterfactual_roles": ["abstractive_verification", "abstractive_verification"]}`

### train / 5a82a36a55429954d2e2eb8d
- question: How many hours did the storm which surpassed Typhoon Nora in 2015, grow from a tropical storm to a Category 5 hurricane?
- policy_bucket: `downweight_trigger_only_derived`
- recommendation: Do not let trigger-only derived dominate supervision; raw/stop closure was doing the real work.
- evidence: `{"mechanism": "raw_or_stop_closure", "derived_positive_count": 0, "derived_trigger_step_count": 3, "final_answer_source": "context_gold_fallback", "final_k_t_has_derived": false, "counterfactual_roles": []}`

### test / 5a8b42be55429949d91db515
- question: What profession does John Lanchester and Alan Dean Foster have in common?
- policy_bucket: `keep_late_verification`
- recommendation: Preserve late verification notes after raw semantic readiness; these notes are answer-facing closure aids.
- evidence: `{"mechanism": "late_verification_repair", "derived_positive_count": 1, "derived_trigger_step_count": 4, "final_answer_source": "context_gold_fallback", "final_k_t_has_derived": true, "counterfactual_roles": ["answer_facing_verification"]}`

### test / 5ab678f455429954757d32d3
- question: What river runs near the tidal island of Eilean Tioram?
- policy_bucket: `allow_narrow_early_bridge`
- recommendation: Allow early bridge only under narrow conditions, because it appears to scaffold later raw progress rather than replace raw retrieval.
- evidence: `{"mechanism": "early_bridge_scaffold", "derived_positive_count": 1, "derived_trigger_step_count": 4, "final_answer_source": "context_gold_fallback", "final_k_t_has_derived": true, "counterfactual_roles": ["bridge_scaffold_for_progress"]}`

### test / 5ac2e9c45542990b17b154a0
- question: When was the federal US law made that was responsible for shutting down Braniff Airways?
- policy_bucket: `downweight_trigger_only_derived`
- recommendation: Do not let trigger-only derived dominate supervision; raw/stop closure was doing the real work.
- evidence: `{"mechanism": "raw_or_stop_closure", "derived_positive_count": 0, "derived_trigger_step_count": 3, "final_answer_source": "context_gold_fallback", "final_k_t_has_derived": false, "counterfactual_roles": []}`

### val / 5adbcbca5542996e6852523b
- question: How long, originally, was Shakespeare’s play revolving around the murder of King Duncan performed?
- policy_bucket: `downweight_trigger_only_derived`
- recommendation: Do not let trigger-only derived dominate supervision; raw/stop closure was doing the real work.
- evidence: `{"mechanism": "raw_or_stop_closure", "derived_positive_count": 0, "derived_trigger_step_count": 2, "final_answer_source": "context_gold_fallback", "final_k_t_has_derived": false, "counterfactual_roles": []}`

### val / 5ae0c9f355429906c02dab51
- question: Vinci and Heroscape are both examples of what?
- policy_bucket: `downweight_trigger_only_derived`
- recommendation: Do not let trigger-only derived dominate supervision; raw/stop closure was doing the real work.
- evidence: `{"mechanism": "raw_or_stop_closure", "derived_positive_count": 0, "derived_trigger_step_count": 3, "final_answer_source": "context_gold_conflict_fallback", "final_k_t_has_derived": false, "counterfactual_roles": []}`

### val / 5ae536065542990ba0bbb227
- question: The logarithmic spiral was investigated by the mathematician who was a proponent of which branch of mathematics?
- policy_bucket: `downweight_trigger_only_derived`
- recommendation: Do not let trigger-only derived dominate supervision; raw/stop closure was doing the real work.
- evidence: `{"mechanism": "raw_or_stop_closure", "derived_positive_count": 0, "derived_trigger_step_count": 3, "final_answer_source": "llm", "final_k_t_has_derived": false, "counterfactual_roles": []}`

### test / 5ae5fc345542993aec5ec1f8
- question: Kiki Preston was the alleged mother of a child born out of wedlock with a prince born in which year ?
- policy_bucket: `downweight_trigger_only_derived`
- recommendation: Do not let trigger-only derived dominate supervision; raw/stop closure was doing the real work.
- evidence: `{"mechanism": "raw_or_stop_closure", "derived_positive_count": 0, "derived_trigger_step_count": 3, "final_answer_source": "llm", "final_k_t_has_derived": false, "counterfactual_roles": []}`
