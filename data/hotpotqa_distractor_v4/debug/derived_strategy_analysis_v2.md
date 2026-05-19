# Derived Strategy Analysis v2

- run_id: `20260421_120911_hotpotqa_v2`
- trajectory_count: `30`
- trajectories_with_derived_positive: `7`
- trajectories_triggered_but_no_positive_derived: `23`

## Overall Stats

- derived_positive_position_counts: `{"middle_late": 9, "early": 1}`
- derived_positive_type_counts: `{"bridge_note": 4, "verification_note": 6}`
- derived_followup_counts: `{"derived_then_immediate_raw": 5, "derived_then_future_raw": 5, "next_raw_already_visible_in_c_t": 3, "next_raw_not_yet_visible_in_c_t": 7}`

## Hypothesis Checks

### overall
- derived_is_usually_middle_late: supported=`True`, ratio=`0.9`, evidence=`{"middle_late": 9, "total": 10}`
- derived_often_followed_by_raw: supported=`True`, ratio=`0.5`, evidence=`{"derived_then_immediate_raw": 5, "total": 10}`
- next_raw_not_always_visible_before_derived: supported=`True`, ratio=`0.7`, evidence=`{"next_raw_not_yet_visible_in_c_t": 7, "total": 10}`

### historical_failure_to_success
- derived_is_usually_middle_late: supported=`True`, ratio=`0.75`, evidence=`{"middle_late": 3, "total": 4}`
- verification_dominates_when_derived_enters_winning_path: supported=`True`, ratio=`0.75`, evidence=`{"verification_note": 3, "bridge_note": 1, "total": 4}`
- early_bridge_exists_but_is_not_mainstream: supported=`True`, ratio=`0.25`, evidence=`{"early": 1, "middle_late": 3, "total": 4}`
- some_derived_steps_are_followed_by_new_raw_discovery: supported=`True`, ratio=`0.75`, evidence=`{"next_raw_not_yet_visible_in_c_t": 3, "total": 4}`

## Historical Failure -> Success Focus Set

- qid_count: `10`
- derived_positive_position_counts: `{"middle_late": 3, "early": 1}`
- derived_positive_type_counts: `{"verification_note": 3, "bridge_note": 1}`
- derived_followup_counts: `{"next_raw_not_yet_visible_in_c_t": 3, "derived_then_immediate_raw": 2, "derived_then_future_raw": 2, "next_raw_already_visible_in_c_t": 1}`

## Case Notes

### train / 5a7630cb554299109176e6af
- question: Ulf Merbold and Lodewijk van den Berg, both flew in ?
- case_mechanism: `raw_or_stop_closure`
- terminal_t: `3`
- final_answer_source: `context_gold_fallback`
- final_support_rule: `multi_hop_answer_in_evidence`
- final_k_t_has_derived: `False`
- derived_positive_ts: `[]`
- derived_positive: none
- derived_trigger_step_count: `3`

### val / 5a7bbe76554299042af8f7d3
- question: Did Manny Coto, a Cuban American writer or Lina Wertmüller, an Italian screenwriter get nominated for an Academy Award for Directing "Seven Beauties"?
- case_mechanism: `late_verification_repair`
- terminal_t: `5`
- final_answer_source: `context_gold_fallback`
- final_support_rule: `explicit_answer_in_evidence`
- final_k_t_has_derived: `True`
- derived_positive_ts: `[3, 4]`
- derived@t=3: type=`verification_note`, position=`middle_late`, goal=`generic_bridge_or_verification`, next_is_raw=`False`, next_raw_in_current_c_t=`False`
- note_text: The film 'Seven Beauties' is an Italian language film from 1975, but no direct nomination details are provided in the current raw evidence.
- derived@t=4: type=`verification_note`, position=`middle_late`, goal=`generic_bridge_or_verification`, next_is_raw=`False`, next_raw_in_current_c_t=`False`
- note_text: The film 'Seven Beauties' is an Italian language film from 1975, but no direct nomination details are provided in the current raw evidence.
- derived_trigger_step_count: `5`

### train / 5a82a36a55429954d2e2eb8d
- question: How many hours did the storm which surpassed Typhoon Nora in 2015, grow from a tropical storm to a Category 5 hurricane?
- case_mechanism: `raw_or_stop_closure`
- terminal_t: `3`
- final_answer_source: `context_gold_fallback`
- final_support_rule: `multi_hop_answer_in_evidence`
- final_k_t_has_derived: `False`
- derived_positive_ts: `[]`
- derived_positive: none
- derived_trigger_step_count: `3`

### test / 5a8b42be55429949d91db515
- question: What profession does John Lanchester and Alan Dean Foster have in common?
- case_mechanism: `late_verification_repair`
- terminal_t: `4`
- final_answer_source: `context_gold_fallback`
- final_support_rule: `multi_hop_answer_in_evidence`
- final_k_t_has_derived: `True`
- derived_positive_ts: `[2]`
- derived@t=2: type=`verification_note`, position=`middle_late`, goal=`bridge_query_entity_to_answer_candidate`, next_is_raw=`True`, next_raw_in_current_c_t=`False`
- note_text: Alan Dean Foster is an American writer of fantasy and science fiction, indicating he is a novelist.
- derived_trigger_step_count: `4`

### test / 5ab678f455429954757d32d3
- question: What river runs near the tidal island of Eilean Tioram?
- case_mechanism: `early_bridge_scaffold`
- terminal_t: `4`
- final_answer_source: `context_gold_fallback`
- final_support_rule: `multi_hop_answer_in_evidence`
- final_k_t_has_derived: `True`
- derived_positive_ts: `[0]`
- derived@t=0: type=`bridge_note`, position=`early`, goal=`target_type_disambiguation`, next_is_raw=`True`, next_raw_in_current_c_t=`True`
- note_text: Castle Island is described as a small tidal island in the Firth of Clyde, which may help contextualize tidal island geography in Scotland.
- derived_trigger_step_count: `4`

### test / 5ac2e9c45542990b17b154a0
- question: When was the federal US law made that was responsible for shutting down Braniff Airways?
- case_mechanism: `raw_or_stop_closure`
- terminal_t: `3`
- final_answer_source: `context_gold_fallback`
- final_support_rule: `multi_hop_answer_in_evidence`
- final_k_t_has_derived: `False`
- derived_positive_ts: `[]`
- derived_positive: none
- derived_trigger_step_count: `3`

### val / 5adbcbca5542996e6852523b
- question: How long, originally, was Shakespeare’s play revolving around the murder of King Duncan performed?
- case_mechanism: `raw_or_stop_closure`
- terminal_t: `3`
- final_answer_source: `context_gold_fallback`
- final_support_rule: `multi_hop_answer_in_evidence`
- final_k_t_has_derived: `False`
- derived_positive_ts: `[]`
- derived_positive: none
- derived_trigger_step_count: `2`

### val / 5ae0c9f355429906c02dab51
- question: Vinci and Heroscape are both examples of what?
- case_mechanism: `raw_or_stop_closure`
- terminal_t: `3`
- final_answer_source: `context_gold_conflict_fallback`
- final_support_rule: `multi_hop_answer_in_evidence`
- final_k_t_has_derived: `False`
- derived_positive_ts: `[]`
- derived_positive: none
- derived_trigger_step_count: `3`

### val / 5ae536065542990ba0bbb227
- question: The logarithmic spiral was investigated by the mathematician who was a proponent of which branch of mathematics?
- case_mechanism: `raw_or_stop_closure`
- terminal_t: `3`
- final_answer_source: `llm`
- final_support_rule: `multi_hop_answer_in_evidence`
- final_k_t_has_derived: `False`
- derived_positive_ts: `[]`
- derived_positive: none
- derived_trigger_step_count: `3`

### test / 5ae5fc345542993aec5ec1f8
- question: Kiki Preston was the alleged mother of a child born out of wedlock with a prince born in which year ?
- case_mechanism: `raw_or_stop_closure`
- terminal_t: `3`
- final_answer_source: `llm`
- final_support_rule: `multi_hop_answer_in_evidence`
- final_k_t_has_derived: `False`
- derived_positive_ts: `[]`
- derived_positive: none
- derived_trigger_step_count: `3`
