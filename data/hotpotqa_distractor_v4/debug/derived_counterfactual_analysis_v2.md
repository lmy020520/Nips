# Derived Counterfactual Analysis v2

- run_id: `20260421_120911_hotpotqa_v2`

## Overall

- summary: `{"event_count": 10, "role_counts": {"bridge_contextualization": 2, "abstractive_verification": 5, "answer_facing_verification": 1, "bridge_scaffold_for_progress": 1, "bridge_scaffold_for_new_raw": 1}, "events_where_note_mentions_gold_but_sources_do_not": 2, "events_where_final_k_t_still_has_gold_without_derived": 4, "events_where_next_raw_not_yet_visible": 2}`

## Historical Failure -> Success Focus Set

- summary: `{"event_count": 4, "role_counts": {"abstractive_verification": 2, "answer_facing_verification": 1, "bridge_scaffold_for_progress": 1}, "events_where_note_mentions_gold_but_sources_do_not": 1, "events_where_final_k_t_still_has_gold_without_derived": 3, "events_where_next_raw_not_yet_visible": 1}`

### val / 5a7bbe76554299042af8f7d3 / t=3
- question: Did Manny Coto, a Cuban American writer or Lina Wertmüller, an Italian screenwriter get nominated for an Academy Award for Directing "Seven Beauties"?
- note_type: `verification_note`
- counterfactual_role: `abstractive_verification`
- next_is_raw: `False`
- next_raw_positive_in_current_c_t: `False`
- note_mentions_gold: `False`
- source_mentions_gold: `True`
- final_k_t_has_gold: `True`
- final_k_t_without_derived_has_gold: `True`
- novel_tokens_vs_sources: `['1975', 'beauties', 'language', 'nomination', 'seven']`
- note_text: The film 'Seven Beauties' is an Italian language film from 1975, but no direct nomination details are provided in the current raw evidence.

### val / 5a7bbe76554299042af8f7d3 / t=4
- question: Did Manny Coto, a Cuban American writer or Lina Wertmüller, an Italian screenwriter get nominated for an Academy Award for Directing "Seven Beauties"?
- note_type: `verification_note`
- counterfactual_role: `abstractive_verification`
- next_is_raw: `False`
- next_raw_positive_in_current_c_t: `False`
- note_mentions_gold: `False`
- source_mentions_gold: `True`
- final_k_t_has_gold: `True`
- final_k_t_without_derived_has_gold: `True`
- novel_tokens_vs_sources: `['1975', 'beauties', 'language', 'nomination', 'seven']`
- note_text: The film 'Seven Beauties' is an Italian language film from 1975, but no direct nomination details are provided in the current raw evidence.

### test / 5a8b42be55429949d91db515 / t=2
- question: What profession does John Lanchester and Alan Dean Foster have in common?
- note_type: `verification_note`
- counterfactual_role: `answer_facing_verification`
- next_is_raw: `True`
- next_raw_positive_in_current_c_t: `False`
- note_mentions_gold: `True`
- source_mentions_gold: `False`
- final_k_t_has_gold: `True`
- final_k_t_without_derived_has_gold: `False`
- novel_tokens_vs_sources: `['indicating', 'novelist']`
- note_text: Alan Dean Foster is an American writer of fantasy and science fiction, indicating he is a novelist.

### test / 5ab678f455429954757d32d3 / t=0
- question: What river runs near the tidal island of Eilean Tioram?
- note_type: `bridge_note`
- counterfactual_role: `bridge_scaffold_for_progress`
- next_is_raw: `True`
- next_raw_positive_in_current_c_t: `True`
- note_mentions_gold: `False`
- source_mentions_gold: `False`
- final_k_t_has_gold: `True`
- final_k_t_without_derived_has_gold: `True`
- novel_tokens_vs_sources: `['contextualize', 'described', 'geography', 'scotland']`
- note_text: Castle Island is described as a small tidal island in the Firth of Clyde, which may help contextualize tidal island geography in Scotland.
