# Stage 9.5: Downstream State Rollout

## Current status

- Status: exact-context cache audit passed; bounded answer smoke authorized.
- Dataset: fixed 3,000-qid HotpotQA evaluation subset.
- Model: frozen v27 seed-42 checkpoint.
- Operating point: Compact, with candidate budget 10, front pool 30,
  selected top-5, state-write top-1, and front--policy weight 0.5.

## Registered intervention

All conditions execute complete, independent retrieval rollouts. They differ
only in the state text visible to state-conditioned retrieval and policy
scoring:

| Condition | Visible state at step t |
|---|---|
| Correct online state | State accumulated by the method's own preceding top-1 decisions |
| Query only | Question without a notebook |
| Frozen initial state | The question's initial empty notebook at every step |
| Other-question state | Correct-rollout state from the cyclically next qid at the same step |
| Previous evidence only | Only the immediately preceding predicted evidence unit |

For an other-question trajectory shorter than the target trajectory, the most
recent available source step is used. The qid pairing is fixed before outcomes
are observed and never maps a question to itself.

## Claim boundary

This experiment measures task-level consequences of state quality. It must
report Supporting-Fact F1, Joint F1, full support coverage, ClosureSuccess,
and acquired-evidence reselection, with paired confidence intervals against
the correct online-state condition. Earlier teacher-relative fixed-pool
interventions remain mechanism diagnostics and cannot substitute for this
rollout experiment.

## Next gate

The readiness audit passed. It verified 3,000 matching query/sample qids,
7,296 evaluation states, all 7,296 saved pre-step online states, the complete
v27 Compact protocol, and all five runtime conditions. The correct report has
no duplicate or missing qids and no missing state steps. Run the 20-qid
selection-only smoke next; do not generate answers.

## Selection smoke

All five conditions completed the same 20 qids and 50 teacher states. Ordered
qid and teacher-target hashes match, all state metadata passed, and no API call
or failure occurred.

| Condition | Step@1 | Step@5 | MRR | Full unit | Top-1 reselection |
|---|---:|---:|---:|---:|---:|
| Correct online state | 0.48 | 0.90 | 0.6571 | 0.85 | 0.26 |
| Query only | 0.28 | 0.90 | 0.5386 | 0.75 | 0.36 |
| Frozen initial state | 0.32 | 0.90 | 0.5563 | 0.75 | 0.36 |
| Other-question state | 0.36 | 0.88 | 0.5769 | 0.80 | 0.32 |
| Previous evidence only | 0.52 | 0.90 | 0.6771 | 0.85 | 0.30 |

These smoke values establish execution only. In particular, the apparent
previous-only advantage cannot be interpreted scientifically at 20 qids. The
next gate is five complete 3,000-qid selection-only rollouts with 10,000-sample
paired bootstrap intervals; answer generation remains locked.

## Complete selection results

All five conditions completed the same 3,000 qids and 7,296 states with
identical qid/target hashes, zero skipped states, and no API calls.

| Condition | Step@1 | Step@5 | MRR | Full unit | Full doc | Top-1 reselection |
|---|---:|---:|---:|---:|---:|---:|
| Correct online state | 0.5515 | 0.8677 | 0.6900 | 0.7823 | 0.9393 | 0.2046 |
| Query only | 0.4666 | 0.8316 | 0.6262 | 0.6697 | 0.8610 | 0.2658 |
| Frozen initial state | 0.4633 | 0.8266 | 0.6217 | 0.6590 | 0.8567 | 0.2667 |
| Other-question state | 0.4561 | 0.7939 | 0.6053 | 0.6720 | 0.8687 | 0.2351 |
| Previous evidence only | 0.5510 | 0.8703 | 0.6908 | 0.7880 | 0.9437 | 0.2015 |

Correct online state significantly exceeds query-only, frozen-initial, and
other-question state on Step@1/5, MRR, full-unit coverage, and full-document
coverage, while significantly reducing Top-1 acquired-evidence reselection.
For example, relative to query-only it gains 0.0850 Step@1 [0.0745, 0.0958],
0.0362 Step@5 [0.0288, 0.0435], and 0.1127 full-unit coverage
[0.0993, 0.1263], while changing Top-1 reselection by -0.0611
[-0.0684, -0.0539].

Correct online state does not outperform previous-evidence-only: Step@1 and
MRR are statistically tied, while previous-only has small but significant
advantages of 0.0057 full-unit and 0.0043 full-document coverage. The
registered interpretation is therefore `STATE_RELEVANCE_SUPPORTED`, not
`FULL_HISTORY_SUPERIOR`: relevant, dynamically updated evidence state matters,
but this benchmark does not show that retaining the full accumulated history
is better than conditioning on the latest evidence.

The next gate is an offline exact-context cache audit across all five
conditions. No answer call is authorized until its reuse and deduplication
counts are reviewed.

## Exact-context cache audit

All 15 historical source reports match the frozen V4-Flash answer protocol,
providing 28,982 eligible unique contexts. Of 15,000 state-condition targets,
6,126 can be reused immediately. The remaining 8,874 target files collapse to
7,924 unique fresh contexts after removing 950 cross-condition duplicates.

| Condition | Exact reuse | Fresh before cross-condition propagation |
|---|---:|---:|
| Correct online state | 3,000 | 0 |
| Query only | 279 | 2,721 |
| Frozen initial state | 386 | 2,614 |
| Other-question state | 200 | 2,800 |
| Previous evidence only | 2,261 | 739 |

The 413 duplicate-source raw-answer disagreements are resolved by the frozen,
outcome-independent source priority; no answer is selected by correctness.
The audit made no API calls and found no failures. A 20-qid
other-question-state answer smoke is authorized because this condition uses
the newest runtime path and has the lowest exact reuse among the interventions.
