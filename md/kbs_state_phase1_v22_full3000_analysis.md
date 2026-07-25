# KBS State Phase 1: v22 Full3000 Analysis

## 1. Experiment status

- Status: `OK`
- Checkpoint: `deberta_v3_large_v22_state_focused`
- Questions: 3,000
- States: 7,296
- Later states (`t>=1`): 4,296
- Fixed-pool eligible later states: 3,932
- Fixed-pool gold retention: 91.37% overall and 91.53% at `t>=1`
- Bootstrap resamples: 10,000

The candidate pool and retrieval-front scores are fixed across interventions.
Only the policy context changes. Gold retention is identical to the v21
diagnostic, so the comparison is not affected by a different front-end pool.

## 2. Main v22 state effect

For `t>=1` states whose teacher-positive evidence remains in the fixed pool:

| Condition | Step@1 | Step@5 | MRR | Gold margin |
|---|---:|---:|---:|---:|
| Correct state | 0.7881 | 0.9921 | 0.8783 | 2.1524 |
| Query only | 0.4415 | 0.9242 | 0.6388 | -0.2871 |
| Empty/frozen state | 0.4382 | 0.9255 | 0.6367 | -0.2791 |
| Other-question state | 0.4761 | 0.9415 | 0.6688 | -0.1049 |
| Previous evidence only | 0.7510 | 0.9908 | 0.8562 | 1.8254 |

Question-level paired bootstrap results for correct state minus query only:

| Metric | Delta | 95% CI |
|---|---:|---:|
| Step@1 | +0.3639 | [0.3460, 0.3814] |
| Step@5 | +0.0620 | [0.0538, 0.0704] |
| MRR | +0.2465 | [0.2355, 0.2574] |
| Gold margin | +2.6237 | [2.5402, 2.7087] |

All four intervals exclude zero by a wide margin. The v22 Student therefore
has a strong and directly measurable dependence on the correct knowledge
state under controlled candidate conditions.

## 3. Improvement over v21

The same fixed-pool protocol was used for v21 and v22.

| Measurement at `t>=1` | v21 | v22 | Change |
|---|---:|---:|---:|
| Correct-state Step@1 | 0.6691 | 0.7881 | +0.1190 |
| Correct-state Step@5 | 0.9395 | 0.9921 | +0.0526 |
| Correct-state MRR | 0.7832 | 0.8783 | +0.0951 |
| Correct-state margin | 2.0154 | 2.1524 | +0.1370 |
| Paired Step@1 state effect | +0.1052 | +0.3639 | +0.2587 |
| Paired Step@5 state effect | +0.0075 | +0.0620 | +0.0545 |
| Paired MRR state effect | +0.0664 | +0.2465 | +0.1801 |
| Conditional rank reversal | 0.3626 | 0.7474 | +0.3848 |

The increase is not caused only by query-only degradation: the absolute
correct-state metrics also improve substantially. Query-only performance does
decrease because the v22 fixed-pool training data deliberately presents the
same question and candidates with different positives at different prefixes.
This is the intended counterfactual property: a static query cannot solve the
trajectory consistently.

## 4. State necessity

Among the 3,932 eligible later states:

| Event | v21 | v22 |
|---|---:|---:|
| Correct state rescues query-only at Step@1 | 14.06% | 37.56% |
| Correct state harms query-only at Step@1 | 5.95% | 2.90% |
| Correct state rescues query-only at Step@5 | 4.07% | 7.15% |
| Correct state harms query-only at Step@5 | 3.74% | 0.36% |
| Gold rank improves | 22.91% | 48.55% |
| Gold rank degrades | 15.41% | 4.81% |

The net Step@1 rescue rate changes from approximately 8.1 percentage points
for v21 to 34.7 points for v22. This directly addresses the earlier concern
that the state mechanism had little discriminative value.

## 5. Accumulated history at `t>=2`

There are 1,162 fixed-pool eligible states at `t>=2`. Their aggregate results
are:

| Condition | Step@1 | MRR |
|---|---:|---:|
| Correct full history | 0.8021 | 0.8848 |
| Query only | 0.4028 | 0.6027 |
| Previous evidence only | 0.6764 | 0.8101 |

Question-level paired bootstrap results:

| Comparison | Metric | Delta | 95% CI |
|---|---|---:|---:|
| Correct minus query only | Step@1 | +0.4134 | [0.3801, 0.4456] |
| Correct minus query only | Step@5 | +0.0938 | [0.0762, 0.1120] |
| Correct minus query only | MRR | +0.2886 | [0.2673, 0.3098] |
| Correct minus previous-only | Step@1 | +0.1239 | [0.1002, 0.1478] |
| Correct minus previous-only | Step@5 | +0.0021 | [-0.0041, 0.0085] |
| Correct minus previous-only | MRR | +0.0730 | [0.0599, 0.0864] |

The significant Step@1 and MRR gains over previous-only show that v22 uses
accumulated multi-step history rather than merely detecting the most recently
selected unit. Step@5 is saturated and does not distinguish full history from
the latest evidence alone.

## 6. Conditional rank reversal

Across 4,181 consecutive-prefix pairs:

| Context mode | Current preference | Next preference | Correct reversal |
|---|---:|---:|---:|
| Correct state | 0.7477 | 0.9998 | 0.7474 |
| Query only | 0.7369 | 0.2631 | 0.0000 |
| Frozen state | 0.7352 | 0.2648 | 0.0000 |
| Previous evidence only | 0.7460 | 1.0000 | 0.7460 |

The correct-state reversal rate more than doubles from v21's 0.3626 to
v22's 0.7474. Query-only remains unable to reverse a static ranking.

Previous-only is sufficient for most adjacent reversals because the immediate
transition mainly requires avoiding the latest selected unit. The separate
`t>=2` analysis is therefore necessary to establish the added value of full
history.

## 7. Question type and representation limitations

The state effect is heterogeneous:

- For bridge states at `t>=1`, correct-state Step@1 is 0.7841 versus 0.3882
  for query-only.
- For comparison states at `t>=1`, correct-state Step@1 is 0.8069 versus
  0.6881 for query-only.

The effect is positive for both types but much larger for bridge questions.

Shuffling evidence lines has almost no effect: correct-state Step@1 is 0.7881
and shuffled-state Step@1 is 0.7820. The model uses state content strongly but
still has little sensitivity to evidence ordering. The paper should claim
content-conditioned knowledge-state guidance, not order-aware reasoning.

## 8. Decision

The policy-level state mechanism now strongly supports the
knowledge-state-guided positioning. The earlier nearly flat state ablation is
no longer an accurate characterization of the final state-focused Student.

However, v22 should not replace v21 in the paper until it passes end-to-end
checks:

1. Rerun HotpotQA KSG-EA-Compact and KSG-EA-Recall with the v22 checkpoint.
2. Rerun the query-only policy ablation with the same v22 checkpoint.
3. Verify that answer EM/F1, Step@5, and full unit coverage do not materially
   regress.
4. If HotpotQA passes, rerun 2Wiki Compact/Recall to test zero-shot transfer.

The current result closes the mechanism-level title risk. The remaining risk
is whether stronger state specialization survives front-policy blending and
improves final RAG behavior.
