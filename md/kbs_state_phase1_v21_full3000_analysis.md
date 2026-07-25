# KBS State Phase 1: v21 Full3000 Analysis

## 1. Experiment status

- Status: `OK`
- Questions: 3,000
- States: 7,296
- Initial states (`t=0`): 3,000
- Later states (`t>=1`): 4,296
- Evaluated states: 7,296
- Skipped states: 0
- Gold retained in the fixed compressed pool:
  - all states: 91.3651%
  - later states: 91.5270%

The experiment holds the candidate pool and front-end scores fixed and changes
only the policy context. It therefore measures state use more directly than the
previous end-to-end ablations.

## 2. Main finding

The v21 student does use the rendered knowledge state, especially to sharpen
the top-ranked evidence at later acquisition steps.

For `t>=1` states whose gold evidence is retained in the fixed pool:

| Condition | Step@1 | Step@5 | MRR | Gold margin |
|---|---:|---:|---:|---:|
| Correct state | 0.6691 | 0.9395 | 0.7832 | 2.0154 |
| Query only | 0.5880 | 0.9362 | 0.7330 | 0.8215 |
| Empty/frozen state | 0.5918 | 0.9390 | 0.7352 | 0.8687 |
| Other-question state | 0.6063 | 0.9479 | 0.7482 | 1.0507 |
| Previous evidence only | 0.6165 | 0.9125 | 0.7412 | 1.2410 |

The paired question-level bootstrap comparison between correct state and
query-only gives:

| Metric | Delta | 95% CI |
|---|---:|---:|
| Step@1 | +0.1052 | [0.0912, 0.1194] |
| Step@5 | +0.0075 | [0.0004, 0.0149] |
| MRR | +0.0664 | [0.0576, 0.0753] |
| Gold margin | +1.8648 | [1.7056, 2.0280] |

The state effect is therefore clear for top-rank quality, reciprocal rank, and
score separation. Its average Step@5 effect is positive but much smaller
because top-five recall is already close to saturation in the fixed pool.

## 3. State necessity

Among 3,932 eligible later states:

| Event | Count | Rate |
|---|---:|---:|
| Correct state rescues query-only at Step@1 | 553 | 14.06% |
| Correct state harms query-only at Step@1 | 234 | 5.95% |
| Correct state rescues query-only at Step@5 | 160 | 4.07% |
| Correct state harms query-only at Step@5 | 147 | 3.74% |
| Gold rank improves | 901 | 22.91% |
| Gold rank degrades | 606 | 15.41% |

The net state effect is substantially stronger at rank 1 than at rank 5. This
explains why the previous end-to-end Step@5 ablation almost hid the state
value.

## 4. Accumulated history

At `t=1`, the full state contains one previous evidence unit, so correct state
and previous-evidence-only are identical.

At `t>=2`, full accumulated history is materially better than retaining only
the latest evidence:

| Comparison: correct minus baseline | Step@1 | Step@5 | MRR |
|---|---:|---:|---:|
| Query only | +0.0921 | +0.0404 | +0.0710 |
| Previous evidence only | +0.1781 | +0.0912 | +0.1421 |

This is direct evidence that multi-step accumulated state becomes useful after
more than one evidence acquisition.

## 5. Conditional rank reversal

The diagnostic contains 4,181 consecutive-prefix pairs where the next
teacher-positive evidence was already available.

| Context mode | Current preference | Next preference | Correct reversal |
|---|---:|---:|---:|
| Correct state | 0.3664 | 0.9945 | 0.3626 |
| Query only | 0.4102 | 0.5898 | 0.0000 |
| Frozen state | 0.4064 | 0.5936 | 0.0000 |
| Previous evidence only | 0.3657 | 0.9928 | 0.3609 |

A static query cannot reverse the ordering across prefixes, whereas the
state-conditioned scorer does reverse it in 36.26% of eligible pairs. However,
the low current-prefix preference accuracy shows that dynamic ranking is not
yet fully learned.

## 6. Important limitations

1. Shuffling evidence lines changes almost nothing. The model uses state
   content but is nearly insensitive to its ordering.
2. State gains are concentrated in Step@1, MRR, and margin; Step@5 is close to
   saturation.
3. For comparison questions at `t>=1`, correct state improves Step@1 but is
   worse than query-only at Step@5. State handling is heterogeneous across
   question types.
4. Previous-evidence-only nearly matches full state in the rank-reversal
   diagnostic, although full history is clearly better at `t>=2` in the fixed
   pool evaluation.
5. The experiment establishes policy-level state use, not yet a large
   end-to-end gain under front-policy blending.

## 7. Why the previous ablation looked flat

The earlier end-to-end comparison mixed several effects:

- all `t=0` states, where state is empty;
- a Step@5 metric that is already near saturation;
- retrieval-front scores that dominate 65% of the final blended score;
- possible candidate-pool changes between runs;
- state value expressed mainly as top-rank sharpening rather than top-five
  inclusion.

The fixed-pool intervention removes these confounders and shows that state use
is real but partially hidden by the deployed aggregation and evaluation setup.

## 8. Decision and next step

Phase 1 passes the prerequisite for Phase 2: state-sensitive samples exist in
meaningful numbers.

The next model should not be produced by another loss-weight sweep. Build a
state-focused v22 training set with:

1. oversampling of `t>=1`, especially `t>=2`;
2. approximately balanced prefix-step buckets;
3. fixed candidate pools across state interventions;
4. counterfactual contexts: empty, correct partial, redundant, wrong previous
   evidence, missing bridge, and missing support;
5. pairwise examples whose ordering changes with `K_t`;
6. hard negatives drawn from the 234 Step@1 harm cases and other-question
   interventions;
7. separate bridge/comparison balancing and reporting;
8. explicit evaluation of Step@1, MRR, gold margin, State Necessity Rate, and
   conditional rank reversal in addition to Step@5.

The paper can retain a knowledge-state-guided positioning, but it should claim
that state improves adaptive evidence prioritization and trajectory
completeness, not that every state representation uniformly improves all
end-to-end metrics.
