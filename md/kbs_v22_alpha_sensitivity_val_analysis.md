# v22 Alpha-Sensitivity Validation Analysis

## Protocol

- Model: `deberta_v3_large_v22_state_focused`
- Split: question-disjoint HotpotQA blending validation set
- Questions: 1,000
- Teacher states: 2,449
- Candidate budget: 10
- Final evidence budget per state: 5
- Alpha values: `0.0, 0.2, 0.35, 0.5, 0.8, 1.0`
- Answer generation: disabled
- Selection rule: maximize Teacher Next-Unit Alignment@5; when a second
  operating point is within strictly less than 0.002, prefer the shorter
  selected context.

All six reports contain the same 1,000 qids, whose sorted-qid SHA-256 prefix is
`50e0aca27c38e848`. They use the same fixed configuration and the v22
state-focused checkpoint. Every report has `answer_judged=0` and
`answer_errors=0`, as required for the no-API selection stage.

## Results

| Alpha | Alignment@1 | Alignment@5 | State MRR | Full Doc | Full Unit | Context units | Context lexical tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.331564 | 0.748877 | 0.509075 | 0.766 | 0.546 | 6.420 | 175.978 |
| 0.20 | **0.363005** | 0.777869 | 0.538310 | 0.812 | 0.607 | 6.631 | 181.324 |
| **0.35** | 0.355655 | **0.797060** | **0.541021** | 0.864 | 0.678 | 7.115 | 194.022 |
| 0.50 | 0.324214 | 0.777460 | 0.512493 | 0.915 | 0.749 | 8.098 | 219.715 |
| 0.80 | 0.342997 | 0.649653 | 0.487861 | **0.934** | **0.772** | 9.604 | 261.356 |
| 1.00 | 0.341364 | 0.596570 | 0.478953 | **0.934** | 0.767 | 10.082 | 276.112 |

`Context lexical tokens` is an offline whitespace-token count of the selected
evidence context, not an LLM API token count.

## Frozen Decision

Freeze `alpha=0.35`.

Its Alignment@5 is the unique maximum at `0.797060`. The next-best value is
`0.777869` at `alpha=0.20`, a gap of `0.019191`, so the pre-registered
near-tie rule is not invoked. The selected point also has the highest state
MRR (`0.541021`).

The sweep exposes a genuine operating tradeoff. Larger policy weights retain
more complete final evidence sets but substantially reduce teacher next-unit
alignment and increase selected-context length. This result supports keeping
Compact and Recall as separate operating points rather than claiming that one
alpha jointly optimizes every metric.
