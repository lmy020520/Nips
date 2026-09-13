# Stage 9.5: Downstream State Rollout

## Current status

- Status: readiness tooling prepared; server audit pending.
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

Run the readiness audit. It must verify the existing correct online-state
report, all saved pre-step states, the v27 Compact protocol, and the five
runtime conditions without training, GPU inference, or API calls. Only a
clean `OK` authorizes a 20-qid selection-only smoke.
