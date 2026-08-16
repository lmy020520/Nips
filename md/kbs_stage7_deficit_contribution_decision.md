# Stage 7 Deficit and Contribution Decision

## Final decision

`DIAGNOSTIC_DOWNGRADE`

Typed deficit and candidate contribution are retained as exploratory
trajectory diagnostics. They are not headline contributions, verified
performance mechanisms, or online control signals in the final v27 system.
No Deficit/Contribution 2.0 training run is authorized.

## Evidence

The matched v21 auxiliary-objective study uses the same initialization, data,
qid set, candidate pools, optimizer, and answer protocol.

| Variant | Answer F1 | Alignment@5 | Full-unit coverage |
|---|---:|---:|---:|
| Full v21 | 0.740117 | 0.827714 | 0.717000 |
| Ranking only | 0.741568 | 0.829221 | 0.717333 |
| Without deficit | 0.743097 | 0.830044 | 0.717333 |
| Without contribution | 0.740701 | 0.827577 | 0.716667 |

Ranking-only and without-contribution differences are negligible. Removing
deficit increases F1 by 0.002980 and Alignment@5 by 0.002330; the paired 95%
intervals are `[0.000087, 0.006026]` and `[0.000136, 0.004651]`. These results
do not support a downstream benefit from either auxiliary objective.

The v21 deficit head obtains MAE 0.261983, Pearson correlation 0.165130, and
monotonic non-increase rate 0.697368 on 459 internal-test states. The
contribution head obtains MAE 0.215344 over 1,836 state-dimension labels.
These values are useful diagnostics but do not establish reliable calibration
or a deployable controller.

The accepted v27 system disables the auxiliary heads. Its state-sensitive
gains therefore come from the dual state/candidate ranking architecture,
counterfactual acquired-evidence supervision, and online state-conditioned
selection, not from deficit or contribution predictions.

## Required paper changes

1. Remove deficit and contribution from the title, abstract headline, and
   numbered core contributions.
2. Describe them as exploratory structured auxiliary supervision evaluated in
   the v21 diagnostic model, not as part of the final v27 controller.
3. Move detailed label/loss formulas, MAE, monotonicity, and matched ablations
   to the appendix or a compact diagnostic subsection.
4. State explicitly that controlled ablations do not show a consistent
   downstream gain and that the final deployed policy does not use these
   outputs.
5. Keep state-only deficit calibration and hard-negative contribution learning
   as future work rather than reporting an unvalidated 2.0 mechanism.

## Allowed claim

> We investigate typed deficit and contribution as exploratory trajectory
> diagnostics. Their current calibration is limited, controlled ablations do
> not establish a downstream gain, and the final v27 policy does not depend on
> either auxiliary output.

## Forbidden claims

- Deficit supervision improves final answer quality.
- Contribution supervision is responsible for the v27 gain.
- The deficit head accurately estimates knowledge completeness.
- Deficit or contribution provides a reliable online stopping or selection
  controller.
