# KBS State Mechanism Phase 1 Protocol

## Objective

Determine whether the v21 student uses the current knowledge state when the
candidate pool, candidate order, and retrieval-front scores are held fixed.
This is a mechanism diagnostic, not a replacement for the official blended
end-to-end evaluation.

## Fixed-Pool Control

For every state, the hybrid retrieval front-end runs exactly once using the
correct dataset `K_t`. Its compressed top-10 candidate pool is then reused for
all policy interventions. Only the policy context changes:

1. `correct`: the matching dataset state;
2. `query_only`: question without a notebook;
3. `empty`: question plus an empty notebook;
4. `frozen`: the first available state for the same question;
5. `shuffled`: evidence lines from the correct state in shuffled order;
6. `other_question`: a state from another question at the same step;
7. `previous_evidence_only`: only the most recent evidence unit.

The policy is evaluated without front-policy score blending so that retrieval
scores cannot hide state sensitivity.

## Reported Diagnostics

- Prefix counts and state proportions for each `t`;
- Step@1, Step@3, Step@5, MRR, and gold-score margin;
- Results over all states and conditional on gold surviving compression;
- Separate `t>=1`, per-step, bridge, and comparison slices;
- State Necessity Rate based on correct-state rescues over query-only;
- Question-level paired bootstrap intervals for state interventions;
- Conditional Rank-Reversal Accuracy over eligible consecutive prefixes.

## Interpretation Gate

Proceed to state-balanced v22 training only if the data contain enough
eligible later states and at least one of the following holds:

- correct state improves `t>=1` ranking or margin over empty/shuffled states;
- correct state rescues a meaningful subset of query-only failures;
- conditional rank-reversal accuracy exceeds stateless controls.

If eligible state-sensitive cases are rare, the current question-local setting
does not strongly identify the proposed mechanism. If cases are common but the
model does not respond to interventions, the primary problem is representation
or training rather than the benchmark.

## Server Commands

Data-only smoke test:

```bash
DATA_ONLY=1 MAX_QIDS=20 \
OUTPUT_DIR=outputs/analysis/kbs_state_phase1_data_smoke20 \
bash scripts/run_kbs_state_phase1.sh
```

Model smoke test:

```bash
CUDA_DEVICE=0 MAX_QIDS=20 N_BOOTSTRAP=200 \
OUTPUT_DIR=outputs/analysis/kbs_state_phase1_v21_smoke20 \
bash scripts/run_kbs_state_phase1.sh
```

Full diagnostic:

```bash
CUDA_DEVICE=0 MAX_QIDS=0 N_BOOTSTRAP=10000 \
OUTPUT_DIR=outputs/analysis/kbs_state_phase1_v21_full3000 \
bash scripts/run_kbs_state_phase1.sh
```

No DeepSeek API key is required.
