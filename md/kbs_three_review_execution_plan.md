# KBS Three-Document Unified Execution Plan

## 0. Plan control

```yaml
plan_id: kbs-three-review-plan-v1
created_at: 2026-07-25
plan_read_required_before_every_run: true
current_stage: 3X
current_status: v27 multiseed gate and seed-42 20-qid fresh-answer protocol smoke passed; authorize exactly one final 3,000-qid/API evaluation with the frozen Compact protocol
```

This file is the single execution plan synthesized from:

1. `KBS-State+Defcit+Contribution效果不佳的原因分析与实验方案-2026-7-22.md`
2. `KBS-论文优化建议-2026-7-21.md`
3. `KBS-论文原创性分析与优化意见-2026-7-21.md`

### Mandatory operating rule

Before proposing, launching, or modifying any experiment, the assistant must:

1. Read this plan.
2. Identify the current stage and its unmet acceptance criteria.
3. Check required files, checkpoints, data, GPU, API, and expected outputs.
4. Run only work belonging to the current stage unless a dependency must be
   repaired.
5. Record the result and update `current_stage`, `current_status`, and the run
   ledger in this file.

If the user explicitly says that reading the plan is no longer required, the
assistant must change:

```yaml
plan_read_required_before_every_run: false
```

After that instruction, the assistant must not read this plan before later
runs unless the user explicitly re-enables the rule.

### Evidence discipline

- Never replace an old result silently. Record model version, checkpoint,
  split, qids, seed, selector, candidate budget, answer generator, and cache
  policy.
- Never compare methods that use different qid sets, evidence budgets, or
  answer-generation settings without marking the comparison as non-paired.
- Do not use answer caches as if they were fresh API calls when the selected
  context changed.
- Do not call an inspired baseline a faithful reproduction of the original
  method.
- Negative results must be retained and reflected in the paper.
- A stage passes only when its acceptance criteria are met, not merely when a
  script finishes.

## 1. Unified research position

The paper must not claim novelty from broad ideas such as sequential
retrieval, evidence chains, stateful retrieval, teacher trajectories, or a
lightweight alternative to online LLM selection. These ideas overlap with
ACRA, ECDR, FiSKE, TreeQA, PRIME, and Flow-RAG.

The final position is:

> KSG-EA learns how to close an answer-facing compiled context under a
> candidate and context budget, rather than merely learning how to append the
> next node to an evidence chain.

The four intended contributions are:

1. **Objective:** answer-identifiable compiled-context closure rather than
   path continuation alone.
2. **State:** an answer-facing acquisition state represented by accumulated
   evidence and compiled context.
3. **Supervision:** offline distillation of a cost-aware closure teacher into
   a local selector.
4. **Deployment:** inexpensive online evidence control with Compact and Recall
   operating points, without online LLM evidence planning.

Typed deficit and contribution are diagnostic mechanisms by default. They
must not be restored as headline performance contributions unless Stage 7
passes its explicit gate.

## 2. Execution overview

| Stage | Goal | Current status | API | Training |
|---|---|---|---|---|
| 1 | Establish causal state use | Passed with v22 | No | Completed |
| 2 | Validate v22 end to end | Passed on HotpotQA and zero-shot 2Wiki | Completed | Completed |
| 3 | Controlled mechanism baselines | Closed: v23/v24 outperform Full; v25 and v26 gates failed | No further API | Completed for v23/v24/v25/v26 seed 42 |
| 3X | High-cost state architecture repair | Multiseed gate and API smoke passed; final 3,000-qid run authorized | One final run | Completed for seeds 42/43/44 |
| 4 | Standard and closure-aware metrics | Tooling ready; paused until Stage 3X gate closes | Mostly offline | No |
| 5 | Multi-seed robustness | Pending | Yes for end-to-end tables | Yes |
| 6 | Teacher, compression, and cost evidence | Pending | Mixed | Possibly |
| 7 | Decide deficit/contribution status | Pending | No initially | Conditional |
| 8 | Rewrite and submission audit | Pending | No | No |

Stages must be executed in order. Work within the same stage may run in
parallel only when it uses independent output directories and GPUs.

---

# Stage 1: Causal State Mechanism

## Purpose

Determine whether the Student actually reads `K_t`, whether state-sensitive
samples exist, and whether full accumulated history has value beyond a static
query or previous-evidence anchor.

## Required experiments

- Prefix distribution by step and question type.
- Fixed-pool interventions:
  - correct state;
  - query only;
  - empty state;
  - frozen initial state;
  - shuffled within-question state;
  - another-question state;
  - previous-evidence-only state.
- State Necessity Rate.
- Conditional Rank-Reversal Accuracy.
- Separate `t=0`, `t>=1`, `t>=2`, bridge, comparison, and
  gold-retained-in-pool slices.

## Completed evidence

v21 diagnostics established a real but partly hidden state effect. The v22
state-focused Student then strengthened that effect.

For v22 at `t>=1`, conditioned on gold retention:

| Measurement | Result |
|---|---:|
| Correct-state Step@1 | 0.7881 |
| Query-only Step@1 | 0.4415 |
| Paired Step@1 delta | +0.3639 `[0.3460, 0.3814]` |
| Paired Step@5 delta | +0.0620 `[0.0538, 0.0704]` |
| Paired MRR delta | +0.2465 `[0.2355, 0.2574]` |
| State rescue at Step@1 | 37.56% |
| Conditional rank reversal | 0.7474 |

At `t>=2`, correct full history minus previous-evidence-only gives:

| Metric | Delta | 95% CI |
|---|---:|---:|
| Step@1 | +0.1239 | `[0.1002, 0.1478]` |
| MRR | +0.0730 | `[0.0599, 0.0864]` |

## Stage decision

**Passed.** State-sensitive samples exist, v22 uses state content, and full
history adds value after the first hop.

## Remaining limitation carried forward

Evidence-line shuffling has almost no effect. The supported claim is
content-conditioned accumulated-state guidance, not order-aware reasoning.

---

# Stage 2: v22 End-to-End Validation

## Purpose

Verify that stronger policy-level state specialization survives
front-policy blending and improves or preserves final RAG behavior.

## Step 2.1: Re-select blending weight on validation

Run the v22 checkpoint on the existing question-disjoint blending validation
set for:

```text
alpha = 0.0, 0.2, 0.35, 0.5, 0.8, 1.0
```

Use no answer API during the initial selection. Report:

- Step@1;
- teacher next-unit alignment at 5;
- MRR;
- full document coverage;
- full unit coverage;
- selected context length.

Choose the alpha that maximizes validation teacher next-unit alignment at 5.
If two values differ by less than 0.002, choose the lower-token operating
point. Freeze the selected alpha before test evaluation.

Implementation entry point:

```bash
bash scripts/run_kbs_v22_alpha_sensitivity_val.sh
```

This entry point forces `GENERATE_ANSWERS=0`, uses the v22 state-focused
checkpoint by default, and derives state-level MRR plus selected-context
length offline from each report.

### Step 2.1 completed evidence

The six reports use the same 1,000 qids and 2,449 teacher states, the same
fixed configuration, and the v22 state-focused checkpoint. Answer generation
was disabled.

| Alpha | Alignment@1 | Alignment@5 | State MRR | Full unit coverage | Context lexical tokens |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.331564 | 0.748877 | 0.509075 | 0.546 | 175.978 |
| 0.20 | 0.363005 | 0.777869 | 0.538310 | 0.607 | 181.324 |
| **0.35** | 0.355655 | **0.797060** | **0.541021** | 0.678 | 194.022 |
| 0.50 | 0.324214 | 0.777460 | 0.512493 | 0.749 | 219.715 |
| 0.80 | 0.342997 | 0.649653 | 0.487861 | **0.772** | 261.356 |
| 1.00 | 0.341364 | 0.596570 | 0.478953 | 0.767 | 276.112 |

**Frozen decision:** `alpha=0.35`. Its Alignment@5 is the unique maximum.
The next-best value differs by `0.019191`, so the `<0.002` near-tie rule does
not apply.

## Step 2.2: HotpotQA end-to-end runs

Using the frozen alpha, same 3,000 qids, same DeepSeek prompt, fresh API calls,
and separate output/cache directories, run:

1. v22 KSG-EA-Compact.
2. v22 KSG-EA-Recall.
3. v22 query-only policy with otherwise identical settings.
4. v22 previous-evidence-only/anchor inference if supported by the runtime.

The budget-matched executable run IDs are:

```text
full_compact
full_recall
query_only_compact
query_only_recall
```

The optional inference diagnostics are `previous_only_compact` and
`previous_only_recall`. They use only the previous online top-1 prediction as
the policy anchor and must not be described as separately trained ACRA-style
baselines.

Implementation entry point:

```bash
RUN=<run-id> CUDA_DEVICE=<gpu> bash scripts/run_kbs_v22_stage2_hotpot.sh
```

The runner fixes `alpha=0.35`, requires fresh DeepSeek calls, uses independent
report/cache paths, refuses silent output reuse, and records selection runtime
and peak GPU memory. `SMOKE=1` creates an isolated 20-qid preflight report.
Each completed report is automatically checked by
`scripts/check_kbs_v22_stage2_report.py`; a run is not accepted merely because
the RAG process exits.

Report:

- Answer EM, Contains, and F1.
- Teacher next-unit alignment at 1/3/5.
- Full document and full unit coverage.
- Supporting-fact and joint metrics after Stage 4 tooling is available.
- Answer API tokens and latency.
- Selection latency and peak GPU memory.

## Step 2.3: Paired statistics

Run 10,000-sample qid-level paired bootstrap intervals for:

- Full-state Compact minus query-only.
- Full-state Recall minus query-only.
- v22 Compact/Recall minus Hybrid.
- v22 Compact/Recall minus BGE.
- v22 Compact/Recall minus the corresponding v21 result.

Metrics:

```text
answer_em, answer_f1, step_at_5, full_unit_coverage
```

## Step 2.4: 2Wiki transfer

Only after HotpotQA passes, evaluate the unchanged v22 checkpoint without
2Wiki fine-tuning:

1. KSG-EA-Compact.
2. KSG-EA-Recall.
3. Query-only policy.

Use the same 1,000 qids, fresh DeepSeek calls, and paired bootstrap against
Hybrid and BGE.

## Acceptance criteria

Stage 2 passes when all are true:

1. The frozen-alpha full-state policy has higher full unit coverage than
   query-only, with a paired 95% interval excluding zero.
2. At least one of Compact or Recall improves answer F1 over query-only with a
   paired interval excluding zero.
3. v22 answer F1 and full unit coverage do not regress by more than 0.01
   absolute from the corresponding v21 operating point.
4. HotpotQA has 3,000 judged answers and zero answer errors.
5. If 2Wiki is run, it has 1,000 judged answers and zero answer errors.

## Failure rule

If mechanism metrics pass but end-to-end metrics fail:

- do not discard the Stage 1 result;
- inspect blending and rollout-state mismatch;
- do not tune on the 3,000-question test set;
- revise only on validation, then rerun one frozen test configuration.

---

# Stage 3: Controlled Mechanism Baselines

## Purpose

Establish originality against recent chain- and state-oriented retrieval
mechanisms under the same textual setting.

## Common protocol

All methods must use:

- the same train/validation/test qids;
- DeBERTa-v3-large or an explicitly matched-capacity encoder;
- the same fixed candidate pool;
- the same candidate and final evidence budgets;
- the same answer generator and prompt;
- the same seed set used in Stage 5;
- independent output directories.

## Baseline 3.1: ACRA-style anchor policy

Input:

```text
(question, previous evidence, candidate)
```

It must be trained as an anchor policy. The v22 intervention that replaces
full state with previous evidence is a useful diagnostic but is not a
substitute for a separately trained baseline.

### Stage 3.1 decision

The separately trained anchor significantly outperformed the original v22
Full rollout on the 3,000-question evaluation set. Diagnosis found a
train--inference mismatch: v22 training adds one teacher-positive unit per
state transition, whereas the original online rollout wrote all five selected
units into the next state.

A single pre-registered validation-only repair changed only the number of
selected units written into the next state from five to one. On the
question-disjoint 1,000-qid validation set, this increased Full hop-2+
Step@1 from `0.171843` to `0.403037` and MRR from `0.385377` to `0.565392`.
However, repaired Full did not materially outperform the trained anchor:

| Metric | Repaired Full | Trained anchor | Anchor minus Full, 95% CI |
|---|---:|---:|---:|
| Hop-2+ Step@1 | 0.403037 | 0.396135 | -0.006901 `[-0.030179, 0.016290]` |
| Hop-2+ MRR | 0.565392 | 0.567883 | +0.002491 `[-0.013482, 0.018735]` |
| All-state Step@5 | 0.839118 | 0.850551 | not bootstrapped |
| Full unit coverage | 0.766000 | 0.784000 | not bootstrapped |

**Decision:** the repair validates rollout mismatch as the main cause of the
earlier collapse, but it fails the pre-registered originality gate. Do not
run the repaired selector on the 3,000-question evaluation set. The paper may
claim that rollout-aligned accumulated state is competitive with the trained
anchor and clearly stronger than query-only state, but it must not claim that
full compiled state outperforms the anchor.

## Baseline 3.2: ECDR-style direct-indirect policy

Implement:

1. Query-only direct evidence selection.
2. Indirect evidence scoring conditioned on the selected direct evidence.

Call it `ECDR-inspired textual direct-indirect baseline`, not a faithful ECDR
reproduction.

### Frozen Stage 3.2 protocol

The textual adaptation uses one model with stage-dependent context:

1. At `t=0`, rank the direct evidence using `(question, candidate)`.
2. Freeze the predicted top-1 unit from `t=0` as the direct anchor.
3. At every `t>0`, score indirect candidates using
   `(question, frozen predicted direct evidence, candidate)`.
4. Do not replace the direct anchor with later predictions and do not expose
   full accumulated history.
5. Training uses `H_t[0]` as teacher forcing only after verifying that it
   equals the `t=0` teacher-positive unit for the same qid. Evaluation uses
   only the predicted `t=0` unit and never a gold supporting fact.

The model uses the same v21 initialization, DeBERTa backbone, v22 fixed
candidate pools, ranking labels, seed, optimizer, losses, epochs, and
candidate/final evidence budgets. Its output is isolated under
`deberta_v3_large_v24_ecdr_direct_indirect`.

### Stage 3.2 decision

The ECDR-inspired baseline significantly outperforms the original v22 Full
rollout on answer EM/F1, Alignment@5, full unit coverage, and hop-2+
Step@1/MRR. Against the separately trained v23 anchor, it ties answer EM/F1,
improves hop-2+ Step@1, ties hop-2+ MRR, and loses Alignment@5 and full unit
coverage.

The bridge/comparison slices show the same sharpness--coverage tradeoff:

| Slice | ECDR minus anchor Step@1 | Step@5 | MRR |
|---|---:|---:|---:|
| Bridge, hop-2+ | +0.026866 | -0.034502 | +0.008507 |
| Comparison, hop-2+ | +0.038158 | -0.046053 | +0.004391 |

**Decision:** Stage 3.2 is complete. Full compiled state does not outperform
the controlled chain baselines, so the broad state-superiority originality
claim is rejected. Retain the significant Full-versus-query-only and 2Wiki
state-value results, but narrow the main contribution to closure-oriented
control, explicit state diagnostics, and the candidate/context budget
frontier.

## Stage 3R: Rollout-aware knowledge-state repair

### Purpose

Before adding another mechanism baseline, repair the identified
train--inference mismatch in the Full knowledge-state policy. This is a
targeted optimization of the existing `Knowledge-State-Guided` contribution,
not a new contribution or a post-hoc test-set search.

### Frozen repair scope

The v25 repair is limited to four already diagnosed mismatches:

1. Replace pure gold-prefix teacher forcing with a deterministic mixture of
   clean teacher states and frozen-v22 model-rollout states.
2. Write exactly the online top-1 selection into the next policy state.
3. Use the same latest-first notebook renderer and the same
   `max_raw=8`, `max_chars_per_item=260` truncation during data construction
   and online inference.
4. Separate the final top-5 answer-context set from the top-1 state-write
   ledger. A unit previously present only in the answer-context set remains
   eligible for a later state write.

The model-rollout states must be generated only from the v22
train/validation/internal-test qids with the frozen v22 checkpoint. The
3,000-question HotpotQA evaluation subset and the 1,000-question blending
validation set must not be used to construct v25 training examples.

### Training-data protocol

- Canonicalize v22 rows to one row per `(qid, t)` before rollout.
- Generate rollouts with the deployed `hybrid_policy` selector,
  `alpha=0.5`, `candidate_top_k=10`, `select_top_k=5`, and
  `state_update_top_k=1`.
- At `t=0`, keep one clean row because clean and rollout state are identical.
- At `t>=1`, train on one online-rendered teacher state and one frozen-model
  rollout state.
- For a rollout state, supervise the first teacher-ordered gold unit not
  already present in the rollout state. Skip the rollout row if all gold
  units have already been acquired.
- Use rollout-only validation and internal-test states so checkpoint
  selection reflects deployment conditions.
- Keep the v22 candidate pool, backbone, ranking loss, optimizer, seed, two
  epochs, and candidate budget. Initialize from the v22 checkpoint and use an
  independent v25 output directory.

### Leakage and readiness gate

Before training, the readiness report must prove:

- zero duplicate canonical `(qid, t)` states;
- zero train/validation/internal-test qid overlap;
- zero overlap between v25 training qids and the 3,000-question evaluation
  subset;
- every positive target is present in its fixed candidate pool;
- no rollout target is already present in the corresponding rollout state;
- all rendered `K_t` values exactly match the shared online renderer;
- the rollout source checkpoint and all frozen selector parameters are
  recorded in the manifest.

### Validation-only acceptance gate

After one seed-42 training run, evaluate v25 Full, v23 anchor, and v24
direct-indirect on the same question-disjoint 1,000-qid validation set,
without answer generation. Proceed to one final 3,000-qid v25 evaluation only
if:

1. v25 significantly improves hop-2+ Step@1 or MRR over at least one of the
   two chain baselines under qid-clustered paired bootstrap; and
2. v25 does not show a clear full-unit-coverage regression against that
   baseline.

If the gate fails, stop v25 test evaluation, retain the negative result, and
continue with the narrowed closure-control claim. Do not tune additional
training mixtures or inspect the 3,000-question evaluation set.

### Stage 3R decision

**Failed and closed.** On the fixed 1,000-qid validation set, v25 Full does
not satisfy the pre-registered chain-baseline gate.

Against v23 anchor, v25 ties full-unit coverage at `0.784`, but loses
hop-2+ Step@1 by `-0.024845`
`[-0.048936, -0.000692]` and MRR by `-0.029613`
`[-0.046304, -0.012383]`. Against v24 direct-indirect, v25 improves
full-unit coverage by `+0.016` `[0.003, 0.029]`, but loses hop-2+ Step@1
by `-0.048999` `[-0.072866, -0.026074]` and MRR by `-0.039211`
`[-0.055932, -0.022727]`.

The repair therefore preserves or improves trajectory completeness but does
not make accumulated Full state a stronger next-unit ranking mechanism than
the controlled chain baselines. Do not run v25 on the 3,000-question
evaluation subset, do not tune another state-mixture ratio, and do not use
v25 as the paper's main result. Retain it as a negative rollout-alignment
diagnostic. The supported state claim remains Full versus query-only and
cross-dataset closure/coverage, not Full-state superiority over anchor or
direct-indirect policies.

## Baseline 3.3: FiSKE-inspired clue-state policy

Construct an explicit textual clue coverage vector and rank candidates against
unresolved clues. The clue generator must not use gold supporting facts at
test time.

Call it `FiSKE-inspired textual clue-state baseline`.

**Status:** failed and closed after the pre-registered validation-only gate.
Epoch 2 was selected with validation accuracy `0.545153`; internal-test
accuracy is `0.503609`. The 1,000-qid online gate had no protocol or clue
transition failures, but v26 significantly regressed full-unit coverage
against v22, v23, and v24. Do not run v26 on the 3,000-qid evaluation subset
or call the answer API for this checkpoint.

### Frozen Stage 3.3 protocol

- Call the method `FiSKE-inspired textual clue-state baseline`; it is not a
  faithful FiSKE reproduction.
- Extract a frozen list of entity, relation, and answer-type clues using only
  the question text and deterministic rules. No LLM or API is used.
- At state `t`, mark clues using only the text of evidence units already
  present in `H_t`. Gold supporting facts not yet acquired, answers, teacher
  roles, positive labels, and future trajectory units are forbidden inputs.
- Render the complete binary coverage vector plus the unresolved clue list
  into `K_t`.
- Copy v22 qids, row order, `H_t`, fixed candidate pools, and ranking labels
  exactly. Keep auxiliary labels masked.
- Match the v22 seed-42 initialization, optimizer, ranking losses, two epochs,
  and candidate budget. Use an independent v26 output directory.

Implementation:

```text
src/clue_state.py
scripts/build_kbs_v26_fiske_clue_state_data.py
scripts/check_kbs_v26_fiske_clue_state.py
scripts/run_kbs_stage3_fiske_clue_state.sh
configs/train_ranker_deberta_v26_fiske_clue_state.yaml
```

The readiness audit must pass exact source/output pairing, deterministic
renderer replay, monotonic clue coverage, zero forbidden clue-state fields,
split and evaluation disjointness, and matched configuration checks. Run a
20-qid data/readiness smoke first. Do not train until that report is recorded
as `status: OK`.

## Controlled reporting

Report:

- teacher next-unit alignment at 1/5;
- hop-2+ Step@1/5 and MRR;
- bridge and comparison slices;
- full unit coverage;
- Supporting Fact F1/EM;
- Joint EM/F1;
- Conditional Rank-Reversal Accuracy;
- answer F1;
- selection latency, tokens, and GPU memory.

## Acceptance criterion

The originality claim is supported if full compiled state significantly
outperforms query-only and at least the anchor baseline on hop-2+ Step@1/MRR
or full unit coverage. If ECDR- or FiSKE-inspired methods win, report that
honestly and narrow the contribution to the conditions where closure state is
advantageous.

---

## Stage 3X: High-cost counterfactual state architecture repair

### Motivation

The v23 anchor and v24 direct-indirect controls outperform the existing Full
state policy, while v25 rollout mixing and v26 clue-state rendering fail their
validation gates. The remaining high-cost route must therefore change both
the learning signal and the state-candidate interaction architecture. Another
text-only renderer variation is forbidden.

### Frozen v27 protocol

1. Start from the audited v22 fixed-pool train/validation/internal-test data;
   do not use the 3,000-question evaluation subset or answer API.
2. Canonicalize to one row per `(qid,t)`. Keep `t=0` once and oversample every
   train state with `t>=1` exactly twice. Validation and internal test remain
   canonical and unweighted.
3. For every later state, identify acquired evidence units in `H_t` that
   remain in the same candidate pool. Add an explicit margin requiring the
   current teacher-positive unit to outrank those acquired units.
4. Use one shared DeBERTa-v3-large backbone in two explicit passes: encode
   `(q,K_t)` as the state vector and `(q,u)` as the candidate vector. Score
   their concatenation, element-wise product, and absolute difference through
   a learned interaction head.
5. Initialize the shared encoder from v21 Full. Freeze encoder LR at `5e-7`,
   new interaction-head LR at `5e-5`, projection dimension at 256, two epochs,
   seed 42, acquired-negative margin 0.20 and weight 0.50. Deficit,
   contribution, and role auxiliary losses remain disabled.
6. Report ordinary ranking accuracy and acquired-pair reversal accuracy.
   Runtime must infer the architecture from checkpoint metadata so training,
   online rollout, and state intervention use identical scoring.

### Execution gates

1. Run a 20-qid data/readiness smoke with no GPU training.
2. Build and audit the complete v27 data. Require fixed pools, exact v22
   labels/state text, all adjacent preference switches, nonzero acquired
   counterfactual pairs, and zero split/evaluation overlap.
3. Run exactly one seed-42 training job. If it OOMs, only memory-preserving
   implementation changes are allowed; objective weights and data protocol
   remain frozen.
4. On the question-disjoint 1,000-qid validation set, compare v27 against v22
   Full, v23 anchor, and v24 direct-indirect without answers. Proceed only if
   v27 significantly improves hop-2+ Step@1 or MRR over both controlled chain
   baselines under qid-clustered paired bootstrap and has no significant
   full-unit-coverage regression.
5. Only after that gate may seeds 43/44 and one final 3,000-qid/API evaluation
   be authorized. A failed gate closes v27 and restores the narrowed
   closure-control claim.

### Current authorization

The three-seed validation review is complete. Seeds 42, 43, and 44 all pass
the same strict gate with zero protocol failures. Across the three runs,
hop-2+ Step@1 is `0.533241 +/- 0.005788`, Alignment@5 is
`0.858293 +/- 0.002218`, and MRR is `0.676870 +/- 0.002974` (mean and sample
standard deviation). Relative to v23 anchor, the mean hop-2+ Step@1/MRR gains
are `+0.137106/+0.108987`; relative to v24 direct-indirect, they are
`+0.112951/+0.099390`. Mean full-unit differences are `-0.004333` and
`+0.011667`, respectively, and no individual seed has a significant
full-unit regression against either baseline.

Seed 42 remains the pre-registered primary checkpoint; seeds 43/44 are
robustness evidence and must not be selected by their validation scores. The
20-qid fresh-answer smoke passed the exact frozen protocol: 20 judged answers,
zero answer errors, zero empty raw answers, EM 0.600000, F1 0.692857,
Alignment@5 0.900000, full-unit coverage 0.850000, positive API token/latency
measurements, and the expected v27 checkpoint, alpha 0.5, 10/5 evidence
budgets, and top-1 state write. Exactly one final 3,000-qid/API evaluation is
now authorized with a fresh cache.

---

# Stage 4: Standard and Closure-Aware Evaluation

## Purpose

Remove the self-reference risk of relying on a teacher-defined Step@5 metric.

## Required implementation

For HotpotQA, add:

- Supporting Fact Precision.
- Supporting Fact Recall.
- Supporting Fact F1.
- Supporting Fact EM.
- Selected Evidence Precision and Recall.
- Answer EM and F1.
- Joint EM and Joint F1 using the official normalization and combination
  rule.

Rename the existing `Step@5` in the paper to:

> Teacher Next-Unit Alignment@5

Add:

```text
ClosureSuccess@B =
answer correct
AND all supporting facts covered
AND acquisition/context cost <= B
```

Implementation entry points:

```text
scripts/check_kbs_stage4_metric_readiness.py
scripts/evaluate_kbs_standard_metrics.py
scripts/run_kbs_stage4_standard_metrics.sh
```

Supporting-fact predictions are the unique per-step top-1
`predicted_unit_id` chain. The complete unique `selected_unit_ids` set is the
answer context and is evaluated separately as selected-evidence
precision/recall/F1 and full coverage. This separation prevents forced top-5
context distractors from making every Supporting Fact EM and Joint EM zero.
`ClosureSuccess@B` uses answer exact match, full selected-context support
coverage, and at most `B` unique selected evidence units.

## Validation requirements

- Unit IDs must map exactly to `(title, sentence_id)`.
- Supporting-fact metrics must be checked on Gold Oracle, where recall and EM
  should reach their expected upper bound.
- Per-qid records must reproduce every aggregate summary.
- Metric definitions and denominators must be written into the paper and
  supplementary material.

## Acceptance criterion

KSG-EA must retain its main conclusion on at least one standard
supporting-fact or joint metric. Teacher alignment may support mechanism
analysis but cannot be the only positive evidence.

---

# Stage 5: Multi-Seed and Sampling Robustness

## Purpose

Measure training randomness rather than relying only on bootstrap uncertainty
over one trained checkpoint.

## Required seeds

```text
42, 43, 44
```

Train the complete final Student pipeline for all three seeds. Fine-tuning
three v22 models from one identical v21 checkpoint measures only v22
fine-tuning variance and must be labeled as such; the preferred result trains
the complete final pipeline independently.

## Required reporting

- Mean and standard deviation for Compact and Recall.
- At minimum: answer EM/F1, full unit coverage, Supporting Fact F1, Joint F1,
  hop-2+ Step@1, and rank reversal.
- Ranking-only or anchor baseline under the same seeds.
- Per-seed checkpoints and manifests.

## Sampling robustness

Preferred:

- Full available HotpotQA validation evaluation.

Minimum fallback:

- two additional fixed 3,000-question subsets with recorded seeds and no
  training overlap.

2Wiki may remain a fixed 1,000-question subset if its sampling limitation is
stated explicitly.

## Acceptance criterion

The main state and answer-quality conclusions must not depend on one seed. A
mean advantage without consistent seed-level direction must be reported as
unstable.

---

# Stage 6: Closure, Compression, and Cost Evidence

## Purpose

Show that the method is more than a learned reranker and substantiate its
budget/control claims.

## Step 6.1: Minimal teacher objective ablation

Compare:

1. Relevance/front-score teacher.
2. Coverage-greedy teacher.
3. Closure utility teacher.
4. Full repair teacher.

First compare offline trajectory statistics:

- success/abort/stall/max-step/false-stop rates;
- average trajectory length;
- label agreement;
- derived-note use;
- construction time.

Then train at least Coverage and Full Closure students if their teacher labels
materially differ.

## Step 6.2: Compression funnel

Using the final model, report:

| Stage | Target recall | Policy alignment@5 | Full unit coverage |
|---|---:|---:|---:|
| Raw pool | | | |
| RRF | | | |
| RRF + MMR | | | |
| Learned/BGE compression | | | |
| No compression | | | |
| Oracle target-preserving compression | | | |

## Step 6.3: Cost-closure frontier

Use the final checkpoint for candidate budgets such as:

```text
10, 15, 20, 50
```

Plot answer F1, Joint F1, full unit coverage, and ClosureSuccess@B against:

- selected context tokens;
- selection latency;
- total end-to-end latency;
- peak GPU memory.

## Step 6.4: Agentic cost accounting

For the LLM-guided iterative selector, record:

- selection LLM calls/question;
- selection input/output tokens;
- final-answer input/output tokens;
- total API tokens;
- retries/failures;
- wall-clock latency;
- estimated API cost.

Use `LLM-guided iterative selection` unless the implementation is verified as
a faithful IRCoT reproduction.

## Optional strengthening

Use a second answer generator such as a fixed Qwen/Llama-family model without
retraining the policy. This is strongly recommended but not a blocker if cost
or hardware prevents it.

---

# Stage 7: Deficit and Contribution Decision

## Default route: diagnostic downgrade

Current evidence shows:

- auxiliary ablations do not improve end-to-end metrics;
- deficit correlation and calibration are weak;
- the old state-level label is produced by a candidate-conditioned head;
- contribution supervises insufficient negatives;
- neither output is a reliable online controller.

Unless the pilot below passes, the paper must:

- remove deficit/contribution from headline contributions;
- describe them as interpretable trajectory diagnostics;
- move detailed formulas and negative ablations to the appendix;
- avoid claiming performance improvement from them.

## Conditional Deficit/Contribution 2.0 pilot

Only run after Stages 2-6 are stable.

Deficit 2.0:

- state-only encoder;
- masked role dimensions;
- no future-dependent derived target;
- exact remaining proportion or ordinal target;
- greater weight on `t>=1`;
- calibration and monotonicity evaluation.

Contribution 2.0:

- candidate-conditioned head;
- positive plus top-3 hard-negative supervision;
- pairwise gain loss;
- optional calibrated score
  `rank + lambda * dot(deficit, contribution)`.

## Upgrade gate

Deficit/contribution may return to the core method only if:

1. deficit correlation and monotonicity materially improve on held-out data;
2. the pairwise contribution objective separates positive from hard negatives;
3. adding calibrated gain improves a pre-registered validation metric;
4. the improvement survives a frozen test run and paired confidence interval.

Otherwise retain the diagnostic route and perform no further weight sweeps.

---

# Stage 8: Paper Rewrite and Submission Audit

## Method and novelty rewrite

- Center the paper on compiled-context closure.
- Explain the difference from ACRA anchor continuation, ECDR
  direct-indirect retrieval, and FiSKE clue-to-KG state.
- Add a mechanism comparison table.
- Avoid `first stateful`, `first evidence chain`, and similar broad claims.
- Present Compact as the efficiency/alignment operating point.
- Present Recall as the accuracy/completeness operating point.

## Experiment rewrite

- Replace all stale v17/v21 main results with the final accepted checkpoint.
- Keep model versions explicit when diagnostic results use a different
  checkpoint.
- Rename Step@5 to Teacher Next-Unit Alignment@5.
- Add standard supporting-fact and joint metrics.
- Add multi-seed mean and standard deviation.
- Add paired confidence intervals near claims of significance.
- Include complete online cost for the agentic baseline.
- Report negative auxiliary results without overclaiming.

## Reproducibility audit

Confirm:

- source split, qid sampling, and seeds;
- train/validation/test disjointness;
- teacher selection and terminal replay;
- candidate memory construction;
- model initialization and loss weights;
- answer prompt and generation parameters;
- all per-qid records and manifests;
- code release commands;
- no report uses stale answer caches for changed contexts.

## Final lock criterion

The paper may be locked only when:

1. Stages 2-5 pass.
2. Every main table uses the final checkpoint and matched evaluation settings.
3. Stage 6 blockers are complete or explicitly documented as limitations.
4. Stage 7 has a final downgrade/upgrade decision.
5. Every quantitative claim maps to a report and per-qid record file.

---

# 3. Priority classification

## Submission blockers

1. v22 end-to-end state validation.
2. ACRA-style anchor and ECDR-style direct-indirect controlled baselines.
3. Standard supporting-fact and joint metrics.
4. Three-seed robustness.
5. Correct final paper tables and claims.

## High-value strengthening

1. FiSKE-inspired clue-state baseline.
2. Teacher objective ablation.
3. Compression funnel and cost-closure frontier.
4. Complete LLM-guided selection cost.

## Optional

1. Second answer generator.
2. Additional datasets beyond HotpotQA and 2Wiki.
3. Deficit/Contribution 2.0 after the upgrade gate.

---

# 4. Current run ledger

| Run ID | Stage | Model/data | Result | Decision |
|---|---:|---|---|---|
| `state-phase1-v21-full3000` | 1 | v21 Full, HotpotQA 3,000 | Correct state beats query-only mainly at Step@1/MRR | Build state-focused v22 |
| `v22-state-focused-train-seed42` | 1 | 10,000 train qids, 27,730 rows | Best epoch 2; val acc 0.7017; test acc 0.6632 | Run fixed-pool diagnosis |
| `state-phase1-v22-full3000` | 1 | v22, HotpotQA 3,000 | Strong state effect; rank reversal 0.7474 | Stage 1 passed |
| `v22-alpha-sensitivity-val1000` | 2.1 | v22, HotpotQA validation 1,000 | Alpha 0.35 uniquely maximizes Alignment@5 at 0.7971 | Freeze alpha 0.35 |
| `v22-stage2-full-compact-smoke20-attempt1` | 2.2 | v22 Compact, HotpotQA 20 | All answers, API tokens, and API latency are zero; no HTTP answer calls occurred | Reject; make blank key fail loudly |
| `v22-stage2-full-compact-smoke20-attempt2` | 2.2 | v22 Compact, HotpotQA 20 | Preflight HTTP 400 before retrieval: the retired `deepseek-chat` alias was rejected | Reject; explicitly freeze V4-Flash non-thinking |
| `v22-stage2-full-compact-smoke20-attempt3` | 2.2 | v22 Compact, HotpotQA 20 | PASS: 20 judged, 0 errors, EM 0.55, F1 0.642857, Alignment@5 0.86, positive API tokens/latency | Authorize frozen 3,000-qid Stage 2.2 runs |
| `v22-stage2-hotpot-full-compact` | 2.2 | v22 Full-state Compact, HotpotQA 3,000 | PASS: EM 0.579667, F1 0.723282, Alignment@5 0.808114, full-unit 0.698667 | Include in paired state comparison |
| `v22-stage2-hotpot-query-compact` | 2.2 | v22 Query-only Compact, HotpotQA 3,000 | PASS: EM 0.564, F1 0.705241, Alignment@5 0.817023, full-unit 0.639 | Full state improves answer/coverage but not Alignment@5 |
| `v22-stage2-hotpot-full-recall` | 2.2 | v22 Full-state Recall, HotpotQA 3,000 | PASS: EM 0.611333, F1 0.760448, Alignment@5 0.781250, full-unit 0.827 | Include in paired state comparison |
| `v22-stage2-hotpot-query-recall` | 2.2 | v22 Query-only Recall, HotpotQA 3,000 | PASS: EM 0.577, F1 0.721138, Alignment@5 0.851288, full-unit 0.704 | Full state improves answer/coverage but not Alignment@5 |
| `v22-stage2-state-ci-compact` | 2.3 | Full-state minus Query-only Compact, paired 10,000 bootstrap | F1 +0.018041 [0.008961, 0.027384]; full-unit +0.059667 [0.046992, 0.073000]; Alignment@5 -0.008909 [-0.016708, -0.000826] | State value supported for answer quality and trajectory coverage |
| `v22-stage2-state-ci-recall` | 2.3 | Full-state minus Query-only Recall, paired 10,000 bootstrap | F1 +0.039310 [0.029927, 0.048772]; full-unit +0.123000 [0.110000, 0.136333]; Alignment@5 -0.070038 [-0.080211, -0.060165] | Strong state benefit with a next-unit-alignment tradeoff |
| `v21-matched-stage2-hotpot-compact` | 2.3 | v21 Compact with V4-Flash non-thinking, HotpotQA 3,000 | PASS: EM 0.591667, F1 0.736479, Alignment@5 0.827440, full-unit 0.716667 | Use as matched v21 reference |
| `v21-matched-stage2-hotpot-recall` | 2.3 | v21 Recall with V4-Flash non-thinking, HotpotQA 3,000 | PASS: EM 0.630667, F1 0.779862, Alignment@5 0.820861, full-unit 0.833333 | Use as matched v21 reference |
| `v22-vs-v21-compact-ci` | 2.3 | v22 minus matched v21 Compact, paired 10,000 bootstrap | F1 -0.013197 [-0.020671, -0.005771]; full-unit -0.018000 [-0.027333, -0.009000] | Strict non-regression criterion fails |
| `v22-vs-v21-recall-ci` | 2.3 | v22 minus matched v21 Recall, paired 10,000 bootstrap | F1 -0.019414 [-0.026912, -0.011984]; full-unit -0.006333 [-0.016667, 0.004333] | F1 non-regression criterion fails |
| `v21-alpha-diagnostic-val1000` | 2.3 | v21, question-disjoint validation 1,000, no API | Alpha 0.35 maximizes Alignment@5 at 0.814210; alpha 0.5 gives full-unit 0.744 | Compare with v22 validation surface |
| `v21-v22-alpha-surface-diagnosis` | 2.3 | Matched validation summaries | At alpha 0.5, v22 gains Step@1 +0.051041, MRR +0.023181, full-unit +0.005; loses Alignment@5 -0.024092 | Regression is not uniform; alpha objective is misaligned |
| `v22-closure-balanced-hotpot-full-compact` | 2.3 | v22 Full-state Compact, alpha 0.5, HotpotQA 3,000 | PASS: EM 0.594667, F1 0.740488, Alignment@5 0.781935, full-unit 0.761000 | Accept repaired Compact operating point |
| `v22-closure-balanced-hotpot-query-compact` | 2.3 | v22 Query-only Compact, alpha 0.5, HotpotQA 3,000 | PASS: EM 0.575000, F1 0.717850, Alignment@5 0.835526, full-unit 0.677000 | Use for final state-value comparison |
| `v22-closure-balanced-state-ci` | 2.3 | Full-state minus Query-only, alpha 0.5, paired 10,000 bootstrap | F1 +0.022638 [0.013072, 0.032147]; full-unit +0.084000 [0.071325, 0.097000]; Alignment@5 -0.053591 [-0.062797, -0.044160] | State improves answer quality and closure with alignment tradeoff |
| `v22-closure-balanced-vs-v21-ci` | 2.3 | v22 alpha 0.5 minus matched v21 alpha 0.35, paired 10,000 bootstrap | F1 +0.004008 [-0.003646, 0.011568]; full-unit +0.044333 [0.034333, 0.054333] | Non-regression gate passes; closure significantly improves |
| `v22-stage2-2wiki-full-compact` | 2.4 | v22 Full-state Compact, alpha 0.5, 2Wiki 1,000, zero-shot | PASS: EM 0.571000, F1 0.639719, Alignment@5 0.724752, full-unit 0.633000 | Accept zero-shot Compact |
| `v22-stage2-2wiki-full-recall` | 2.4 | v22 Full-state Recall, alpha 0.5, 2Wiki 1,000, zero-shot | PASS: EM 0.639000, F1 0.727514, Alignment@5 0.711921, full-unit 0.882000 | Accept zero-shot Recall |
| `v22-stage2-2wiki-query-compact` | 2.4 | v22 Query-only Compact, alpha 0.5, 2Wiki 1,000, zero-shot | PASS: EM 0.491000, F1 0.551609, Alignment@5 0.681291, full-unit 0.409000 | Use for state-transfer comparison |
| `v22-stage2-2wiki-state-ci` | 2.4 | Full-state minus Query-only Compact, paired 10,000 bootstrap | EM +0.080 [0.058, 0.102]; F1 +0.088109 [0.067209, 0.109832]; Alignment@5 +0.043460 [0.027340, 0.060191]; full-unit +0.224 [0.196, 0.251] | Cross-dataset state value passes |
| `v22-stage2-2wiki-vs-hybrid-ci` | 2.4 | Compact/Recall minus Hybrid, paired 10,000 bootstrap | Compact F1 +0.137131 [0.113846, 0.160951], full-unit +0.312 [0.283, 0.342]; Recall F1 +0.224927 [0.197323, 0.252026], full-unit +0.561 [0.530, 0.592] | Both operating points significantly outperform Hybrid |
| `v22-stage2-2wiki-vs-bge-ci` | 2.4 | Compact/Recall minus BGE, paired 10,000 bootstrap | Compact F1 -0.008764 [-0.032967, 0.015400], Alignment@5 +0.060430 [0.040345, 0.080565], full-unit +0.068 [0.034, 0.102]; Recall F1 +0.079032 [0.056608, 0.101730], full-unit +0.317 [0.286, 0.349] | Compact answer quality ties BGE; Recall significantly wins all four metrics |
| `stage2-final-decision` | 2 | HotpotQA plus zero-shot 2Wiki | All acceptance gates pass with 3,000/1,000 judged answers and zero errors | Close Stage 2; advance to controlled mechanism baselines |
| `v22-closure-balanced-compact-smoke20` | 2.3 | v22 Full-state Compact, alpha 0.5, HotpotQA 20 | PASS: 20 judged, 0 errors, F1 0.642857, Alignment@5 0.78, full-unit 0.75 | Authorize the two frozen 3,000-qid Compact runs |
| `stage3-acra-anchor-readiness-tooling` | 3.1 | Matched v22 fixed-pool data; previous-evidence-only context | Training context switch, matched config, split/anchor audit, and guarded runner prepared | Run readiness only; do not train until status is OK |
| `stage3-acra-anchor-readiness` | 3.1 | v22 fixed-pool train/validation/internal-test data and matched v22 Full configuration | PASS: readiness checker returned `status: OK`; training was not started | Authorize independent seed-42 anchor training |
| `stage3-acra-anchor-train-seed42` | 3.1 | Previous-evidence-only input; matched v22 Full initialization/data/losses; 2 epochs | PASS: best epoch 2, validation acc 0.6761, internal-test acc 0.6488; checkpoint saved | Authorize matched 20-qid end-to-end smoke |
| `stage3-acra-anchor-smoke20` | 3.1 | v23 anchor, previous-evidence-only, Compact alpha 0.5, HotpotQA 20 | PASS: 20 judged, 0 errors, EM 0.6000, F1 0.692857, Alignment@5 0.8400, full-unit 0.8000; positive API tokens/latency | Authorize frozen 3,000-qid evaluation |
| `stage3-acra-anchor-hotpot3000` | 3.1 | v23 anchor, previous-evidence-only, Compact alpha 0.5, HotpotQA 3,000 | PASS protocol: 3,000 judged, 0 errors; EM 0.612000, F1 0.757822, Step@1 0.448876, Step@5 0.854167, full-unit 0.792667 | Run paired comparison against Full v22 |
| `stage3-acra-anchor-vs-full-ci` | 3.1 | Anchor minus Full v22, paired 10,000 bootstrap | Anchor wins: EM +0.017333 [0.008000, 0.026667], F1 +0.017334 [0.009041, 0.025619], Step@5 +0.072231 [0.062674, 0.081413], full-unit +0.031667 [0.020667, 0.042667] | Stage 3 originality gate fails for Full v22 |
| `stage3-acra-hop2-diagnostic` | 3.1 | Anchor minus Full v22 on t>=1 states, qid-clustered 10,000 bootstrap | Anchor gains Step@1 +0.224395 [0.205447, 0.243378] and MRR +0.181735 [0.167487, 0.195786] | Diagnose rollout/state-noise failure before Stage 3.2 |
| `stage3-acra-failure-diagnosis` | 3.1 | Full v22 versus anchor, t0/hop-2+/type/anchor-conditioning slices | Full wins slightly at t0, but collapses at t>=1; online Full writes top-5 per step while training writes one gold unit; anchor has 2,498 versus 1,118 hop-2+ rank wins | Test a pre-registered top-1 state-update repair on validation only |
| `stage3-rollout-aligned-state-val1000` | 3.1 | Repaired Full top-1 state write versus trained anchor; alpha 0.5; validation 1,000; no API | Full hop-2+ Step@1 0.403037 versus 0.396135 and MRR 0.565392 versus 0.567883; both paired intervals include zero; full-unit 0.766 versus 0.784 | Repair confirms rollout mismatch but does not beat anchor; do not run repaired Full on evaluation 3,000 |
| `stage3-ecdr-readiness-tooling` | 3.2 | Query-only direct stage; frozen predicted direct anchor for every indirect stage; matched v22 training protocol | Dataset/runtime mode, matched v24 config, guarded readiness checker, and runner prepared | Run server readiness with `CHECK_ONLY=1`; do not train until status is OK |
| `stage3-ecdr-readiness-attempt1` | 3.2 | v22 fixed-pool data with `deep_repeat=2` | FAILED only because 3,865 expected repeated deep-state rows were treated as duplicate errors; direct-anchor match 1.0, all memory resolved, no split/evaluation leakage | Correct checker to permit structurally identical repeats while rejecting conflicting repeats; rerun readiness only |
| `stage3-ecdr-readiness` | 3.2 | v22 train 27,730 rows, validation 1,207, internal test 1,247; matched v24 configuration | PASS: failures empty; direct-anchor match 1.0 on all splits; 3,865 expected train repeats and zero conflicting repeats; no qid leakage | Authorize independent v24 seed-42 training; no answer API |
| `stage3-ecdr-train-seed42` | 3.2 | Query-only t=0 and frozen teacher-direct context at t>0; matched v22 initialization/data/losses; 2 epochs | PASS training: best epoch 2, validation acc 0.6123, internal-test acc 0.5838; checkpoint saved | Lower offline accuracy than v23 anchor; run protocol-checked 20-qid online smoke before final evaluation |
| `stage3-ecdr-smoke20` | 3.2 | v24 direct-indirect Compact alpha 0.5; fresh V4-Flash non-thinking answers | PASS: 20 judged, 0 errors, no empty answers, zero direct-anchor violations; EM 0.650000, F1 0.742857, Alignment@5 0.820000, full-unit 0.800000 | Freeze protocol and authorize one 3,000-qid evaluation |
| `stage3-ecdr-hotpot3000` | 3.2 | v24 direct-indirect Compact alpha 0.5; HotpotQA 3,000; fresh V4-Flash non-thinking answers | PASS: 3,000 judged, 0 errors, zero direct-anchor violations; EM 0.608000, F1 0.755806, Alignment@1/5 0.471902/0.833333, full-unit 0.776667 | Compute paired comparisons and mechanism slices |
| `stage3-ecdr-vs-full-ci` | 3.2 | ECDR minus v22 Full alpha 0.5; paired 10,000 bootstrap | ECDR wins EM +0.013333 [0.003333, 0.023000], F1 +0.015319 [0.006716, 0.023679], Alignment@5 +0.051398 [0.042173, 0.060710], full-unit +0.015667 [0.005000, 0.026333] | ECDR-inspired chain baseline significantly outperforms the original Full rollout |
| `stage3-ecdr-vs-anchor-ci` | 3.2 | ECDR minus v23 trained anchor; paired 10,000 bootstrap | EM -0.004000 [-0.011000, 0.003000] and F1 -0.002016 [-0.007782, 0.003955] tie; Alignment@5 -0.020833 [-0.026839, -0.014827], full-unit -0.016000 [-0.023333, -0.009000] | ECDR ties answer quality but loses top-5 alignment and coverage |
| `stage3-ecdr-hop2-bootstrap` | 3.2 | State-weighted metrics with 10,000 qid-clustered bootstrap samples over t>=1 states | ECDR minus Full: Step@1 +0.253259 [0.236464, 0.270310], MRR +0.189514 [0.176816, 0.202210]; ECDR minus anchor: Step@1 +0.028864 [0.014792, 0.042941], MRR +0.007779 [-0.002281, 0.017761] | ECDR improves sharp hop-2 ranking; its MRR gain over anchor is not significant |
| `stage3-ecdr-type-slices` | 3.2 | Bridge 2,423 qids/3,536 hop-2+ states; comparison 577 qids/760 states | Versus anchor, ECDR bridge Step@1/5/MRR deltas +0.026866/-0.034502/+0.008507; comparison +0.038158/-0.046053/+0.004391 | Consistent Top-1 sharpness versus Top-5 coverage tradeoff; close Stage 3.2 |
| `stage3r-v25-protocol-authorized` | 3R | Rollout-aware Full-state repair; no evaluation data and no answer API | Frozen clean/model-rollout mixture, exact online renderer, independent top-1 state ledger, and validation-only gate | Implement data/readiness tooling before any training |
| `stage3r-v25-tooling` | 3R | Shared online state renderer; corrected state-write ledger; canonical rollout collector; mixed-data builder; readiness checker; matched v25 config | Local syntax, import, renderer-order, and rewritten-target smoke checks pass; no server data, training, or API run yet | Authorize isolated 20-qid data/readiness smoke only |
| `stage3r-v25-readiness-smoke20` | 3R | First 20 qids from each v22 train/validation/internal-test split; frozen v22 hybrid-policy rollout; no API and no training | PASS: canonical rows 46/43/47, zero conflicting repeats and skipped rollout steps; all state writes top-1; reconstructed v25 rows 72/43/47; 162/162 renderer matches; zero acquired-positive violations; split overlap zero. Full v22 source audit already proves train/val/test overlap with the 3,000-qid evaluation subset is zero | Authorize full v25 data construction/readiness only |
| `stage3r-v25-full-readiness` | 3R | 10,000/500/500 train/validation/internal-test qids; frozen-v22 hybrid-policy top-1 rollout; no API and no training | PASS: 37,730/1,207/1,247 rows; 40,184/40,184 renderer matches; zero acquired-positive, row, pool, and split errors; zero overlap with the 3,000-qid evaluation subset; train mixture contains 23,865 teacher and 13,865 rollout rows; 7,774 rollout targets rewritten | Authorize exactly one matched v25 seed-42 training run |
| `stage3r-v25-train-seed42` | 3R | v25 mixed teacher/rollout train states; rollout-only validation/internal test; v22 initialization; matched optimizer/ranking losses; two epochs | PASS training: epoch 2 selected; train accuracy 0.699655, validation accuracy 0.678542, internal-test accuracy 0.605453; validation loss improved from 0.954928 to 0.942863 | Authorize one question-disjoint 1,000-qid validation-only gate; no answer API |
| `stage3r-v25-validation-gate-tooling` | 3R | v25 Full versus v23 trained anchor and v24 direct-indirect; same 1,000 qids, alpha 0.5, top-5 evidence budget, top-1 state write | Three-way runner and qid-clustered paired bootstrap analyzer prepared; gate requires significant v25 hop-2+ Step@1 or MRR gain over at least one baseline without significant full-unit regression | Run once without answers; do not inspect the 3,000-qid evaluation subset |
| `stage3r-v25-validation-gate` | 3R | v25 Full versus v23 anchor and v24 direct-indirect; question-disjoint validation 1,000; 10,000 qid-clustered bootstrap samples; no API | FAIL: v25 ties v23 full-unit but significantly loses hop-2+ Step@1/MRR; v25 significantly improves v24 full-unit by +0.016 [0.003, 0.029] but significantly loses hop-2+ Step@1/MRR | Close Stage 3R; forbid v25 evaluation on 3,000 qids and further mixture tuning |
| `stage3-fiske-clue-state-tooling` | 3.3 | Deterministic question-only clue generator; current-prefix coverage updater; v22 row/pool/label-preserving rewrite; matched v26 config | Local syntax and synthetic paired audit PASS: renderer replay 6/6, zero forbidden fields, zero non-monotonic transitions, zero config mismatches | Authorize isolated 20-qid data/readiness smoke only; no training or API |
| `stage3-fiske-clue-state-smoke20` | 3.3 | First 20 qids from each v22 train/validation/internal-test split; deterministic clue state; no API and no training | PASS: 52/43/47 source-output rows paired exactly; renderer 142/142; zero forbidden fields, non-monotonic transitions, row errors, split/evaluation overlap, and config mismatches | Authorize full v26 clue-state data construction/readiness only; training remains forbidden |
| `stage3-fiske-clue-state-full-readiness` | 3.3 | v22-matched train/validation/internal-test qids and fixed pools rewritten as deterministic clue states; no API and no training | PASS: 27,730/1,207/1,247 rows paired and rendered exactly; 10,000/500/500 qids; zero forbidden fields, non-monotonic transitions, row errors, split/evaluation overlap, and 24-field config mismatches | Authorize exactly one matched v26 seed-42 two-epoch training run; answer API forbidden |
| `stage3-fiske-clue-state-train-attempt1` | 3.3 | Frozen v26 seed-42 configuration on GPU 0 | FAILED before epoch completion: current process reached about 22.66 GiB while another process occupied 838 MiB; only 52 MiB remained and a 64 MiB allocation failed | Do not change training protocol; resume once on a GPU with no competing process |
| `stage3-fiske-clue-state-checkpoint-audit` | 3.3 | Existing v26 output after interrupted launch | INCOMPLETE: `best_model.pt` (1,736,267,788 bytes) and `best_val_metrics.json` exist, but `train_history.json` and final `test_metrics.json` are absent | Archive the incomplete directory; restart the identical seed-42 run on an empty GPU; forbid online evaluation |
| `stage3-fiske-clue-state-train-attempt2` | 3.3 | Identical frozen v26 run launched in the foreground | INCOMPLETE: training log ends at epoch-1 row 23,131/27,730 (83%) without traceback; no training process remains and GPUs 0-3/5-7 use only 24 MiB | Treat as terminal/session interruption; preserve residuals and restart under `nohup` on an empty GPU |
| `stage3-fiske-clue-state-train-seed42` | 3.3 | Deterministic clue-state input; matched v21 initialization, v22 pools/qids/losses/optimizer; 2 epochs | PASS training completeness: epoch 2 selected; train accuracy 0.494050, validation accuracy 0.545153, internal-test accuracy 0.503609; all auxiliary labeled counts zero | Offline ranking is substantially below v22; authorize only a question-disjoint 1,000-qid no-answer validation gate before any test/API run |
| `stage3-fiske-clue-state-online-tooling` | 3.3 | Shared training/online clue renderer; top-1 state-write coverage updates; saved before/after transitions; v26/v22/v23/v24 paired gate | Syntax and synthetic transition/gate audits PASS; initial coverage zero, transitions monotonic and linked, no API path enabled | Four-way 20-qid no-answer runtime smoke authorized |
| `stage3-fiske-clue-state-runtime-smoke20` | 3.3 | v26 clue state vs v22 full state, v23 previous-evidence anchor, and v24 direct/indirect anchor; alpha 0.5; candidate 10; select 5; top-1 state write; no answers | PASS: 20 identical qids and 56 states; zero protocol failures; all 56 clue updates monotonic; all 36 cross-step transitions linked; v26 is competitive against all three controls under the smoke gate | Authorize the same paired gate on the full question-disjoint 1,000-qid validation set; smoke metrics are not paper evidence |
| `stage3-fiske-clue-state-validation-gate` | 3.3 | Same four methods on 1,000 question-disjoint validation qids and 2,449 states; alpha 0.5; top-1 state write; 10,000 qid-clustered bootstrap samples; no answers | FAIL: zero protocol/transition failures, but v26 full-unit coverage is 0.664 versus 0.782/0.784/0.768; deltas are -0.118 [-0.145,-0.092], -0.120 [-0.148,-0.093], and -0.104 [-0.131,-0.078]. Versus v22, v26 improves hop-2+ Step@1 by +0.037957 [0.004071,0.071179] and MRR by +0.027656 [0.002655,0.052216], but all-state Alignment@5 falls by -0.018375 [-0.034414,-0.002448] | Close v26 as a negative sharp-ranking/completeness tradeoff; forbid 3,000-qid and API evaluation; advance to Stage 4 |
| `stage4-standard-metric-tooling` | 4 | Per-step top-1 supporting-fact chain; selected-context evidence metrics; official-style answer and joint products; unit-budget ClosureSuccess; source-summary replay; Gold Oracle gate | Local syntax, synthetic oracle, and real v22 Compact replay PASS. Recomputed Answer EM/F1 and full coverage exactly match the source report. Preliminary v22 Compact values: Supporting Fact F1 0.464404, Supporting Fact EM 0.086333, Joint F1 0.356303, Joint EM 0.056667 | Authorize server readiness and one offline five-method evaluation; no API or training |
| `stage3x-v27-protocol-authorized` | 3X | v22 fixed-pool counterfactual states; shared-backbone dual state/candidate encoder; acquired-evidence reversal margin | Protocol frozen before training; Stage 4 paused at user request; no API, training, validation, or evaluation run yet | Implement tooling and authorize only a 20-qid data/readiness smoke |
| `stage3x-v27-tooling` | 3X | v27 canonicalizer, counterfactual labels, dual-tower model/trainer/runtime compatibility, config, readiness checker, guarded runner | Python/shell syntax and diff checks pass; isolated synthetic audit passes t>=1 oversampling, acquired-negative pairs, adjacent preference switches, and split disjointness. Local tensor forward was unavailable because the workstation environment has no PyTorch | Authorize server 20-qid data/readiness smoke only; no training or API |
| `stage3x-v27-readiness-smoke20` | 3X | First 20 qids from each v22 train/validation/internal-test split; v27 canonical fixed-pool states; no API and no training | PASS: 72/43/47 output rows from 46/43/47 canonical states; 66/27/36 acquired-negative pairs; adjacent preference switches 26/26, 23/23, and 27/27; zero row errors, split overlap, and overlap with the 3,000-qid evaluation subset | Authorize complete v27 data construction and readiness audit only; training remains forbidden |
| `stage3x-v27-full-readiness` | 3X | Complete v27 train/validation/internal-test data derived from audited v22 fixed pools; no API and no training | PASS: 37,730/1,207/1,247 rows over 10,000/500/500 qids; 37,642/975/1,085 acquired-negative pairs; adjacent switches 13,865/13,865, 707/707, and 747/747; zero row errors, split overlap, and overlap with the 3,000-qid evaluation subset | Authorize exactly one frozen seed-42 two-epoch dual-tower training run; API and evaluation forbidden |
| `stage3x-v27-train-seed42` | 3X | Shared-backbone dual state/candidate encoder; v21 initialization; v27 counterfactual data; two epochs; auxiliary heads disabled | PASS training completeness: epoch 2 selected; validation accuracy 0.664457 and acquired-pair accuracy 0.902564 over 975 pairs; internal-test accuracy 0.607057 and acquired-pair accuracy 0.886636 over 1,085 pairs. Reversal learning is strong, but ordinary accuracy remains below historical v22/v23 values | Implement paired no-answer gate and authorize only a 20-qid runtime smoke before the 1,000-qid validation gate |
| `stage3x-v27-validation-gate-tooling` | 3X | v27 dual versus v22 Full, v23 anchor, and v24 direct-indirect; alpha 0.5; top-1 state write; qid-clustered bootstrap | Syntax and synthetic four-method pairing/protocol audit pass. Frozen gate requires significant v27 hop-2+ Step@1 or MRR gains over both chain baselines with no significant full-unit regression against either | Authorize 20-qid no-answer runtime smoke only |
| `stage3x-v27-validation-runtime-smoke20` | 3X | v27 dual, v22 Full, v23 anchor, and v24 direct-indirect; same 20 validation qids; alpha 0.5; top-1 state write; no answers | PASS runtime: `SMOKE_OK`, zero protocol failures, and exact qid/step pairing. v27 has higher hop-2+ Step@1/MRR point estimates than all three references; intervals are inconclusive as expected at 20 qids. Full-unit point estimate ties v22/direct and is -0.05 versus anchor without significant regression | Authorize exactly one complete 1,000-qid no-answer validation gate; smoke metrics are not paper evidence |
| `stage3x-v27-validation-gate` | 3X | Same 1,000 question-disjoint validation qids; v27 dual versus v22 Full, v23 anchor, and v24 direct-indirect; alpha 0.5; top-1 state write; 10,000 qid-clustered bootstrap samples; no answers | PASS strict gate with zero protocol failures. v27 hop-2+ Step@1/MRR gains are +0.158730/+0.129810 versus v22, +0.140097/+0.109983 versus anchor, and +0.115942/+0.100386 versus direct; every 95% CI is strictly positive. Full-unit is 0.776 versus 0.782/0.784/0.768, with no significant regression against any reference | Core state-superiority risk is repaired on validation; authorize matched seeds 43/44 before any final evaluation/API run |
| `stage3x-v27-multiseed-configs` | 3X | Seed 43 and 44 configurations with independent output directories | Config audit confirms both are byte-equivalent in parsed protocol to seed 42 except seed/output directory; data, initialization, architecture, optimizer, losses, epochs, and margins remain frozen | Authorize parallel seed-43/44 training; no evaluation or API yet |
| `stage3x-v27-train-seeds43-44` | 3X | Matched v27 dual training with seeds 43 and 44; two epochs; independent outputs | PASS training completeness for both. Across seeds 42/43/44, validation accuracy is 0.661419 +/- 0.003348 and acquired-pair accuracy 0.908718 +/- 0.005427; internal-test accuracy is 0.609195 +/- 0.002018 and acquired-pair accuracy 0.892780 +/- 0.007674 (sample standard deviation) | Authorize seed-43/44 1,000-qid no-answer robustness gates using frozen baseline reports |
| `stage3x-v27-multiseed-gate-tooling` | 3X | Run only v27 seeds 43/44 online; reuse seed-42 gate's v22/v23/v24 reports; separate 10,000-bootstrap gates and three-seed aggregate | Syntax and synthetic aggregate audits pass; analyzer accepts an explicit expected checkpoint per seed and the summary requires all three strict gates to pass | Run once without answers; no 3,000-qid/API work yet |
| `stage3x-v27-multiseed-validation-gate` | 3X | Seeds 42/43/44 on the same 1,000 question-disjoint validation qids; alpha 0.5; top-1 state write; no answers; 10,000 qid-clustered bootstrap samples per gate | PASS for all seeds with zero protocol failures. Three-seed hop-2+ Step@1/Alignment@5/MRR are 0.533241 +/- 0.005788, 0.858293 +/- 0.002218, and 0.676870 +/- 0.002974. Mean Step@1/MRR gains are +0.137106/+0.108987 versus v23 and +0.112951/+0.099390 versus v24; no seed has significant full-unit regression | Close the robustness gate; retain seed 42 as the pre-registered primary model and authorize only a 20-qid fresh-answer final-protocol smoke |
| `stage3x-v27-final-hotpot-tooling` | 3X | Seed-42 v27 checkpoint; Compact alpha 0.5; candidate/select/state-write budgets 10/5/1; fresh V4-Flash answers | Dedicated guarded runner reuses the strict end-to-end report audit, explicitly checks top-1 state writes, isolates v27 reports/caches, and keeps `final3000` locked behind post-smoke authorization; shell/Python syntax and diff checks pass | Run `MODE=smoke20` once; final 3,000-qid/API mode remains locked |
| `stage3x-v27-final-hotpot-smoke20` | 3X | Seed-42 v27 Compact; HotpotQA first 20 evaluation qids; alpha 0.5; candidate/select/state-write budgets 10/5/1; fresh V4-Flash answers | PASS strict report audit: 20 judged, zero errors and empty answers; EM 0.600000, F1 0.692857, Alignment@1/5 0.480000/0.900000, full-unit 0.850000, average API tokens 412.0 and latency 0.730 s; checkpoint and all frozen protocol fields match | Authorize exactly one final 3,000-qid/API evaluation using seed 42 and a fresh isolated cache |

## One-time closure-balance repair

The original alpha rule maximized only teacher next-unit Alignment@5. This
conflicts with the paper's compiled-context closure objective and selected a
v22 operating point that underuses the state-sensitive policy. For the
one-time validation-only repair, define:

```text
closure_balance(alpha)
  = harmonic_mean(Alignment@5(alpha), full_unit_coverage(alpha))
```

On the fixed 1,000-qid validation set, the v22 scores are:

| Alpha | Alignment@5 | Full unit | Closure balance |
|---:|---:|---:|---:|
| 0.00 | 0.748877 | 0.546 | 0.631545 |
| 0.20 | 0.777869 | 0.607 | 0.681893 |
| 0.35 | 0.797060 | 0.678 | 0.732725 |
| **0.50** | **0.777460** | **0.749** | **0.762965** |
| 0.80 | 0.649653 | 0.772 | 0.705562 |
| 1.00 | 0.596570 | 0.767 | 0.671134 |

Freeze `alpha=0.5`. This is the only authorized repaired test operating
point. No other alpha may be tested if it fails.

## DeepSeek model migration note

DeepSeek retired the legacy `deepseek-chat` model name after 2026-07-24.
The earlier June 2026 experiments used that alias when it mapped to V4-Flash
in non-thinking mode. From Stage 2.2 attempt 3 onward, all fresh answer
generation is therefore frozen to:

- requested model: `deepseek-v4-flash`;
- thinking mode: `disabled`;
- temperature: `0.0`;
- prompt version: `kbs_extractive_answer_json_v1`.

This is the closest protocol-continuity setting available after alias
retirement. Reports and answer-cache records must retain the requested model
and thinking mode. The migration must be disclosed in the reproducibility
notes because the API does not expose a frozen historical backend snapshot.

## Next authorized run

```text
Run exactly one final 3,000-qid/API evaluation with the seed-42 v27 checkpoint,
alpha 0.5, candidate budget 10, selected-evidence budget 5, top-1 state write,
V4-Flash non-thinking, temperature 0.0, and a new isolated answer cache. Do
not launch seeds 43/44 or another alpha on the evaluation subset.

```bash
KBS_V27_FINAL_AUTHORIZED=1 MODE=final3000 CUDA_DEVICE=0 \
  bash scripts/run_kbs_v27_final_hotpot.sh
```
```

No Stage 3 or later experiment should begin before Stage 2 acceptance is
recorded, unless the user explicitly changes this plan.
