# KBS Three-Document Unified Execution Plan

## 0. Plan control

```yaml
plan_id: kbs-three-review-plan-v1
created_at: 2026-07-25
plan_read_required_before_every_run: true
current_stage: 3
current_status: Stage 3.1 anchor smoke passed; matched 3,000-qid evaluation authorized
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

## 2. Eight-stage execution overview

| Stage | Goal | Current status | API | Training |
|---|---|---|---|---|
| 1 | Establish causal state use | Passed with v22 | No | Completed |
| 2 | Validate v22 end to end | Passed on HotpotQA and zero-shot 2Wiki | Completed | Completed |
| 3 | Controlled mechanism baselines | Stage 3.1 anchor readiness pending | Yes for final answers | Yes |
| 4 | Standard and closure-aware metrics | Pending | Mostly offline | No |
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

## Baseline 3.2: ECDR-style direct-indirect policy

Implement:

1. Query-only direct evidence selection.
2. Indirect evidence scoring conditioned on the selected direct evidence.

Call it `ECDR-inspired textual direct-indirect baseline`, not a faithful ECDR
reproduction.

## Baseline 3.3: FiSKE-inspired clue-state policy

Construct an explicit textual clue coverage vector and rank candidates against
unresolved clues. The clue generator must not use gold supporting facts at
test time.

Call it `FiSKE-inspired textual clue-state baseline`.

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
Run the frozen 3,000-qid Stage 3.1 ACRA-style anchor evaluation through
`scripts/run_kbs_stage3_acra_anchor_eval.sh` with `SMOKE=0`. Keep the trained
v23 anchor checkpoint, previous-evidence-only online context, Compact
candidate/evidence budgets, alpha 0.5, and fresh DeepSeek V4-Flash
non-thinking answers. The report must contain 3,000 unique judged qids, zero
answer errors, no empty raw answers, positive API tokens/latency, and the
expected checkpoint/context metadata. Do not begin Stage 3.2 until the full
report and paired comparison against Full v22 have been reviewed.
```

No Stage 3 or later experiment should begin before Stage 2 acceptance is
recorded, unless the user explicitly changes this plan.
