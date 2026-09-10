# KBS 75--85% Experimental Closure Plan

## Scope and authority

This plan implements the experimental requests in
`md/kbs_review_todo_report.pdf`. It extends, but does not rewrite, the
completed Stage 1--8 ledger in `md/kbs_three_review_execution_plan.md`.
Theoretical strengthening is excluded at the user's request. Every runner
must read both plans and fail closed when an artifact or protocol field is
missing.

The target is review-ready experimental completeness, not a guaranteed
acceptance probability. Negative results are retained and reported.

## Frozen common protocol

- HotpotQA primary evaluation: the existing fixed 3,000-qid distractor-
  validation subset and question-local sentence memory.
- Final policy architecture: dual state--candidate interaction.
- State input: online accumulated state; state write top-1.
- Retrieval front: BM25--BGE RRF, with local expansion and MMR where the
  KSG-EA operating point requires it.
- Policy blend: alpha 0.5.
- Answer evidence: unique selected top-5 units per acquisition state.
- Answer generator: DeepSeek V4-Flash, thinking disabled, temperature 0,
  prompt `kbs_extractive_answer_json_v1`.
- Primary uncertainty: paired qid bootstrap with 10,000 samples.
- Primary downstream metrics: Answer F1, Supporting-Fact F1/EM, Joint
  F1/EM, Full Supporting-Unit Coverage, and ClosureSuccess at the registered
  budget.
- API answers may be reused only when question, gold answer, answer protocol,
  and the full ordered selected-unit sequence match exactly.

## Stage 9.1: Closure teacher versus Coverage teacher

### Scientific question

Does changing only the teacher objective from answer-facing Closure to
coverage-greedy supervision improve downstream compiled-context quality?

### Matched design

- Full Closure: existing v27 seeds 42/43/44.
- Coverage teacher: v29 seeds 42/43/44.
- Hold fixed row order/repetition, qids, rollout state, candidate pools,
  architecture, v21 initialization, optimizer, epochs, and loss weights.
- Change only ranking positives and dependent ranking metadata.
- Use Compact budget 10 as the pre-registered causal operating point.
- Report the full three-seed result regardless of direction.

### Execution gates

1. `pretrain_readiness`: audit data/config parity; no training, GPU, or API.
2. Train v29 seeds 43 and 44; seed 42 already exists.
3. `posttrain_readiness`: require all checkpoints, histories, validation,
   and internal-test metrics.
4. Run a 20-qid selection-only paired smoke; no API.
5. Run 3,000-qid selection-only reports for all Coverage seeds.
6. Prepare exact-context caches from the corresponding Full Closure seed.
7. Run one 20-qid answer smoke, then the three 3,000-qid answer reports.
8. Compute standard metrics, redundant/acquired-evidence reselection rate,
   and paired bootstrap intervals against Full Closure.

### Interpretation gate

The experiment supports Closure superiority only if downstream effects, not
cross-label training accuracy, favor Closure consistently. Otherwise report
objective distinctness and the observed negative or mixed result.

## Stage 9.2: Counterfactual acquired-evidence loss ablation

Use v27 data, architecture, initialization, seeds, and optimizer for:

| Variant | CE | Margin | Acquired-evidence margin |
|---|---:|---:|---:|
| Ranking only | yes | no | no |
| CE + margin | yes | yes | no |
| CE + acquired | yes | no | yes |
| Full | yes | yes | yes |

Train seeds 42/43/44. The primary causal contrast is Full versus CE+margin.
Before answer generation, report teacher alignment, acquired-pair accuracy,
and redundant/acquired evidence reselection. End-to-end claims require
Supporting-Fact F1, Joint F1, Full Coverage, ClosureSuccess, and paired CIs.

## Stage 9.3: Same-generator final-protocol strong baselines

Run BM25-RAG, Dense-RAG, Hybrid-RAG, Iterative-Hybrid-RAG, and
BGE-Reranker-RAG on the same 3,000 qids with the frozen answer protocol.
Keep native retrieval/reranking behavior visible, but freeze question-local
memory and final selected top-5. Report standard downstream metrics, local
selection cost, answer tokens/latency, and paired CIs versus Compact and
Recall. Historical alias rows remain descriptive and cannot satisfy this
stage.

## Stage 9.4: Final-policy 2Wiki zero-shot transfer

Apply v27 without 2Wiki-specific fine-tuning. Evaluate seed-42 Compact
(budget 10), balanced-knee (15), and Recall (50), plus final-protocol Hybrid
and BGE baselines. Report standard answer/support/joint/coverage/closure
metrics and paired CIs. Add selection-only seeds 43/44 to demonstrate policy
robustness before deciding whether extra answer calls are justified.

## Stage 9.5: Downstream state rollout

Using one frozen v27 checkpoint and matched candidate protocol, run complete
rollouts for correct online state, query-only, frozen initial state,
other-question state, and previous-evidence-only state. Compare Supporting-
Fact F1, Joint F1, Full Coverage, ClosureSuccess, and reselection rate.
Teacher-relative fixed-pool interventions remain mechanism diagnostics and
must not be presented as this task-level experiment.

## Stage 9.6: One breadth enhancement

After Stages 9.1--9.5, select exactly one:

- preferred when data construction is reliable: MuSiQue zero-shot transfer;
- preferred when a second backend is already available: repeat the frozen
  answer-context comparison with a non-DeepSeek generator.

Do not use another DeepSeek service tier as an independent generator family.

## Stop conditions

- Never choose a run, seed, alpha, or cache based on test-set outcomes.
- A failed scientific gate is a valid result, not permission to change the
  protocol post hoc.
- Do not start answer generation before selection records and exact cache
  reuse have passed audit.
- Do not mix v22 transfer, historical DeepSeek aliases, or v21 auxiliary
  heads into final-v27 inferential claims.

## Stage 9 run ledger

| Date | Stage | Gate | Result | Evidence |
|---|---|---|---|---|
| 2026-09-09 | 9.1 | Coverage pre-training readiness | Passed | No missing paths; seed parity and Closure--Coverage config parity audits contain no unexpected differences; v29 seed 42 and v27 seeds 42/43/44 are complete. |
| 2026-09-10 | 9.1 | Coverage post-training readiness | Passed | v29 seeds 42/43/44 and v27 seeds 42/43/44 each contain a non-empty checkpoint, validation metrics, internal-test metrics, and training history; no protocol failures. |
| 2026-09-10 | 9.1 | Paired selection smoke | Passed | On the same 20 qids, both methods completed the frozen Compact protocol with no API calls or protocol failures. The observed metric direction is diagnostic only and is not used as a scientific gate. |
| 2026-09-10 | 9.1 | Three-seed paired selection | Passed | All seeds share identical ordered-qid and step-target hashes. Closure exceeds Coverage for Alignment@1, Alignment@5, MRR, full unit coverage, and full document coverage for every seed; Coverage has more acquired-evidence reselection for every seed. All per-seed paired intervals exclude zero for these contrasts. |

## Current authorized action

Only Stage 9.1 exact-context answer-cache preparation is authorized. It must
audit the existing v27 Closure answers against the frozen generator protocol
and reuse an answer only when the question, gold answer, and full ordered
selected-unit sequence match the corresponding v29 Coverage selection.
Report the fresh-call requirement per seed. Do not start answer generation
before the cache-preparation report is reviewed. No new API calls are
authorized.
