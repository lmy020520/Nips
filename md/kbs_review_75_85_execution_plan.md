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
| 2026-09-10 | 9.1 | Exact-context cache preparation | Passed | The historical Closure answer reports match the frozen generator and rerun selection protocols. Exact ordered-context reuse is 32/37/38 qids for seeds 42/43/44, leaving 8,893 fresh answers across the registered three-seed evaluation. |
| 2026-09-10 | 9.1 | Coverage answer smoke | Passed | Seed 42 completed 20/20 answers with the frozen V4-Flash protocol; all 20 cache files and ordered selection contexts passed audit, with no empty/error answers or invalid reuse metadata. Smoke EM/F1 are execution diagnostics only. |
| 2026-09-10 | 9.1 | Three-seed Coverage answers | Passed | Seeds 42/43/44 each completed all 3,000 qids with no answer errors or invalid cache entries. Exact Closure-context reuse was 32/37/38 qids; the remaining 2,968/2,963/2,962 answers were generated for Coverage contexts under the frozen protocol. |
| 2026-09-10 | 9.1 | Matched downstream objective comparison | Passed; Stage 9.1 closed | Closure beats Coverage for every seed on Answer F1, Supporting-Fact F1/EM, Joint F1/EM, full support coverage, and ClosureSuccess@10. Mean Coverage-minus-Closure deltas are -0.0078 Answer F1, -0.1772 Supporting-Fact F1, -0.1481 Joint F1, -0.0413 full coverage, and -0.0198 ClosureSuccess@10. The latter four evidence/joint/closure metrics have per-seed paired 95% CIs strictly below zero; Answer F1 CIs cross zero, and Answer EM is indistinguishable. Coverage also increases top-1 acquired-evidence reselection by +0.1161. The registered interpretation gate is `CLOSURE_SUPERIOR`. |
| 2026-09-10 | 9.2 | Acquired-loss pre-training readiness | Passed | The matched v27 data contain 37,730/1,207/1,247 train/validation/internal-test rows and 37,642/975/1,085 acquired-negative pairs, with no split or primary-evaluation qid overlap. Full seeds 42/43/44 are complete. Nine resolved configs differ only in output directory and the registered ordinary/acquired margin weights; all hashes are frozen and no partial variant artifacts exist. |
| 2026-09-11 | 9.2 | Acquired-loss post-training readiness | Passed | All nine registered Ranking-only, CE+Margin, and CE+Acquired runs for seeds 42/43/44 contain non-empty checkpoints, best-validation metrics, internal-test metrics, and training histories. Together with the three existing Full v27 seeds, all 12 matched model instances are complete; no paths, runs, or protocol checks are missing. No inference or API call was made by this audit. |
| 2026-09-11 | 9.2 | Training-metric review | Passed; selection smoke authorized | In the primary Full-minus-CE+Margin contrast, acquired-pair accuracy improves for every seed: mean +0.0202 on validation and +0.0301 on internal test. Ordinary ranking accuracy changes are small and mixed (-0.0025 validation, +0.0040 internal test). CE+Acquired similarly exceeds Ranking-only acquired-pair accuracy by +0.0236/+0.0338. This is a targeted discrimination signal only; online reselection and downstream effects remain untested. |
| 2026-09-11 | 9.2 | Four-variant selection smoke | Passed; full selection authorized | The same 20 qids and teacher targets pass the frozen Compact protocol for Full, Ranking-only, CE+Margin, and CE+Acquired with no API calls or failures. Full and CE+Margin tie alignment and coverage; Full top-1 acquired-evidence reselection is 0.26 versus 0.36 for CE+Margin. These values validate execution only and are not used as scientific evidence. |
| 2026-09-12 | 9.2 | Three-seed 3,000-qid selection | Passed; primary cache audit authorized | All methods and seeds share identical ordered-qid and step-target hashes. In the registered Full-minus-CE+Margin contrast, Full reduces Top-1 acquired-evidence reselection by 0.0179 on average, with every seed's paired 95% CI strictly below zero. Step@5, coverage, and MRR are tied or mixed; CE+Acquired alone has the lowest reselection rate. This supports a targeted anti-reselection effect, not a broad retrieval-quality gain. No API calls were made. |
| 2026-09-12 | 9.2 | Primary exact-context cache audit | Passed; answer smoke authorized | Full answers can be reused for exactly matching ordered CE+Margin contexts on 2,016/1,947/1,966 qids for seeds 42/43/44. This leaves 984/1,053/1,034 fresh answers, or 3,071 total API calls instead of 9,000. Protocol, qid, question, answer target, and ordered-context checks passed with no API calls or failures. |
| 2026-09-12 | 9.2 | CE+Margin answer smoke | Passed; full answers authorized | Seed 42 completed the frozen 20-qid answer smoke with no errors or invalid cache records. The audit exercised both paths: 11 exact Full-context answers were reused and 9 CE+Margin-context answers were freshly generated. Smoke EM/F1 are execution diagnostics only. |
| 2026-09-12 | 9.2 | Matched downstream acquired-loss comparison | Passed; Stage 9.2 closed with targeted-only decision | Full reduces Top-1 acquired-evidence reselection by 0.0179 with every seed's paired CI below zero. Full also improves Supporting-Fact F1 by 0.0052 on average for every seed, although only seeds 42/43 exclude zero, and improves Joint F1 by 0.0045 on average with only seed 42 excluding zero. Answer F1, full coverage, and ClosureSuccess are mixed or tied. The registered decision is `TARGETED_ANTI_RESELECTION_ONLY`; no broad downstream gain is claimed. |
| 2026-09-12 | 9.3 | Same-generator strong-baseline readiness | Passed; selection smoke authorized | The 3,000-qid/7,296-state/124,523-unit HotpotQA evaluation data and all five selector implementations are present. The answer protocol is frozen to DeepSeek V4-Flash, thinking disabled, JSON prompt v1, and temperature 0. Historical reports omit required generator-protocol metadata and are descriptive only; none qualifies for final-protocol reuse. No GPU inference or API call was made. |
| 2026-09-12 | 9.3 | Five-baseline selection smoke | Passed; full selection authorized | All five methods completed the same 20 qids and 50 teacher states with identical ordered-qid and teacher-target hashes, zero skipped states, and no answer/API calls. BM25/Dense/Hybrid/Iterative-Hybrid/BGE Step@5 values are 0.58/0.70/0.76/0.76/0.64; these are execution diagnostics only. |
| 2026-09-12 | 9.3 | Five-baseline 3,000-qid selection | Passed; exact-context cache audit authorized | All methods completed the same 3,000 qids/7,296 states with identical ordered qids and targets, zero skipped states, and no API calls. Hybrid has the highest Step@1/5 (0.3912/0.7767), while BGE-Reranker has the highest full-unit coverage (0.7353). Selection latency is 9.00/484.05/492.05/484.85/1,011.14 ms/qid for BM25/Dense/Hybrid/Iterative-Hybrid/BGE. Downstream ordering remains untested. |
| 2026-09-12 | 9.3 | Exact-context answer-cache audit | Passed; bounded answer smoke authorized | Ten frozen-protocol reports provide 15,143 unique source contexts, but only 3/14/43/45/6 target contexts can be reused for BM25/Dense/Hybrid/Iterative-Hybrid/BGE. The naive fresh requirement is 14,889; cross-baseline context deduplication lowers the theoretical minimum to 13,839 by removing 1,050 duplicate targets. The deterministic source priority resolves 413 duplicate-source raw-answer disagreements without outcome selection. No API call was made. |
| 2026-09-12 | 9.3 | BM25 bounded answer smoke | Passed; full answer chain authorized | The registry-first method completed 20/20 frozen-protocol answers with zero context mismatches, empty/error answers, or invalid caches. All 20 were fresh calls. Smoke EM/F1 0.5500/0.6429 and 367.05 tokens are execution diagnostics only. |
| 2026-09-12 | 9.3 | Five-baseline complete answers | Passed; offline finalization authorized | All five methods completed 3,000 qids with no answer or cache-audit failures. BM25/Dense/Hybrid/Iterative-Hybrid/BGE Answer F1 is 0.6686/0.7046/0.7185/0.7114/0.7709. The fixed answer chain made exactly 13,839 fresh calls and reused 1,161 exact contexts, fully realizing the registered cross-method deduplication. |
| 2026-09-12 | 9.3 | Final-protocol downstream comparison | Passed; Stage 9.3 closed | Compact is statistically tied with BGE-Reranker on Answer EM/F1 but significantly improves Supporting-Fact F1 by 0.1805, Joint F1 by 0.1537, full support coverage by 0.0470, and ClosureSuccess@10 by 0.0397. Recall exceeds all five baselines on all 40 registered metric--baseline comparisons; versus BGE it gains 0.0195 Answer F1, 0.2356 Supporting-Fact F1, 0.2038 Joint F1, 0.1377 full coverage, and 0.0847 ClosureSuccess@50. All reports contain the same 3,000 qids, all source-summary checks pass, and the Gold Oracle reaches 1.0 supporting-fact recall/EM. Decision: `FINAL_PROTOCOL_BASELINES_COMPLETE`. |
| 2026-09-12 | 9.4 | Final-policy 2Wiki readiness | Passed; selection smoke authorized | The fixed 2Wiki transfer set contains 1,000 matching query/sample qids, 2,416 states, and 31,522 memory units. All v27 seeds, model assets, and required selectors are present; no output collision exists. v22 KSG reports may seed only exact-context caches, while legacy Hybrid/BGE/Gold reports lack the frozen generator metadata and are not reusable. No training, inference, or API call was made. |
| 2026-09-12 | 9.4 | Five-method selection smoke | Passed; full multiseed selection authorized | Seed-42 Compact/Balanced/Recall and Hybrid/BGE completed the same 20 qids/50 states with identical ordered-qid and target hashes, zero skipped states, and no API calls. Recall has the strongest smoke Step@5/full-unit values (0.82/0.70), but all smoke metrics remain execution diagnostics only. |
| 2026-09-13 | 9.4 | Complete multiseed 1,000-qid selection | Passed; exact-context cache audit authorized | All 11 reports share identical ordered-qid and target hashes and contain 1,000 qids/2,416 states with no skips or API calls. Compact/Balanced/Recall mean Step@5 is 0.7689/0.8005/0.8462 and mean full-unit coverage is 0.6103/0.6623/0.7500 across seeds 42/43/44. Seed-42 Step@5 exceeds BGE by 0.1126/0.1457/0.1970 at the three operating points. These are selection-only results. |

## Current authorized action

Stages 9.1--9.3 are closed. Stage 9.2 supports a replicated targeted anti-
reselection effect but not a broad end-to-end gain. Stage 9.3 establishes
same-generator final-protocol superiority in evidence, joint, coverage, and
closure metrics, with Recall also significantly improving answer quality over
the strongest BGE-Reranker baseline. Stage 9.4 complete multiseed selection
also passed and shows stable Compact--Balanced--Recall ordering. The next
authorized action is the exact-context answer-cache audit for seed-42
Compact/Balanced/Recall plus Hybrid and BGE-Reranker. Seeds 43/44 remain
selection-only robustness runs. No answer API call is authorized at this gate.
