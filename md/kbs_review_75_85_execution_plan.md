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
| 2026-09-13 | 9.4 | Exact-context answer-cache audit | Passed; bounded Compact answer smoke authorized | Two protocol-complete v22 reports provide 1,919 eligible source contexts, but exact reuse covers only 48 of 5,000 final target files. Cross-method deduplication reduces the remaining requirement from 4,952 target files to 4,431 unique fresh contexts by eliminating 521 duplicates. There are no protocol differences, answer disagreements, invalid caches, failures, or API calls. |
| 2026-09-13 | 9.4 | Seed-42 Compact answer smoke | Passed; complete five-method answer chain authorized | The frozen 20-qid run completed without answer, context, protocol, or cache failures. It exercised one exact historical reuse and 19 fresh answers. Smoke EM/F1 0.6500/0.7207 are execution diagnostics only. The full chain must run sequentially with cache propagation before each method. |
| 2026-09-13 | 9.4 | Complete five-method answers | Passed; offline finalization authorized | Compact/Balanced/Recall/Hybrid/BGE completed 1,000 qids with Answer F1 0.6711/0.6999/0.7330/0.5475/0.6912 and no answer, context, protocol, or cache failures. The chain made exactly 4,431 fresh calls and 569 exact-context reuses, accounting for all 5,000 method--qid targets. |
| 2026-09-13 | 9.4 | Final-policy 2Wiki downstream comparison | Passed; Stage 9.4 closed | Against Hybrid, every registered metric is significantly higher for all three KSG-EA operating points. Against BGE, Compact and Balanced are tied on answer metrics but significantly improve evidence and joint metrics; Recall significantly improves all eight metrics, including +0.0418 Answer F1, +0.1760 Support F1, +0.1650 Joint F1, +0.205 full support, and +0.124 ClosureSuccess@50. Decision: `FINAL_POLICY_2WIKI_COMPLETE`. |
| 2026-09-13 | 9.5 | Downstream state-rollout readiness | Passed; selection smoke authorized | The frozen HotpotQA set contains 3,000 matching query/sample qids and 7,296 states. The existing v27 Compact correct-state report matches every registered protocol field and preserves all 7,296 pre-step online states, with no missing or duplicate qids. All five task-level state conditions are implemented. No training, inference, or API call was made. |
| 2026-09-13 | 9.5 | Five-condition selection smoke | Passed; full selection authorized | All five conditions completed the same 20 qids/50 states with identical ordered-qid and teacher-target hashes, valid state-intervention metadata, and no skips or API calls. Correct state exceeds query-only on Step@1, MRR, full-unit coverage, and Top-1 reselection, while previous-only is slightly higher on Step@1/MRR. These small-sample directions are diagnostics only. |
| 2026-09-13 | 9.5 | Five-condition 3,000-qid selection | Passed; exact-context cache audit authorized | Correct online state significantly exceeds query-only, frozen-initial, and other-question state on Step@1/5, MRR, full-unit/document coverage, and Top-1 acquired-evidence reselection. Against query-only it gains 0.0850 Step@1, 0.0362 Step@5, and 0.1127 full-unit coverage while reducing Top-1 reselection by 0.0611. It ties previous-only on Step@1/MRR and is slightly lower on full-unit/document coverage. Interpretation: `STATE_RELEVANCE_SUPPORTED`, not full-history superiority. |
| 2026-09-13 | 9.5 | Exact-context answer-cache audit | Passed; bounded answer smoke authorized | All 15 source reports match the frozen answer protocol and provide 28,982 eligible unique contexts. Exact reuse covers 6,126 of 15,000 targets; the remaining 8,874 target files contain 7,924 unique fresh contexts after 950 cross-condition duplicates are removed. Online state is fully reusable, and previous-only reuses 2,261 qids. No API call or audit failure occurred. |
| 2026-09-13 | 9.5 | Initial other-question answer smoke | Rejected; corrected retry authorized | All 20 answers were valid, but one selected context differed from the frozen full-run prefix. The bounded rerun paired its final qid against the first qid of the 20-qid prefix rather than the next qid in the complete 3,000-qid ordering. This was a boundary-scope implementation error, not a model or API failure. The runtime now derives cyclic pairing from the complete external report, and the guarded retry removes only the one invalid-context cache. |
| 2026-09-13 | 9.5 | Corrected other-question answer smoke | Passed; complete answer chain authorized | The corrected 20-qid run matches every frozen selection context and completes all answers under the registered V4-Flash protocol. It has no empty/error answers, invalid caches, context mismatches, or audit failures. Smoke EM/F1 are execution diagnostics only. |
| 2026-09-14 | 9.5 | First complete-answer-chain attempt | Interrupted; metadata-priority fix and resume authorized | Online-state reuse and the complete other-question/query-only reports passed audit. Before frozen-initial generation, cache preparation rejected propagated files because newly available query-only reports were ordered ahead of the earlier other-question source, changing only `source_report` provenance. The source priority is now frozen to the execution order, so later reports can fill missing contexts but cannot supersede prior provenance. No completed answer report is invalidated and cached answers are retained. |
| 2026-09-14 | 9.5 | Complete five-condition answers | Passed; offline finalization authorized | All five conditions contain 3,000 valid caches and clean answer audits, and the sequential chain reached `STAGE9_5_ALL_STATE_ANSWERS_OK`. The remaining work is deterministic standard-metric replay and 10,000-sample paired qid bootstrap; no further answer call is authorized. |
| 2026-09-14 | 9.5 | Final downstream state intervention | Passed; Stage 9.5 closed | Correct online state significantly exceeds query-only, frozen-initial, and other-question state on all eight registered downstream metrics. Relative to query-only, it gains 0.0336 Answer F1, 0.1288 Supporting-Fact F1, 0.1241 Joint F1, 0.1127 full support, and 0.0570 Closure@10; every paired interval excludes zero. The corresponding gains over frozen and mismatched states are also all significant. Previous-only is tied on Support F1/Closure and slightly better on Answer/Joint/full support, so the decision is `STATE_RELEVANCE_DOWNSTREAM_SUPPORTED`, not `FULL_HISTORY_SUPERIOR`. Exact accounting records 7,924 fresh and 7,076 reused answers. |
| 2026-09-14 | 9.6 | Breadth-readiness instrumentation | Prepared; server audit authorized | The no-run audit discovers official MuSiQue-Ans dev data, validates answerable rows and decomposition-to-support mappings, checks paragraph-budget feasibility and frozen v27 prerequisites, and records that MuSiQue evidence metrics are paragraph-level. It also verifies that the current answer runtime is not yet an independent non-DeepSeek backend. The audit performs no download, training, GPU inference, or API call. |
| 2026-09-14 | 9.6 | Initial breadth-readiness audit | Blocked only on official MuSiQue data | All plans, evaluators, the frozen v27 checkpoint, and DeBERTa/BGE model assets are present. No MuSiQue-Ans dev file or archive was found, and the current DeepSeek-specific runtime does not qualify as an independent second-generator branch. The next authorized action is official MuSiQue v1.0 data acquisition followed by the same no-run schema audit; no model run or API call is authorized. |
| 2026-09-14 | 9.6 | MuSiQue data readiness | Passed; deterministic adapter build authorized | The official MuSiQue-Ans dev file contains 2,417/2,417 schema-valid answerable rows, no duplicate ids or exclusions, and exactly matched 2/3/4-hop decomposition and support counts. All rows support budgets 10/15, 2,401 support 20, and none support 50. Compact-10 is primary, Balanced-15 is optional, and Recall-50 is prohibited. The next action builds and independently audits a fixed 1,000-qid paragraph-unit subset without training, GPU inference, or API calls. |
| 2026-09-14 | 9.6 | MuSiQue paragraph adapter | Passed; bounded Compact selection smoke authorized | Deterministic construction and independent replay agree on 1,000 qids, 2,629 states, 19,995 paragraph units, source/qid hashes, candidate pools, and ordered paragraph targets. The policy-facing files contain zero teacher-only decomposition keys. Candidate pools contain 15--20 units after acquired-support removal, so Compact-10 is feasible throughout. One seed-42 20-qid selection-only smoke is authorized; answer generation remains locked. |
| 2026-09-14 | 9.6 | Initial Compact smoke audit | Audit-only mismatch; offline re-audit authorized | Runtime completed 20 qids/40 states with no skips, answer calls, or missing online states, but the checker expected the first 20 physical query rows (55 states). The runtime actually uses lexicographically sorted grouped sample qids before applying `max_qids`. The raw GPU report is retained and only the corrected offline audit may be rerun. |
| 2026-09-14 | 9.6 | Corrected Compact selection smoke | Passed; complete four-method selection authorized | The retained runtime report passes the corrected qid/target audit on 20 qids and 40 paragraph states, with no skips, answer outputs, non-paragraph units, or missing online states. Compact paragraph Alignment@1/5 is 0.6500/0.9500 and full paragraph/title coverage is 0.9000/0.9000; these are smoke diagnostics only. Full selection is pre-registered for Compact-10, Balanced-15, Hybrid, and BGE-Reranker on all 1,000 qids; Recall-50 remains prohibited. |
| 2026-09-14 | 9.6 | Complete MuSiQue zero-shot selection | Passed; paired selection Bootstrap authorized | All four methods complete the same 1,000 qids/2,629 paragraph states with identical qid/target hashes, no skips, and no answer output. Compact/Balanced paragraph Alignment@5 is 0.8186/0.8383 versus 0.6946 Hybrid and 0.7155 BGE; full paragraph coverage is 0.700/0.717 versus 0.436/0.669. Compact and Balanced exceed BGE on all four point estimates, while costing 1,474.97/1,775.28 ms per qid versus 1,010.21 ms. Significance and answer claims remain locked pending offline paired Bootstrap and cache audit. |
| 2026-09-15 | 9.6 | MuSiQue paired selection Bootstrap | Passed; exact-context cache audit authorized | The 10,000-sample paired qid-cluster Bootstrap passed on all 1,000 qids. Balanced exceeds Hybrid and BGE on all four paragraph metrics with intervals excluding zero. Compact exceeds Hybrid on all four and BGE on Alignment@1/5 and full-title coverage; Compact's +0.031 full-paragraph gain over BGE has CI [0.000, 0.063] and is treated as borderline rather than strictly significant. Balanced versus Compact is tied on Alignment@1 but significantly improves the other three metrics at higher latency. The next action is a no-API cache audit with canonical-plus-alias scoring locked for 287 alias-bearing qids. |
| 2026-09-15 | 9.6 | MuSiQue exact-context cache audit | Passed; bounded Compact answer smoke authorized | All 4,000 method--qid targets and 1,000 query records passed, including 287 alias-bearing qids. No historical MuSiQue answer context is reusable. Cross-method exact-context deduplication removes 242 duplicate target files, reducing the theoretical fresh requirement from 4,000 to 3,758 calls. The answer scorer now maximizes over the canonical answer and aliases without changing historical single-reference datasets. One 20-qid Compact answer smoke is authorized; complete answers remain locked. |
| 2026-09-15 | 9.6 | Compact answer smoke | Passed; sequential complete answer chain authorized | All 20 frozen Compact contexts and caches passed, including five alias-bearing qids and independent per-qid score replay. There were 20 fresh answers, zero invalid/error answers, and zero context or alias mismatches. Smoke EM/F1 0.7500/0.8167 are diagnostics only. Complete answers must run Compact, Balanced, Hybrid, then BGE sequentially with exact-context propagation; concurrent execution is prohibited because it would discard the 242-context saving. |
| 2026-09-15 | 9.6 | First complete-answer-chain attempt | Interrupted by one API error; surgical Hybrid repair authorized | Compact and Balanced passed 1,000-qid audits. Hybrid wrote all 1,000 rows and reproduced the frozen evidence metrics, but one fresh answer exhausted retries, yielding exactly one empty/error answer; its other 999 caches are valid. BGE was not started. The guarded repair archives the failed report, audit, log, pid, and one bad cache while preserving 999 valid Hybrid caches, then the sequential chain may resume. No wholesale rerun or cache deletion is allowed. |
| 2026-09-15 | 9.6 | Complete four-method MuSiQue answers | Passed; offline finalization authorized | The surgical Hybrid retry and resumed BGE run closed all 4,000 method--qid targets with clean protocol and cache audits. Compact/Balanced/Hybrid/BGE used 1,000/759/999/1,000 fresh answers and 0/241/1/0 exact-context reuses, totaling exactly 3,758 fresh calls and 242 reuses with no raw-answer disagreements or unresolved targets. No further answer call is authorized; final canonical-plus-alias downstream metrics and paired Bootstrap must now be computed offline. |

## Historical authorized action before answer-chain completion

Stages 9.1--9.4 are closed. Stage 9.2 supports a replicated targeted anti-
reselection effect but not a broad end-to-end gain. Stage 9.3 establishes
same-generator final-protocol superiority in evidence, joint, coverage, and
closure metrics, with Recall also significantly improving answer quality over
the strongest BGE-Reranker baseline. Stage 9.4 establishes zero-shot 2Wiki
transfer: Compact and Balanced significantly improve evidence and joint
metrics over BGE while tying on answer metrics, and Recall significantly
improves all eight registered metrics. Stage 9.5 readiness and the five-
condition smoke and complete 3,000-qid selection have passed. Correct online
state significantly outperforms query-only, frozen-initial, and mismatched
other-question states, but does not outperform previous-evidence-only. The
cache audit also passed: 6,126 of 15,000 targets are immediately reusable and
cross-condition propagation reduces 8,874 unresolved target files to 7,924
unique fresh contexts. The initial other-question answer smoke was rejected
because its final boundary qid used a 20-qid rather than 3,000-qid cyclic
pairing universe; after correcting that scope, all 20 contexts and answers pass
audit. The first complete-chain attempt finished and audited online-state,
other-question, and query-only outputs, then stopped before frozen-initial
generation because dynamic source ordering changed propagated provenance. The
priority is now frozen to execution order. The resumed chain completed all
15,000 condition--qid targets with clean audits. Offline finalization confirms
significant gains on all eight downstream metrics over query-only, frozen, and
mismatched states, while previous-only remains tied or slightly stronger. Stage
9.5 is closed as `STATE_RELEVANCE_DOWNSTREAM_SUPPORTED`. The next authorized
action is the no-run Stage 9.6 breadth-readiness audit now implemented in
`scripts/run_kbs_stage9_breadth.sh`. MuSiQue is preferred only if its official
question-local paragraphs and decomposition-to-support mapping pass the
registered checks. No download, training, GPU inference, or API call is
authorized by this audit. The initial server audit passed every runtime check
but found no MuSiQue data asset. Official MuSiQue v1.0 data acquisition and a
repeat of the no-run audit were authorized. The repeated audit passed as
`READY_MUSIQUE`: all 2,417 dev rows are schema-valid, while candidate budget 50
is impossible. Deterministic construction and independent replay audit of the
fixed 1,000-qid paragraph-unit subset are now authorized; all model and answer
runs remain locked. The adapter audit subsequently passed with exact replay and
zero teacher-key leakage. The retained 20-qid Compact report passes the
corrected audit with 40 expected states and no protocol failures. The next
authorized four-method selection-only evaluation has now passed on all 1,000
qids. Both KSG-EA operating points exceed Hybrid and BGE on every registered
paragraph-level evidence point estimate, with the expected latency cost. The
registered 10,000-sample paired qid-cluster Bootstrap has now passed. Balanced
significantly exceeds both baselines on all four paragraph metrics; Compact
significantly exceeds Hybrid on all four and BGE on three, with the full-
paragraph interval touching zero. The exact-context answer-cache audit has also
passed under canonical-plus-alias scoring: no historical answer is reusable,
while 242 exact cross-method duplicates reduce 4,000 target files to 3,758
unique fresh contexts. Recall-50 remains unsupported. The next authorized
Compact-10 answer smoke has now passed all 20 qids, including five alias-
bearing cases, with exact context, cache, and score replay. The next authorized
Compact and Balanced complete reports have passed. The first chain attempt
stopped after Hybrid produced one exhausted API retry among 1,000 rows; BGE was
not started. The next authorized action is a no-API surgical repair that
archives only the bad Hybrid cache and failed derived artifacts while retaining
999 valid Hybrid answers. The guarded sequential chain may then resume, skip
Compact/Balanced, repair Hybrid with one answer call, and continue to BGE. No
scientific answer claim is allowed until all 4,000 targets pass final offline
audit and paired Bootstrap.

## Current authorized action

Stages 9.1--9.5 are closed. Stage 9.6 has completed its fixed MuSiQue paragraph
adapter, four-method selection, selection Bootstrap, answer-cache audit,
bounded answer smoke, and all 4,000 final answer targets. The final cache ledger
contains exactly 3,758 fresh answers and 242 exact-context reuses, with zero
unresolved targets or answer disagreements. Recall-50 remains unsupported
because MuSiQue memories contain at most 20 paragraphs.

The only authorized next action is `ACTION=finalize_answers` in
`scripts/run_kbs_stage9_breadth.sh`. It performs canonical-plus-alias standard-
metric replay and a 10,000-sample paired qid Bootstrap entirely offline. It
must not receive an API key and does not start training or GPU inference. No
final MuSiQue answer, joint, or closure claim may be written into the paper
until those artifacts pass and are reviewed.
