# Stage 9.4: Final-Policy 2Wiki Zero-Shot Transfer

## Current status

- Status: closed with `FINAL_POLICY_2WIKI_COMPLETE`.
- Primary model: v27 seed 42, trained only on HotpotQA.
- Robustness models: v27 seeds 43 and 44, selection-only until reviewed.
- Transfer dataset: fixed 1,000-qid 2Wiki question-local evaluation memory.

## Registered comparison

| Family | Operating point | Candidate budget | Front pool | Answer generation |
|---|---|---:|---:|---|
| KSG-EA seed 42 | Compact | 10 | 30 | required after selection audit |
| KSG-EA seed 42 | Balanced knee | 15 | 30 | required after selection audit |
| KSG-EA seed 42 | Recall | 50 | 50 | required after selection audit |
| KSG-EA seeds 43/44 | all three points | 10/15/50 | 30/30/50 | selection only initially |
| Hybrid-RAG | native baseline | 8 | native | required after selection audit |
| BGE-Reranker-RAG | native baseline | 8 | native | required after selection audit |

All KSG-EA checkpoints remain HotpotQA-trained and receive no 2Wiki-specific
fine-tuning. The answer protocol remains V4-Flash, thinking disabled,
temperature 0, and `kbs_extractive_answer_json_v1`. Historical v22 transfer
reports cannot be reported as final v27 results; they may be used only as
exact-context cache sources after a complete protocol audit.

## Readiness gate

The readiness audit passed with no failures, training, inference, or API calls.
The fixed 2Wiki set contains 1,000 qids, 2,416 states, and 31,522 memory units;
sample and query qid sets match exactly. All three v27 checkpoints and all
required model and selector implementations are present.

The historical v22 Compact and Recall reports match the frozen answer protocol
and may be considered only for exact ordered-context cache reuse. They remain
ineligible as final v27 results because their checkpoint and state-write
protocol differ. Legacy Hybrid, BGE, and Gold reports also lack the frozen
generator metadata and cannot seed answer caches.

## Selection smoke

All five methods completed the same 20 qids and 50 teacher states. Ordered-qid
and teacher-target hashes match exactly, with no skipped states, failures, or
API calls.

| Method | Step@1 | Step@5 | Full doc | Full unit |
|---|---:|---:|---:|---:|
| KSG-EA-Compact | 0.40 | 0.74 | 0.70 | 0.55 |
| KSG-EA-Balanced | 0.44 | 0.76 | 0.70 | 0.60 |
| KSG-EA-Recall | 0.46 | 0.82 | 0.85 | 0.70 |
| Hybrid-RAG | 0.24 | 0.60 | 0.35 | 0.30 |
| BGE-Reranker-RAG | 0.28 | 0.76 | 0.70 | 0.65 |

These values validate execution only and are not scientific estimates. The
next gate is the complete 1,000-qid selection-only evaluation for all three
operating points and all three v27 seeds, plus Hybrid and BGE-Reranker.

## Complete selection results

All 11 reports completed the same 1,000 qids and 2,416 teacher states with
identical ordered-qid and teacher-target hashes, zero skipped states, and no
API calls. The operating-point ordering is stable for every seed.

| Operating point | Step@1 mean (SD) | Step@5 mean (SD) | Full doc mean (SD) | Full unit mean (SD) |
|---|---:|---:|---:|---:|
| Compact | 0.4884 (0.0075) | 0.7689 (0.0071) | 0.7893 (0.0061) | 0.6103 (0.0076) |
| Balanced | 0.5229 (0.0033) | 0.8005 (0.0090) | 0.8297 (0.0110) | 0.6623 (0.0118) |
| Recall | 0.5708 (0.0033) | 0.8462 (0.0138) | 0.8827 (0.0124) | 0.7500 (0.0187) |

The pre-registered seed-42 primary rows and final baselines are:

| Method | Step@1 | Step@5 | Full doc | Full unit | Selection ms/qid |
|---|---:|---:|---:|---:|---:|
| KSG-EA-Compact | 0.4946 | 0.7769 | 0.796 | 0.619 | 804.35 |
| KSG-EA-Balanced | 0.5265 | 0.8100 | 0.842 | 0.676 | 937.33 |
| KSG-EA-Recall | 0.5745 | 0.8613 | 0.897 | 0.770 | 1168.51 |
| Hybrid-RAG | 0.2442 | 0.6035 | 0.505 | 0.321 | 358.46 |
| BGE-Reranker-RAG | 0.3593 | 0.6643 | 0.740 | 0.565 | 694.54 |

Relative to BGE-Reranker, seed-42 Compact/Balanced/Recall improve Step@5 by
0.1126/0.1457/0.1970 and full-unit coverage by 0.054/0.111/0.205. These are
selection-only observations; answer, support, joint, closure, and paired-CI
claims remain pending.

## Next gate

The frozen v22 Compact and Recall reports provide 1,919 eligible unique source
contexts. Exact historical reuse covers only 19/18/10/1/0 targets for
Compact/Balanced/Recall/Hybrid/BGE. Of the remaining 4,952 target files, 521
are duplicates across final methods; deterministic cross-method propagation
therefore lowers the theoretical fresh-call requirement to 4,431. No protocol
differences, answer disagreements, invalid caches, or audit failures were
found.

Run one bounded 20-qid answer smoke for seed-42 Compact. Complete answer
generation remains locked until this smoke validates both the frozen context
and cache records.

## Answer smoke

The seed-42 Compact smoke completed 20/20 qids with the frozen generator
protocol and no failures. It exercised both cache paths: one exact historical
context was reused and 19 answers were freshly generated. All 20 cache files
matched the frozen selection contexts and protocol metadata. Smoke Answer
EM/F1 is 0.6500/0.7207, but these 20-qid values are execution diagnostics and
are not scientific estimates.

The complete sequential five-method answer chain is now authorized. It must
re-run exact-context propagation before each method and preserve the fixed
method order; final downstream claims remain locked until all reports and
paired confidence intervals pass offline finalization.

## Complete answer reports

All five final methods completed 1,000 qids with no answer, protocol, context,
or cache failures. The chain used exactly 4,431 fresh answers and 569 exact
context reuses, matching all 5,000 method--qid targets.

| Method | Answer EM | Answer F1 | Full unit | Avg. API tokens | API latency (s/qid) |
|---|---:|---:|---:|---:|---:|
| KSG-EA-Compact | 0.609 | 0.6711 | 0.619 | 384.05 | 0.680 |
| KSG-EA-Balanced | 0.631 | 0.6999 | 0.676 | 385.29 | 0.683 |
| KSG-EA-Recall | 0.658 | 0.7330 | 0.770 | 388.79 | 0.666 |
| Hybrid-RAG | 0.497 | 0.5475 | 0.321 | 330.31 | 0.649 |
| BGE-Reranker-RAG | 0.619 | 0.6912 | 0.565 | 384.25 | 0.669 |

## Final downstream metrics

Offline finalization completed with no API calls or audit failures. All five
methods contain the same 1,000 qids, and all paired intervals use 10,000
question-level bootstrap samples.

| Method | Answer F1 | Support F1 | Support EM | Joint F1 | Joint EM | Full support | ClosureSuccess |
|---|---:|---:|---:|---:|---:|---:|---:|
| KSG-EA-Compact | 0.6711 | 0.5445 | 0.278 | 0.4069 | 0.212 | 0.619 | 0.463@10 |
| KSG-EA-Balanced | 0.6999 | 0.5734 | 0.308 | 0.4372 | 0.234 | 0.676 | 0.502@15 |
| KSG-EA-Recall | 0.7330 | 0.6209 | 0.360 | 0.4813 | 0.269 | 0.770 | 0.563@50 |
| Hybrid-RAG | 0.5475 | 0.2853 | 0.026 | 0.1893 | 0.022 | 0.321 | 0.259 |
| BGE-Reranker-RAG | 0.6912 | 0.4450 | 0.061 | 0.3162 | 0.046 | 0.565 | 0.439 |

Against Hybrid-RAG, every registered downstream metric is significantly
higher for all three KSG-EA operating points. Against BGE-Reranker, Compact is
tied on Answer EM/F1 and ClosureSuccess@10 but significantly improves Support
F1 by 0.0995 [0.0774, 0.1214], Joint F1 by 0.0907 [0.0684, 0.1130], and full
support coverage by 0.054 [0.023, 0.085]. Balanced is tied on Answer EM/F1 and
significantly improves the remaining six metrics. Recall significantly
improves all eight metrics; its BGE-relative deltas include +0.0418 Answer F1
[0.0196, 0.0642], +0.1760 Support F1 [0.1546, 0.1973], +0.1650 Joint F1
[0.1437, 0.1867], +0.205 full support [0.174, 0.235], and +0.124
ClosureSuccess@50 [0.097, 0.152].

The final decision is `FINAL_POLICY_2WIKI_COMPLETE`. The defensible claim is
zero-shot policy transfer with strong evidence and joint-quality gains;
Compact and Balanced do not establish answer-quality superiority over BGE.
