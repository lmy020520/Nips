# Stage 9.4: Final-Policy 2Wiki Zero-Shot Transfer

## Current status

- Status: exact-context cache audit passed; bounded answer smoke authorized.
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
