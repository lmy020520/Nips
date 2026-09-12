# Stage 9.4: Final-Policy 2Wiki Zero-Shot Transfer

## Current status

- Status: five-method selection smoke passed; full multiseed selection authorized.
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

## Next gate

The readiness audit passed with no failures, training, inference, or API calls.
The fixed 2Wiki set contains 1,000 qids, 2,416 states, and 31,522 memory units;
sample and query qid sets match exactly. All three v27 checkpoints and all
required model and selector implementations are present.

The historical v22 Compact and Recall reports match the frozen answer protocol
and may be considered only for exact ordered-context cache reuse. They remain
ineligible as final v27 results because their checkpoint and state-write
protocol differ. Legacy Hybrid, BGE, and Gold reports also lack the frozen
generator metadata and cannot seed answer caches.

Run one 20-qid selection-only smoke for seed-42 Compact/Balanced/Recall,
Hybrid, and BGE-Reranker. No answer API call is authorized at this gate.

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
