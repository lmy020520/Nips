# Stage 9.4: Final-Policy 2Wiki Zero-Shot Transfer

## Current status

- Status: readiness passed; five-method selection smoke authorized.
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
