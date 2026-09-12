# Stage 9.4: Final-Policy 2Wiki Zero-Shot Transfer

## Current status

- Status: readiness implementation prepared; no Stage 9.4 run has started.
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

Run readiness only. If the data, three checkpoints, model assets, selector
implementations, and proposed output paths pass, authorize one 20-qid
selection-only smoke for seed-42 Compact/Balanced/Recall and the two strong
baselines. No answer API call is authorized at readiness.
