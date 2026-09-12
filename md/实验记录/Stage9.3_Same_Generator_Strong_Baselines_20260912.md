# Stage 9.3: Same-Generator Strong Baselines

## Current status

- Status: 20-qid selection-only smoke passed; full selection authorized.
- Readiness date: 2026-09-12.
- API calls during readiness: 0.
- GPU inference runs during readiness: 0.
- Evaluation data: 3,000 HotpotQA qids, 7,296 states, and 124,523 memory units.

## Frozen comparison protocol

| Item | Frozen value |
|---|---|
| Answer model | `deepseek-v4-flash` |
| Thinking mode | `disabled` |
| Answer mode | `json` |
| Temperature | 0.0 |
| Prompt version | `kbs_extractive_answer_json_v1` |
| Final evidence count | Top 5 |

The registered comparison contains BM25-RAG, Dense-RAG, Hybrid-RAG,
Iterative-Hybrid-RAG, and BGE-Reranker-RAG. All five selector implementations
are present. The final runs must use the same ordered 3,000 qids and the same
question-local candidate memory.

## Readiness audit

| Method | Selector | Candidate top-k | Required local model | Historical report reusable as final? |
|---|---|---:|---|---|
| BM25-RAG | `bm25` | 8 | none | no |
| Dense-RAG | `dense` | 8 | BGE-large | no |
| Hybrid-RAG | `hybrid` | 8 | BGE-large | no |
| Iterative-Hybrid-RAG | `iterative_hybrid` | 8 | BGE-large | no |
| BGE-Reranker-RAG | `generic_reranker` | 8 | BGE-reranker-large | no |

The historical reports exist, but they do not record the current answer-model,
thinking-mode, temperature, prompt-version, or answer-generation fields.
Consequently, they remain descriptive evidence only and cannot be silently
reused as final-protocol reports. Exact-context cache reuse may be considered
only after the newly generated selection contexts and the complete answer
protocol have passed an explicit audit.

## Selection smoke

All five methods completed the same 20 qids and 50 teacher states. Ordered-qid
and ordered-teacher-target SHA-256 values match across every report. No state
was skipped, no answer was generated, and no API call was made.

| Method | Step@1 | Step@5 | Full doc coverage | Full unit coverage |
|---|---:|---:|---:|---:|
| BM25-RAG | 0.26 | 0.58 | 0.70 | 0.25 |
| Dense-RAG | 0.28 | 0.70 | 0.60 | 0.45 |
| Hybrid-RAG | 0.34 | 0.76 | 0.70 | 0.45 |
| Iterative-Hybrid-RAG | 0.32 | 0.76 | 0.70 | 0.45 |
| BGE-Reranker-RAG | 0.24 | 0.64 | 0.95 | 0.60 |

These small-sample values are execution diagnostics only. They are not used
for method selection, scientific claims, or paper tables.

## Full selection results

All five methods completed the same 3,000 qids and 7,296 teacher states with
identical ordered-qid and teacher-target hashes, zero skipped states, and no
answer/API calls.

| Method | Step@1 | Step@5 | Full doc | Full unit | Selection ms/qid | QID/s | Peak GPU MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25-RAG | 0.3261 | 0.6473 | 0.6230 | 0.3697 | 9.00 | 111.126 | 0.00 |
| Dense-RAG | 0.3529 | 0.7560 | 0.7523 | 0.5700 | 484.05 | 2.066 | 2018.61 |
| Hybrid-RAG | **0.3912** | **0.7767** | 0.7877 | 0.5810 | 492.05 | 2.032 | 2018.61 |
| Iterative-Hybrid-RAG | 0.3458 | 0.7758 | 0.7807 | 0.5810 | 484.85 | 2.063 | 2018.61 |
| BGE-Reranker-RAG | 0.3355 | 0.7166 | **0.9217** | **0.7353** | 1011.14 | 0.989 | 2500.20 |

Hybrid has the strongest teacher-alignment values, whereas BGE-Reranker has
the strongest complete evidence coverage. This tradeoff must be resolved with
the frozen downstream answer, support, joint, and closure metrics; selection
metrics alone do not establish the final method ordering.

## Authorized next gate

Audit exact-context reuse from existing reports that contain the complete
frozen V4-Flash answer protocol. Report both the naive fresh target count and
the unique count after cross-baseline context deduplication.

Do not generate answers until the cache-readiness report has been reviewed.
