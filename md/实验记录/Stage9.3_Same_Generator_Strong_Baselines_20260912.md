# Stage 9.3: Same-Generator Strong Baselines

## Current status

- Status: readiness passed; 20-qid selection-only smoke authorized.
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

## Authorized next gate

Run the five methods on the same 20 qids with answer generation disabled. The
smoke must verify selectors, candidate and selection budgets, state-update
behavior, ordered qids, teacher targets, zero skipped states, and zero answer
calls. Smoke metric values validate execution only and are not paper results.

Do not start the 3,000-qid selection runs or any answer generation until the
smoke summary has been reviewed.
