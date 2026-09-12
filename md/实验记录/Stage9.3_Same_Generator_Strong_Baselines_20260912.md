# Stage 9.3: Same-Generator Strong Baselines

## Current status

- Status: closed as `FINAL_PROTOCOL_BASELINES_COMPLETE`.
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

## Exact-context cache audit

All ten available source reports pass the complete frozen answer protocol,
providing 15,143 unique eligible source contexts. Exact reuse remains limited
for the baseline targets:

| Method | Directly reusable | Fresh without cross-method propagation |
|---|---:|---:|
| BM25-RAG | 3 | 2,997 |
| Dense-RAG | 14 | 2,986 |
| Hybrid-RAG | 43 | 2,957 |
| Iterative-Hybrid-RAG | 45 | 2,955 |
| BGE-Reranker-RAG | 6 | 2,994 |

Across 15,000 target reports, 111 exact-context answers were written into the
new caches. The naive remaining requirement is 14,889 target answers. There
are 1,050 duplicate target files across baseline methods, so deterministic
cross-method propagation can reduce the theoretical number of fresh contexts
to 13,839.

There are 413 duplicate eligible source contexts whose raw API strings differ.
The audit keeps the pre-registered source priority and never selects an answer
by its measured correctness. This preserves protocol validity despite residual
API nondeterminism at temperature zero.

## Answer smoke

The registry-first BM25 method completed the frozen 20-qid answer smoke. All
20 answer caches match the frozen selection prefix and protocol; all were
fresh calls, with zero empty/error answers and zero invalid cache entries.
Smoke EM/F1 are 0.5500/0.6429, average API tokens are 367.05, and average API
latency is 0.739 seconds. These are execution diagnostics only.

## Complete answer reports

All five methods completed the frozen 3,000-qid answer protocol with no empty
answers, answer errors, context mismatches, or invalid cache records.

| Method | Answer EM | Answer F1 | Avg API tokens | Avg API latency (s) | Exact reuse | Fresh calls |
|---|---:|---:|---:|---:|---:|---:|
| BM25-RAG | 0.5373 | 0.6686 | 357.30 | 0.788 | 3 | 2,997 |
| Dense-RAG | 0.5763 | 0.7046 | 363.61 | 0.782 | 15 | 2,985 |
| Hybrid-RAG | 0.5897 | 0.7185 | 362.84 | 0.761 | 66 | 2,934 |
| Iterative-Hybrid-RAG | 0.5813 | 0.7114 | 359.22 | 0.718 | 1,071 | 1,929 |
| BGE-Reranker-RAG | **0.6290** | **0.7709** | 421.68 | 0.731 | 6 | 2,994 |

The chain made exactly 13,839 fresh API calls, matching the pre-run
cross-baseline deduplication estimate. The remaining 1,161 target reports used
exactly matched contexts: 111 were available before the chain and 1,050 were
propagated across completed baselines. The final cache audit covers all 15,000
method--qid targets. These answer-only point estimates do not replace the
registered supporting-fact, joint, closure, and paired-bootstrap analysis.

## Standard downstream results

The offline evaluator reproduced all source answer and coverage summaries,
found 3,000 identical qids for every method, and verified the Gold Oracle at
1.0 Supporting-Fact recall and EM. The table reports final-protocol results.

| Method | Answer F1 | SF F1 | SF EM | Joint F1 | Joint EM | Full support | Closure |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25-RAG | 0.6686 | 0.3948 | 0.0543 | 0.2888 | 0.0373 | 0.3697 | 0.2603 @10 |
| Dense-RAG | 0.7046 | 0.4407 | 0.0937 | 0.3344 | 0.0667 | 0.5700 | 0.3983 @10 |
| Hybrid-RAG | 0.7185 | 0.4714 | **0.0970** | 0.3581 | **0.0667** | 0.5810 | 0.4020 @10 |
| Iterative-Hybrid-RAG | 0.7114 | 0.4614 | 0.0927 | 0.3464 | 0.0630 | 0.5810 | 0.3990 @10 |
| BGE-Reranker-RAG | **0.7709** | **0.4756** | 0.0570 | **0.3756** | 0.0400 | **0.7353** | **0.4757 @10** |
| KSG-EA-Compact | 0.7678 | 0.6561 | 0.3623 | 0.5293 | 0.2520 | 0.7823 | 0.5153 @10 |
| KSG-EA-Recall | **0.7904** | **0.7113** | **0.4340** | **0.5794** | **0.3030** | **0.8730** | **0.5873 @50** |

Bold within the first five rows denotes the strongest conventional baseline;
bold KSG-EA-Recall values denote the strongest overall operating point. The
Closure column uses each KSG operating point's registered budget, so Compact
comparisons use ClosureSuccess@10 and Recall comparisons use
ClosureSuccess@50.

## Paired uncertainty and conclusion

Each comparison uses 10,000 paired qid-bootstrap samples. KSG-EA-Compact is
significantly higher than BM25, Dense, Hybrid, and Iterative-Hybrid on all
eight registered metrics. Against the strongest BGE-Reranker baseline,
Compact Answer EM/F1 are statistically indistinguishable: `-0.0080`
`[-0.0207,0.0043]` and `-0.0031` `[-0.0140,0.0078]`. In contrast, Compact
significantly improves Supporting-Fact F1 by `+0.1805`
`[0.1683,0.1928]`, Joint F1 by `+0.1537` `[0.1415,0.1656]`, full support
coverage by `+0.0470` `[0.0307,0.0633]`, and ClosureSuccess@10 by `+0.0397`
`[0.0243,0.0550]`.

KSG-EA-Recall is significantly higher than every baseline on all 40 registered
metric--baseline comparisons. Against BGE-Reranker specifically, Recall gains
`+0.0195` Answer F1 `[0.0089,0.0302]`, `+0.2356` Supporting-Fact F1
`[0.2239,0.2473]`, `+0.2038` Joint F1 `[0.1921,0.2157]`, `+0.1377` full
support coverage `[0.1227,0.1533]`, and `+0.0847` ClosureSuccess@50
`[0.0697,0.0997]`.

The final decision is `FINAL_PROTOCOL_BASELINES_COMPLETE`. These results
support the paper's core compiled-evidence claim: Compact preserves answer
quality relative to the strongest reranker while materially improving support
and joint quality, whereas Recall provides a statistically reliable quality
gain at the higher-cost operating point. The result does not imply that
Compact dominates BGE-Reranker on answer-only metrics.
