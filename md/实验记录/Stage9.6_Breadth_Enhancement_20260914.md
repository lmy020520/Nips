# Stage 9.6: One Breadth Enhancement

## Current status

- Status: closed as `MUSIQUE_ZERO_SHOT_BREADTH_SUPPORTED`.
- Authorized action: integrate the frozen Stage 9 results into the paper.
- Preferred branch: MuSiQue zero-shot transfer when its official development
  data and paragraph-level support mapping pass the registered checks.
- Alternative branch: one non-DeepSeek generator replication only when an
  independent backend and frozen protocol already exist.

## MuSiQue transfer contract

The proposed evaluation uses the official MuSiQue-Ans development split. One
official paragraph is one evidence unit, and the question-local candidate
memory consists only of the paragraphs supplied with that question. Gold
units are paragraphs marked `is_supporting`; their acquisition order is read
from `question_decomposition.paragraph_support_idx`. Decomposition question
text and answers are evaluation metadata and must never be exposed to the
retrieval policy.

The v27 Student remains frozen after HotpotQA training, with no MuSiQue
fine-tuning. Compact with candidate budget 10 is the primary operating point.
Balanced is allowed only if at least 1,000 schema-valid questions contain 15
paragraphs. Recall-50 must not be fabricated when question-local memories
contain fewer than 50 paragraphs.

## Metric boundary

MuSiQue publishes paragraph-level supporting annotations rather than the
HotpotQA/2Wiki sentence indices used by the current evaluator. MuSiQue evidence
alignment and coverage must therefore be labelled paragraph-level. They cannot
be presented as directly identical to the sentence-level Step@5 values in the
main HotpotQA and 2Wiki tables.

## Readiness gate

The audit must verify all of the following before an adapter is implemented:

1. The official MuSiQue-Ans development file is present and readable.
2. At least 1,000 answerable rows contain valid questions, answers, paragraphs,
   supporting flags, and decomposition-to-paragraph mappings.
3. Paragraph indices are unique, every decomposition support index exists, and
   decomposition supports match the `is_supporting` set one-to-one.
4. At least 1,000 valid rows support candidate budget 10.
5. MuSiQue qids do not overlap with the available HotpotQA train/evaluation
   qids, and all frozen v27 runtime/model prerequisites are present.

The readiness action performs no download, training, GPU inference, or API
call. A missing dataset produces a structured blocked report rather than an
experiment failure.

## Next gate

If the report returns `READY_MUSIQUE`, implement and separately audit the
deterministic paragraph-unit adapter. Do not begin selection inference until
the generated qids, memory, gold mapping, candidate sizes, and split manifest
have passed that second audit. If neither MuSiQue nor an independent generator
is ready, Stage 9.6 remains blocked rather than substituting another DeepSeek
service tier.

## Initial readiness result

The server audit returned `BLOCKED_MISSING_PREREQUISITES` without performing
any download, training, GPU inference, or API call. Every registered runtime
prerequisite passed: the experiment plans, Policy-RAG runtime, standard metric
and bootstrap evaluators, frozen v27 checkpoint, DeBERTa model, and BGE model
are present. The sole MuSiQue-branch failure is that no official MuSiQue-Ans
development JSONL or archive was found.

The alternative generator branch is also not currently admissible. The answer
runtime remains DeepSeek-specific and no independent non-DeepSeek adapter or
frozen protocol is registered. Consequently, the authorized next action is
data acquisition only: download and extract the official `musique_v1.0.zip`,
then rerun the same no-run readiness audit. No adapter implementation,
selection inference, or answer generation is authorized yet.

## Successful data readiness

After the official archive was provided, the repeated audit returned
`READY_MUSIQUE`. All 2,417 MuSiQue-Ans dev rows are answerable and schema-valid,
with no exclusions or duplicate ids. The split contains 1,252 two-hop, 760
three-hop, and 405 four-hop questions, and the support-count distribution is
identical to the decomposition-hop distribution. There is no observed qid
overlap with the available HotpotQA evaluation qids.

All 2,417 rows support candidate budgets 10 and 15; 2,401 support budget 20,
and none support budget 50. Stage 9.6 therefore registers Compact-10 as the
primary point and may use Balanced-15 as a secondary point, but Recall-50 is
prohibited. The next authorized action is deterministic construction and
independent replay audit of a fixed 1,000-qid paragraph-unit subset. This
action remains CPU-only and makes no API call.

## Adapter audit

The deterministic adapter and independent replay audit passed on the server.
The fixed subset contains 1,000 qids, 2,629 trajectory states, and 19,995
paragraph units. It preserves 287 rows with answer aliases. The generated
source SHA-256 and ordered-qid SHA-256 match independent replay, all candidate
pools and ordered positive targets are reproducible, and no teacher-only
decomposition key appears in policy-facing queries or samples.

Candidate pools range from 15 to 20 units after previously acquired teacher
supports are removed. Compact-10 is therefore feasible for every state. The
next authorized action is one seed-42, 20-qid Compact selection-only smoke
under the frozen v27 protocol. Its evidence metrics must be labelled paragraph
alignment and paragraph coverage. No answer generation is authorized.

## Initial smoke-audit correction

The first smoke audit returned `FAIL` even though the runtime completed 20
qids and 40 states with no skipped states, answer calls, or missing online
states. All four reported failures had one audit-only cause: the runtime sorts
the grouped sample qids lexicographically before applying `max_qids=20`, while
the checker incorrectly treated the first 20 physical query rows as the
runtime prefix and therefore expected 55 states.

The raw GPU report is retained. The checker now mirrors
`sorted(grouped_qids)[:20]`, and a dedicated audit-only action rechecks the
existing report without model inference or an API call. The preliminary
paragraph-alignment and coverage values remain smoke diagnostics rather than
scientific results until this corrected audit passes.

## Corrected Compact smoke result

The corrected audit passed as `SMOKE_OK`. The frozen v27 Compact policy
completed 20 qids and 40 paragraph-level states with no skipped states, answer
outputs, non-paragraph memory rows, or missing online-state records. Paragraph
Alignment@1/5 was 0.6500/0.9500, and full support-paragraph/title coverage was
0.9000/0.9000. These values validate the execution path only and are not used
as scientific results.

The next registered gate evaluates exactly four methods on all 1,000 fixed
MuSiQue qids: KSG-EA-Compact-10 as the primary method, KSG-EA-Balanced-15 as a
secondary operating point, Hybrid-RAG, and BGE-Reranker-RAG. Recall-50 remains
prohibited because the official question-local memories contain at most 20
paragraphs. All runs are selection-only and use the same ordered qids and
paragraph targets; answer generation remains locked.

## Complete zero-shot selection result

All four methods completed the same 1,000 qids and 2,629 paragraph-level
states. Ordered qid and target hashes agree, no state was skipped, and no
answer output or invalid evidence granularity was observed.

| Method | Para Align@1 | Para Align@5 | Full paragraph | Full title | ms/qid |
|---|---:|---:|---:|---:|---:|
| KSG-EA-Compact-10 | 0.4892 | 0.8186 | 0.700 | 0.721 | 1474.97 |
| KSG-EA-Balanced-15 | 0.4941 | 0.8383 | 0.717 | 0.735 | 1775.28 |
| Hybrid-RAG | 0.3275 | 0.6946 | 0.436 | 0.463 | 730.85 |
| BGE-Reranker-RAG | 0.3827 | 0.7155 | 0.669 | 0.684 | 1010.21 |

Compact minus BGE is +0.1065/+0.1031 on paragraph Alignment@1/5 and
+0.031/+0.037 on full paragraph/title coverage. Balanced minus BGE is
+0.1114/+0.1229 and +0.048/+0.051, respectively. The gains over Hybrid are
larger, while Balanced provides a small quality gain over Compact at greater
latency. These are currently point estimates; significance language remains
locked until the registered 10,000-sample paired qid-cluster Bootstrap passes.

Because MuSiQue uses paragraph units, none of these values may be merged into
the sentence-level HotpotQA/2Wiki Step@k table. They support a separate
zero-shot paragraph-evidence transfer analysis.

## Paired selection Bootstrap result

The registered 10,000-sample paired qid-cluster Bootstrap completed on all
1,000 qids with no failures. The intervals below are for method differences;
an interval excluding zero supports a directional significance statement.

| Contrast | Para Align@1 | Para Align@5 | Full paragraph | Full title |
|---|---:|---:|---:|---:|
| Compact - Hybrid | +0.1617 [0.1418, 0.1814] | +0.1240 [0.1085, 0.1396] | +0.264 [0.236, 0.293] | +0.258 [0.229, 0.287] |
| Compact - BGE | +0.1065 [0.0860, 0.1276] | +0.1031 [0.0833, 0.1225] | +0.031 [0.000, 0.063] | +0.037 [0.007, 0.068] |
| Balanced - Hybrid | +0.1666 [0.1466, 0.1865] | +0.1438 [0.1280, 0.1599] | +0.281 [0.251, 0.311] | +0.272 [0.242, 0.301] |
| Balanced - BGE | +0.1114 [0.0904, 0.1322] | +0.1229 [0.1041, 0.1419] | +0.048 [0.018, 0.079] | +0.051 [0.022, 0.080] |
| Balanced - Compact | +0.0049 [-0.0023, 0.0124] | +0.0198 [0.0118, 0.0280] | +0.017 [0.003, 0.031] | +0.014 [0.001, 0.027] |

Balanced significantly exceeds both baselines on all four paragraph-level
metrics. Compact significantly exceeds Hybrid on all four metrics and BGE on
Alignment@1, Alignment@5, and full-title coverage. Its full-paragraph gain over
BGE is borderline because the percentile interval touches zero; it must not be
described as strictly significant. Balanced and Compact are tied on
Alignment@1, while Balanced has small significant gains on the other three
metrics at an additional 300.31 ms per qid.

The adapter preserves answer aliases for 287 qids. Before any answer call, the
runtime and offline evaluator must score each prediction against the maximum
over the canonical answer and its aliases. Exact-context cache identity must
also include the ordered alias list. The next action is therefore a CPU-only
cache-readiness audit over Compact, Balanced, Hybrid, and BGE; answer generation
remains locked until its unique fresh-call count is reviewed.

## Exact-context answer-cache audit

The no-API cache audit passed on all 4,000 method--qid targets. No historical
MuSiQue answer report exists, so the initial exact reuse count is zero. Across
the four target methods, 242 target files share an identical question,
canonical answer, ordered alias list, and ordered selected paragraph sequence.
Sequential cross-method propagation therefore reduces the theoretical fresh
answer requirement from 4,000 to 3,758 calls.

The audit confirms 1,000 query records and 287 qids with aliases. The frozen
answer protocol is DeepSeek V4-Flash, thinking disabled, JSON extraction,
temperature 0, and prompt version `kbs_extractive_answer_json_v1`. Reporting
must use `max_over_canonical_and_aliases`; canonical-only MuSiQue EM/F1 is not
an admissible final result.

The next gate is one Compact-10 answer smoke on the first 20 runtime-ordered
qids. It must reproduce the frozen selection contexts and pass cache, alias,
and per-qid score replay. Smoke values are execution diagnostics only. Complete
four-method answer generation remains unauthorized until this smoke passes.

## Compact answer smoke

The bounded Compact-10 answer smoke passed all 20 runtime-ordered qids. It
reproduced every frozen selection context, generated 20 fresh answers, and had
zero empty answers, API errors, invalid caches, alias mismatches, or score-
replay mismatches. Five of the 20 qids contain answer aliases, directly testing
the canonical-plus-alias scoring path.

| Metric | Smoke value |
|---|---:|
| Answer EM | 0.7500 |
| Answer F1 | 0.8167 |
| Paragraph Alignment@1 | 0.6500 |
| Paragraph Alignment@5 | 0.9500 |
| Full support-title coverage | 0.9000 |
| Full support-paragraph coverage | 0.9000 |
| Average API tokens | 880.55 |
| Average answer latency | 5.158 s |

These 20-qid values validate execution only and are not scientific results.
The complete chain is authorized in fixed order: Compact, Balanced, Hybrid,
then BGE-Reranker. Before each method, completed reports are propagated only
when the question, canonical answer, ordered aliases, selected paragraphs, and
answer protocol match exactly. Running the four methods concurrently is not
allowed because it would forfeit the registered 242-context deduplication.

## First complete-chain interruption

Compact and Balanced completed and passed all 1,000-qid audits. Hybrid also
generated 1,000 records and preserved the frozen paragraph-selection metrics,
but its audit found one exhausted API retry: 999 answers are valid and one row
has an empty/error answer. Hybrid's provisional EM/F1 of 0.4340/0.5409 is not
reportable until the failed row is repaired. BGE was not started.

This is an operational API failure rather than a model, context, alias, or
cache-protocol failure. The authorized repair identifies the single failed qid,
moves its bad cache together with the failed report, audit, launcher log, and
pid into an immutable failed-attempt directory, and preserves the other 999
Hybrid caches. It performs no inference or API call. The resumed chain must
skip clean Compact/Balanced reports, regenerate only the missing Hybrid answer,
reaudit all 1,000 Hybrid rows, and then continue to BGE.

## Complete answer-chain result

The surgical retry repaired the single Hybrid API failure, and the resumed
chain subsequently completed BGE-Reranker. The final cache-propagation audit
passes all four 1,000-qid reports with no protocol differences, raw-answer
disagreements, unresolved targets, or failures. All 4,000 method--qid targets
are now accounted for.

| Method | Freshly generated | Exact-context reuse | Total |
|---|---:|---:|---:|
| KSG-EA-Compact-10 | 1,000 | 0 | 1,000 |
| KSG-EA-Balanced-15 | 759 | 241 | 1,000 |
| Hybrid-RAG | 999 | 1 | 1,000 |
| BGE-Reranker-RAG | 1,000 | 0 | 1,000 |
| Total | 3,758 | 242 | 4,000 |

The answer-generation phase is closed. No further API call is authorized.
The next action is deterministic offline replay of canonical-plus-alias answer
metrics, paragraph evidence, joint, and ClosureSuccess metrics, followed by a
10,000-sample paired qid Bootstrap. Final answer and downstream values must be
read from those generated artifacts rather than copied from provisional or
smoke reports.

## Final downstream result

The alias-aware offline replay and 10,000-sample paired qid Bootstrap passed on
all 1,000 MuSiQue questions. Every source-summary check matches, all reports
use paragraph units, and no API call was made during finalization.

| Method | Answer EM | Answer F1 | Support Para F1 | Support Para EM | Joint F1 | Joint EM | Full Para | Full Title | Closure@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KSG-EA-Compact-10 | 0.523 | 0.6292 | 0.5738 | 0.248 | 0.4026 | 0.165 | 0.700 | 0.721 | 0.433 |
| KSG-EA-Balanced-15 | 0.518 | 0.6251 | 0.5795 | 0.246 | 0.3993 | 0.161 | 0.717 | 0.735 | 0.436 |
| Hybrid-RAG | 0.434 | 0.5409 | 0.4072 | 0.067 | 0.2497 | 0.052 | 0.436 | 0.463 | 0.279 |
| BGE-Reranker-RAG | 0.543 | 0.6393 | 0.4978 | 0.104 | 0.3390 | 0.078 | 0.669 | 0.684 | 0.376 |

Key paired differences are shown below. Intervals are percentile 95% CIs over
1,000 paired qids.

| Contrast | Answer F1 | Support Para F1 | Joint F1 | Full Para | Closure@10 |
|---|---:|---:|---:|---:|---:|
| Compact - Hybrid | +0.0883 [0.0652, 0.1110] | +0.1666 [0.1461, 0.1870] | +0.1528 [0.1317, 0.1734] | +0.264 [0.234, 0.294] | +0.154 [0.128, 0.181] |
| Compact - BGE | -0.0101 [-0.0325, 0.0121] | +0.0760 [0.0550, 0.0965] | +0.0636 [0.0436, 0.0840] | +0.031 [0.000, 0.062] | +0.057 [0.028, 0.086] |
| Balanced - Hybrid | +0.0843 [0.0601, 0.1081] | +0.1723 [0.1519, 0.1932] | +0.1496 [0.1278, 0.1716] | +0.281 [0.251, 0.311] | +0.157 [0.131, 0.184] |
| Balanced - BGE | -0.0142 [-0.0366, 0.0082] | +0.0817 [0.0612, 0.1023] | +0.0604 [0.0398, 0.0809] | +0.048 [0.018, 0.077] | +0.060 [0.033, 0.087] |

Compact significantly outperforms Hybrid on every registered downstream
metric. Against BGE-Reranker, Compact is statistically tied on Answer EM/F1,
but significantly improves supporting-paragraph F1/EM, Joint F1/EM,
full-title coverage, and Closure@10. Its full-paragraph coverage interval
touches zero and is therefore described as borderline. Closure@15 is also a
tie. Balanced follows the same pattern, except its Answer EM is significantly
lower than BGE by 0.025, while its paragraph and title coverage gains are
significant.

Balanced significantly improves full-paragraph coverage by 0.017 and
full-title coverage by 0.014 over Compact, but their answer, support F1/EM,
joint, and closure metrics are tied. Compact therefore remains the primary
efficiency-oriented operating point; Balanced is reported only as the
completeness-oriented point. Together with the selection results, this closes
Stage 9.6 as `MUSIQUE_ZERO_SHOT_BREADTH_SUPPORTED`. The claim is specifically
zero-shot paragraph-evidence and joint-quality transfer from a HotpotQA-only
Student; it is not a claim of universal answer-score superiority or direct
equivalence with sentence-level HotpotQA/2Wiki metrics.
