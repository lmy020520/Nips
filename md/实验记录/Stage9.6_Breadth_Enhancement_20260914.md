# Stage 9.6: One Breadth Enhancement

## Current status

- Status: MuSiQue adapter and corrected Compact selection smoke passed.
- Authorized action: four-method 1,000-qid selection-only evaluation.
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
