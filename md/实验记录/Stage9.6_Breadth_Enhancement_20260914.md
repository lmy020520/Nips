# Stage 9.6: One Breadth Enhancement

## Current status

- Status: readiness instrumentation prepared; server audit pending.
- Authorized action: no-run breadth-readiness audit only.
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
