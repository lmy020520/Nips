# Stage 9.1: Closure Teacher vs Coverage Teacher

## Experiment status

- Status: completed and passed the registered interpretation gate.
- Date closed: 2026-09-10.
- Dataset: fixed 3,000-qid HotpotQA distractor-validation subset.
- Seeds: 42, 43, 44.
- Operating point: Compact, candidate top-k 10, selected top-k 5,
  state-update top-k 1, front-policy blend 0.5.
- Answer protocol: DeepSeek V4-Flash, thinking disabled, temperature 0,
  JSON mode, prompt `kbs_extractive_answer_json_v1`.
- Uncertainty: paired qid bootstrap, 10,000 samples independently for each
  training seed.

The matched experiment holds questions, states, candidate pools, row order
and repetition, architecture, initialization, optimizer, epochs, loss
weights, retrieval front, inference budget, and answer generator fixed. It
changes only the teacher ranking objective and dependent ranking labels.

## Three-seed downstream results

| Metric | Closure mean +/- SD | Coverage mean +/- SD | Coverage - Closure |
|---|---:|---:|---:|
| Answer EM | 0.6214 +/- 0.0008 | 0.6223 +/- 0.0009 | +0.0009 |
| Answer F1 | 0.7669 +/- 0.0010 | 0.7591 +/- 0.0008 | -0.0078 |
| Supporting-Fact F1 | 0.6578 +/- 0.0020 | 0.4806 +/- 0.0032 | -0.1772 |
| Supporting-Fact EM | 0.3650 +/- 0.0023 | 0.1114 +/- 0.0017 | -0.2536 |
| Joint F1 | 0.5292 +/- 0.0002 | 0.3811 +/- 0.0029 | -0.1481 |
| Joint EM | 0.2524 +/- 0.0008 | 0.0789 +/- 0.0020 | -0.1736 |
| Full Supporting-Unit Coverage | 0.7808 +/- 0.0024 | 0.7394 +/- 0.0032 | -0.0413 |
| ClosureSuccess@10 | 0.5140 +/- 0.0013 | 0.4942 +/- 0.0034 | -0.0198 |

The reported standard deviation is the sample standard deviation across the
three training seeds. It is not a pooled confidence interval.

## Paired bootstrap evidence

Intervals below use the delta definition Coverage minus Closure.

| Metric | Seed 42 delta [95% CI] | Seed 43 delta [95% CI] | Seed 44 delta [95% CI] |
|---|---:|---:|---:|
| Answer EM | +0.0023 [-0.0087, 0.0137] | +0.0010 [-0.0103, 0.0127] | -0.0007 [-0.0120, 0.0103] |
| Answer F1 | -0.0078 [-0.0171, 0.0015] | -0.0071 [-0.0166, 0.0028] | -0.0085 [-0.0180, 0.0010] |
| Supporting-Fact F1 | -0.1722 [-0.1831, -0.1615] | -0.1799 [-0.1908, -0.1696] | -0.1795 [-0.1901, -0.1687] |
| Joint F1 | -0.1450 [-0.1556, -0.1343] | -0.1501 [-0.1607, -0.1393] | -0.1493 [-0.1601, -0.1387] |
| Full Coverage | -0.0397 [-0.0503, -0.0287] | -0.0417 [-0.0527, -0.0307] | -0.0427 [-0.0537, -0.0320] |
| ClosureSuccess@10 | -0.0173 [-0.0297, -0.0047] | -0.0227 [-0.0350, -0.0100] | -0.0193 [-0.0317, -0.0070] |

## Reselection diagnostic

| Metric | Closure | Coverage | Coverage - Closure |
|---|---:|---:|---:|
| Top-1 acquired-evidence reselection | 0.1982 | 0.3143 | +0.1161 |
| Top-5 acquired-evidence slot rate | 0.0651 | 0.0734 | +0.0083 |

Both reselection increases have paired intervals above zero for every seed.

## Scientific conclusion and claim boundary

The registered decision is `CLOSURE_SUPERIOR`. Answer-facing Closure
supervision consistently improves support quality, joint answer-support
quality, full support coverage, and budgeted closure success relative to the
matched coverage-greedy objective. It also substantially reduces reselection
of evidence that is already present in the accumulated state.

The result does not establish a significant standalone Answer EM gain.
Answer F1 favors Closure for all three seeds, but each seed's paired interval
crosses zero. The paper should therefore center the claim on compiled-context
quality, support sufficiency, joint quality, and closure under budget rather
than claim a significant answer-only improvement.

## Primary artifacts

- Final summary:
  `outputs/analysis/kbs_stage9_teacher_objective/downstream3000/multiseed_summary.json`
- Standard metrics:
  `outputs/analysis/kbs_stage9_teacher_objective/downstream3000/standard_metrics.json`
- Per-seed paired bootstrap:
  `outputs/analysis/kbs_stage9_teacher_objective/downstream3000/seed{42,43,44}_paired_bootstrap.json`
- Selection diagnostics:
  `outputs/analysis/kbs_stage9_teacher_objective/selection3000/multiseed_summary.json`
- Coverage answer reports:
  `outputs/rag/kbs_stage9_teacher_objective/coverage_seed{42,43,44}_full3000.json`
- Matched Closure answer reports:
  `outputs/rag/kbs_v27_final_hotpot/full_compact.json` and
  `outputs/rag/kbs_v27_stage5_multiseed/seed{43,44}/full_compact.json`

