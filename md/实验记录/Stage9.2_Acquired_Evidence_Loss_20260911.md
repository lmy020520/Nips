# Stage 9.2: Counterfactual Acquired-Evidence Loss Ablation

## Current status

- Status: completed and closed with `TARGETED_ANTI_RESELECTION_ONLY`.
- Post-training readiness date: 2026-09-11.
- API calls during readiness: 0.
- GPU inference runs during readiness: 0.
- Dataset: `data/hotpotqa_distractor_v27_counterfactual_dual`.
- Architecture: dual state--candidate interaction with full accumulated state.
- Initialization: v21 unified Full checkpoint.
- Seeds: 42, 43, and 44.

## Registered loss matrix

| Variant | Cross entropy | Ordinary margin | Acquired-evidence margin |
|---|---:|---:|---:|
| Ranking only | yes | no | no |
| CE + margin | yes | yes | no |
| CE + acquired | yes | no | yes |
| Full v27 | yes | yes | yes |

The ordinary margin and weight are 0.20 and 0.20. The acquired-evidence
margin and weight are 0.20 and 0.50. The primary causal contrast is Full
minus CE+Margin.

## Data and artifact audit

| Split | Rows | QIDs | Acquired-negative pairs |
|---|---:|---:|---:|
| Train | 37,730 | 10,000 | 37,642 |
| Validation | 1,207 | 500 | 975 |
| Internal test | 1,247 | 500 | 1,085 |

All split overlaps and overlaps with the 3,000-qid primary evaluation set are
zero. All auxiliary labels are masked. Each of the nine newly trained models
and the three existing Full models contains `best_model.pt`,
`best_val_metrics.json`, `test_metrics.json`, and `train_history.json`.
There are no pending runs, missing paths, partial artifacts, or readiness
failures.

## Training diagnostics

| Variant | Validation rank acc. | Validation acquired-pair acc. | Test rank acc. | Test acquired-pair acc. |
|---|---:|---:|---:|---:|
| Ranking only | 0.6689 | 0.8892 | 0.6041 | 0.8670 |
| CE + margin | 0.6639 | 0.8885 | 0.6052 | 0.8627 |
| CE + acquired | 0.6636 | 0.9128 | 0.6127 | 0.9008 |
| Full v27 | 0.6614 | 0.9087 | 0.6092 | 0.8928 |

For the registered Full-minus-CE+Margin contrast, acquired-pair accuracy
improves for every seed, with mean deltas of +0.0202 on validation and
+0.0301 on internal test. Ranking accuracy changes by -0.0025 and +0.0040,
respectively. CE+Acquired minus Ranking-only yields acquired-pair gains of
+0.0236 and +0.0338. These results establish that the acquired-evidence loss
targets its intended pairwise discrimination. The full selection evaluation
below determines whether that discrimination changes online behavior.

## Three-seed selection results

All four variants use the same 3,000 qids, 7,296 state targets, frozen Compact
protocol, and no answer generation. Ordered-qid and step-target hashes match
across every seed.

| Variant | Step@1 | Step@5 | MRR | Full unit coverage | Top-1 acquired reselection |
|---|---:|---:|---:|---:|---:|
| Ranking only | 0.5560 | 0.8697 | 0.6939 | 0.7814 | 0.2070 |
| CE + margin | 0.5498 | 0.8688 | 0.6898 | 0.7806 | 0.2161 |
| CE + acquired | 0.5524 | 0.8686 | 0.6906 | 0.7807 | **0.1848** |
| Full v27 | 0.5481 | 0.8686 | 0.6880 | 0.7808 | 0.1982 |

For the registered Full-minus-CE+Margin contrast, the mean Top-1
acquired-evidence reselection delta is `-0.0179`. The paired 95% intervals are
strictly below zero for seed 42 `[-0.0199,-0.0127]`, seed 43
`[-0.0233,-0.0152]`, and seed 44 `[-0.0226,-0.0142]`. Thus, the acquired loss
has a reproducible targeted online effect on the Top-1 item that is written
into the next state.

The same contrast does not improve Step@5, MRR, full unit coverage, or full
document coverage. Full also has a slightly higher Top-5 acquired-slot rate
than CE+Margin. CE+Acquired alone gives the lowest Top-1 reselection rate and
slightly stronger Step@1/MRR than Full, indicating interaction with the
ordinary margin objective. The supported claim is therefore narrow: the
acquired-specific loss reduces Top-1 acquired-evidence reselection, not that
the combined Full loss universally improves retrieval or coverage.

## Remaining gates

The 20-qid seed-42 smoke passed with identical qid/target hashes, the frozen
Compact protocol, no answer generation, and no failures. Full and CE+Margin
tie Step@1, Step@5, MRR, and coverage in this smoke. Full reduces top-1
acquired-evidence reselection from 0.36 to 0.26. This is an execution signal,
not a scientific result.

The exact-context audit passed. Full answers can be reused for 2,016, 1,947,
and 1,966 CE+Margin contexts for seeds 42, 43, and 44. The corresponding fresh
answer requirements are 984, 1,053, and 1,034, totaling 3,071 API calls. Thus,
65.9% of the 9,000 seed--qid answers require no new generation. Reuse requires
the same seed, question, gold answer, frozen generator protocol, and full
ordered selected-unit sequence.

The seed-42 20-qid answer smoke passed under the frozen V4-Flash protocol. It
audited 11 exact-context Full answer reuses and 9 fresh CE+Margin answers, with
no empty answers, API errors, context mismatches, or invalid cache metadata.
Its EM and F1 values are execution diagnostics and are not scientific results.

1. Run the three 3,000-qid CE+Margin reports in the background. Do not
   generate answers for Ranking-only or CE+Acquired.
2. Compute downstream support, joint, coverage, closure, and paired intervals
   against the existing matched Full reports.

Ranking-only and CE+Acquired remain factorial selection diagnostics; they do
not require answer generation unless a later registered analysis needs them.

## Downstream results and final decision

The three 3,000-qid CE+Margin answer reports completed under the same frozen
generator protocol as Full. Per-qid uncertainty uses 10,000 paired bootstrap
samples independently for each training seed.

| Metric | Full mean | CE+Margin mean | Full minus CE+Margin |
|---|---:|---:|---:|
| Answer EM | 0.6214 | 0.6226 | -0.0011 |
| Answer F1 | 0.7669 | 0.7658 | +0.0011 |
| Supporting-Fact F1 | 0.6578 | 0.6526 | +0.0052 |
| Supporting-Fact EM | 0.3650 | 0.3630 | +0.0020 |
| Joint F1 | 0.5292 | 0.5247 | +0.0045 |
| Joint EM | 0.2524 | 0.2508 | +0.0017 |
| Full support coverage | 0.7808 | 0.7806 | +0.0002 |
| ClosureSuccess@10 | 0.5140 | 0.5152 | -0.0012 |

Supporting-Fact F1 favors Full for every seed, with paired intervals excluding
zero for seeds 42 and 43 but narrowly crossing zero for seed 44. Joint F1 also
favors Full for every seed, but only seed 42 excludes zero. Answer F1 is mixed
and all intervals cross zero. Full support coverage and ClosureSuccess@10 are
effectively tied.

The final registered decision is `TARGETED_ANTI_RESELECTION_ONLY`. The acquired-
evidence loss reproducibly reduces the Top-1 already-acquired item written into
the next state and yields small evidence/joint improvements, but it does not
establish a broad answer, coverage, or closure improvement. It should be
reported as a targeted auxiliary control rather than a headline performance
contribution.
