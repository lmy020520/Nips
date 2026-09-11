# Stage 9.2: Counterfactual Acquired-Evidence Loss Ablation

## Current status

- Status: all matched training runs and the three-seed 3,000-qid no-answer
  selection evaluation completed; primary exact-context cache audit authorized.
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

1. Audit exact ordered-context reuse for the primary Full versus CE+Margin
   contrast without API calls.
2. Review the resulting fresh-answer requirement and decide whether the narrow
   but significant Top-1 behavior effect justifies downstream answer calls.
3. If authorized, run one 20-qid answer smoke before any full report, then
   compute downstream support, joint, coverage, closure, and paired intervals.

Ranking-only and CE+Acquired remain factorial selection diagnostics; they do
not require answer generation unless a later registered analysis needs them.
