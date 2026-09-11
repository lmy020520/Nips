# Stage 9.2: Counterfactual Acquired-Evidence Loss Ablation

## Current status

- Status: all matched training runs completed; training-metric review pending.
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

## Remaining gates

1. Summarize and review best-validation and internal-test metrics for all 12
   model instances without inference.
2. Run a 20-qid no-answer selection smoke under the frozen Compact protocol.
3. Run 3,000-qid no-answer selection diagnostics for all variants and seeds.
4. Report teacher alignment, acquired-pair accuracy, and acquired-evidence
   reselection before deciding whether answer generation is justified.
5. If authorized, compute downstream support, joint, coverage, closure, and
   paired-bootstrap results using exact-context answer-cache rules.

The completed-training audit does not by itself establish a scientific gain.
