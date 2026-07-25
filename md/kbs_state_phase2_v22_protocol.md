# KBS State Phase 2: v22 State-Focused Student

## Motivation

The Phase 1 fixed-pool intervention shows that the v21 Student uses the
rendered knowledge state, but its training distribution is shallow:
selection samples cover only `t=0` and `t=1`, whereas evaluation trajectories
also contain `t>=2` states. This mismatch can make the full-state and
query-only systems appear nearly identical in aggregate RAG metrics.

## Data construction

The v22 builder uses only the existing HotpotQA training, validation, and
internal-test splits. It does not use the 3,000-question evaluation subset.

For each question:

1. Resolve every annotated supporting fact to a sentence-level memory unit.
2. Order gold units by typed role priority and original supporting-fact order.
3. Build one state for every non-terminal gold prefix, including `t>=2`.
4. Reuse exactly the same candidate pool at every prefix.
5. Include all gold units and deterministic hard distractors in that pool.
6. Keep already selected evidence in the pool as repetition negatives.
7. Repeat deeper training states (`t>=2`) twice by default.

Because adjacent states have the same question and candidate pool but a
different positive unit, they form counterfactual state-dependent ranking
pairs. Their labels cannot be solved from the question or front-end rank
alone.

## Label boundary

The reconstructed prefixes have reliable ranking labels but are not replayed
through the original teacher's typed-deficit procedure. Every generated row
therefore sets `build_meta.mask_auxiliary_labels=true`. The dataset loader
masks deficit and contribution targets with `-100`, and v22 trains only the
ranking and margin objectives.

The model is initialized from the final v21 Full checkpoint. This preserves
the learned encoder and auxiliary heads while adding focused ranking
supervision for knowledge-state sensitivity.

## Run

```bash
cd ~/lmyproject/Nips
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CUDA_DEVICE=0 \
bash scripts/run_kbs_v22_state_focused.sh
```

Data-only smoke test:

```bash
DATA_ROOT=data/hotpotqa_distractor_v22_state_focused_smoke20 \
MAX_QIDS=20 \
FORCE_DATA=1 \
TRAIN=0 \
bash scripts/run_kbs_v22_state_focused.sh
```

## Acceptance criteria

Before replacing v21 in the paper, v22 must satisfy all of the following:

1. Phase 1 fixed-pool correct-state gains over query-only remain significant
   and become larger, especially for `t>=2`.
2. HotpotQA end-to-end Compact/Recall performance does not materially regress.
3. Full-state outperforms a query-only policy under the same v22 checkpoint.
4. Candidate pools remain identical across prefixes and split qids remain
   disjoint.

If criterion 3 still fails, the next change should be an explicit
cross-state pairwise objective or state-delta encoder, not another loss-weight
sweep.

## Post-training validation

First rerun the no-API fixed-pool mechanism diagnostic with v22:

```bash
CHECKPOINT=outputs/ranker/deberta_v3_large_v22_state_focused/best_model.pt \
OUTPUT_DIR=outputs/analysis/kbs_state_phase1_v22_full3000 \
CUDA_DEVICE=0 \
MAX_QIDS=3000 \
N_BOOTSTRAP=10000 \
bash scripts/run_kbs_state_phase1.sh
```

Compare `summary.json` against the v21 Phase 1 report, especially the
correct-state minus query-only paired differences for all later states and
for `t>=2`. Only after that check passes should the Compact and Recall RAG
evaluations be rerun with the v22 checkpoint.
