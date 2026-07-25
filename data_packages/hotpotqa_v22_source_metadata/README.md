# HotpotQA v22 source metadata

These gzip-compressed JSONL files contain only the fields required to build
the KBS v22 fixed-pool state-focused training data:

- `qid`
- `question`
- `answer`
- `type`
- `level`
- `supporting_facts`

They were deterministically projected from the existing
`hotpotqa_distractor_v6_10k_source/processed` train, validation, and internal
test files. Document contexts are intentionally excluded because sentence
texts and candidate memories are read from the v7 unit registry.

The package avoids transferring the 68 MB processed source directory to a
training server while preserving the exact supporting-fact annotations.
