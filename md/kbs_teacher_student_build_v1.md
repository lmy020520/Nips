# KBS Teacher-Student Build v1

本文档固定 teacher-student 数据构建入口。它的目标是把数据重建、schema 检查、contribution scope 检查和可选 student 训练统一到一条路线里。

## 1. 统一入口

```bash
bash scripts/run_kbs_teacher_student_build.sh
```

默认执行：

```text
rebuild frontend dataset
-> validate KBS sample schema
-> check contribution supervision scope
```

默认不训练 student，避免误占 GPU。需要训练时显式设置：

```bash
RUN_TRAIN=1 bash scripts/run_kbs_teacher_student_build.sh
```

## 2. Manifest

```text
configs/kbs_teacher_student_build_v1_manifest.json
```

该 manifest 固定：

* source dataset
* output dataset
* hybrid front-end 参数
* hard negative 策略
* preserved teacher/student supervision fields
* student training config

## 3. 默认数据构建策略

| Component | Value |
|---|---|
| source root | `data/hotpotqa_distractor_v7_10k_llm_prestep` |
| output root | `data/kbs_teacher_student_v1_hybrid_frontend` |
| dense model | `models/bge-large-en-v1.5` |
| front pool | BM25 top-30 + Dense top-30 |
| fusion | RRF |
| local expansion | window 1 |
| compression | MMR |
| candidate top-k | 10 |
| hard negative source | mixed |
| hard negative count | 4 |
| natural-only | false |
| corrupted-state rollout variants | 0 |

`natural-only` 和 `corrupt_state_variants` 不是 official 默认项，它们保留为后续消融或鲁棒性分析开关。

## 4. 训练配置

```text
configs/train_ranker_deberta_kbs_official_student_v1.yaml
```

该配置使用重建后的数据：

```text
data/kbs_teacher_student_v1_hybrid_frontend
```

并从当前 strong student 初始化：

```text
outputs/ranker/deberta_v3_large_v17_candidate_contribution_lr5e7/best_model.pt
```

## 5. 常用命令

只构建和检查：

```bash
RUN_BUILD=1 \
RUN_SCHEMA_VALIDATE=1 \
RUN_CONTRIBUTION_SCOPE_CHECK=1 \
RUN_TRAIN=0 \
bash scripts/run_kbs_teacher_student_build.sh
```

构建后训练 student：

```bash
RUN_BUILD=1 \
RUN_SCHEMA_VALIDATE=1 \
RUN_CONTRIBUTION_SCOPE_CHECK=1 \
RUN_TRAIN=1 \
CUDA_DEVICE=5 \
bash scripts/run_kbs_teacher_student_build.sh
```

只训练已经构建好的数据：

```bash
RUN_BUILD=0 \
RUN_SCHEMA_VALIDATE=0 \
RUN_CONTRIBUTION_SCOPE_CHECK=0 \
RUN_TRAIN=1 \
CUDA_DEVICE=5 \
bash scripts/run_kbs_teacher_student_build.sh
```

## 6. 与整体流程对应关系

| Pipeline step | Build v1 implementation |
|---|---|
| Retrieval front-end 生成 `R_t` | `rebuild_hotpotqa_frontend_dataset.py` |
| Teacher trajectory backbone | preserved existing teacher positive sequence |
| RankingLabel | regenerated under fixed hybrid front-end |
| `d_t*` | preserved from source samples |
| `c_t*` | preserved as positive-only contribution |
| StopLabel | preserved from source samples |
| Student learning | `train_ranker.py` with ranking / deficit / contribution heads |

## 7. Scope

Build v1 的目标是统一 teacher-student 数据入口，不承诺提升性能。性能比较应放到后续对比实验和消融实验中进行。
