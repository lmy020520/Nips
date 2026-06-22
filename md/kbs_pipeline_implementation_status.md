# KBS Pipeline 搭建状态

本文档只记录完整流程是否搭好，不评价实验效果。

目标流程：

```text
Question
  ↓
Retrieval front-end 生成候选证据
  ↓
构建 / 更新知识状态 H_t / A_t / S_t / K_t
  ↓
Teacher 离线构建 trajectory
  ↓
派生 RankingLabel / d_t* / c_t* / StopLabel
  ↓
Student 学习 evidence selection policy
  ↓
接入 RAG，逐步选择证据
  ↓
最终 context 给 LLM 回答
```

---

## 1. Question 输入层

### 当前状态

已接通。

### 已有输入

```text
queries/{split}.jsonl
samples/{split}.jsonl
```

### 典型数据目录

```text
data/hotpotqa_distractor_v7_10k_llm_prestep
data/hotpotqa_distractor_eval_3000_cand50
```

### 当前缺口

无关键工程缺口。

---

## 2. Retrieval front-end 候选生成

### 当前状态

已接通，但不是最终稳定形态。

### 已实现能力

```text
BM25
Dense retrieval: BAAI/bge-large-en-v1.5
Hybrid score fusion
RRF fusion
Local expansion
MMR compression
cand10 / cand15 / cand20 / full50 stress setting
```

### 相关脚本

```text
scripts/rebuild_hotpotqa_frontend_dataset.py
scripts/run_hotpotqa_policy_rag.py
scripts/analyze_hotpotqa_frontend_policy_ranks.py
scripts/run_eval3000_hybrid_policy_curve.sh
```

### 当前缺口

```text
1. front-end 已经能召回 gold，但对 policy 的候选分布仍不够稳定。
2. ColBERTv2 compression 暂未实现。
3. front-end 曲线还没有形成固定一键汇总流程。
```

---

## 3. 知识状态 H_t / A_t / S_t / K_t

### 当前状态

已接通，属于可用雏形。

### 已有字段

```text
state.H_t
state.A_t
state.S_t
state.K_t
K_t
```

### 已有能力

```text
1. 样本中已有 teacher-side state。
2. RAG 推理时可以用 selected evidence 构造 online notebook。
3. train dataset 会把 Question + K_t 作为 student 输入。
```

### 相关脚本/模块

```text
src/datasets/prefix_dataset.py
scripts/run_hotpotqa_policy_rag.py
md/update_render_ledger_design.md
```

### 当前缺口

```text
1. online H_t / A_t / S_t / K_t 还不是完整 deterministic Update/Ledger/Render 实现。
2. 当前 RAG 主要维护 selected evidence notebook，尚未完整维护 A_t typed ledger。
```

---

## 4. Teacher 离线 trajectory 构建

### 当前状态

已接通雏形，但未完全升级成最终 teacher pipeline。

### 已有能力

```text
1. 可以构建 teacher positive trajectory。
2. 可以生成 step-level samples。
3. 可以构建 full trajectories / init states / stop labels / contribution labels / deficit labels 的旧版脚本。
```

### 相关脚本

```text
scripts/build_hotpotqa_full_trajectories_v4.py
scripts/build_hotpotqa_samples_v4.py
scripts/build_hotpotqa_teacher_select_v2.py
scripts/build_hotpotqa_teacher_step0.py
scripts/build_hotpotqa_teacher_step1.py
```

### 当前缺口

```text
1. Gate / ProposeDerived / LegalityFilter / FinalRetain 还没有并入一个统一 teacher pipeline。
2. failed / stalled / progressive trajectory 的利用还没有成为主流程。
3. teacher 构建质量分析还没有一键化。
```

---

## 5. 结构化监督派生

### 当前状态

部分完成，是当前最需要补齐的核心层。

### 已完成

```text
RankingLabel:
  已稳定接入训练。

d_t*:
  已有 deficit label 计算逻辑。
  v16/v17 已训练 deficit head。

c_t*:
  已有 positive role 监督。
  v17 已加入 candidate-level contribution role supervision。

StopLabel:
  数据接口已有 stop_label。
  但还没有作为核心训练和推理控制模块使用。
```

### 相关脚本/模块

```text
src/datasets/prefix_dataset.py
src/models/ranker.py
src/train/train_ranker.py
scripts/build_hotpotqa_ranking_labels_v4.py
scripts/build_hotpotqa_deficit_labels_v4.py
scripts/build_hotpotqa_contribution_labels_v4.py
scripts/build_hotpotqa_stop_labels_v4.py
```

### 当前缺口

```text
1. d_t* 当前是简化版 raw role deficit，还不是完整 derived deficit。
2. c_t* 当前主要是 candidate role supervision，还不是完整 utility/contribution value。
3. StopLabel 尚未系统进入 student stop/closure-aware control。
```

---

## 6. Student evidence selection policy

### 当前状态

已接通，正在从 ranking-only 升级到 multitask。

### 已实现模型

```text
CrossEncoderRanker
  ranking scorer
  role / contribution classifier
  deficit regressor
```

### 相关文件

```text
src/models/ranker.py
src/train/train_ranker.py
configs/train_ranker_deberta_v16_deficit_contribution.yaml
configs/train_ranker_deberta_v17_candidate_contribution.yaml
```

### 当前 checkpoint 主线

```text
v7: ranking-only strong baseline
v16: ranking + positive contribution + deficit
v17: ranking + candidate-level contribution + deficit
```

### 当前缺口

```text
1. stop head 尚未正式加入模型。
2. contribution 目前是 role classification，不是完整 value-based utility。
3. inference 中 deficit/contribution 融合仍是 heuristic，而不是 learned value scorer。
```

---

## 7. Online Policy-RAG

### 当前状态

已接通。

### 已实现能力

```text
policy selector
dense selector
hybrid selector
dense_policy
hybrid_policy
deficit_role scoring mode
top-k evidence selection
answer generation with DeepSeek
answer EM / F1 / contains evaluation
coverage metrics
```

### 相关脚本

```text
scripts/run_hotpotqa_policy_rag.py
scripts/run_standalone_hotpot_rag.py
```

### 当前缺口

```text
1. online typed state update 还不完整。
2. stop / closure-aware decision 还没系统接入。
3. 最终 answer generation 已能跑，但还不是完整主实验流程。
```

---

## 8. 最终 context → LLM answer

### 当前状态

已接通基础版。

### 已实现能力

```text
DeepSeek answer generation
JSON answer mode
answer cache
answer_em / answer_f1 / answer_contains
avg_answer_tokens / avg_answer_latency
```

### 当前缺口

```text
1. 还需要固定正式 answer prompt。
2. 还需要确定主实验是否生成答案，还是先只做 evidence selection。
3. 大规模 answer 评测需要缓存和失败重试策略进一步稳定。
```

---

## 9. 当前整体搭建进度

| 模块 | 搭建状态 | 完成度 |
|---|---|---:|
| Question 输入 | 已接通 | 90% |
| Retrieval front-end | 已接通，需稳定 | 75% |
| 知识状态 | 可用雏形 | 70% |
| Teacher trajectory | 可用雏形 | 60% |
| RankingLabel | 已完成 | 90% |
| d_t* | 初版完成 | 60% |
| c_t* | 初版完成，v17 已增强 | 65% |
| StopLabel | 接口有，主流程弱 | 35% |
| Student policy | 已接通，多任务版已有 | 80% |
| Online RAG | 已接通 | 75% |
| LLM answer | 基础版已接通 | 60% |

整体判断：

```text
工程主链路已跑通。
论文级完整框架仍需补齐 stop/closure-aware control、teacher 质量分析、deficit/contribution 的严格定义和系统 ablation。
```

---

## 10. 下一步不做实验时，应优先补的工程流程

如果暂时不做新实验，优先补下面三件事。

### 10.1 统一 pipeline readiness 检查

目标：

```text
输入 data_root + checkpoint
检查 samples / memory / queries / targets / model / dense model 是否齐全
输出当前 pipeline 哪些环节 ready，哪些缺文件
```

### 10.2 固化 teacher → student label 接口

目标：

```text
把 RankingLabel / d_t* / c_t* / StopLabel 的字段规范写死
让 dataset loader 不依赖临时字段猜测
```

### 10.3 固化 online state update 接口

目标：

```text
把 online RAG 中的 selected evidence notebook
升级成显式 H_t / A_t / S_t / K_t update
```

这三件做完后，才适合系统性开展大规模实验。
