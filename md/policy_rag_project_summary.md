# Policy-RAG 项目阶段性总结

## 1. 项目定位

本项目面向多跳问答场景下的 Retrieval-Augmented Generation（RAG），重点研究 **如何为大语言模型构造更有效的上下文**。

传统 RAG 通常采用如下流程：

```text
Question -> Retriever -> Top-k documents/passages -> LLM answer
```

这种范式在单跳问答中较为有效，但在多跳问答中存在明显不足。多跳问题通常需要多个证据之间形成推理链，例如先找到 bridge entity，再找到最终 answer evidence。如果仅根据原始问题一次性检索 top-k 证据，检索结果可能相关但不完整，也可能包含大量噪声，最终导致 LLM 无法稳定回答。

因此，本项目的核心问题不是单纯提高检索召回率，而是：

```text
如何在有限上下文预算下，动态选择并组织多跳问答所需的证据，使最终输入 LLM 的 context 更适合回答问题。
```

我们将这个问题定义为：

```text
Trajectory-aware Context Construction for Multi-hop RAG
```

即：把多跳 RAG 的上下文构建过程建模为一个带有证据状态的序列决策问题。

## 2. 核心思想

### 2.1 从一次性检索到序列化 context 构建

传统检索器通常只学习：

```text
score(q, u)
```

其中 `q` 是问题，`u` 是候选证据。这类方法只判断候选证据和问题之间的静态相关性。

但在多跳问答中，候选证据是否有用，取决于当前已经选中了哪些证据。例如，当系统已经找到 bridge evidence 后，下一步更需要的是 support / answer evidence，而不是重复选择另一个表面相关的 bridge 句子。

因此，我们建模的不是普通相关性分数，而是一个带状态的证据价值函数：

```text
Q_theta(s_t, u) = score(q, K_t, u)
```

其中：

- `q`：原始问题；
- `K_t`：当前已经构造出的 evidence context / notebook；
- `u`：当前候选证据；
- `t`：多跳证据选择过程中的步骤。
- `s_t = (q, H_t, K_t)`：当前 RAG 证据构建状态。

也就是说，模型不是只做 query-passage relevance，而是学习：

```text
在当前证据状态下，下一条最应该进入上下文的 evidence 是什么。
```

更准确地说，这里的 score 可以理解为：

```text
在状态 s_t 下，如果选择候选证据 u 加入上下文，
它对最终完成多跳证据覆盖和回答问题的预期价值有多大。
```

因此，本项目中的 score 本质上是 **state-action value**，而不是普通检索相关性分数。

### 2.2 Value-based 视角

为了突出本项目的原理，可以将多跳 context construction 写成一个有限步序列决策问题。

#### State

```text
s_t = (q, H_t, K_t)
```

其中：

- `q` 是当前问题；
- `H_t` 是已经选择过的 evidence history；
- `K_t` 是由已选证据渲染出的当前 context / notebook。

#### Action

```text
a_t = u,  u in C_t
```

也就是从当前候选证据集合 `C_t` 中选择一条 evidence unit 加入上下文。

#### Transition

选择 `u` 之后，系统更新 evidence state：

```text
H_{t+1} = H_t ∪ {u}
K_{t+1} = Render(H_{t+1})
```

#### Value

我们希望学习：

```text
Q_theta(s_t, u)
```

它表示在当前状态 `s_t` 下选择证据 `u` 的价值。这个价值不是局部文本相关性，而是和最终目标相关：

- 是否补齐缺失的 bridge evidence；
- 是否补齐 answer-facing support evidence；
- 是否减少当前 context 的 evidence deficit；
- 是否提高最终 gold evidence coverage；
- 是否帮助 LLM 在最终 context 中回答正确。

因此，Policy-RAG 的选择过程可以写成：

```text
u_{t+1} = argmax_{u in C_t} Q_theta(s_t, u)
```

这就是 value-based 的核心：模型输出的是每个候选 action 在当前 context state 下的价值估计，然后根据价值选择下一条 evidence。

#### 与普通 score/rerank 的区别

普通 reranker 的 score 通常表示：

```text
relevance(q, u)
```

而我们的 score 表示：

```text
value(s_t, u)
```

二者区别在于：

```text
同一条 evidence u，在不同 K_t 下价值可能不同。
```

例如：

- 如果当前 context 还没有 bridge evidence，则 bridge sentence 的 value 高；
- 如果 bridge evidence 已经选入 context，则重复 bridge sentence 的 value 应下降；
- 如果当前缺少 answer support，则 answer-facing support sentence 的 value 应上升。

这就是本项目相对于普通检索和 reranking 的核心差异。

### 2.3 Policy-RAG 的基本流程

当前系统可以抽象为：

```text
Question
  -> Candidate Evidence Pool
  -> Trajectory-aware Evidence Selection Policy
  -> Evidence State / Notebook Update
  -> Answer-facing Context
  -> LLM Answer
```

更具体地说：

```text
输入：q, K_t, C_t
输出：每个候选 u 的 Q_theta(s_t, u)，并选择价值最高的 u_{t+1}
```

其中：

- `q` 是问题；
- `K_t` 是当前已经选择并组织好的证据上下文；
- `C_t` 是当前候选证据集合；
- `u_{t+1}` 是下一条被选择进入 context 的证据。

最终，系统将多步选择出的证据集合组织成 answer-facing context，再交给 LLM 生成答案。

### 2.4 与传统 reranker 的区别

本项目中的模型容易被误解为 reranker，但二者的学习目标不同。

传统 reranker 通常是：

```text
rerank(q, candidates)
```

它对一批候选 passage 做一次性排序。

我们的模型是：

```text
Q_theta(s_t, u) = value(q, K_t, u)
select_next = argmax_u Q_theta(s_t, u)
```

它在当前 evidence state 下选择下一条最有增益的证据。因此，它更适合称为：

```text
Trajectory-aware evidence selection policy
```

或：

```text
Value-guided context construction policy
```

而不是传统意义上的 one-shot reranker。

## 3. 方法框架

### 3.1 核心对象

项目中使用以下核心对象描述多跳 RAG 的证据构建过程。

#### Query `q`

问题对象，包含：

```text
qid
question
answer
metadata
```

#### Candidate Evidence `C_t`

当前步骤可供选择的候选证据集合。每个候选通常是 sentence-level evidence unit。

#### History `H_t`

已经选择过的 evidence unit 序列：

```text
H_t = [u_1, u_2, ..., u_t]
```

#### Notebook / Context State `K_t`

由已选证据渲染得到的当前上下文状态。它表示 LLM 当前已经“知道”的证据内容。

#### Policy `pi`

学习一个选择策略：

```text
pi(u | q, K_t, C_t)
```

用于判断候选证据 `u` 是否应该作为下一条证据进入上下文。

### 3.2 Teacher-Student 设计

项目采用 teacher-student 思路。

Teacher 阶段用于离线构造高质量监督轨迹：

```text
(q, K_t, C_t) -> u_{t+1}^*
```

其中 `u_{t+1}^*` 是当前状态下的 gold next evidence。

Student 阶段训练一个轻量模型，学习 teacher 生成的 evidence selection decision。在线推理时，student 不需要调用复杂 teacher 逻辑，只需要根据当前 context state 对候选证据打分并选择。

这个设计的好处是：

- 将复杂的多跳上下文构建过程转化为监督学习任务；
- 将昂贵的标注和轨迹构建放到离线阶段；
- 在线阶段只需要小模型完成 evidence selection，成本低于 agentic RAG；
- 模型显式学习 context state，而不是只学习静态相关性。

## 4. 数据集整理过程

### 4.1 原始数据来源

项目主要使用 HotpotQA distractor 数据集。HotpotQA 适合作为本项目实验数据，原因是：

- 它是典型多跳问答数据集；
- 每个问题包含多个 distractor documents；
- 标注了 supporting facts；
- 可以用于构造 evidence selection trajectory；
- 适合评估多跳证据覆盖和最终答案质量。

原始样本包含：

```text
question
answer
type
level
context
supporting_facts
```

其中 `context` 包含多个候选文档及其句子，`supporting_facts` 给出 gold evidence 所在文档和句子编号。

### 4.2 统一数据格式

为了训练和评测 Policy-RAG，我们将 HotpotQA 转换成以下几类文件。

#### Queries

路径示例：

```text
queries/test.jsonl
```

每行表示一个问题：

```json
{
  "qid": "...",
  "question": "...",
  "answer": "...",
  "metadata": {
    "dataset": "hotpotqa_distractor",
    "split": "test",
    "type": "bridge",
    "level": "hard"
  }
}
```

#### Raw Units

路径示例：

```text
unit_registry/raw_units_test.jsonl
```

将 HotpotQA context 中的句子拆成 sentence-level evidence units：

```json
{
  "unit_id": "qid::doc_id::sent_id",
  "text": "...",
  "doc_id": "...",
  "parent_chunk_id": "...",
  "provenance": "raw",
  "candidate_granularity": "sentence"
}
```

#### Targets

路径示例：

```text
targets/test.jsonl
```

由 HotpotQA supporting facts 构造 gold evidence targets：

```json
{
  "qid": "...",
  "question": "...",
  "T_q_raw": [
    {
      "unit_id": "...",
      "text": "...",
      "doc_id": "...",
      "primary_role": "support"
    }
  ]
}
```

#### Samples

路径示例：

```text
samples/test.jsonl
```

这是训练和评测的核心文件。每一条 sample 对应一个 trajectory step：

```json
{
  "qid": "...",
  "t": 0,
  "question": "...",
  "state": {
    "K_t": "current evidence context"
  },
  "candidates": {
    "C_t": ["unit_id_1", "unit_id_2", "..."]
  },
  "labels": {
    "ranking_label": {
      "positive_unit_id": "gold_next_unit",
      "negative_unit_ids": ["..."]
    }
  }
}
```

这样就把多跳证据选择问题转成了监督学习问题：

```text
给定 q、K_t 和候选 C_t，选择 positive_unit_id。
```

### 4.3 数据版本演进

项目中逐步构建了多个数据版本。

| 数据版本 | 主要作用 |
|---|---|
| `v3/v4` | 初步构建 trajectories、raw units、targets、samples |
| `v5_llm_dataset` | 引入 LLM role 标注，增强 bridge/support 区分 |
| `v6_10k_llm` | 扩大到约 10k qids，提高训练稳定性 |
| `v7_10k_llm_prestep` | 引入 pre-step trajectory state，是当前主力训练数据 |
| `v8_15k_llm_prestep` | 扩大训练数据到约 15k，作为扩展尝试 |
| `eval_3000_cand10` | 当前用于正式 RAG 对比实验的 3000 条独立评测集 |

当前最佳训练数据为：

```text
data/hotpotqa_distractor_v7_10k_llm_prestep
```

当前主要评测数据为：

```text
data/hotpotqa_distractor_eval_3000_cand10
```

### 4.4 为什么使用 cand10 评测设置

`cand10` 表示每一步候选集合约为 10 条 evidence units。

采用 cand10 的原因是：

- 与模型训练时的候选规模一致；
- 便于公平比较不同选择策略；
- 可以聚焦评估 context construction policy，而不是把问题变成完全开放检索；
- 和之前 383 条测试实验保持一致。

另外也构建过 full-candidate 设置，即每一步使用该问题原始 HotpotQA context 中的所有 sentence 作为候选，平均约 40-60 个候选。该设置远难于训练时分布，更适合作为 stress test，而不是当前主实验。

## 5. 模型训练过程

### 5.1 模型结构

当前 student policy 使用 cross-encoder 结构，对：

```text
(question + current context, candidate evidence)
```

进行联合编码，然后输出候选证据的分数。

当前主力 backbone：

```text
microsoft/deberta-v3-large
```

本地模型目录：

```text
models/deberta-v3-large
```

当前最佳 checkpoint：

```text
outputs/ranker/frozen_deberta_v3_large_v7_main_val08252/best_model.pt
```

### 5.2 训练目标

每个训练样本对应一个多跳轨迹 step。模型需要在候选集合中将 gold next evidence 排在前面。

训练目标可以理解为：

```text
maximize score(q, K_t, u_positive)
minimize score(q, K_t, u_negative)
```

也就是学习：

```text
当前 context state 下，哪条 evidence 对完成多跳推理最有贡献。
```

### 5.3 训练过程中的主要调整

训练过程中进行了多轮优化：

- 从 DeBERTa-v3-base 切换到 DeBERTa-v3-large；
- 处理 Hugging Face 下载、tokenizer、protobuf、tiktoken 等环境问题；
- 处理 RTX 3090 上 large 模型 OOM 问题；
- 调整 `batch_size`、`max_length`、`grad_accum_steps`；
- 尝试不同学习率、epoch、seed；
- 对错误样本进行分析，修复 unlabeled positive 和 near-miss hard negatives；
- 引入 pre-step state，显著提升 trajectory-aware selection 效果。

### 5.4 当前训练效果

当前最佳模型来自 v7 pre-step 数据版本。

代表性结果：

```text
Best epoch: 4
best val acc: 0.8252
test acc: 0.8257
```

这个结果说明模型已经能够较好地学习 trajectory-aware evidence selection。

## 6. RAG 系统接入

### 6.1 Policy-RAG 推理流程

在 RAG 推理阶段，系统逐步选择 evidence。

对于每个问题：

1. 读取当前候选集合 `C_t`；
2. 根据已经选择的 evidence 构造 `K_t`；
3. 使用 policy model 对每条候选 evidence 打分；
4. 选择 top-k evidence 加入上下文；
5. 多步完成后，将 selected evidence 组织成 answer-facing context；
6. 调用 LLM 输出答案。

当前核心脚本：

```text
scripts/run_hotpotqa_policy_rag.py
```

### 6.2 支持的对比方法

当前脚本支持多种 selector，用于公平比较。

| Selector | 说明 |
|---|---|
| `policy` | 我们的 trajectory-aware context construction policy |
| `bm25` | BM25 词法检索 |
| `dense` | BGE dense retrieval |
| `hybrid` | BM25 + dense 分数融合 |
| `multi_query_dense` | 多查询视角 dense retrieval |
| `iterative_dense` | 基于当前 state 的 iterative dense retrieval |
| `generic_reranker` | 通用 cross-encoder reranker |
| `dense_policy` | dense shortlist 后使用 policy 选择 |
| `first` | 直接选择候选前几条 |
| `random` | 随机选择候选 |
| `gold_oracle` | Gold evidence only reference |

### 6.3 答案生成

证据选择完成后，使用 LLM 根据 selected evidence 生成答案。

当前答案生成采用 JSON 模式：

```json
{"answer": "..."}
```

这样可以减少解释性文本对 EM/F1 评测的影响。

系统还加入了 answer cache 和 retry 机制：

```text
--answer-cache-dir
--llm-max-retries
```

避免 API 中断导致长时间实验需要重跑。

## 7. 实验指标

项目同时评估 evidence quality 和 answer quality。

### 7.1 Evidence Quality

| 指标 | 含义 |
|---|---|
| `step_acc@1` | 每一步 top-1 是否选中 gold evidence |
| `step_acc@5` | 每一步 top-5 是否包含 gold evidence |
| `step_selected_contains_gold` | 每一步最终选入 context 的证据是否包含 gold evidence |
| `full_gold_doc_coverage` | 最终 context 是否覆盖完整 gold documents |
| `full_gold_unit_coverage` | 最终 context 是否覆盖完整 gold sentence units |

这些指标直接衡量 context construction 是否成功。

### 7.2 Answer Quality

| 指标 | 含义 |
|---|---|
| `answer_em` | Exact Match |
| `answer_f1` | token-level F1 |
| `answer_contains` | 生成答案是否包含 gold answer |
| `avg_answer_tokens` | 平均回答消耗 token |
| `avg_answer_latency` | 平均回答延迟 |

这些指标衡量 evidence context 是否真正帮助 LLM 生成正确答案。

## 8. 已完成的主要实验

### 8.1 383 条测试集实验

在 v7 test split 上完成了 383 个 qid 的完整对比。

| Method | QIDs | Answer EM | Answer F1 | Answer Contains | Full Gold Doc Coverage | Full Gold Unit Coverage | Step Selected Contains Gold | Avg Tokens | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| First | 383 | 0.3368 | 0.4462 | 0.3838 | 0.4752 | 0.4360 | 0.4815 | 336.68 | 0.852 |
| Random | 383 | 0.4047 | 0.5211 | 0.4595 | 0.6423 | 0.5561 | 0.5970 | 342.76 | 0.838 |
| BM25-RAG | 383 | 0.4648 | 0.5963 | 0.5248 | 0.8068 | 0.7311 | 0.7756 | 332.74 | 1.361 |
| Dense-RAG | 383 | 0.4883 | 0.6312 | 0.5770 | 0.9347 | 0.9034 | 0.9194 | 329.18 | 1.325 |
| Gold-only | 383 | 0.4935 | 0.6377 | 0.5744 | 1.0000 | 1.0000 | 1.0000 | 342.23 | 0.871 |
| Dense+Policy | 383 | 0.5535 | 0.6907 | 0.6319 | 0.9948 | 0.9843 | 0.9346 | 345.09 | 1.215 |
| **Policy-RAG** | 383 | **0.5587** | **0.6955** | **0.6371** | **0.9974** | **0.9843** | **0.9346** | 345.76 | **0.827** |

主要观察：

- Policy-RAG 明显优于 BM25-RAG 和 Dense-RAG；
- Policy-RAG 的 evidence coverage 接近 Gold-only；
- Policy-RAG 的答案质量高于 Gold-only，说明额外上下文可能提供了有益的消歧信息；
- Dense+Policy 与 Policy-RAG 接近，但 Policy-RAG 速度更快。

### 8.2 3000 条 cand10 smoke test

在 3000 条独立评测集的 cand10 设置下，先进行了 20 条 smoke test。

结果：

```text
qids = 20
steps = 50
trajectory_all_steps_correct = 0.45
any_gold_doc_selected = 1.0
full_gold_doc_coverage = 1.0
full_gold_unit_coverage = 1.0
step_selected_contains_gold = 0.98
step_acc@1 = 0.70
step_acc@2 = 0.84
step_acc@3 = 0.94
step_acc@5 = 0.98
answer_em = 0.65
answer_contains = 0.65
answer_f1 = 0.742857
avg_answer_tokens = 587.2
avg_answer_latency = 1.206
answer_errors = 0
```

该结果说明：

- 新评测数据格式正确；
- Policy-RAG 可以在独立 3000 条数据上正常运行；
- cand10 设置下证据覆盖和答案生成效果较好；
- 可以进入完整 3000 条对比实验。

## 9. 当前工作的主要特点

### 9.1 显式建模多跳 context state

与传统检索器相比，本项目不是只看 `question-candidate` 相关性，而是显式引入当前 evidence state。

这使模型能够学习：

```text
当前 context 还缺什么证据？
下一条 evidence 应该补充哪个推理环节？
```

### 9.2 将 context construction 转化为监督学习

通过 teacher trajectory，将多跳证据构建过程转化为：

```text
(q, K_t, C_t) -> u_{t+1}^*
```

这使得系统既保留了多跳推理过程，又避免在线阶段依赖昂贵的 LLM agent。

### 9.3 同时优化 evidence coverage 和 answer quality

项目不是只报告检索 recall，也不是只报告答案 EM/F1，而是同时评估：

- 每一步证据选择是否正确；
- 最终 context 是否覆盖完整 gold evidence；
- LLM 是否能基于该 context 输出正确答案。

这更符合 RAG 系统的真实目标。

## 10. 后续实验计划

### 10.1 完整 3000 条主实验

在：

```text
data/hotpotqa_distractor_eval_3000_cand10
```

上跑完整对比：

- BM25-RAG
- Dense-RAG
- Hybrid-RAG
- MultiQuery-Dense
- Iterative-Dense
- Generic-Reranker
- Dense+Policy
- Policy-RAG
- First
- Random
- Gold-only

### 10.2 Context ablation

为了进一步证明 context state 的作用，需要增加消融实验：

| Ablation | 目的 |
|---|---|
| no-state policy | 去掉 `K_t`，只用 question 和 candidate |
| dataset-state policy | 使用 teacher-provided state |
| self-state policy | 使用模型自己选择出来的 state |
| top-1 / top-3 / top-5 context budget | 分析上下文预算对结果影响 |
| no-order context | 打乱 evidence 顺序，分析顺序信息作用 |

### 10.3 Candidate difficulty analysis

为了展示方法在不同候选难度下的表现，可以比较：

| Setting | 候选规模 | 作用 |
|---|---:|---|
| cand10 | 约 10 | 主实验，接近训练分布 |
| cand20 | 约 20 | 中等难度 |
| full-candidate | 约 40-60 | stress test |

这样可以形成候选规模难度曲线，说明方法的适用范围和当前限制。

## 11. 可以用于论文的贡献表述

当前工作可以总结为三点贡献。

### Contribution 1: Trajectory-aware context construction formulation

将多跳 RAG 的上下文构建问题建模为一个 trajectory-aware sequential evidence selection problem，而不是一次性检索或普通 reranking。

### Contribution 2: Teacher-guided supervised policy learning

通过离线 teacher trajectory 构造监督样本，使 student policy 学习在当前 evidence state 下选择下一条最有增益的证据。

### Contribution 3: Improved evidence coverage and answer quality

实验表明，Policy-RAG 相比 BM25-RAG、Dense-RAG 和其他 baseline，在 evidence coverage 与 answer quality 上均有提升。

## 12. 一句话总结

本项目研究的是多跳 RAG 中的 context construction 问题。我们将证据选择建模为一个 trajectory-aware sequential decision process，通过离线 teacher 构造 evidence-state trajectories，并训练 student policy 在当前 context 下选择下一条最有增益的证据，从而在有限上下文预算内提高多跳证据覆盖率和最终答案质量。
