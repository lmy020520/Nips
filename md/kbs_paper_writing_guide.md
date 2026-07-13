# KBS 论文写作指南：Knowledge-State-Guided Evidence Acquisition for Multi-Hop RAG

本文档用于指导后续论文写作，目标期刊为 **Knowledge-Based Systems (KBS)**。文档重点说明我们当前工作的技术主线、方法贡献、实验结果、论文结构，以及写作时应如何突出创新点、规避薄弱点。

## 0. 2026-07-13 最终实验口径更新

后文部分表格保留了早期 v17 结果用于追踪实验演化。正式论文必须以统一训练的 **v21 Full Student** 为准，不能混用 v17 与 v21 数值。

当前正式主结果：

| Setting | EM | F1 | Step@5 | Full Unit Cov. | Avg. API Tokens |
|---|---:|---:|---:|---:|---:|
| KSG-EA-Compact (v21) | 0.5957 | 0.7401 | 0.8277 | 0.7170 | 385.08 |
| KSG-EA-Recall (v21) | 0.6270 | 0.7777 | 0.8205 | 0.8337 | 423.97 |

正式写作必须遵守以下结论边界：

* `policy_blend_weight=0.35` 已通过独立 1,000-qid validation sensitivity 确定，选择准则为最大化 Step@5。
* v21 统一消融表明 ranking-only、w/o deficit、w/o contribution 与 Full 基本持平；不能声称 deficit/contribution auxiliary losses 显著提升最终性能。
* v21 deficit MAE 为 `0.2620`，Pearson 为 `0.1651`；其校准质量较弱，且未超过简单均值预测基线。因此 typed deficit 只能定位为探索性的结构化辅助信号和诊断接口。
* KSG-EA-Recall 在 HotpotQA 和 2Wiki 上相对 Hybrid、BGE 的主要指标均获得 paired-bootstrap 显著提升；Compact 是效率与证据精度导向配置，不应声称全面超过 BGE。
* 2Wiki 结果属于 question-local candidate memory 下、直接使用 HotpotQA checkpoint 的 zero-shot transfer，不属于开放域检索。
* Question-only 与 Iterative-Hybrid baselines、统一 latency/显存/吞吐量分析均已完成。

后文出现的 `KBS Official = 0.5943/0.7390`、旧 cand10 数值以及 v17 deficit 变体，均应在最终论文表格整理时替换为对应 v21 report。

---

## 1. 论文主线定位

### 1.1 论文不应被写成 reranker

本工作的核心不应表述为：

```text
我们训练了一个 reranker 来重排候选证据。
```

这种说法会把贡献压低成普通的 passage reranking，容易被认为创新不足。

更准确的定位应该是：

```text
我们提出了一种 knowledge-state-guided evidence acquisition framework，
用于 multi-hop RAG 中的逐步证据获取。
```

也就是说，我们解决的不是静态相关性排序问题，而是：

```text
在当前已经获得部分证据的情况下，系统应该继续选择哪条证据，才能推进多跳推理并构建更完整的最终上下文。
```

普通 reranker 学的是：

```text
score(q, passage)
```

我们的方法学的是：

```text
score(q, K_t, u)
```

其中：

```text
q   = 用户问题
K_t = 当前已经构建出的知识状态 / 当前证据上下文
u   = 候选证据单元
```

因此，本工作应被包装成：

```text
Knowledge-State-Guided Evidence Acquisition for Multi-Hop Retrieval-Augmented Generation
```

---

## 2. 推荐论文标题

### 2.1 首选标题

```text
Knowledge-State-Guided Evidence Acquisition for Multi-Hop Retrieval-Augmented Generation
```

这个标题最稳，适合 KBS。它突出：

```text
knowledge state
evidence acquisition
multi-hop RAG
```

而不是突出 reranking。

### 2.2 备选标题

```text
Trajectory-Aware Evidence Selection with Typed Deficit Supervision for Multi-Hop RAG
```

这个标题更强调 trajectory 和 typed deficit。

```text
Learning to Acquire Evidence via Knowledge-State Trajectories for Multi-Hop Question Answering
```

这个标题更偏 teacher-student trajectory learning。

```text
A Knowledge-State-Aware Framework for Sequential Evidence Acquisition in Multi-Hop RAG
```

这个标题更系统化，也比较适合期刊。

---

## 3. 一句话贡献总结

可以这样概括整篇论文：

```text
We propose a knowledge-state-guided evidence acquisition framework for multi-hop RAG, where an offline teacher constructs evidence acquisition trajectories and derives typed supervision signals, and a student policy learns to select evidence conditioned on the current knowledge state. Experiments on HotpotQA show that the proposed method substantially improves evidence coverage and answer quality compared with BM25, dense retrieval, and hybrid retrieval baselines.
```

中文理解：

```text
我们提出了一个知识状态驱动的多跳 RAG 证据获取框架。该框架先用 teacher 离线构建证据获取轨迹，并派生结构化监督；然后训练 student policy，使其在在线 RAG 过程中根据当前知识状态逐步选择证据。HotpotQA 实验表明，该方法相比 BM25、Dense 和 Hybrid RAG 显著提升了证据覆盖率和最终答案质量。
```

---

## 4. 方法流程

论文中应明确给出完整流程：

```text
Question
  ↓
Retrieval front-end 生成 raw candidate pool R_t
  ↓
构建 / 更新知识状态 H_t, A_t, S_t, K_t
  ↓
Teacher 离线构建 evidence acquisition trajectory
  ↓
派生结构化监督:
RankingLabel / d_t* / c_t* / StopLabel
  ↓
Student 学习:
ranking head + deficit head + contribution head
  ↓
Online RAG 中:
根据当前 K_t 和候选证据逐步选择 evidence
  ↓
更新知识状态并生成 final context
  ↓
最终 context 给 LLM 回答
```

---

## 5. 核心概念定义

### 5.1 Retrieval front-end

front-end 负责从原始候选池中形成可供 student policy 判断的候选证据集合。

当前正式设置：

```text
BM25 top-k
Dense top-k
RRF fusion
Local expansion
MMR compression
Candidate pool C_t
```

正式系统使用：

```text
dense model = BAAI/bge-large-en-v1.5
front fusion = RRF
front_pool_k = 30
local_expansion_window = 1
candidate_top_k = 10
select_top_k = 5
```

front-end 的作用：

```text
1. 保证 gold evidence 有较高概率进入候选池
2. 压缩 full candidate pool 中的噪声
3. 为 student policy 提供更稳定的候选分布
```

### 5.2 Knowledge State

本工作不是简单把历史证据拼进 prompt，而是显式维护知识状态：

```text
H_t: historical evidence units
A_t: raw evidence ledger
S_t: evidence-centric notebook state
K_t: compiled context / rendered knowledge state
```

论文中应强调：

```text
K_t 是在线 RAG 中 student policy 的条件输入。
Student 不是只看 question 和 candidate，而是看 question + current knowledge state + candidate。
```

### 5.3 Teacher trajectory

Teacher 是离线使用的轨迹构造器。它不参与在线推理。

Teacher 的作用：

```text
1. 构造 evidence acquisition trajectory
2. 决定每一步应该选择哪条 evidence
3. 记录当前知识状态下还缺什么
4. 派生 student 训练所需的结构化监督
```

Teacher 生成的监督包括：

```text
RankingLabel
d_t*
c_t*
StopLabel
```

### 5.4 Student policy

Student 是最终接入 RAG 的模型。

Student 输入：

```text
question q
current knowledge state K_t
candidate evidence u
```

Student 输出：

```text
ranking score
typed deficit prediction
contribution prediction
```

正式 online RAG 中使用的是 trained student checkpoint：

```text
outputs/ranker/deberta_v3_large_v17_candidate_contribution_lr5e7/best_model.pt
```

### 5.5 Typed deficit

Typed deficit 表示当前知识状态还缺什么类型的信息。

包括：

```text
d_br  = bridge deficit
d_dis = distinguish deficit
d_sup = support deficit
d_der = derived deficit
```

写作时可以这样解释：

```text
Typed deficit encourages the student to reason about what information is still missing, rather than merely selecting locally relevant passages.
```

需要注意：

```text
当前 v21 deficit head 已完成 teacher-label 对齐评估：overall MAE 为 `0.2620`，各角色 MAE 为 `0.2255/0.3143/0.2421/0.2660`，Pearson 为 `0.1651`。这些结果说明校准仍较弱，且统一损失消融没有证明 deficit supervision 带来性能收益。因此论文只能将其描述为探索性的结构化辅助监督和诊断信号，不能写成已验证的主要性能贡献或可靠解释器。
```

### 5.6 Front-policy blend

正式系统不是完全相信 policy score，也不是完全相信 front-end score，而是融合两者：

```text
final_score = (1 - w) * front_score + w * policy_score
```

当前设置：

```text
w = 0.35
```

这点非常重要，因为 full50 setting 下，纯 policy 会受到候选分布漂移影响，而 front-policy blend 可以利用 front-end 的稳定召回能力，同时保留 student 的 trajectory-aware selection 能力。

---

## 6. 论文贡献点设计

建议写成四个贡献。

### Contribution 1: Knowledge-state-guided evidence acquisition formulation

将 multi-hop RAG 的 evidence selection 形式化为一个 sequential evidence acquisition problem：

```text
u_{t+1} = argmax_{u in C_t} f(q, K_t, u)
```

与普通 reranker 不同，选择证据依赖当前知识状态。

### Contribution 2: Teacher-student trajectory learning framework

提出 teacher-student 框架：

```text
teacher: offline trajectory construction
student: online evidence selection policy
```

Teacher 提供结构化监督，student 学习轻量化在线策略。

### Exploratory component: Typed deficit and contribution supervision

不仅训练 ranking label，还引入：

```text
d_t*: typed deficit
c_t*: candidate contribution
StopLabel: auxiliary stopping supervision
```

该设计尝试让 student 表达当前状态下候选证据对缺口的补充作用。不过 v21 统一消融未观察到稳定的最终性能收益，因此它不应再列为与 knowledge-state policy 同等级的已验证贡献。

### Contribution 4: Systematic empirical study under noisy candidate pools

在 HotpotQA 3000 条评测上系统比较：

```text
BM25-RAG
Dense-RAG
Hybrid-RAG
First
Random
Gold Oracle
KBS Official
```

并做：

```text
ablation study
candidate size sensitivity
failure / trajectory analysis
```

证明方法在证据覆盖率和答案质量上优于常规 RAG baseline。

---

## 7. 实验设置

### 7.1 Dataset

使用 HotpotQA distractor setting。

正式评测数据：

```text
data/hotpotqa_distractor_eval_3000_cand50
```

规模：

```text
qids = 3000
steps = 7296
```

### 7.2 Models

Student policy backbone：

```text
DeBERTa-v3-large
```

Dense retriever：

```text
BAAI/bge-large-en-v1.5
```

Answer generator：

```text
DeepSeek Chat
```

### 7.3 Main official setting

```text
selector = hybrid_policy
policy_context_source = online_state
dense_query_mode = state
front_pool_k = 30
front_fusion = rrf
local_expansion_window = 1
candidate_top_k = 10
select_top_k = 5
policy_score_mode = front_policy_blend
policy_blend_weight = 0.35
answer_mode = json
```

---

## 8. Evaluation Metrics

论文中建议将指标分成三层。

### 8.1 Step-level evidence selection metrics

```text
step_acc@1
step_acc@2
step_acc@3
step_acc@5
step_selected_contains_gold
```

含义：

```text
衡量每一步 evidence selection 是否能选中 gold evidence。
```

### 8.2 Trajectory-level metrics

```text
trajectory_all_steps_correct
full_gold_doc_coverage
full_gold_unit_coverage
any_gold_doc_selected
avg_trajectory_length
```

含义：

```text
衡量整个多步证据获取轨迹是否覆盖了完整支撑证据。
```

### 8.3 Answer-level metrics

```text
answer_em
answer_f1
answer_contains
avg_answer_tokens
avg_answer_latency
```

含义：

```text
衡量最终 context 给 LLM 后，答案是否正确，以及上下文成本。
```

---

## 9. 主实验结果

### 9.1 Main comparison

| Method | EM | F1 | Step@5 | Full Doc Cov. | Full Unit Cov. | Avg Tokens |
|---|---:|---:|---:|---:|---:|---:|
| First | 0.3570 | 0.4551 | 0.1364 | 0.2977 | 0.0250 | 516.74 |
| Random | 0.3553 | 0.4617 | 0.1409 | 0.3133 | 0.0250 | 521.39 |
| BM25-RAG | 0.5017 | 0.6297 | 0.6473 | 0.6230 | 0.3697 | 356.52 |
| Dense-RAG | 0.5457 | 0.6775 | 0.7560 | 0.7523 | 0.5700 | 362.82 |
| Hybrid-RAG | 0.5563 | 0.6884 | 0.7767 | 0.7877 | 0.5810 | 362.03 |
| KBS Official | 0.5943 | 0.7390 | 0.8300 | 0.8903 | 0.7113 | 382.15 |
| Gold Oracle | 0.6520 | 0.8023 | 1.0000 | 1.0000 | 1.0000 | 532.43 |

### 9.2 Main result interpretation

KBS Official 明显优于 BM25、Dense 和 Hybrid：

```text
相比 BM25:
EM: 0.5017 -> 0.5943
F1: 0.6297 -> 0.7390
Full Unit Coverage: 0.3697 -> 0.7113

相比 Dense:
EM: 0.5457 -> 0.5943
F1: 0.6775 -> 0.7390
Full Unit Coverage: 0.5700 -> 0.7113

相比 Hybrid:
EM: 0.5563 -> 0.5943
F1: 0.6884 -> 0.7390
Full Unit Coverage: 0.5810 -> 0.7113
```

可以写成：

```text
The proposed KBS system consistently improves both evidence coverage and answer quality over lexical, dense, and hybrid retrieval baselines. This indicates that knowledge-state-guided evidence acquisition is more effective than one-shot retrieval for multi-hop RAG.
```

---

## 10. Ablation Study

### 10.1 Ablation results

| Variant | EM | F1 | Step@5 | Full Unit Cov. | Avg Tokens |
|---|---:|---:|---:|---:|---:|
| Official | 0.5930 | 0.7386 | 0.8300 | 0.7113 | 382.16 |
| w/o online_state | 0.5973 | 0.7406 | 0.8309 | 0.7130 | 381.49 |
| w/o front-policy blend | 0.6053 | 0.7498 | 0.6916 | 0.7487 | 487.03 |
| w/o policy | 0.5540 | 0.6873 | 0.7767 | 0.5810 | 362.01 |
| w/o local expansion | 0.5950 | 0.7389 | 0.8300 | 0.7113 | 382.11 |
| w/o dense | 0.5013 | 0.6281 | 0.6473 | 0.3697 | 356.53 |
| w/o BM25 | 0.5457 | 0.6778 | 0.7560 | 0.5700 | 362.81 |
| deficit role score | 0.6047 | 0.7489 | 0.6938 | 0.7477 | 486.26 |
| deficit contribution score | 0.6030 | 0.7464 | 0.6845 | 0.6787 | 463.62 |

### 10.2 How to interpret ablation

#### Hybrid front-end is important

`w/o dense` 和 `w/o BM25` 明显低于 Official，说明 BM25 和 Dense 互补。

可以写：

```text
Removing either BM25 or dense retrieval degrades both evidence coverage and answer quality, confirming that lexical and semantic signals are complementary in the candidate generation stage.
```

#### Student policy is useful

`w/o policy` 低于 Official：

```text
F1: 0.6873 -> 0.7386
Full Unit Coverage: 0.5810 -> 0.7113
```

说明 student policy 对证据选择有实质提升。

#### Front-policy blend improves evidence precision

`w/o front-policy blend` 的 answer F1 略高，但 Step@5 明显下降：

```text
Official Step@5 = 0.8300
w/o front-policy blend Step@5 = 0.6916
```

同时 token 明显增加：

```text
Official tokens = 382.16
w/o blend tokens = 487.03
```

解释：

```text
Pure policy ranking selects longer/noisier contexts that may sometimes help the LLM answer, but it is less precise in selecting gold evidence. Front-policy blending provides a better trade-off between evidence accuracy and context cost.
```

注意不要写成：

```text
front-policy blend 在所有指标上都更好。
```

更严谨写法：

```text
Front-policy blending substantially improves step-level evidence selection while maintaining competitive answer quality with lower context cost.
```

#### online_state 当前不是强提升点

`w/o online_state` 和 Official 基本持平，甚至略高。

因此不能强行写：

```text
online_state 显著提升效果。
```

可以写：

```text
The online state implementation achieves performance comparable to the legacy state construction while enabling a cleaner and fully online evidence acquisition pipeline.
```

重点强调它的系统一致性，而不是性能提升。

#### local expansion 当前不是主要贡献

`w/o local expansion` 与 Official 几乎一致。

可以写：

```text
Local expansion has limited effect under the current HotpotQA sentence-level candidate setting, suggesting that most useful neighboring evidence has already been captured by the fused retrieval pool.
```

---

## 11. Candidate Size Sensitivity

### 11.1 Results

| Candidate Size | EM | F1 | Step@1 | Step@5 | Full Unit Cov. | Avg Tokens |
|---|---:|---:|---:|---:|---:|---:|
| cand10 | 0.5950 | 0.7397 | 0.3751 | 0.8300 | 0.7113 | 382.15 |
| cand15 | 0.6040 | 0.7503 | 0.3614 | 0.8446 | 0.7530 | 388.57 |
| cand20 | 0.6133 | 0.7610 | 0.3461 | 0.8494 | 0.7743 | 393.74 |
| cand50 | 0.6293 | 0.7788 | 0.3083 | 0.8496 | 0.8250 | 411.97 |

### 11.2 Interpretation

候选池变大后：

```text
Full Unit Coverage 提升
Answer F1 提升
Step@1 下降
Avg Tokens 增加
```

这说明：

```text
更大的候选池带来更高召回，但也引入更多噪声，使 top-1 精确选择更难。
```

论文中要强调：

```text
Candidate pool size creates a trade-off between recall and selection precision.
```

可以写：

```text
Increasing the candidate pool improves evidence recall and downstream answer quality, but reduces top-1 selection precision. This confirms that candidate pool distribution is a crucial factor in trajectory-aware evidence acquisition.
```

注意：之前我们预计 full50 可能会表现差，但最终 cand50 的答案指标最高。这不是坏事，反而说明：

```text
我们的 front-end compression + policy 在 full50 stress setting 下仍然能保持较强鲁棒性。
```

---

## 12. Failure / Trajectory Analysis

### 12.1 Front-end trace results

核心结果：

```text
p_gold_in_pool50 = 1.0000
p_gold_in_compressed = 0.8958
p_policy_hit_at_select_top_k = 0.8343
p_policy_hit_given_compressed = 0.9313
```

解释：

```text
原始 full50 候选池中几乎总能包含 gold evidence。
经过 compression 后仍保留约 89.6% gold evidence。
当 gold evidence 存在于 compressed pool 中时，policy 有 93.1% 概率能在 top-5 选中它。
```

这说明：

```text
主要损失来自 compression 阶段丢掉 gold evidence，以及少量 policy ranking failure。
```

### 12.2 Stage hit rates

| Stage | Hit@1 | Hit@5 | Hit@10 | Hit@50 |
|---|---:|---:|---:|---:|
| BM25 | 0.2917 | 0.7442 | 0.8472 | 1.0000 |
| Dense | 0.3912 | 0.8019 | 0.9047 | 1.0000 |
| Fused | 0.3601 | 0.7995 | 0.8953 | 1.0000 |
| Compressed Pool | 0.3503 | 0.7992 | 0.8958 | 0.8958 |
| Policy | 0.3775 | 0.8343 | 0.8958 | 0.8958 |

可以写：

```text
The policy improves top-5 evidence selection over the compressed pool, indicating that the student policy contributes beyond front-end retrieval.
```

### 12.3 Failure types

Failure types:

```text
correct: 2754
answer_string_visible_but_not_gold: 840
other_distractor: 1888
same_doc_wrong_sentence: 930
gold_missing_after_compression: 760
same_entity_or_title_overlap: 124
```

解释：

```text
1. same_doc_wrong_sentence 表示模型找到了正确文档但句子粒度不准确。
2. answer_string_visible_but_not_gold 表示候选包含答案字符串，但不是标注 gold evidence。
3. gold_missing_after_compression 表示 front-end compression 阶段已经丢失 gold evidence。
4. other_distractor 表示普通干扰项。
```

可写结论：

```text
Many errors are not simple retrieval failures, but fine-grained evidence discrimination failures within the same document or semantically similar distractors.
```

---

## 13. Deficit Analysis 的写法

当前 deficit 汇总结果：

```text
predicted_deficit_steps = 7296
teacher_deficit_steps = 0
overall_mae = null
role-wise MAE = null
monotonic_non_increase_rate = null
```

这说明目前的 report 记录了 predicted deficit，但没有成功和 teacher d_t* 对齐。

因此论文中暂时不要写：

```text
Our deficit predictor achieves low MAE.
```

可以写：

```text
We incorporate typed deficit supervision into student training and expose deficit prediction as an auxiliary signal. A more fine-grained calibration analysis of predicted deficits is left for future work.
```

如果后续补齐 teacher deficit 对齐，可以再加 deficit MAE 表。

---

## 14. 论文结构建议

### Abstract

摘要需要包含：

```text
问题：multi-hop RAG 需要逐步获取证据，普通 retrieval/reranking 忽略当前知识状态。
方法：提出 knowledge-state-guided evidence acquisition framework。
技术：teacher trajectory + typed deficit/contribution supervision + student policy。
实验：HotpotQA 3000 条，超过 BM25/Dense/Hybrid baselines。
结论：提升 evidence coverage 和 answer quality。
```

### 1. Introduction

写作逻辑：

```text
1. RAG 在复杂问答中很重要。
2. Multi-hop QA 需要多个互补证据。
3. 现有 retriever/reranker 多是 single-shot relevance matching。
4. 它们没有显式建模当前已经知道什么、还缺什么。
5. 因此提出 knowledge-state-guided evidence acquisition。
6. 我们使用 teacher 构造轨迹，用 student 学习在线策略。
7. 实验显示优于主流 baselines。
```

### 2. Related Work

建议分为：

```text
2.1 Retrieval-Augmented Generation
2.2 Multi-Hop Question Answering
2.3 Neural Retrieval and Reranking
2.4 Iterative / Agentic Retrieval
2.5 Knowledge State and Trajectory Learning
```

### 3. Problem Formulation

需要形式化：

```text
Question q
Candidate pool R_t
Knowledge state K_t
Candidate evidence u
Policy f(q, K_t, u)
Trajectory T = (u_1, u_2, ..., u_T)
Final answer generation p(a | q, K_T)
```

重点是：

```text
evidence selection is sequential and state-dependent.
```

### 4. Method

建议小节：

```text
4.1 Framework Overview
4.2 Retrieval Front-end
4.3 Knowledge State Construction
4.4 Teacher Trajectory Construction
4.5 Typed Deficit and Contribution Supervision
4.6 Student Policy Learning
4.7 Online RAG Inference
```

### 5. Experiments

建议小节：

```text
5.1 Dataset and Evaluation Setup
5.2 Baselines
5.3 Metrics
5.4 Implementation Details
5.5 Main Results
5.6 Ablation Study
5.7 Candidate Size Sensitivity
5.8 Failure and Trajectory Analysis
```

### 6. Discussion

讨论：

```text
1. 为什么 evidence coverage 提升会带来 answer F1 提升。
2. 为什么 full50 下 Step@1 降低但 answer F1 提升。
3. 为什么 pure policy 有时答案 F1 高但 evidence precision 差。
4. 当前 deficit calibration 分析还不充分。
```

### 7. Conclusion

总结：

```text
本文提出 knowledge-state-guided evidence acquisition framework。
通过 teacher-student trajectory learning 训练 student policy。
在 HotpotQA 上显著提升 evidence coverage 和 answer quality。
未来工作包括更强 deficit calibration、跨数据集验证、开放域检索扩展。
```

---

## 15. KBS 投稿时应强化的点

KBS 更喜欢系统性、知识建模、可解释分析，而不是单纯刷分。

因此要强化：

```text
1. Knowledge state modeling
2. Sequential evidence acquisition
3. Teacher-student structured supervision
4. Typed deficit / contribution
5. System-level evaluation
6. Failure analysis
```

不要只强调：

```text
我们训练了一个 DeBERTa reranker。
```

应该强调：

```text
我们提出了一个 knowledge-state-aware evidence acquisition framework。
DeBERTa 只是 student policy 的实现方式。
```

---

## 16. 当前结果中需要谨慎处理的地方

### 16.1 online_state 没有明显提升

不要写：

```text
online_state significantly improves performance.
```

建议写：

```text
online_state provides a fully online and internally consistent inference route, achieving performance comparable to the legacy state construction.
```

### 16.2 local expansion 没有明显提升

不要强行说 local expansion 很关键。

建议写：

```text
Local expansion has limited effect in the sentence-level HotpotQA setting, but it provides a general mechanism for exposing neighboring evidence in less structured corpora.
```

### 16.3 pure policy / deficit score 的 answer F1 略高

这个现象要解释为 trade-off：

```text
Pure policy and deficit-aware scoring may introduce more context and improve answer generation, but they reduce step-level evidence precision and increase token cost.
```

### 16.4 deficit MAE 还没形成有效结果

可以写设计和训练，不要写强实验结论。

---

## 17. 论文最核心的图表

建议至少做 5 张表/图。

### Table 1: Main Results

内容：

```text
BM25 / Dense / Hybrid / KBS Official / Gold Oracle
EM / F1 / Step@5 / Full Unit Coverage / Tokens
```

### Table 2: Ablation Study

内容：

```text
Official
w/o online_state
w/o front-policy blend
w/o policy
w/o dense
w/o BM25
deficit variants
```

### Table 3: Candidate Size Sensitivity

内容：

```text
cand10 / cand15 / cand20 / cand50
EM / F1 / Step@1 / Step@5 / Full Unit Coverage / Tokens
```

### Table 4: Front-end and Policy Stage Hit Rates

内容：

```text
BM25 / Dense / Fused / Compressed Pool / Policy
Hit@1 / Hit@5 / Hit@10 / Hit@50
```

### Table 5: Failure Type Distribution

内容：

```text
same_doc_wrong_sentence
answer_string_visible_but_not_gold
other_distractor
gold_missing_after_compression
same_entity_or_title_overlap
```

### Figure 1: Framework Overview

画法：

```text
Question
  → Retrieval front-end
  → Candidate pool
  → Knowledge state K_t
  → Student policy
  → Selected evidence
  → Updated K_t
  → Final context
  → LLM answer

Teacher trajectory construction 放在训练阶段支线。
```

### Figure 2: Candidate Pool Trade-off

横轴：

```text
candidate size
```

纵轴：

```text
Answer F1
Step@1
Step@5
Full Unit Coverage
```

展示：

```text
candidate size 增大，coverage/F1 提升，但 Step@1 下降。
```

---

## 18. 推荐摘要初稿

```text
Retrieval-augmented generation (RAG) has become a widely used paradigm for knowledge-intensive question answering. However, existing retrieval and reranking methods often select evidence based on static query-passage relevance, which is insufficient for multi-hop reasoning where the next useful evidence depends on what has already been acquired. In this paper, we propose a knowledge-state-guided evidence acquisition framework for multi-hop RAG. The framework maintains an explicit knowledge state during evidence acquisition and trains a student evidence selection policy from teacher-constructed trajectories. Beyond standard ranking labels, the teacher derives typed deficit and contribution signals, encouraging the student to select evidence that complements the current knowledge state. During online inference, a hybrid retrieval front-end first constructs a candidate pool, and the student policy selects evidence conditioned on the current knowledge state to form the final context for answer generation. Experiments on 3,000 HotpotQA examples show that our method substantially improves both evidence coverage and answer quality compared with BM25, dense retrieval, and hybrid retrieval baselines. Further analyses reveal the impact of candidate pool size and identify major failure modes in multi-hop evidence acquisition.
```

---

## 19. 推荐 Introduction 逻辑

Introduction 可以按 6 段写。

### Paragraph 1: RAG 背景

```text
RAG 通过外部知识增强 LLM，在知识密集型任务中表现很好。
```

### Paragraph 2: Multi-hop RAG 难点

```text
Multi-hop QA 需要多个证据共同支撑答案。
单次 retrieval 很难一次性找齐所有证据。
```

### Paragraph 3: 现有方法局限

```text
BM25 / dense retriever / reranker 多基于静态 query-passage relevance。
它们没有显式考虑当前已经获得哪些证据，以及还缺什么证据。
```

### Paragraph 4: 我们的观点

```text
Evidence selection should be modeled as sequential evidence acquisition conditioned on a knowledge state.
```

### Paragraph 5: 我们的方法

```text
提出 knowledge-state-guided evidence acquisition framework。
Teacher 离线构造轨迹。
Student 学习 ranking / deficit / contribution。
Online RAG 中逐步选择证据并更新 K_t。
```

### Paragraph 6: 实验和贡献

```text
HotpotQA 3000 条实验。
相比 BM25、Dense、Hybrid 显著提升。
提供消融、candidate size sensitivity、failure analysis。
```

---

## 20. 最终写作建议

现在可以开始论文初稿。

优先顺序：

```text
1. 先写 Method，确保主线清楚。
2. 再写 Experiment，先把表格放进去。
3. 再写 Introduction，围绕“不是 reranker，而是 knowledge-state-guided evidence acquisition”展开。
4. 最后写 Related Work 和 Abstract。
```

写作时最重要的一句话：

```text
Our contribution is not a stronger standalone retriever, but a knowledge-state-guided evidence acquisition framework that learns to select complementary evidence along a multi-hop reasoning trajectory.
```
