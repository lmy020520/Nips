# Policy-RAG 项目阶段性总结

本文档总结当前项目从数据构建、模型训练、错误分析、RAG 接入到对比实验设计的完整工作流。当前项目的核心结论是：我们训练的模型不应被简单表述为传统 reranker，而应定位为一个 **trajectory-aware evidence selection policy model**，并将其接入 RAG 的检索/证据选择阶段，形成 **Policy-RAG**。

## 1. 项目目标

HotpotQA 是典型的多跳问答任务。传统 RAG 常见流程是：

```text
Question -> Retriever -> Top-k passages -> LLM answer
```

但多跳问答的问题在于：一次性检索 top-k 往往不能准确覆盖完整推理链，尤其是 bridge question、comparison question 和需要逐步补全证据状态的场景。

因此我们希望构建一个更适合多跳问答的 RAG 流程：

```text
Question
  -> Candidate Generation
  -> Trajectory-aware Evidence Selection Policy
  -> Evidence Notebook / State Update
  -> LLM Answer
```

其中，模型不是只判断 query 和 passage 的静态相关性，而是根据：

- 当前问题 `Question`
- 当前已经选中的证据状态 `K_t / Notebook`
- 当前候选证据集合 `C_t`
- 下一步应该选择的 gold evidence/action

来决定下一步应该选择哪条证据。因此更准确的表述是：

```text
Trajectory-aware retrieval policy model
```

或者：

```text
Policy-RAG evidence selector
```

## 2. 核心方法定位

### 2.1 不是传统 reranker

传统 cross-encoder reranker 通常学习：

```text
score(query, passage)
```

我们的模型学习的是：

```text
score(question, current_state, candidate_evidence)
```

其中 `current_state` 包含当前已经选择的证据、上下文 notebook、trajectory step 等信息。因此它学习的不只是相关性，而是 **retrieval decision policy**。

### 2.2 在 RAG 系统中的位置

我们的 Policy-RAG 组件位于 RAG 的检索阶段内部：

```text
Candidate evidence pool
  -> Policy model selects evidence conditioned on current trajectory state
  -> Selected evidence forms context
  -> LLM generates answer
```

它不是替代 LLM，也不是答案生成模块，而是负责把多跳问答需要的证据更准确地组织进上下文。

## 3. 数据集构建过程

### 3.1 早期数据集版本

项目从 HotpotQA distractor 数据集开始，逐步构建了多个版本的数据：

| Version | 作用 |
|---|---|
| `v3/v4` | 初步构建 trajectories、raw units、targets、samples |
| `v5_llm_dataset` | 引入 LLM role 标注，形成可训练的 ranking/policy 数据 |
| `v6_10k_llm` | 扩大到约 10k qids，用于更稳定训练 |
| `v7_10k_llm_prestep` | 关键版本，引入 pre-step trajectory/state 表示，模型性能显著提升 |
| `v8_15k_llm_prestep` | 扩大训练数据尝试，但效果不一定稳定提升 |

### 3.2 当前最佳训练数据

当前效果最好的主线数据是：

```text
data/hotpotqa_distractor_v7_10k_llm_prestep
```

核心文件包括：

```text
samples/train.jsonl
samples/val.jsonl
samples/test.jsonl
unit_registry/raw_units_train.jsonl
unit_registry/raw_units_val.jsonl
unit_registry/raw_units_test.jsonl
queries/test.jsonl
targets/test.jsonl
```

其中 `samples/*.jsonl` 是模型训练和评测的核心，每条样本表示一个 trajectory step 下的 evidence selection decision。

### 3.3 新的 3000 条独立评测集

为了做更正式的 RAG 对比实验，我们新增了独立评测集构建脚本：

```text
scripts/prepare_hotpotqa_policy_rag_eval.py
scripts/validate_hotpotqa_policy_rag_eval.py
```

目标是从 HotpotQA validation/dev 中抽取 3000 条，构建成 Policy-RAG 可直接评测的格式。

当前推荐的正式评测集路径：

```text
data/hotpotqa_distractor_eval_3000_cand10
```

选择 `cand10` 的原因是：它与模型训练阶段的候选规模更接近，每一步约 10 个候选证据，因此更适合作为主实验设置。全候选版本也保留为更难的 open-candidate setting。

## 4. 模型训练过程

### 4.1 基础模型

主要尝试过：

| Model | 结果 |
|---|---|
| `microsoft/deberta-v3-base` | 可以跑通，但准确率较低 |
| `microsoft/deberta-v3-large` | 当前主力模型，效果最好 |
| BGE reranker / 其他更强模型 | 有尝试，但没有超过当前 DeBERTa v3 large 主线 |

当前最佳模型使用：

```text
models/deberta-v3-large
```

当前最佳 checkpoint：

```text
outputs/ranker/frozen_deberta_v3_large_v7_main_val08252/best_model.pt
```

### 4.2 关键训练结果

早期 v5/v6 训练准确率大致在 0.60-0.70 区间。引入 v7 pre-step trajectory 表示后，效果明显提升。

关键结果之一：

```text
Best epoch: 4, best val acc: 0.8252
test_acc: 0.8257
```

这成为后续 RAG 接入的主模型。

### 4.3 训练中解决的问题

训练过程中遇到并解决过：

- `protobuf` / `tiktoken` 缺失导致 tokenizer 加载失败
- Hugging Face 下载超时，改用 `HF_ENDPOINT=https://hf-mirror.com`
- `fp16` 下出现 `Attempting to unscale FP16 gradients`
- DeBERTa large 在 RTX 3090 上 OOM，调整 `batch_size`、`max_length`、`grad_accum_steps`
- 多卡可用但最终主要采用单卡稳定训练
- Git LFS 下载大文件卡死，改用 tar 包或 raw 脚本下载

## 5. 错误分析与性能提升

### 5.1 早期错误分析

我们使用：

```text
scripts/analyze_hotpotqa_ranker_errors.py
```

分析模型错误，发现：

- `unlabeled` 正样本是早期数据构造中的主要问题之一
- `bridge` 和 `support` 的同角色近邻混淆较多
- 第二步 trajectory 选择比第一步更难
- 大候选集合下容易出现 near-miss

### 5.2 数据修复方向

主要修复策略：

- 清理 `unlabeled positive`
- 增加 pre-step state
- 加 hard negative / near-miss mining
- 调整学习率、epoch、max length
- 尝试扩大数据到 15k

最终发现，**数据构造方式和 trajectory state 设计** 比单纯扩大模型或训练轮数更重要。

## 6. LLM verifier 尝试

曾经尝试用 DeepSeek 作为 top-2 verifier，对模型 top-2 候选做二次判断。

结果示例：

```text
base_acc: 0.825708
basic_fixed: 26
targeted_extra_fixed: 15
combined_correct: 420
combined_acc: 0.915033
```

这说明 LLM verifier 可以显著提升最终选择准确率，但我们最终认为它不适合作为主方法，因为：

- 需要额外 API 成本
- 推理延迟增加
- 论文贡献会偏向大模型后处理，而不是小模型本身

因此主线仍然回到：提升小模型自身的 evidence selection 能力，并将其作为 RAG 检索策略模块。

## 7. Policy-RAG 系统实现

核心脚本：

```text
scripts/run_hotpotqa_policy_rag.py
```

该脚本支持多种 selector：

| Selector | 含义 |
|---|---|
| `policy` | 我们的 trajectory-aware policy model |
| `bm25` | BM25 baseline |
| `dense` | BGE dense retriever baseline |
| `dense_policy` | Dense 召回后再用 policy 选择 |
| `first` | 直接取候选前几条 |
| `random` | 随机选择候选 |
| `gold_oracle` | Gold evidence only reference |

支持生成答案：

```text
--generate-answers
--answer-mode json
```

并支持 API 断线恢复：

```text
--llm-max-retries 10
--answer-cache-dir ...
```

这样即使 DeepSeek API 中途断开，也可以重复同一命令继续跑，已完成的 qid 会从 cache 读取。

## 8. 383 条测试集对比实验

基于 `v7_10k_llm_prestep` 的 test split，我们完成了 383 个 qid 的完整对比实验。

### 8.1 实验结果

| Method | QIDs | Answer EM | Answer F1 | Answer Contains | Full Gold Doc Coverage | Full Gold Unit Coverage | Step Selected Contains Gold | Avg Tokens | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| First | 383 | 0.3368 | 0.4462 | 0.3838 | 0.4752 | 0.4360 | 0.4815 | 336.68 | 0.852 |
| Random | 383 | 0.4047 | 0.5211 | 0.4595 | 0.6423 | 0.5561 | 0.5970 | 342.76 | 0.838 |
| BM25-RAG | 383 | 0.4648 | 0.5963 | 0.5248 | 0.8068 | 0.7311 | 0.7756 | 332.74 | 1.361 |
| Dense-RAG | 383 | 0.4883 | 0.6312 | 0.5770 | 0.9347 | 0.9034 | 0.9194 | 329.18 | 1.325 |
| Gold-only | 383 | 0.4935 | 0.6377 | 0.5744 | 1.0000 | 1.0000 | 1.0000 | 342.23 | 0.871 |
| Dense+Policy | 383 | 0.5535 | 0.6907 | 0.6319 | 0.9948 | 0.9843 | 0.9346 | 345.09 | 1.215 |
| **Policy-RAG** | 383 | **0.5587** | **0.6955** | **0.6371** | **0.9974** | **0.9843** | **0.9346** | 345.76 | **0.827** |

### 8.2 主要结论

Policy-RAG 在答案指标上最好：

```text
Answer EM = 0.5587
Answer F1 = 0.6955
```

相比主流 baseline：

| 对比 | EM 提升 | F1 提升 |
|---|---:|---:|
| vs First | +22.19 | +24.93 |
| vs Random | +15.40 | +17.44 |
| vs BM25-RAG | +9.40 | +9.92 |
| vs Dense-RAG | +7.05 | +6.43 |
| vs Dense+Policy | +0.52 | +0.48 |

关键证据覆盖指标：

```text
full_gold_doc_coverage = 0.9974
full_gold_unit_coverage = 0.9843
step_selected_contains_gold = 0.9346
```

这些指标说明：Policy-RAG 的主要优势不是让 LLM 更强，而是让进入上下文的 evidence 更准确、更完整。

### 8.3 Gold-only 的解释

`Gold-only` 的 coverage 是 1.0，但答案指标低于 Policy-RAG。这不矛盾，因为：

- Gold-only 只提供标注证据
- Policy-RAG top-5 可能包含额外有帮助的上下文
- LLM 回答时额外上下文可能帮助实体消歧和答案表达

因此 `Gold-only` 不应写成 theoretical upper bound，更适合称为：

```text
Gold Evidence Only reference
```

## 9. 3000 条独立评测集

为了得到更正式、更有说服力的实验结果，我们开始构建 3000 条独立评测集。

### 9.1 全候选版本 smoke test

路径：

```text
data/hotpotqa_distractor_eval_3000
```

该版本每一步候选是 HotpotQA 原始 context 中的所有 sentence，平均候选数约 50。

Smoke20 结果：

```text
step_acc@1 = 0.12
step_acc@5 = 0.36
answer_em = 0.0
answer_f1 = 0.0
```

其中答案全 0 的原因是第一次 DeepSeek API key 没有正确传入；检索指标偏低则说明全候选设置远难于训练时的候选规模。

### 9.2 cand10 版本 smoke test

路径：

```text
data/hotpotqa_distractor_eval_3000_cand10
```

该版本每一步保留 10 个候选，更接近训练设置。

Smoke20 结果：

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

这个结果说明：

- 新 3000 数据集构建流程可用
- DeepSeek API 调用正常
- cand10 设置与当前模型能力匹配
- 可以进入 3000 条完整对比实验

## 10. 正式对比实验设计

正式实验建议使用：

```text
data/hotpotqa_distractor_eval_3000_cand10
```

主要方法：

| Method | 类型 | 作用 |
|---|---|---|
| First | Ablation / weak baseline | 不做智能选择，直接取前几个候选 |
| Random | Ablation / weak baseline | 随机选择候选 |
| BM25-RAG | Main baseline | 传统词法检索 RAG |
| Dense-RAG | Main baseline | BGE dense retriever RAG |
| Dense+Policy | Hybrid baseline | Dense 召回后用 policy 选择 |
| Gold-only | Oracle-style reference | 只提供 gold evidence |
| Policy-RAG | Ours | trajectory-aware evidence selection |

主要指标：

| 指标类型 | 指标 |
|---|---|
| Answer Quality | `answer_em`, `answer_f1`, `answer_contains` |
| Evidence Quality | `full_gold_doc_coverage`, `full_gold_unit_coverage`, `step_selected_contains_gold` |
| Step Accuracy | `step_acc@1`, `step_acc@2`, `step_acc@3`, `step_acc@5` |
| Efficiency | `avg_answer_tokens`, `avg_answer_latency` |
| Robustness | `answer_errors` |

## 11. 当前论文贡献点

当前工作可以总结为三个贡献：

### 11.1 提出 trajectory-aware evidence selection policy

区别于传统 reranker，模型显式考虑当前 retrieval trajectory state，从而学习下一步 evidence selection decision。

### 11.2 构建 Policy-RAG 多跳证据组织流程

将 policy model 放入 RAG 检索阶段，形成：

```text
Candidate Generation -> Policy Evidence Selection -> Context Assembly -> LLM Answer
```

在 HotpotQA 多跳问答上显著提升 evidence coverage 和 answer quality。

### 11.3 系统化对比 BM25、Dense、Random、First、Gold-only

383 条实验中，Policy-RAG 超过 BM25-RAG 和 Dense-RAG，并在证据覆盖上接近 Gold-only。

下一步 3000 条实验将进一步验证方法稳定性。

## 12. 当前最重要的路径

### 最佳模型

```text
outputs/ranker/frozen_deberta_v3_large_v7_main_val08252/best_model.pt
```

### 模型目录

```text
models/deberta-v3-large
```

### 当前主评测集

```text
data/hotpotqa_distractor_eval_3000_cand10
```

### RAG 脚本

```text
scripts/run_hotpotqa_policy_rag.py
```

### 构建 3000 eval 数据脚本

```text
scripts/prepare_hotpotqa_policy_rag_eval.py
scripts/validate_hotpotqa_policy_rag_eval.py
```

## 13. 下一步计划

1. 跑完 `eval3000_cand10` 上的完整 Policy-RAG。
2. 跑完以下完整对比：
   - BM25-RAG
   - Dense-RAG
   - Dense+Policy
   - Random
   - First
   - Gold-only
3. 汇总 3000 条正式实验表格。
4. 如时间允许，补充全候选 setting：

```text
data/hotpotqa_distractor_eval_3000
```

5. 将论文表述统一为：

```text
trajectory-aware evidence selection policy
Policy-RAG
multi-hop evidence assembly
```

避免将核心模型简单称为 reranker。

## 14. 给老师看的简短总结

我们的方法不是传统 reranker，而是一个 trajectory-aware evidence selection policy。它根据问题、当前已选证据状态和候选证据集合，逐步选择下一条 evidence，并将最终证据集合交给 LLM 回答。

在 383 个 HotpotQA 测试 qid 上，Policy-RAG 相比 BM25-RAG 和 Dense-RAG 都有明显提升：

```text
Policy-RAG: EM 0.5587, F1 0.6955
Dense-RAG:  EM 0.4883, F1 0.6312
BM25-RAG:   EM 0.4648, F1 0.5963
```

同时，Policy-RAG 的证据覆盖率非常高：

```text
full_gold_doc_coverage = 0.9974
full_gold_unit_coverage = 0.9843
step_selected_contains_gold = 0.9346
```

这说明我们的主要优势在于多跳证据选择和上下文组织，而不是简单依赖更强生成模型。

