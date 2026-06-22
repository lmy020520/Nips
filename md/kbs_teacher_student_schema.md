# KBS Teacher-Student Schema

本文档固定当前 KBS evidence acquisition 流程中，teacher 数据、student 训练样本、policy-RAG 推理之间共用的字段协议。它的目的不是描述某一次实验结果，而是保证后续重建数据集、替换 front-end、训练新模型时，整条流程仍然对齐。

## 1. 总体流程

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

每一条训练样本对应一个 prefix state：

```text
(q, t, H_t, A_t, S_t, K_t, C_t) -> labels
```

其中 `C_t` 是当前步候选池，`labels` 是 teacher 给 student 的结构化监督信号。

## 2. Top-Level Record

每条 sample 至少包含：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `qid` | string | 问题 ID |
| `t` | int | 当前 trajectory step |
| `question` | string | 原始问题 |
| `state` | object | 当前知识状态 |
| `candidates` | object | 当前候选证据池 |
| `labels` | object | teacher 派生监督 |

推荐保留但不作为最小强制项：

| 字段 | 作用 |
| --- | --- |
| `build_meta` | 数据构建版本、front-end、候选池来源 |
| `derived_payloads` | derived unit 的文本和来源 |
| `meta` | trajectory 状态、辅助统计 |

## 3. Knowledge State

`state` 必须包含：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `H_t` | list | 已选 evidence unit 序列 |
| `A_t` | object | raw ledger，记录已覆盖的原始证据 |
| `S_t` | object | evidence-centric notebook state |
| `K_t` | string | compiled context，用于 student/policy 输入 |

`K_t` 不是随意拼接文本，而是由 `H_t/A_t/S_t` 渲染得到的当前知识状态文本。student 的核心输入应是：

```text
context_t = question + K_t
candidate = u
```

## 4. Candidate Pool

`candidates` 必须包含：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `C_t` | list[string] | 当前步最终可选候选 |
| `R_t` | list[string] | raw 候选 |
| `G_t_final` | list[string] | final derived 候选 |
| `G_t_aux` | list[string] | auxiliary derived 候选 |
| `G_t_illegal` | list[string] | illegal derived 候选 |

`candidate_provenance` 推荐覆盖 `C_t` 中所有候选，用于后续可解释分析和错误定位。

基本约束：

1. `C_t` 不能为空。
2. `C_t` 内不应有重复 unit。
3. `C_t` 应等于 `R_t + G_t_final` 的去重结果，或者在重建 front-end 时显式记录压缩策略。
4. 训练 ranking 时，正样本必须在 `C_t` 中。

## 5. RankingLabel

`labels.ranking_label` 是 student candidate scorer 的主监督。

```json
{
  "positive_unit_id": "...",
  "negative_unit_ids": ["...", "..."],
  "positive_provenance": {},
  "negative_provenance": {}
}
```

语义：

| 字段 | 含义 |
| --- | --- |
| `positive_unit_id` | teacher 在当前 prefix 下选择的下一步证据 |
| `negative_unit_ids` | 当前候选池内未被选择的对照候选 |
| `positive_provenance` | 正样本来源 |
| `negative_provenance` | 负样本来源 |

约束：

1. `positive_unit_id in C_t`。
2. `positive_unit_id` 不应出现在 `negative_unit_ids`。
3. `negative_unit_ids` 应是 `C_t - {positive_unit_id}` 的子集。

## 6. Deficit Label `d_t*`

`labels.d_t_star` 是当前 prefix 的 typed residual deficit。

```json
{
  "d_raw": 0.0,
  "d_br": 0.0,
  "d_dis": 0.0,
  "d_sup": 0.0,
  "d_der": 0.0
}
```

含义：

| 字段 | 含义 |
| --- | --- |
| `d_raw` | raw evidence 总缺口 |
| `d_br` | bridge 缺口 |
| `d_dis` | distinguish/comparison 缺口 |
| `d_sup` | support 缺口 |
| `d_der` | derived/context 缺口 |

该标签用于训练 student 的 deficit predictor，也是 closure-aware control 的基础。

## 7. Contribution Label `c_t*`

`labels.c_t_star` 是 teacher 正样本对 typed deficit 的贡献。

```json
{
  "c_raw": 0.0,
  "c_br": 0.0,
  "c_dis": 0.0,
  "c_sup": 0.0,
  "c_der": 0.0
}
```

语义：

```text
c_t*(u_t+) = positive evidence 对当前 typed deficit 的逐维补缺贡献
```

它不是普通相关性标签，而是解释 teacher 为什么在当前 state 下选择这个证据。

## 8. StopLabel

`labels.stop_label` 是辅助监督，不应作为主控制逻辑。

```json
{
  "should_stop": false,
  "label_type": "non-terminal"
}
```

推荐类型：

| `label_type` | 含义 |
| --- | --- |
| `non-terminal` | 当前 prefix 仍需继续选证据 |
| `continue` | 与 `non-terminal` 等价，用于兼容旧版数据 |
| `near-terminal` | 接近完成，但仍未停止 |
| `terminal` | teacher 认为 trajectory 完成 |
| `abort` | trajectory 构建失败或无效 |

主 stop/control 应依赖：

```text
predicted deficit + candidate contribution / gain
```

`StopLabel` 主要用于校准和分析。

## 9. Student Training Interface

student 至少学习：

| Head | 输入 | 监督 |
| --- | --- | --- |
| Ranking head | `(q, K_t, u)` | `RankingLabel` |
| Deficit head | `(q, K_t, u)` 或 `(q, K_t)` | `d_t*` |
| Contribution/role head | `(q, K_t, u)` | `c_t*` 或 candidate role/contribution label |

当前工程中 v16/v17 已经覆盖：

1. ranking score
2. deficit regression
3. candidate-level role/contribution auxiliary supervision

后续要补强的是：

1. 更严格的 candidate-level contribution value，而不只是 role proxy。
2. 更完整的 online deficit update / stop control。
3. failed trajectory / rollout state 的系统利用。

## 10. Pipeline Acceptance Criteria

在进入大规模实验前，数据和流程至少应满足：

1. schema 校验通过，或只有明确可解释的 warning。
2. `positive_unit_id` 在训练候选池中。
3. `d_t* / c_t* / RankingLabel / StopLabel` 都存在。
4. `K_t` 非空，且可作为 student 和 online policy 的状态输入。
5. policy-RAG 能用同一 checkpoint 和同一候选协议完成逐步选择。
