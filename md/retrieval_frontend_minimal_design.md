# Retrieval Frontend Minimal Design

本文档统一总结 current paper 中 retrieval 前端的三个最小模块：

1. `SeedRetrieve`
2. `SlotSummary`
3. `TopKRetrieve`

目标不是把 retrieval 前端做成一个复杂系统，而是给 **closure-aware control** 提供一个**固定、轻量、弱状态条件化**的 raw candidate generation 接口。主学习仍然发生在 retrieval 之后，而不是 retrieval 本身。

---

## 1. 设计边界

这三个模块共同遵循下面四条边界：

- retrieval 前端必须便宜，不能每轮都调用 LLM
- retrieval 前端不负责 deficit / contribution / stop / compile policy
- retrieval 前端只负责 raw candidate generation
- state conditioning 必须很弱，不能演化成复杂 query rewriting

因此，这三个模块都应保持“能跑、够轻、别扩张”的风格。

---

## 2. 基础对象

### 2.1 问题与索引库

对固定问题 `q`：

- `M_q`：当前问题对应的多视图索引库
- 每个 chunk `c_i` 具有三类视图：
  - `v_sum(i)`：summary view vector
  - `v_chunk(i)`：chunk vector
  - `v_atom(i,j)`：local pseudo-atom vector

### 2.2 当前 notebook state

采用 **evidence-centric notebook**，只保存：

- 最近加入的 raw evidence entries
- 少量 derived notes（如 `bridge_note` / `verification_note`）

不要求 notebook 在 raw 侧承担复杂高层语义抽取。

### 2.3 history

- `H_t`：当前前缀已经选中的知识单元序列

---

## 3. 模块一：`SeedRetrieve`

### 3.1 作用

`SeedRetrieve` 用于构造一个 search-initialized 起点：

\[
P_0 = \mathrm{SeedRetrieve}(q, M_q, k_0)
\]

然后：

\[
H_0 = P_0,\qquad
S_0 = \mathrm{Update}(\varnothing, P_0),\qquad
K_0 = \mathrm{Render}(S_0)
\]

它只负责：

- 在 `t=0` 时给系统一个非空起点
- 提供少量高概率有用的 raw units
- 避免 teacher / online rollout 从空状态开始

它不负责：

- derived generation
- LLM planning
- deficit 估计
- stop 判断

### 3.2 输入输出

输入：

- 原问题 `q`
- 多视图索引库 `M_q`
- 初始化返回数 `k_0`
- coarse shortlist 大小 `K_seed`

输出：

- `P_0`：长度为 `k_0` 的初始化 raw units

### 3.3 最小算法

1. 只用原问题 `q` 做一次 coarse retrieval
2. 对每个 chunk 计算：

\[
Score^{seed}_{coarse}(i)
=
0.5\,sim(q, v_{sum}(i))
+
0.5\,sim(q, v_{chunk}(i))
\]

3. 取 top-`K_seed` chunk shortlist
4. 在 shortlist 内，每个 chunk 暴露 1 个 top-1 atom：

\[
Score^{seed}_{atom}(i,j)=sim(q,v_{atom}(i,j))
\]

5. 合并 chunk candidates 与 atom candidates
6. 做轻量 diversity：尽量选不同 `parent_chunk`
7. 返回前 `k_0` 个 raw units

### 3.4 默认配置

- `k_0 = 2`
- `K_seed = 8`
- 每个 shortlist chunk 暴露 `top-1 atom`
- 初始化尽量选不同 `parent_chunk`

### 3.5 设计理由

- 借鉴 SIT 的“先搜一把，再开始”思想
- 但不引入 SIT 的完整 action policy / RL 训练链条
- 只把它作为 search-initialized `H_0` 的最小实现

---

## 4. 模块二：`SlotSummary`

### 4.1 作用

`SlotSummary` 不是一个复杂 summary 系统，也不是语义总结器。
它只是一个 **retrieval hint builder**。

在当前最小版本里，它的功能是：

- 从 `S_t` 中取最近的一条 raw evidence
- 取最近的一条 derived note（如果有）
- 再附上一句极简单的 need hint

然后用于构造：

\[
q_t = q + \mathrm{SlotSummary}(S_t)
\]

### 4.2 输入输出

输入：

- notebook state `S_t`
- 原问题 `q`

输出：

- 一段极短字符串 `h_t`

### 4.3 极简规则

#### 规则 1：取最近 raw
如果 `S_t` 中存在最近加入的 raw evidence，则写：

```text
Evidence: {last_raw_text}
```

#### 规则 2：取最近 derived note
如果 `S_t` 中存在最近加入的 derived note，则再写：

```text
Note: {last_derived_text}
```

#### 规则 3：生成一句 need hint
从原问题 `q` 里抽一个最短目标短语，例如：

- `Which university did X attend?`
  → `Need: university attended by X`
- `Who acquired Y?`
  → `Need: acquirer of Y`

如果不做模板抽取，则直接退化为：

```text
Need: more evidence for the question
```

### 4.4 输出模板

版本 A：只有 raw

```text
Evidence: {last_raw_text}
Need: {goal_from_q}
```

版本 B：raw + derived

```text
Evidence: {last_raw_text}
Note: {last_derived_text}
Need: {goal_from_q}
```

### 4.5 默认配置

- 最多 1 条 raw
- 最多 1 条 derived note
- 总长度不超过约 160 个字符

### 4.6 为什么必须这么简单

因为如果 `SlotSummary` 再做：

- evidence ranking
- note ranking
- novelty / redundancy / anchor 打分
- 复杂 open-need 推断

那它本身就会膨胀成一个工程，不符合 current paper 的最小实现目标。

---

## 5. 模块三：`TopKRetrieve`

### 5.1 作用

`TopKRetrieve` 负责在每一轮根据：

\[
q_t = q + \mathrm{SlotSummary}(S_t)
\]

返回当前 raw candidate pool：

\[
R_t = \mathrm{TopKRetrieve}(q_t, M_q)
\]

它依旧只做 raw candidate generation，不做后续控制。

### 5.2 输入输出

输入：

- 原问题 `q`
- 当前 notebook state `S_t`
- 多视图索引库 `M_q`
- 当前历史 `H_t`

输出：

- 当前 raw candidate pool `R_t`

### 5.3 最小算法

#### Step 1：构造当前查询

先生成：

\[
h_t = \mathrm{SlotSummary}(S_t)
\]

再构造：

\[
q_t = q + h_t
\]

如果 `S_t` 为空或没有可用 hint，则直接：

\[
q_t = q
\]

#### Step 2：检索 chunk top-K

用 `q_t` 对 `M_q` 做 coarse retrieval，得到 top-`K` chunks。

最简单版本可以只做：

\[
Score^{coarse}_t(i)
=
0.5\,sim(q_t,v_{sum}(i))
+
0.5\,sim(q_t,v_{chunk}(i))
\]

#### Step 3：在 shortlist 内暴露 top-1 atom

对 shortlist 中每个 chunk，只取最相关的 1 个 local pseudo-atom：

\[
Score^{atom}_t(i,j)=sim(q_t,v_{atom}(i,j))
\]

#### Step 4：merge

候选池由两部分组成：

- shortlist 中的 chunk candidates
- shortlist 中每个 chunk 的 top-1 atom candidate

#### Step 5：filter

对 merged candidates 做最轻量过滤：

- 去掉已经在 `H_t` 中的 unit
- 做去重
- 返回前 `K_r` 个

### 5.4 默认配置

- chunk shortlist：`K = 8`
- 每个 shortlist chunk 暴露 `top-1 atom`
- 最终返回数：`K_r = 10`

### 5.5 这个版本为什么合理

因为它已经保留了多视图底座最重要的两点：

- coarse retrieval
- shortlist 内 local pseudo-atom exposure

同时又没有把 retrieval 展开成复杂 reranking / policy / multi-stage fusion。

---

## 6. 三个模块的完整调用链

### 初始化阶段

1. 调用：

\[
P_0 = \mathrm{SeedRetrieve}(q, M_q, k_0)
\]

2. 构造：

\[
H_0 = P_0
\]

3. 更新得到：

\[
S_0 = \mathrm{Update}(\varnothing, P_0)
\]

4. 再渲染：

\[
K_0 = \mathrm{Render}(S_0)
\]

### 第 `t` 轮 retrieval

1. 生成极简 hint：

\[
h_t = \mathrm{SlotSummary}(S_t)
\]

2. 构造：

\[
q_t = q + h_t
\]

3. 调用：

\[
R_t = \mathrm{TopKRetrieve}(q_t, M_q)
\]

4. 后续 teacher / student 再在 `R_t` 基础上做：

- stop gate
- derived proposal
- final scoring
- state update

---

## 7. 当前 paper 的最终推荐版本

如果只保留一个最小强版本，我建议固定为：

### `SeedRetrieve`
- 只用 `q`
- top-K chunk shortlist
- 每个 chunk 取 `top-1 atom`
- 返回 `2~3` 个 raw units 作为 `P_0`

### `SlotSummary`
- 最近 `1` 条 raw evidence
- 最近 `1` 条 derived note
- `1` 条固定 need hint

### `TopKRetrieve`
- 用 `q_t = q + SlotSummary(S_t)`
- 检索 top-K chunks
- shortlist 内取 `top-1 atom`
- merge + dedup + filter used units

---

## 8. 一句话总结

这三个模块的最小设计可以概括为：

> `SeedRetrieve` 负责提供一个 search-initialized 起点；`SlotSummary` 只提供极短的 retrieval hint；`TopKRetrieve` 则基于该 hint 做一轮固定多视图 raw candidate generation。三者都应保持轻量，不应演化成新的复杂系统。
