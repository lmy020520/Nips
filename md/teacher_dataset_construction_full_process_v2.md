# 完整的 Teacher 数据集构建流程设计稿（v2）

## 1. 目的与边界

本文档系统整理当前版本中 **teacher 数据集构建** 的完整流程，目标是把已经分别讨论过的核心模块统一到一条清晰的、可执行的 offline teacher trajectory 构建链条中，并补上 teacher→student 接口中之前尚未闭环的部分。

本文档首先明确一个关键边界：

> **以下设计默认针对 offline teacher 数据集构建，而不是 online student 推理。**

也就是说，本文档的目标是：

1. 明确 teacher 数据集构建中每一轮 prefix 的状态递推过程  
2. 把 retrieval、gate、derived proposal、legality / retain、state update、render、ledger、stop / abort policy 串成一个完整流程  
3. 给后续 student 训练准备统一的数据对象与监督标签  
4. 明确 teacher 产出的标签如何被 student 消费  
5. 明确 failed trajectories 如何按 prefix 级别被保留或丢弃  
6. 避免各个 supporting module 各自膨胀成独立工程

---

## 2. teacher 数据集构建的核心目标

teacher 数据集构建的目标不是直接得到一个在线可部署 agent，而是：

1. 构造 prefix trajectory `H_t`
2. 构造 raw 覆盖账本 `A_t`
3. 构造 evidence-centric notebook `S_t`
4. 构造 answer-facing compiled context `K_t`
5. 在每一轮选择 teacher 的下一知识单元 `u_{t+1}^*`
6. 生成可训练的监督标签，包括：
   - 正样本 `u_t^+`
   - deficit `d_t^*`
   - contribution `c_t^*(u_t^+)`
   - ranking labels
   - stop labels

因此，teacher 侧的核心任务是：

> **构造一条高质量、可解释、可监督的前缀轨迹，而不是追求部署期极致效率。**

---

## 3. 一个关键澄清：teacher stop 与 student stop 不是同一个问题

这是 v2 版本中最重要的修正之一。

### 3.1 teacher stop
teacher 侧 stop 解决的是：

> **当前 trajectory 是否可以成功终止，不再继续构建。**

它服务于：
- trajectory 构建终止
- terminal positive 构造
- false-stop / near-terminal 样本记录

### 3.2 student stop
student 侧 stop 解决的是：

> **当前 deficit 是否已经足够小，以至于继续 retrieve / derive 的边际收益已经不值得。**

因此，当前推荐明确规定：

> **student stop 应主要由 deficit / closure-aware control 驱动，`StopLabel` 最多只是辅助监督，而不是主任务。**

也就是说：
- teacher stop 协议可以更重、更保守
- student 推理时不应照搬 teacher 的完整 stop 机制

---

## 4. 核心数据结构

当前冻结的核心数据结构分为四层。

### 4.1 全局静态层

#### `q`
问题对象：

```python
QueryRecord = {
    "qid": str,
    "question": str,
    "answer": str | None,
    "metadata": dict
}
```

#### `T_q^raw`
全局 raw 目标监督集：

```python
RawSpan = {
    "unit_id": str,
    "text": str,
    "doc_id": str,
    "parent_chunk_id": str | None,
    "span_start": int | None,
    "span_end": int | None
}

RawTargetUnit = RawSpan | {
    "provenance": "raw",
    "weight": float,
    "primary_role": str | None,
    "role_label_source": str | None
}

T_q_raw = list[RawTargetUnit]
```

当前版本默认：
- raw target 尤其 atom 需要有 role 标签：
  - `bridge`
  - `distinguish`
  - `support`

#### `M_q`
多视图检索底座索引：

```python
AtomRecord = {
    "atom_id": str,
    "atom_text": str,
    "v_atom": "vector",
    "span_start": int,
    "span_end": int
}

ChunkRecord = {
    "chunk_id": str,
    "doc_id": str,
    "chunk_text": str,
    "summary_text": str,
    "v_sum": "vector",
    "v_chunk": "vector",
    "atoms": list[AtomRecord]
}

M_q = list[ChunkRecord]
```

---

### 4.2 知识单元层

#### `RawUnit`
候选 raw knowledge unit：

```python
RawUnit = RawSpan | {
    "provenance": "raw",
    "candidate_granularity": str   # chunk / atom / sentence
}
```

#### `DerivedUnit`
候选 derived note：

```python
DerivedUnit = {
    "unit_id": str,
    "text": str,
    "provenance": "derived",
    "candidate_granularity": "note",
    "type": str,                  # bridge_note / verification_note
    "source_unit_ids": list[str]
}
```

#### `UnitRegistry`
统一 payload 注册表：

```python
UnitRegistry = {
    "unit_id": RawUnit | DerivedUnit
}
```

---

### 4.3 前缀状态层

#### `H_t`
当前前缀历史：

```python
HistoryStep = {
    "step_id": int,
    "unit_id": str
}

H_t = list[HistoryStep]
```

#### `A_t`
raw ledger：

```python
ALedger = {
    "covered_target_ids": set[str],
    "k_bridge": float,
    "k_distinguish": float,
    "k_support": float,
    "coverage_trace": dict
}
```

#### `S_t`
evidence-centric notebook state：

```python
StateRef = {
    "unit_id": str,
    "added_step": int,
    "used_in_summary_count": int,
    "selected_count": int
}

State = {
    "raw_refs": list[StateRef],
    "derived_refs": list[StateRef],
    "last_added_unit_id": str | None,
    "last_updated_step": int
}
```

#### `K_t`
compiled context：

```python
K_t = str
```

---

### 4.4 当前轮工作层

#### `P_0`
初始化 seed 集合：

```python
P_0 = list[RawUnit]
```

#### `R_t`
当前 raw candidate pool：

```python
R_t = list[RawUnit]
```

#### `G_t^{harvest}, G_t^{final}, G_t^{aux}, G_t^{illegal}`
derived candidate pools：

```python
G_t_harvest = list[DerivedUnit]
G_t_final = list[DerivedUnit]
G_t_aux = list[DerivedUnit]
G_t_illegal = list[DerivedUnit]
```

#### `C_t`
当前前缀可达候选池：

```python
C_t = list[str]   # unit_id list
```

其中：

```python
C_t = R_t ∪ G_t_final
```

---

## 5. teacher 数据集构建的总体时间线

每条 teacher trajectory 从初始化开始，然后按 prefix 逐轮展开。

总体顺序可以概括为：

1. 初始化种子 `P_0`
2. 构造 `H_0, S_0, A_0, K_0`
3. 在每一轮：
   - 生成 retrieval hint
   - 执行 raw retrieval
   - 执行 stop gate
   - 若需要则执行 derived proposal
   - 清洗与保留 derived
   - teacher 在当前候选池中选取下一步 `u_{t+1}^*`
   - 更新 `H_t, S_t, A_t, K_t`
   - 判断 stop / abort
4. 对成功与失败轨迹做 prefix 级筛选与保存
5. 生成 teacher→student 训练记录

---

## 6. 初始化阶段

### 6.1 SeedRetrieve

初始化时执行：

\[
P_0 = \mathrm{SeedRetrieve}(q, M_q, k_0)
\]

最小设计：
- 只用原问题 `q`
- 对 chunk 做 coarse retrieval
- 每个 shortlist chunk 暴露 `top-1 atom`
- 返回 `2~3` 个 raw units

推荐默认值：
- `k_0 = 2`
- `K_seed = 8`

### 6.2 初始化状态

得到 `P_0` 后，构造：

\[
H_0 = P_0
\]

\[
S_0 = Update(\varnothing, P_0)
\]

\[
A_0 = Ledger(\varnothing, P_0, T_q^{raw})
\]

\[
K_0 = Render(q, S_0, UnitRegistry)
\]

此后进入常规 prefix 递推。

---

## 7. 每一轮 prefix 的完整流程

以下描述第 `t` 轮，从已知状态：

- `H_t`
- `S_t`
- `A_t`
- `K_t`

出发。

### 7.1 构造当前 retrieval query

先从 `S_t` 构造一个极简 retrieval hint：

- 最近 1 条 raw evidence
- 最近 1 条 derived note（若存在）
- 1 条简单 need hint

于是得到：

\[
h_t = \mathrm{SlotSummary}(S_t)
\]

然后构造：

\[
q_t = q + h_t
\]

若 `S_t` 为空或无可用 hint，则直接：

\[
q_t = q
\]

### 7.2 执行 TopKRetrieve

根据 `q_t` 和多视图索引库 `M_q`，执行：

\[
R_t = \mathrm{TopKRetrieve}(q_t, M_q)
\]

最小设计：
1. 检索 top-K chunks
2. shortlist 内每个 chunk 暴露 `top-1 atom`
3. merge
4. 去掉已在 `H_t` 中的 units
5. 去重
6. 返回前 `K_r` 个 raw candidates

推荐默认值：
- chunk shortlist `K = 8`
- 每个 chunk 暴露 `top-1 atom`
- 最终返回 `K_r = 10`

### 7.3 执行 cheap stop gate

在当前 raw retrieval 完成后，先执行：

\[
StopCandidate_t = CheapStopGate(q, T_q^{raw}, A_t, S_t, R_t)
\]

这个 gate 的作用是：

> **判断当前前缀是否已经足够接近 terminal，以至于应优先尝试 stop，而不是继续 derive。**

### 7.4 执行 need-derived gate

如果当前不在 `stop_candidate`，则继续判断：

\[
T_t^{der} = NeedDerivedGate(q, T_q^{raw}, A_t, S_t, R_t)
\]

该 gate 用于判断：

> **当前 raw evidence 是否已经积累到值得做一次 derived organization。**

核心依据包括：
- `s_t^{sem}`：raw semantic readiness
- `has_composable_raw(R_t)`
- 最近是否刚做过 verification

若触发，则允许执行 `ProposeDerived`。

### 7.5 执行 ProposeDerived

若 `NeedDerivedGate` 打开，则执行：

\[
G_t^{harvest} = \mathrm{ProposeDerived}(q, S_t, R_t)
\]

当前最小版本边界：
- 只允许两类 derived：
  - `bridge_note`
  - `verification_note`
- harvest pool 最多 4 个
- 只依赖：
  - `q`
  - 轻量 notebook summary
  - top-3 raw candidates
- 只在 teacher offline 数据构建中作为 proposer 出现

### 7.6 执行 LegalityFilter

对 `G_t^{harvest}` 执行 legality filtering，得到：

- `G_t^{legal}`
- `G_t^{illegal}`

`LegalityFilter` 只做硬规则过滤，不做复杂打分。

当前保留的 4 类硬规则：
1. schema / type 合法
2. source 可见且数量合法
3. 单句且长度受控
4. 重复剔除

### 7.7 执行 FinalRetainSelection

对合法 derived 候选执行极简 retained selection：

- 第一名一定保留
- 第二名优先补另一种 type
- 最多保留 2 个

得到：

- `G_t^{final}`
- `G_t^{aux}`

然后构造当前最终候选池：

\[
C_t = R_t \cup G_t^{final}
\]

### 7.8 teacher 选择下一步知识单元

teacher 最终只在：

\[
C_t = R_t \cup G_t^{final}
\]

上做当前轮 utility 精算，选择：

\[
u_{t+1}^* = \arg\max_{u \in C_t} U_t(u)
\]

这里的 `U_t(u)` 代表当前 prefix 下 teacher 的 planning utility / closure utility。

### 7.9 更新 notebook / ledger / compiled context

一旦得到 `u_{t+1}^*`，按固定顺序递推状态：

#### Step 1：Update
\[
S_{t+1} = Update(S_t, u_{t+1}^*, t+1)
\]

#### Step 2：Ledger
\[
A_{t+1} = Ledger(A_t, u_{t+1}^*, T_q^{raw})
\]

#### Step 3：Render
\[
K_{t+1} = Render(q, S_{t+1}, UnitRegistry)
\]

同时更新：

\[
H_{t+1} = H_t \oplus u_{t+1}^*
\]

---

## 8. 三个状态更新函数的职责

### 8.1 Update

`Update` 的目标是：

> **把当前步选中的知识单元写入轻量 notebook state。**

它只负责：
- state 引用更新
- `selected_count`
- `last_added_unit_id`
- `last_updated_step`

它不负责：
- 语义抽取
- note 生成
- context 编写

### 8.2 Ledger

`Ledger` 的目标是：

> **把新加入的 raw unit 对 `T_q^raw` 的覆盖贡献增量累加到 `A_t`。**

当前前提：
- raw 特别是 atom 已有 role 标签：
  - `bridge`
  - `distinguish`
  - `support`

更新逻辑：
- 若 `u_{t+1}` 是 derived，则 `A_t` 不变
- 若是 raw，则：
  - exact match 优先
  - fallback 才用 span overlap
  - 更新：
    - `covered_target_ids`
    - `k_bridge / k_distinguish / k_support`
    - `coverage_trace`

### 8.3 Render

`Render` 的目标是：

> **把当前 notebook 渲染成 answer-facing compiled context `K_t`。**

当前最小策略：
1. 取最新 `bridge_note`
2. 取最新 `verification_note`
3. 先放 note 的支撑 evidence
4. 再补 recent raw evidence

输出模板：

```text
Evidence:
[1] ...
[2] ...
[3] ...

Notes:
[bridge] ...
[verification] ...
```

---

## 9. teacher stop 的完整协议（修正版）

teacher stop 不是每轮都触发，而是在 candidate stop state 下的保守终止判定。

当前推荐：

\[
TeacherStop_t=
StopCandidate_t
\land
AnswerCorrect_t
\land
SupportSufficient_t
\]

### 9.1 `AnswerCorrect_t`

v2 中明确收缩为：

#### Stage A：minimal normalization
只做：
- 小写化
- 空格 / 标点清洗
- 数字 / 日期基础归一化

#### Stage B：local answer equivalence judge
若 Stage A 不能确定，则调用一个本地 equivalence judge或LLM，输出三分类：

- `equivalent`
- `not_equivalent`
- `uncertain`

其中：
- 只有 `equivalent` 才算通过
- `uncertain` 一律不通过

这里明确不再依赖复杂的：
- alias 列表
- canonical entity map
- 大量任务特化规则

### 9.2 `SupportSufficient_t`

即使答案等价，也必须要求当前 `K_t` 对该答案支持充分。

最小版可检查：

1. 答案相关 evidence 已出现在 `K_t`
2. 若答案依赖 bridge，则 bridge 已就绪
3. 若问题需要 distinguish，则 distinguish 覆盖不能过低

### 9.3 false-stop handling

若当前：
- 进入 `stop_candidate`
- 但 stop probe 失败

则视为 false-stop。

推荐维护一个轻量状态：

```python
FalseStopState = {
    "false_stop_count": int,
    "stop_cooldown": int
}
```

规则：
1. false stop 后计数 +1
2. 设置短期 cooldown
3. 连续多次 false-stop 时抑制重复 stop probe

---

## 10. `should_abort` 的明确落地定义（v2 修正）

这是 v2 中最重要的新增修正。

`should_abort` 不是抽象原则，而是一个 teacher-side 的确定性失败终止规则。  
它回答的是：

> **当前 trajectory 继续 rollout 是否已经不值得。**

当前推荐定义：

\[
Abort_t =
BudgetExceeded_t
\lor
Stalled_t
\lor
FalseStopOscillation_t
\]

### 10.1 `BudgetExceeded_t`
当：

\[
t \ge T_{max}
\]

则触发。

推荐默认：
- `T_max = 10`

### 10.2 `Progress_t` 的定义
定义当前步是否产生了有效进展：

\[
Progress_t = \mathbf 1[
\Delta A_t > 0
\lor
{NewUsefulDerived_t} = 1
\lor
\Delta d_t^* < -\epsilon_d
]
\]

其中：

- `\Delta A_t > 0`：raw ledger 有新增覆盖
- `NewUsefulDerived_t = 1`：本步 teacher 选中了真正有用的 derived
- `\Delta d_t^* < -\epsilon_d`：deficit 显著下降

推荐：
- `\epsilon_d = 0.05`

### 10.3 `Stalled_t`
若最近连续 `m` 步都无进展，则触发：

\[
Stalled_t = \mathbf 1[
\sum_{i=t-m+1}^{t} Progress_i = 0
]
\]

推荐：
- `m = 2`

### 10.4 `FalseStopOscillation_t`
若 trajectory 在 terminal 邻域来回 false-stop 且没有新进展，则触发：

\[
FalseStopOscillation_t =
\mathbf 1[
false\_stop\_count_t \ge 2
\land
Progress_t = 0
\land
\Delta A_t = 0
]
\]

### 10.5 伪代码

```python
def should_abort(step_id, T_max, recent_progress_flags, false_stop_state, delta_A, delta_d, new_useful_derived):
    if step_id >= T_max:
        return True, "budget_exceeded"

    progress = (delta_A > 0) or new_useful_derived or (delta_d < -0.05)

    if len(recent_progress_flags) >= 2 and sum(recent_progress_flags[-2:]) == 0 and (not progress):
        return True, "stalled"

    if false_stop_state["false_stop_count"] >= 2 and delta_A == 0 and (not progress):
        return True, "false_stop_oscillation"

    return False, None
```

---

## 11. trajectory abort policy（v2 细化）

teacher trajectory 不一定都能成功到达 terminal positive。

因此推荐为每条 trajectory 记录一个总状态：

```python
TrajectoryStatus = {
    "status": str,          # success / failed_but_progressive / failed_stalled
    "terminal_step": int | None,
    "abort_reason": str | None
}
```

### 11.1 `success`
当某步满足：

\[
TeacherStop_t = 1
\]

则 trajectory 成功终止。

### 11.2 `failed_but_progressive`
若轨迹最终没答对，但过程中仍在取得进展，例如：

- raw coverage 在增长
- derived deficit 在下降
- 候选选择总体合理

则保留这条 trajectory 的大量 prefix 样本，但不产生 terminal positive。

### 11.3 `failed_stalled`
若轨迹很早陷入无进展，例如：

- `A_t` 多步不变
- `R_t` 高度重复
- derived proposal 长时间无效
- stop probe 反复失败

则：
- 截断 trajectory
- 只保留前半段较高质量 prefix

### 11.4 与 `should_abort` 的关系
- `should_abort = True` 只是终止当前 trajectory
- trajectory 的最终状态需根据历史 progress 决定：
  - 若之前长期有进展 → `failed_but_progressive`
  - 若后半段长期无进展 → `failed_stalled`

---

## 12. failed trajectories 的 prefix 级利用策略（v2 细化）

这是 v2 的第二个核心修正。

failed trajectories 不应整条保留或整条丢弃，而应按 prefix 细分。

### 12.1 三段划分

对 failed trajectory，按 prefix 切分为三段：

#### 段 A：高质量进展段
特点：
- `Progress_t = 1`
- teacher 选择的 `u_t^+` 合理
- `A_t` 或 `d_t^*` 明显改善

这段应全部保留。

#### 段 B：临界停滞段
特点：
- 开始出现重复 raw
- derived usefulness 下降
- 但偶尔还有局部进展

这段可有条件保留，但应降低权重。

#### 段 C：失败尾部震荡段
特点：
- 连续无进展
- repeated false-stop
- `R_t` 高度重复
- `A_t` 几乎不动

这段直接截断，不进入训练集。

### 12.2 `failed_but_progressive` 的处理

#### 保留
- 所有高质量进展段 prefix
- 大多数 ranking labels
- contribution labels
- deficit labels
- stop negatives（但无 terminal positive）

#### 不保留
- 最尾部若已进入明显停滞，也应截断
- 不产生 `terminal_positive`
- 不产生 `near_terminal_negative`

### 12.3 `failed_stalled` 的处理

定义最后一个有效进展步：

\[
t_{last\_progress}=
\max\{t: Progress_t = 1\}
\]

只保留：

\[
\{0,1,\dots,t_{last\_progress}\}
\]

丢弃：

\[
\{t_{last\_progress}+1,\dots,t_{abort}\}
\]

### 12.4 failed trajectory 上各类标签如何处理

#### 可以保留
- `u_t^+`
- `RankingLabel`
- `d_t^*`
- `c_t^*(u_t^+)`
- `StopLabel`（负例类）

#### 不能保留
- `terminal_positive`
- `near_terminal_negative`（若无真实 terminal）
- 任何基于成功 closure 的标签

### 12.5 ranking label 的保留条件（v2 新增明确化）

对 failed trajectory，某步的 `RankingLabel` 只有在其后仍带来有效进展时才保留。

也就是：

> **只要某步之后 `Progress_{t+1}=1`，就保留该步的 `RankingLabel`。**

反之：
- 若这一步之后没有任何进展
- 很快进入 stalled tail

则该步 ranking label 可丢弃。

---

## 13. teacher 数据集构建生成的监督标签

### 13.1 正样本 `u_t^+`
```python
PositiveLabel = {
    "step_id": int,
    "unit_id": str
}
```

### 13.2 deficit 标签 `d_t^*`
```python
DeficitLabel = {
    "d_raw": float | None,
    "d_br": float | None,
    "d_dis": float | None,
    "d_sup": float | None,
    "d_der": float
}
```

当 raw role 标签可用时，优先使用三维 raw deficit。

### 13.3 contribution 标签 `c_t^*(u_t^+)`
```python
ContributionLabel = {
    "c_raw": float | None,
    "c_br": float | None,
    "c_dis": float | None,
    "c_sup": float | None,
    "c_der": float
}
```

### 13.4 ranking labels
```python
RankingLabel = {
    "positive_unit_id": str,
    "negative_unit_ids": list[str]
}
```

其中 negatives 可来自：
- raw negatives
- `G_t^{aux}`
- `G_t^{illegal}`（更适合 legality/filter 辅助监督）
- `G_t^{final}` 中未被 teacher 选中的 derived

### 13.5 stop labels
```python
StopLabel = {
    "should_stop": bool,
    "label_type": str
}
```

推荐枚举：
- `terminal_positive`
- `false_stop_negative`
- `near_terminal_negative`
- `continue_negative`

---

## 14. teacher → student 接口：这些标签到底怎么用（v2 修正）

这是 v2 的第三个核心修正。

### 14.1 `RankingLabel` 的用途
`RankingLabel` 应明确作为：

> **student candidate scorer 的主监督。**

student 在每个 prefix 上利用：

- 当前状态：`S_t, A_t, K_t`
- 当前候选池：`C_t`
- 正样本：`u_t^+`
- 负样本：`C_t \setminus \{u_t^+\}`

训练一个 scorer：

\[
Score_t(u)
\]

满足：

\[
Score_t(u_t^+) > Score_t(u^-)
\]

因此，`RankingLabel` 不是附属品，而是 student control policy 学习的主入口之一。

### 14.2 `StopLabel` 的用途
`StopLabel` 不应作为 student 的核心任务，而应作为：

> **辅助 stop / calibration 监督。**

也就是说：

- student 的主 stop 仍由 deficit 与最大可获增益决定
- `StopLabel` 只用于训练一个辅助 stop head，或用于阈值校准与分析

推荐 student stop 形式：

\[
StudentStop_t =
\mathbf 1[
\|d_t\| \le \tau_d
\land
\max_{u\in C_t} Score_t(u) \le \tau_g
\land
p_{stop} \ge \tau_s
]
\]

其中：
- 前两项是主判据
- `p_stop` 只是辅助 calibration 项

### 14.3 `d_t^*` 与 `c_t^*(u_t^+)` 的用途
这两个仍然是 student 侧的核心监督：

- `d_t^*` → 训练 deficit predictor
- `c_t^*(u_t^+)` → 训练 contribution predictor

student 的主体控制逻辑应围绕它们展开，而不是围绕 `StopLabel` 单独展开。

### 14.4 `G_t^{illegal}` 的用途
不建议直接混入主 ranking loss；更适合作为：
- legality/filter 辅助监督
- 或 candidate prefilter head 的训练数据

---

## 15. 统一的数据记录接口（v2 新增）

对每个最终保留下来的 prefix，teacher 应输出：

```python
TeacherPrefixRecord = {
    "state": {
        "H_t": ...,
        "A_t": ...,
        "S_t": ...,
        "K_t": ...
    },
    "candidates": {
        "R_t": ...,
        "G_t_final": ...,
        "G_t_aux": ...,
        "G_t_illegal": ...,
        "C_t": ...
    },
    "labels": {
        "u_t_plus": ...,
        "ranking_label": ...,
        "d_t_star": ...,
        "c_t_star": ...,
        "stop_label": ...
    },
    "meta": {
        "trajectory_status": ...,
        "progress_flag": ...,
        "keep_prefix": bool
    }
}
```

这样 student 训练时就能明确区分：

- 主监督
- 辅助监督
- 元信息过滤

---

## 16. 整个 teacher 数据集构建流程的简化伪代码（v2）

```python
def BuildTeacherTrajectory(q, T_q_raw, M_q, UnitRegistry):
    # init
    P_0 = SeedRetrieve(q, M_q, k_0=2)
    H_t = init_history(P_0)
    S_t = init_state_from_seed(P_0)
    A_t = init_ledger_from_seed(P_0, T_q_raw)
    K_t = Render(q, S_t, UnitRegistry)

    false_stop_state = {"false_stop_count": 0, "stop_cooldown": 0}
    trajectory = []
    terminal_step = None
    status = "failed_stalled"
    abort_reason = None
    recent_progress_flags = []

    for t in range(T_max):
        # retrieval
        h_t = SlotSummary(S_t, q)
        q_t = q if not h_t else q + "\n" + h_t
        R_t = TopKRetrieve(q_t, M_q, H_t)

        # stop / derived gating
        stop_info = CheapStopGate(q, T_q_raw, A_t, S_t, R_t)
        if stop_info["stop_candidate"]:
            probe = RunStopProbe(q, K_t, gold_answer=q.get("answer", None))
            if probe["should_stop"]:
                terminal_step = t
                status = "success"
                trajectory.append(record_step(...))
                break
            else:
                false_stop_state = HandleFalseStop(false_stop_state)

        derived_info = NeedDerivedGate(q, T_q_raw, A_t, S_t, R_t)
        if derived_info["trigger_derived"]:
            G_h = ProposeDerived(q, S_t, R_t, gold_answer=q.get("answer", None))
            legal_out = LegalityFilter(S_t, R_t, G_h["derived_candidates"])
            retain_out = FinalRetainSelection(legal_out["G_t_legal"])
            G_final = retain_out["G_t_final"]
            G_aux = retain_out["G_t_aux"]
            G_illegal = legal_out["G_t_illegal"]
        else:
            G_final, G_aux, G_illegal = [], [], []

        C_t = build_candidate_pool(R_t, G_final)
        u_next = TeacherSelect(q, S_t, A_t, C_t)

        # state update
        A_prev = A_t
        H_t = append_history(H_t, u_next, step_id=t+1)
        S_t = Update(S_t, u_next, step_id=t+1)
        A_t = Ledger(A_t, u_next, T_q_raw)
        K_t = Render(q, S_t, UnitRegistry)

        delta_A = len(A_t["covered_target_ids"]) - len(A_prev["covered_target_ids"])
        new_useful_derived = int(u_next["provenance"] == "derived")
        delta_d = estimate_delta_d(...)   # teacher-side available
        progress_flag = int((delta_A > 0) or new_useful_derived or (delta_d < -0.05))
        recent_progress_flags.append(progress_flag)

        trajectory.append(record_step(...))

        if any(recent_progress_flags):
            status = "failed_but_progressive"

        abort, abort_reason = should_abort(
            step_id=t+1,
            T_max=T_max,
            recent_progress_flags=recent_progress_flags,
            false_stop_state=false_stop_state,
            delta_A=delta_A,
            delta_d=delta_d,
            new_useful_derived=bool(new_useful_derived),
        )
        if abort:
            break

    trajectory = prune_failed_tail_if_needed(trajectory, status)

    return {
        "trajectory": trajectory,
        "terminal_step": terminal_step,
        "status": status,
        "abort_reason": abort_reason
    }
```

---

## 17. 一句话总结

当前推荐的完整 teacher 数据集构建流程（v2）是：

> **从 `SeedRetrieve` 提供的 search-initialized 起点开始，系统在每个 prefix 上先做轻量 retrieval，再通过 stop gate 与 need-derived gate 控制是否终止或触发 derived proposal；随后用 legality / retain 模块清洗并保留极少量 derived notes，teacher 再在当前候选池中选择下一知识单元，并通过 `Update / Ledger / Render` 递推 `S_t / A_t / K_t`；最终结合保守的 teacher stop 协议、明确的 `should_abort` 规则，以及 prefix 级 failed trajectory 利用策略，构造可用于 student 训练的正样本、deficit、contribution、ranking 和 stop supervision，其中 `RankingLabel` 是 student candidate scorer 的主监督，而 `StopLabel` 仅作为辅助 stop / calibration 监督。**
