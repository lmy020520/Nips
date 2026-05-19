# Teacher Stop / Answer Equivalence / Trajectory Abort Policy 设计稿（修正版）

## 1. 目的与边界

本文档整理当前版本中 **teacher stop**、**answer equivalence 判定** 与 **trajectory abort policy** 的统一设计，目标是为 **offline teacher 数据集构建** 提供一套清晰、保守、可执行的终止协议。

本文档首先明确一个关键边界：

> **以下设计默认针对 offline teacher 数据集构建，而不是 online student 推理。**

也就是说，这里的 stop 主要解决的是：

1. teacher 在构建一条 prefix trajectory 时，何时可以安全终止
2. 如何为后续 student 训练准备：
   - terminal positives
   - false-stop negatives
   - near-terminal negatives
   - continue negatives
3. 如何在 trajectory 始终无法到达正确答案时，决定：
   - 是继续 rollout
   - 还是中止该 trajectory
   - 以及哪些部分仍可用于训练

同时，本文档明确一个核心收敛：

> **student 的 stop 主体应由 deficit / closure-aware control 驱动，而不是靠一个孤立的 stop classifier。**

因此：
- 本文档中的 stop 设计首先服务于 teacher 数据集构建
- student 侧最多蒸馏其中的局部经验，而不直接照搬完整 teacher stop 协议

---

## 2. 一个核心澄清：teacher stop 与 student stop 不是同一个问题

## 2.1 teacher stop 的问题

teacher 侧 stop 回答的是：

> **当前 prefix trajectory 是否已经可以结束，不再继续扩展？**

它是一个 **offline stopping criterion**，用于：
- 结束 teacher trajectory
- 构造 terminal supervision
- 控制数据集构建成本

teacher stop 可以依赖：
- candidate stop state
- answer probe
- gold answer
- support sufficiency check

---

## 2.2 student stop 的问题

student 侧 stop 回答的是：

> **当前 deficit 是否已经足够小，以至于继续 retrieve / derive 的边际收益已经不值得？**

student stop 的核心应是：

- 当前 residual deficit `d_t`
- 当前候选池的最大预期收益
- closure-aware control

因此，当前推荐的 student stop 形式不是：
- 单独训练一个 stop classifier 作为核心任务

而是：

> **以 deficit 为主，stop head 为辅。**

更合理的 student-side 形式应当类似：

\[
stop_t =
\mathbf 1[
\|d_t\| \le \tau_d
\land
\max_{u \in C_t} score_t(u) \le \tau_g
]
\]

可选再叠加一个轻量 stop head 做 calibration，但它不应成为主任务。

---

## 3. teacher stop 的总体结构

teacher stop 不是“每轮都问要不要停”，而是一个稀疏触发的协议。

当前推荐的 teacher-side 逻辑是：

1. 先做 raw retrieval
2. 先做 `CheapStopGate`
3. 若当前进入 `stop_candidate`
4. 才执行一次 answer-probe stop
5. 若 probe 成功且支持充分，则 stop
6. 若 probe 失败，则视为 false-stop，继续 rollout

因此：

> **teacher stop 是 candidate-stop-state 下的一次保守终止判定，而不是每轮常规动作。**

---

## 4. teacher stop 的最终定义

当前推荐把 teacher stop 写成：

\[
TeacherStop_t
=
StopCandidate_t
\land
AnswerCorrect_t
\land
SupportSufficient_t
\]

其中：

- `StopCandidate_t`：来自 cheap stop gate
- `AnswerCorrect_t`：当前 probe answer 与 gold answer 在任务意义上等价
- `SupportSufficient_t`：当前 compiled context 对该答案的支持足够

只有三者同时成立，teacher 才真正终止 trajectory。

---

## 5. `StopCandidate_t`：来自 cheap stop gate

当前版本中，`StopCandidate_t` 已由前面的 gate 设计确定。  
这里不重复展开其全部细节，只强调其角色：

> `StopCandidate_t` 只表示“当前前缀已经足够接近 terminal，值得触发一次 stop probe”。

它不是最终 stop truth。

也就是说：
- `StopCandidate_t = 1` 不代表真的要 stop
- 它只意味着“现在值得做一次更强的终止检查”

---

## 6. `AnswerCorrect_t`：修正后的 answer equivalence 设计

这是本次修正中最重要的变化之一。

## 6.1 为什么要修正

之前如果把 answer equivalence 做成：

- normalization
- alias 匹配
- canonical entity map
- task-specific hand rules
- verifier

就很容易演化成一个新的脆弱规则系统。

而现实中，大多数数据集都不会提供：

- alias 列表
- canonical entity mapping
- 专门的同义词表

因此，当前修正后的结论是：

> **answer equivalence 不应由复杂规则系统主导，而应主要由一个保守的本地 judge 完成。**

---

## 6.2 新的两阶段设计

当前推荐的 `AnswerCorrect_t` 采用：

### Stage A：minimal normalization
只保留最机械、最稳定的规则：

- 小写化
- 去首尾空格
- 连续空格合并
- 去明显无意义标点
- 数字 / 日期基础归一化

这一步的作用只是：

> **消除表面格式差异，而不是解决实体级语义等价。**

若经过这一层后，预测答案与 gold answer 完全一致，则可直接认为等价。

---

### Stage B：local answer equivalence judge
对于 Stage A 不能确定的样本，调用一个本地 answer equivalence judge。

judge 的输入建议包括：

- `q`：问题
- `a_q^*`：gold answer
- `\hat a_t`：probe answer
- 与答案最相关的短 evidence（来自 `K_t`）

judge 的输出采用三分类：

- `equivalent`
- `not_equivalent`
- `uncertain`

然后定义：

\[
AnswerCorrect_t = 
ExactNormalizedMatch_t
\lor
JudgeEquivalent_t
\]

其中：

\[
JudgeEquivalent_t = 1[ judge = equivalent ]
\]

也就是说：
- `equivalent` 才算通过
- `not_equivalent` 不通过
- `uncertain` 一律不通过

### 当前关键原则
> **宁可保守地不 stop，也不要把模棱两可的样本误判为 terminal positive。**

---

## 6.3 judge 的职责边界

judge 只负责：

> **判断 predicted answer 是否与 gold answer 在任务意义上等价。**

它不负责：
- 判断当前 trajectory 是否值得停止
- 判断当前 context 是否支持该答案
- 做完整 reasoning correctness 评估

这些都不应混进 equivalence judge。

---

## 6.4 judge 的推荐提示约束

本地 judge 的提示应明确要求：

1. 只有在高把握语义等价时才输出 `equivalent`
2. 若只是“相似”“相关”“可能是同一实体”，一律输出 `uncertain`
3. 不允许依赖未提供的外部常识去扩大等价范围
4. 只能依据：
   - question
   - gold answer
   - predicted answer
   - 当前给定 evidence

这样可以明显提升 teacher terminal positives 的纯度。

---

## 6.5 为什么采用三分类而不是二分类

若 judge 只能输出：
- yes
- no

则它会被迫对大量模糊样本做硬判，容易产生过度自信误判。

采用三分类：
- `equivalent`
- `not_equivalent`
- `uncertain`

可以把真正模糊的情况安全地压到“不停止”。

对于 teacher 数据构建，这是更合理的偏置。

---

## 7. `SupportSufficient_t`：为什么答案等价还不够

即便 `AnswerCorrect_t = 1`，也不意味着当前 prefix 一定已经构成合格的 terminal context。

因为可能出现：

- probe answer 碰巧答对
- 但当前 `K_t` 对这个答案的支持不够
- 或主要歧义尚未被排除

如果这种状态也 stop，会污染 terminal positive 样本。

因此必须再检查：

\[
SupportSufficient_t
\]

---

## 7.1 当前推荐的最小版 support sufficiency

最小版可采用规则化检查：

### 条件 A：答案相关 evidence 已出现在 `K_t`
当前 `K_t` 中应包含与预测答案直接相关的 raw evidence。

### 条件 B：若答案依赖 bridge，则 bridge 结构已出现
也就是：
- notebook / compiled context 中已有 bridge raw support
- 或已有 `bridge_note`

### 条件 C：若问题需要 distinguish，则 distinguish 覆盖不能过低
由于当前已经决定 raw 尤其 atom 需要 role 标签，因此可以利用 role-wise ledger 直接检查：

- `k_distinguish` 是否达到最低门槛

---

## 7.2 更完整但仍保守的定义

当前可以把 `SupportSufficient_t` 设计成一个布尔条件：

\[
SupportSufficient_t
=
AnsSupport_t
\land
BridgeReady_t
\land
DistinguishReady_t
\]

其中不适用的项可以置为真。

也就是说，support sufficiency 仍然应尽量规则化、可解释，而不再引入新的大 judge。

---

## 8. teacher stop probe 的最小协议

## 8.1 触发条件
只在：

\[
StopCandidate_t = 1
\]

时触发一次 stop probe。

---

## 8.2 输入
```python
StopProbeInputs = {
    "q": str,
    "K_t": str,
    "gold_answer": str | None
}
```

---

## 8.3 输出
推荐最小结构：

```python
StopProbeOutput = {
    "pred_answer": str,
    "answer_correct": bool,
    "support_sufficient": bool,
    "should_stop": bool
}
```

其中：

\[
should\_stop = answer\_correct \land support\_sufficient
\]

### 说明
`pred_answer` 可以由 answerer 生成；  
`answer_correct` 使用上文的两阶段 equivalence 判定；  
`support_sufficient` 使用当前最小版 support 检查。

---

## 9. stop supervision 的标签体系

当前推荐仍使用两层标签：

### 第一层：二元监督
```python
should_stop: bool
```

### 第二层：类型标签
```python
label_type: str
```

推荐枚举：

- `terminal_positive`
- `false_stop_negative`
- `near_terminal_negative`
- `continue_negative`

这可以帮助后续：
- 主任务训练 `should_stop`
- 分析不同类型 stop 错误
- 对 hard negatives（尤其 false-stop）做更高权重训练

---

## 10. 四类 stop 标签的定义

## 10.1 `terminal_positive`

当满足：

1. `StopCandidate_t = 1`
2. 执行 stop probe
3. `answer_correct = True`
4. `support_sufficient = True`
5. trajectory 在该步终止

则打标签：

```python
{
    "should_stop": True,
    "label_type": "terminal_positive"
}
```

---

## 10.2 `false_stop_negative`

当满足：

1. `StopCandidate_t = 1`
2. 执行 stop probe
3. `should_stop = False`
4. trajectory 后续继续 rollout
5. 且未来某一步才真正达到 terminal

则当前 prefix 为：

```python
{
    "should_stop": False,
    "label_type": "false_stop_negative"
}
```

这类样本是 stop 任务最关键的 hard negatives。

---

## 10.3 `near_terminal_negative`

若当前步不是 terminal，但离真正 terminal 只差 1–2 步，则记为：

```python
{
    "should_stop": False,
    "label_type": "near_terminal_negative"
}
```

推荐定义：

- 设真正终止步为 `T*`
- 若 `T* - t ∈ {1,2}` 且当前不是 positive，则为 near-terminal negative

这类样本帮助模型学会：
- “接近 closure” ≠ “当前就可以停”

---

## 10.4 `continue_negative`

其余明显不该停的 prefix，记为：

```python
{
    "should_stop": False,
    "label_type": "continue_negative"
}
```

---

## 11. false-stop handling 的设计

false-stop handling 是 teacher rollout 里的一个轻量防震荡机制。

它回答的问题是：

> 当当前进入 candidate stop state，但 stop probe 失败时，teacher 接下来怎么继续？

当前推荐目标只有三个：

1. 明确记录这是一轮 false-stop
2. 短期冻结 stop，避免下一步立刻再次 probe
3. 让 rollout 继续，而不是在 terminal 邻域来回震荡

---

## 11.1 轻量状态变量

推荐在 teacher rollout 临时状态中加入：

```python
FalseStopState = {
    "false_stop_count": int,
    "stop_cooldown": int
}
```

默认：
- `false_stop_count = 0`
- `stop_cooldown = 0`

---

## 11.2 规则

### Rule 1：false stop 后计数 +1
一旦 stop probe 失败：

\[
false\_stop\_count \leftarrow false\_stop\_count + 1
\]

### Rule 2：设置短期 cooldown
推荐：

\[
stop\_cooldown \leftarrow 1
\]

含义：
- 下一步禁止再次执行 stop probe
- 至少再走一步补足内容

### Rule 3：若连续多次 false stop，则抑制重复 stop
若：

- `false_stop_count >= 2`
- 且当前仍处于 `StopCandidate_t = 1`

则在重新回到 `StopCandidate_t = 0` 之前，暂时不再执行 stop probe。

这能防止 trajectory 在 terminal 邻域震荡。

---

## 11.3 伪代码

```python
def HandleFalseStop(false_stop_state):
    next_state = dict(false_stop_state)
    next_state["false_stop_count"] += 1
    next_state["stop_cooldown"] = 1
    return next_state

def UpdateFalseStopCooldown(false_stop_state):
    next_state = dict(false_stop_state)
    if next_state["stop_cooldown"] > 0:
        next_state["stop_cooldown"] -= 1
    return next_state

def CanRunStopProbe(stop_candidate, false_stop_state):
    if not stop_candidate:
        return False
    if false_stop_state["stop_cooldown"] > 0:
        return False
    if false_stop_state["false_stop_count"] >= 2:
        return False
    return True
```

---

## 12. trajectory abort policy：如果一直答不对怎么办

这是当前设计里另一个必须明确的问题。

teacher 轨迹不是一定都能成功到达 terminal positive。  
原因包括：

- raw coverage 虽然在增长，但 judge / answerer 始终答不对
- proposer 产生的 derived 帮助有限
- LLM 本身能力不足
- 数据本身噪声较大

因此不能简单要求“所有 trajectory 都必须成功终止”。

---

## 12.1 三类 trajectory 结果

推荐给每条 teacher trajectory 一个总状态：

```python
TrajectoryStatus = {
    "status": str,          # success / failed_but_progressive / failed_stalled
    "terminal_step": int | None,
    "abort_reason": str | None
}
```

---

## 12.2 `success`
若 trajectory 在某步满足：

\[
TeacherStop_t = 1
\]

则记为：

```python
{
    "status": "success",
    "terminal_step": t,
    "abort_reason": None
}
```

---

## 12.3 `failed_but_progressive`
若 trajectory 最终没有得到正确 terminal answer，但整个过程中仍持续有明显进展，例如：

- raw role coverage 在增长
- derived deficit 在下降
- 选中的候选总体合理
- 只是终点 probe answer 未达标

则记为：

```python
{
    "status": "failed_but_progressive",
    "terminal_step": None,
    "abort_reason": "no_terminal_answer"
}
```

### 这类 trajectory 如何利用
可以保留大量 prefix 样本用于：
- ranking supervision
- deficit / contribution supervision
- stop negatives

但：
- 不能产生 `terminal_positive`

---

## 12.4 `failed_stalled`
若 trajectory 很早就陷入无进展，例如：

- 连续多步 `A_t` 不增长
- `R_t` 高度重复
- `ProposeDerived` 连续产不出合法有用 note
- stop probe 反复失败
- deficit 长时间几乎不下降

则记为：

```python
{
    "status": "failed_stalled",
    "terminal_step": None,
    "abort_reason": "stalled"
}
```

### 这类 trajectory 如何利用
- 只保留前半段较高质量 prefix
- 尾部低质量抖动段应截断
- 不产生 terminal positive

---

## 13. abort 的触发规则

当前推荐 teacher 数据构建中引入三个轻量 abort 条件：

### Abort 1：达到最大步数
\[
t \ge T_{max}
\]

当前建议：
- `T_max = 8 ~ 12`
- 具体取决于任务复杂度

---

### Abort 2：连续 `m` 步无进展
无进展可定义为：

- `A_t` 不增长
- 无新合法 derived
- `d_t^*` 的下降极小
- `R_t` 与近期状态高度重复

若连续 `m` 步成立，则中止 trajectory。

建议：
- `m = 2` 或 `3`

---

### Abort 3：连续多次 false-stop 且无新覆盖
若：
- false-stop 次数持续增加
- 同时 raw ledger 与 state 几乎不再前进

则认为 trajectory 已在 terminal 邻域失效震荡，应终止。

---

## 14. failed trajectories 能否用于训练？

可以，但要分层使用。

## 14.1 可以利用的部分
即使整条 trajectory 最后失败，很多 prefix 仍然有价值：

- ranking positives / negatives
- deficit 递减监督
- contribution labels
- stop negatives
- false-stop / near-terminal cases

---

## 14.2 不能利用的部分
不能把失败轨迹的最后状态伪装成：
- `terminal_positive`
- 或 pseudo-closure state

这会污染：
- stop supervision
- terminal context
- teacher 成功示范

---

## 14.3 推荐策略
- `success`：完整利用
- `failed_but_progressive`：利用 prefix 样本，但无 terminal positive
- `failed_stalled`：截断利用，尾部废弃

---

## 15. 与 student 训练的接口关系

当前修正版强调：

### teacher stop
服务于：
- trajectory 构建终止
- terminal / false-stop / near-terminal 标注

### student stop
服务于：
- closure-aware inference control
- deficit-driven stopping decision

因此：
- teacher stop 的 answer equivalence judge 不应被误当成 student 主模块
- student stop 不应退化成“复制 teacher 的 answer string judge”

student 侧的主目标仍然是：
- deficit `d_t`
- candidate contribution `c_t(u)`
- closure-aware control

stop head 最多是辅助。

---

## 16. 当前推荐的最终协议（简版）

### teacher stop 终止条件
\[
TeacherStop_t
=
StopCandidate_t
\land
AnswerCorrect_t
\land
SupportSufficient_t
\]

### answer correctness
\[
AnswerCorrect_t
=
ExactNormalizedMatch_t
\lor
JudgeEquivalent_t
\]

其中：
- 规则层只保留最机械的 normalization
- 主体由保守的本地三分类 judge 完成
- `uncertain` 一律视作不 stop

### false-stop handling
- false stop 后计数 +1
- 设置 `stop_cooldown = 1`
- 连续 false-stop 过多时抑制重复 probe

### trajectory abort
- 最大步数
- 连续无进展
- repeated false-stop + no progress

### trajectory result
- `success`
- `failed_but_progressive`
- `failed_stalled`

---

## 17. 一句话总结

当前修正版的核心收敛是：

> **teacher stop 是一个保守的离线终止准则：只有在当前 prefix 已进入 candidate stop state，并且 probe answer 与 gold answer 在任务意义上等价、且当前 context 对该答案的支持足够时，才真正停止 trajectory；答案等价判定不再依赖复杂 alias/canonical 规则，而主要依赖一个保守的本地三分类 judge；若 trajectory 始终无法到达正确答案，则根据是否仍在取得进展将其分为 `failed_but_progressive` 或 `failed_stalled`，从而决定哪些 prefix 仍可作为训练样本保留。**
