# Gate + ProposeDerived 设计稿 v3（Typed Derived Trigger）

## 1. 目的

本文档在 v2 基础上进一步收口 derived 策略。  
v3 的核心目标不是继续放宽 derived，而是把当前已经在轨迹中观察到的 derived 使用方式，整理成一个**分类型触发策略**：

1. `late_verification`
2. `early_bridge`
3. `trigger_only_derived`

v3 的中心判断是：

> **derived 不应再被视为单一动作，而应被视为三类不同触发机制的动作，其中前两类需要被更清晰地建模，第三类需要被控制而不是完全消除。**

---

## 2. 为什么需要 v3

v2 已经解决了一个关键问题：

> raw evidence 已接近足够，但 compiled context 仍未 closure；系统发生 false stop，随后 retrieval stagnation，却仍未触发 derived repair。

但我们在成功轨迹分析中发现，当前 derived 的实际使用方式并不单一：

- 多数进入 winning path 的 derived，出现在**中后段**
- 多数真正进入 winning path 的 derived，不是 `bridge_note`，而是 `verification_note`
- 少数 case 存在 `t=0` 或极早期 derived，这类 derived 更像是 **bridge scaffold**
- 还有一大类 case 虽然触发了 derived，但 derived 并没有真正进入 winning path

这说明：

- v2 的统一 `NeedDerivedGate` 已经不够表达真实 teacher 行为
- “是否开 derived” 这个问题，应该改写成：
  - **该不该开 late verification**
  - **该不该允许 narrow early bridge**
  - **哪些 triggered-only derived 只是保留作负样本，而不应被主监督强化**

---

## 3. v3 的基本边界

v3 仍然保持 current paper 的基本边界不变：

- teacher-side only
- 仍只保留两类 note：
  - `bridge_note`
  - `verification_note`
- proposer 仍然只允许一次 unified LLM 调用
- 仍保留：
  - `G_t^{harvest}`
  - legality filtering
  - `G_t^{final} / G_t^{aux} / G_t^{illegal}`
- teacher final scoring 仍只在：

\[
C_t = R_t \cup G_t^{final}
\]

上进行

v3 不是新增 note family，也不是把 proposer 变成 planner。  
v3 只是让 derived 的触发更有类型意识。

---

## 4. v3 的核心观察

### 4.1 主流模式：late verification

derived 的主流使用方式不是开新方向，而是：

- raw 已经接近 semantic sufficiency
- 但 answer-facing closure 还差一点
- 于是插入一个 `verification_note`
- 帮助 stop / closure 正确接住最后一跳

这类 derived 的特点是：

- 出现在中后段
- 通常与：
  - `recent_false_stop`
  - `stagnation`
  - `answer_focus_mismatch`
  相关
- 更像 answer-facing repair，而不是 retrieval exploration

### 4.2 少数模式：early bridge

少数 case 中，derived 会在很早阶段出现。  
这种 derived 通常不是 answer-facing verification，而更像：

- 初始化深搜 raw 已经给出了可 bridge 的候选 bundle
- 但 query anchor 到 answer candidate 之间还缺一层组织
- teacher 先用一个 `bridge_note` 做 scaffold
- 后续 raw 更容易继续走对

这类 derived 的特点是：

- 位置早
- 通常是 `bridge_note`
- 后面往往继续接 raw，而不是立即 stop

### 4.3 第三类：trigger-only derived

还有很多 case 的 derived 虽然被触发了，但并没有真正进入 winning path。  
这些 derived 的特点是：

- gate 允许开 derived
- proposer 也生成了 note
- 但最终真正推动成功的仍是 raw + stop closure

这类 derived 的存在并不是完全无价值，因为：

- 它们仍可作为负样本
- 它们反映了 teacher 在边界状态下的候选探索

但问题是：

> **它们不应在最终 teacher supervision 中被当作强正样本对待。**

---

## 5. v3 的策略总览

v3 把原本统一的 `NeedDerivedGate_v2` 拆成一个 typed trigger policy：

1. `LateVerificationGate`
2. `EarlyBridgeExceptionGate`
3. `TriggerOnlyControl`

整体策略如下：

### 默认策略

默认仍然是：

> **raw-first, then late verification**

也就是说：

- 首先依赖 raw retrieval / raw selection 推进
- 当 raw semantic readiness 已经达到一定水平，但 closure 仍未完成时
- 允许触发 `late_verification`

### 窄例外策略

只在很严格的情况下允许：

> **early bridge scaffold**

也就是说：

- 不是因为 raw 不够而提前开 derived
- 而是因为 initial raw bundle 已经足够 bridgeable，但还缺结构化桥接

### 控制策略

对：

> **triggered but never selected derived**

采用“保留但降权”的策略。

---

## 6. Gate A：LateVerificationGate

### 6.1 目标

用于识别这样一种状态：

> raw evidence 已经基本够用，但 context 还没有 answer-facing closure，因此需要一个 `verification_note` 来帮助收口。

### 6.2 触发条件

推荐使用如下条件：

\[
LateVerificationTrigger_t =
s_t^{sem} \ge \tau_{sem}
\land
bridgeable\_raw_t
\land
answer\_ready\_raw_t
\land
closure\_repair\_needed_t
\]

其中：

- `s_t^{sem} >= tau_sem`
- `bridgeable_raw_t = True`
- `answer_ready_raw_t = True`
- `closure_repair_needed_t = True`

这里：

\[
closure\_repair\_needed_t =
recent\_false\_stop_t
\lor
stagnation_t
\lor
answer\_focus\_mismatch_t
\]

### 6.3 直觉

这类 case 中，系统并不是“缺 raw”，而是：

- raw 已经比较齐
- 但 stop 还接不住
- 需要一个 answer-facing note 来告诉系统：
  - 当前 evidence 支持什么
  - 不支持什么
  - 当前答案焦点应该落在哪里

### 6.4 对应 note type

优先触发：

- `verification_note`

只有在极少数确实需要桥接时，才允许 `bridge_note` 进入同一轮候选。

---

## 7. Gate B：EarlyBridgeExceptionGate

### 7.1 目标

用于识别这样一种少数状态：

> 虽然轨迹仍处于早期，但初始化深搜 raw 已经提供了足够 bridgeable 的结构；如果先加一个桥接 note，后续 raw 更容易沿正确方向推进。

### 7.2 触发条件

推荐把它设计成一个严格例外：

\[
EarlyBridgeTrigger_t =
(t \le 1)
\land
initial\_bundle\_semantic\_ready_t
\land
bridgeable\_raw_t
\land
\neg answer\_ready\_raw_t
\land
bridge\_gap\_explicit_t
\]

其中：

- `t <= 1`
- `initial_bundle_semantic_ready_t = True`
- `bridgeable_raw_t = True`
- `answer_ready_raw_t = False`
- `bridge_gap_explicit_t = True`

### 7.3 `bridge_gap_explicit_t` 的直觉

它表示：

- query anchor 已经出现
- 候选答案类型信号也出现了
- 但二者之间缺一个显式桥

例如：

- raw1 含 query entity
- raw2 含 answer-type signal
- 共享实体 / 最近 state anchor 存在
- 但当前 raw 还没形成清晰 answer-facing chain

### 7.4 对应 note type

优先触发：

- `bridge_note`

### 7.5 重要约束

这不是“让 derived 提前替代 raw”。  
它的真实语义应该是：

> initial raw retrieval 已经做完，并且已经形成了足够强的 raw bundle；teacher 只是决定先组织一下，再继续消费 raw。

---

## 8. Gate C：TriggerOnlyControl

### 8.1 目标

用于控制第三类 derived：

> gate 被打开了，proposer 也生成了 derived，但它并没有真正进入 winning path。

### 8.2 原则

这类 derived 不应被完全消除，因为：

- 我们仍然需要 teacher 提供一部分 derived 负样本
- 这类候选仍有助于训练 student 理解“哪些 derived 没有真正帮助 closure”

但这类 derived 需要被：

- **减少触发**
- **降低 retained 概率**
- **在样本监督中降权**

### 8.3 实现方向

推荐做两层控制：

#### 层 1：gate 端减少泛触发

只有在明确满足：

- `LateVerificationTrigger`
- 或 `EarlyBridgeTrigger`

时，才进入强 derived proposal 通道。  
其余边界状态，即使保留 proposer 调用，也应标记为：

- `exploratory_derived`
- 或 `weak_trigger_derived`

#### 层 2：retain / supervision 端降权

对于：

- `closure_value` 低
- `answer_facing = False`
- 未真正进入 winning path

的 derived，应优先：

- 留在 `G_t^{aux}`
- 不进入 `G_t^{final}`
- 或在 sample label 中作为 harder negative / weak candidate 处理

---

## 9. v3 的统一决策逻辑

推荐顺序如下：

1. `CheapStopGate_v2`
2. `FalseStopRepairOverride`
3. `LateVerificationGate`
4. `EarlyBridgeExceptionGate`
5. `TriggerOnlyControl`

伪代码：

```python
def TypedDerivedPolicy_v3(q, T_q_raw, A_t, S_t, R_t, failure_signals, t):
    stop_info = CheapStopGate_v2(q, T_q_raw, A_t, S_t, R_t, failure_signals)

    if FalseStopRepairOverride(failure_signals):
        return {
            "action": "propose_derived",
            "derived_subtype": "late_verification",
            "derive_mode": "repair_after_false_stop",
        }

    if stop_info["stop_candidate"]:
        return {
            "action": "stop_probe",
            "derived_subtype": None,
            "derive_mode": None,
        }

    if LateVerificationGate(q, T_q_raw, A_t, S_t, R_t, failure_signals):
        return {
            "action": "propose_derived",
            "derived_subtype": "late_verification",
            "derive_mode": "normal",
        }

    if EarlyBridgeExceptionGate(q, T_q_raw, A_t, S_t, R_t, failure_signals, t):
        return {
            "action": "propose_derived",
            "derived_subtype": "early_bridge",
            "derive_mode": "normal",
        }

    if TriggerOnlyControl(q, T_q_raw, A_t, S_t, R_t, failure_signals):
        return {
            "action": "propose_derived_weak",
            "derived_subtype": "trigger_only_candidate",
            "derive_mode": "normal",
        }

    return {
        "action": "raw_only",
        "derived_subtype": None,
        "derive_mode": None,
    }
```

---

## 10. proposer 的 typed conditioning

v3 不要求新增 note type，但要求 proposer 明确感知当前 derived subtype。

推荐新增：

```python
derived_subtype: str
```

允许值：

- `late_verification`
- `early_bridge`
- `trigger_only_candidate`

### 对 proposer 的约束

#### 当 `derived_subtype = late_verification`

优先生成：

- `verification_note`

要求：

- answer-facing
- 明确支持 / 不支持
- 尽量帮助 stop closure

#### 当 `derived_subtype = early_bridge`

优先生成：

- `bridge_note`

要求：

- 只做桥接
- 不要过早 answer-facing
- 目标是为后续 raw selection 提供结构化 scaffold

#### 当 `derived_subtype = trigger_only_candidate`

允许 proposer 生成候选，但后续 retain 更保守。

---

## 11. sample supervision 的建议

v3 不建议把所有 triggered derived 都当作强监督信号。

建议把 derived 监督拆成三档：

### 强正样本

- 真正进入 winning path 的 `late_verification`

### 窄正样本

- 真正进入 winning path 的 `early_bridge`

### 弱负样本 / 辅助负样本

- `triggered but never selected derived`

这样做的好处是：

- 保留 derived negative signal
- 避免 student 学到“只要触发了 derived 就值得优先选”
- 让 teacher supervision 更贴近真实有效的 derived 使用方式

---

## 12. 与 v2 的关系

可以把 v3 看成 v2 的 typed refinement。

v2 的贡献是：

- 修复 false-stop 后 derived 被 stop 压制的问题
- 让 system 能识别 closure failure after semantic sufficiency

v3 的贡献是：

- 进一步承认 derived 不是单一动作
- 把 derived 的真实使用方式整理成三类
- 让 gate、proposer、retain、supervision 四层逻辑更一致

所以：

> v2 解决的是“derived 该不该被允许开出来”；  
> v3 解决的是“derived 开出来时，到底是哪一种 derived，以及后续该怎么对待它”。

---

## 13. 一句话总结

v3 的核心变化可以概括为：

> **derived 不再被视为单一的 repair 动作，而被明确拆分成 late verification、early bridge 与 trigger-only derived 三种子类型。默认策略仍然是 raw-first + late verification；early bridge 只在严格条件下作为窄例外允许；triggered but never selected derived 则被保留但降权。这样可以让 teacher 的 gate、proposer、retain 与 supervision 更一致，也更贴近当前成功轨迹中观察到的真实 derived 使用方式。**
