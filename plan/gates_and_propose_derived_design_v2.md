# Gate + ProposeDerived 设计稿 v2（面向数据集构建）

## 1. 目的与边界

本文档给出 **cheap stop gate**、**need-derived gate** 与 **ProposeDerived** 的 v2 设计，用于 **offline teacher 数据集构建**。v2 的核心目标不是扩张 current paper 的 scope，而是在保持 teacher-side 最小闭环的前提下，修复这样一类关键 failure：

> **raw evidence 已接近足够，但 compiled context 仍未 closure；系统发生 false stop，随后 retrieval stagnation，却仍未触发 derived repair。**

v2 的重点修补三处：

1. **false-stop repair override**  
   一旦系统已经出现 stop-probe failure 且 raw coverage 不再增长，允许 derived proposal 通过一个 repair 通道被强制触发，而不是继续被 stop 优先级压制。

2. **bridgeable raw 识别**  
   用更贴近 bridge / verification 生成需求的 `has_bridgeable_raw(...)`，替换过于句法化的 `has_composable_raw(...)`。

3. **goal-conditioned proposer**  
   `ProposeDerived` 不再只做泛化 note proposal，而是接收当前 repair goal，例如：
   - `bridge_query_entity_to_answer_candidate`
   - `answer_focus_verification`
   - `target_type_disambiguation`

需要强调：

- 本文档默认仍然是 **teacher-side dataset construction design**
- 不把 current paper 改写成复杂 tree search / agent planner
- 不新增更多 note types，当前仍只保留：
  - `bridge_note`
  - `verification_note`

---

## 2. v2 的设计动机：当前版本在哪类状态下失效

v1 的整体顺序是：

1. raw retrieval
2. cheap stop gate
3. need-derived gate
4. 若触发，则 `ProposeDerived`

这一顺序的优点是成本低、逻辑清楚；但它对下面这类状态不够敏感：

- 当前 raw semantic coverage 已经较高
- raw retrieval 重复度高
- 新 raw target 不再增长
- stop probe 已经失败
- 失败原因不是 evidence 缺失，而是 context 没有被 answer-facing 地组织起来

换句话说，v1 更擅长识别：

- “raw 还不够”的状态

但不擅长识别：

- “raw 基本够了，但还缺一个 bridge / verification repair step”的状态

因此 v2 的核心不是把 proposer 变强，而是：

> **让 gate 能稳定识别 closure failure after semantic sufficiency，并为 derived repair 留出通道。**

---

## 3. 仍然保持的基础边界

v2 明确保留以下 current paper 边界，不做扩张：

### 3.1 teacher-side only
当前设计默认用于 offline teacher 数据集构建，而不是在线 student 推理。

### 3.2 仍只保留两类 note
- `bridge_note`
- `verification_note`

### 3.3 仍只保留一次 proposer LLM 调用
当触发 proposal 时，仍然只进行 **一次 unified proposer 调用**。

### 3.4 仍维持 harvest / final-retained 两层
`ProposeDerived` 输出的仍然只是 `G_t^{harvest}`，之后还需经过：

1. legality filtering
2. coarse priority selection
3. 分流为：
   - `G_t^{final}`
   - `G_t^{aux}`
   - `G_t^{illegal}`

### 3.5 teacher 最终仍只在极少量候选上精算
teacher final scoring 仍只发生在：

\[
C_t = R_t \cup G_t^{final}
\]

上，不把 proposer 扩张成 planner。

---

## 4. v2 中新增的 failure-aware 输入对象

为了让 gate 与 proposer 感知 closure failure，v2 引入一个新的轻量输入对象：

```python
FailureSignals_t = {
    "recent_false_stop": bool,
    "false_stop_count_recent": int,
    "last_delta_covered_targets": int,
    "last_retrieval_repeat_ratio": float,
    "last_probe_pred_answer": str | None,
    "last_probe_answer_correct": bool | None,
    "answer_focus_mismatch": bool,
    "stagnation": bool
}
```

### 字段解释

- `recent_false_stop`  
  最近一个 stop probe 是否失败。

- `false_stop_count_recent`  
  最近窗口内 false stop 次数。

- `last_delta_covered_targets`  
  最近一轮相对上轮新增覆盖的 raw target 数量。

- `last_retrieval_repeat_ratio`  
  最近一次 retrieval 与上一轮或最近窗口相比的重复比例。

- `last_probe_pred_answer`  
  上次 stop probe 输出的答案。

- `last_probe_answer_correct`  
  上次 stop probe 是否正确。

- `answer_focus_mismatch`  
  当前问题期待的答案类型，与上次 probe 输出的答案类型是否明显不一致。  
  例如问题问 `What river...`，但 probe 给出地点、湖泊、区域等实体。

- `stagnation`  
  一个派生布尔量，定义为：
  - `last_delta_covered_targets == 0`
  - 且 `last_retrieval_repeat_ratio` 足够高

推荐默认阈值：
- `stagnation_repeat_threshold = 0.6`

---

## 5. Gate 设计总览（v2）

v2 推荐采用 **三段式顺序**：

1. `CheapStopGate_v2`
2. `FalseStopRepairOverride`
3. `NeedDerivedGate_v2`

整体逻辑如下：

### Step 1：先做 cheap stop gate
用于判断当前是否是一个 terminal candidate state。

### Step 2：若近期已发生 false stop 且检索停滞
则优先走 `propose_derived_repair`，绕过普通 stop 优先级。

### Step 3：若无 repair override
再决定：
- stop probe
- normal derived proposal
- raw only

也就是说，v2 不改变“stop 优先于 derive”的大原则，但增加了一个非常关键的例外：

> **一旦系统已经证明自己“停不下来、又补不进新 raw”，就不该继续让 stop gate 压住 derived。**

---

## 6. Gate A：CheapStopGate_v2

### 6.1 目标

`CheapStopGate_v2` 仍然是：

> 一个便宜的、保守的 terminal 候选判定器。

但与 v1 不同的是，v2 明确要求：

> **近期若已出现 closure failure signal，则 cheap stop gate 不能继续维持 stop_candidate。**

### 6.2 输入

```python
CheapStopGateV2Inputs = {
    "q": str,
    "T_q_raw": object,
    "A_t": object,
    "S_t": object,
    "R_t": list,
    "failure_signals": FailureSignals_t
}
```

### 6.3 输出

```python
CheapStopGateV2Output = {
    "raw_complete": bool,
    "no_obvious_derived_need": bool,
    "no_recent_closure_failure": bool,
    "stop_candidate": bool
}
```

### 6.4 条件 1：raw completion

仍沿用 role-wise 版本。

对每个需要的 role `r`：

\[
\tilde s_t^r=
\frac{k_t^r + \alpha}{N_q^r + 2\alpha}
\]

并定义：

\[
RawComplete_t =
\mathbf{1}\Big[
\forall r\; \text{s.t.}\; m_q^r = 1,\;
\tilde s_t^r \ge \tau_r
\Big]
\]

默认阈值仍取：
- `tau_r = 0.70`

### 6.5 条件 2：no obvious derived need

v2 中，这一项不再叫 `NoDerivedNeed_t`，而更推荐叫：

\[
NoObviousDerivedNeed_t
\]

因为它只是一个 cheap heuristic，不等价于“确实完全不需要 derived”。

沿用 v1 中最小规则：

- 最近 1–2 步已有 `verification_note`
- 或者 raw pool 高度冗余，且最近已有 `bridge_note`

定义：

\[
NoObviousDerivedNeed_t =
has\_recent\_verification(S_t)
\;\lor\;
(raw\_redundant(R_t,S_t) \land has\_recent\_bridge(S_t))
\]

### 6.6 条件 3：no recent closure failure

这是 v2 的新增项。

定义：

\[
NoRecentClosureFailure_t=
\neg recent\_false\_stop_t
\land
\neg stagnation_t
\land
\neg answer\_focus\_mismatch_t
\]

更保守的版本可写成：

\[
NoRecentClosureFailure_t=
\neg(
recent\_false\_stop_t
\lor
stagnation_t
\lor
answer\_focus\_mismatch_t
)
\]

其意义是：

- 一旦系统已经显露出“答不出来但又补不进新 raw”的信号，
- stop gate 不应继续把当前状态视为正常 terminal candidate。

### 6.7 最终定义

\[
StopCandidate_t=
RawComplete_t
\land
NoObviousDerivedNeed_t
\land
NoRecentClosureFailure_t
\]

这比 v1 更稳，因为它不会在 false-stop 之后继续顽固地把 derived 压掉。

### 6.8 伪代码

```python
def CheapStopGate_v2(q, T_q_raw, A_t, S_t, R_t, failure_signals):
    role_scores = get_rolewise_progress(A_t, T_q_raw)
    required_roles = get_required_roles(T_q_raw)

    raw_complete = all(role_scores[r] >= 0.70 for r in required_roles)

    has_recent_verif = has_recent_note(S_t, note_type="verification_note", window=2)
    has_recent_bridge = has_recent_note(S_t, note_type="bridge_note", window=2)
    raw_redundant = is_raw_pool_redundant(R_t, S_t)

    no_obvious_derived_need = has_recent_verif or (raw_redundant and has_recent_bridge)

    no_recent_closure_failure = not (
        failure_signals["recent_false_stop"]
        or failure_signals["stagnation"]
        or failure_signals["answer_focus_mismatch"]
    )

    stop_candidate = (
        raw_complete
        and no_obvious_derived_need
        and no_recent_closure_failure
    )

    return {
        "raw_complete": raw_complete,
        "no_obvious_derived_need": no_obvious_derived_need,
        "no_recent_closure_failure": no_recent_closure_failure,
        "stop_candidate": stop_candidate
    }
```

---

## 7. Repair Override：FalseStopRepairOverride

这是 v2 最重要的新模块。

### 7.1 目标

用于识别这样一种状态：

> **teacher 已经尝试过 stop，但 answer-probe 失败；同时 retrieval 开始停滞，raw coverage 不再增长。此时系统应优先进行一次 derived repair，而不是继续 stop 或 raw-only。**

### 7.2 触发条件

推荐使用如下最小版本：

\[
RepairOverride_t =
recent\_false\_stop_t
\land
(last\_delta\_covered\_targets_t = 0)
\land
(last\_retrieval\_repeat\_ratio_t \ge \tau_{repeat})
\]

其中：
- `tau_repeat = 0.6`

更强版本可以加入：
- `false_stop_count_recent >= 2`
- 或 `answer_focus_mismatch = True`

### 7.3 作用

一旦触发：
- 当前轮不做 stop probe
- 不走普通 `NeedDerivedGate_v2`
- 直接执行：
  - `action = "propose_derived_repair"`

并向 proposer 传入 repair mode 与 derive goal。

### 7.4 伪代码

```python
def FalseStopRepairOverride(failure_signals, repeat_threshold=0.6):
    return (
        failure_signals["recent_false_stop"]
        and failure_signals["last_delta_covered_targets"] == 0
        and failure_signals["last_retrieval_repeat_ratio"] >= repeat_threshold
    )
```

---

## 8. Gate B：NeedDerivedGate_v2

### 8.1 目标

`NeedDerivedGate_v2` 仍然回答：

> 当前不该 stop 时，是否值得触发一次 derived proposal？

但 v2 明确不再只看“能不能组合”，而是进一步看：

- 是否存在 **bridgeable raw**
- 当前是否真的处于 **closure repair needed** 状态

### 8.2 输入

```python
NeedDerivedGateV2Inputs = {
    "q": str,
    "T_q_raw": object,
    "A_t": object,
    "S_t": object,
    "R_t": list,
    "failure_signals": FailureSignals_t
}
```

### 8.3 输出

```python
NeedDerivedGateV2Output = {
    "s_sem": float,
    "bridgeable_raw": bool,
    "has_recent_verification": bool,
    "recent_false_stop": bool,
    "stagnation": bool,
    "answer_focus_mismatch": bool,
    "derived_need": bool,
    "trigger_derived": bool,
    "derive_goal": str
}
```

### 8.4 Step 1：raw semantic readiness

保持 v1 版本。

对需要的 raw roles 做平均：

\[
s_t^{sem}=
\frac{1}{|R(q)|}
\sum_{r\in R(q)} \tilde s_t^r
\]

其中 `R(q)` 表示当前问题真正需要的 raw roles 集合。

默认阈值仍取：
- `tau_sem = 0.5`

### 8.5 Step 2：从 composable raw 升级为 bridgeable raw

v1 的 `has_composable_raw(R_t)` 只检查：

- top raw 是否不重复
- 是否来自不同 parent chunk

这对 derived generation 来说过于句法化。  
v2 推荐改为：

\[
bridgeable\_raw_t = has\_bridgeable\_raw(q, S_t, R_t)
\]

### 直觉定义

若当前 top raw 中存在一对或一组三元关系，使得：

1. 至少一个 raw 明确锚定 query entity / query target object
2. 至少一个 raw 引入候选答案实体或答案类型
3. 它们共享中间桥接实体，或可由最近 notebook state 中已有实体链连接

则记为 bridgeable。

### 例子
- raw1: `Eilean Tioram ↔ Castle Tioram`
- raw2: `Castle Tioram ↔ River Shiel`
- 共享桥接实体：`Castle Tioram`
- 因此 `bridgeable_raw = True`

### 8.6 一个极简实现草案

工程上不必一开始就做复杂 relation extraction。  
可以先用一个便宜版本：

1. 从 `q`、`S_t`、`R_t[:3]` 中抽实体字符串
2. 检查 top raw 中是否存在：
   - 一个 raw 包含 query anchor
   - 另一个 raw 包含候选答案实体或答案类型触发词
   - 二者共享实体，或共享最近 state summary 中的 anchor

```python
def has_bridgeable_raw(q, S_t, R_t):
    top_raw = R_t[:3]
    if len(top_raw) < 2:
        return False

    q_anchors = extract_query_anchors(q)
    state_anchors = extract_state_anchors(S_t)

    for i in range(len(top_raw)):
        for j in range(i + 1, len(top_raw)):
            e_i = extract_entities(top_raw[i]["text"])
            e_j = extract_entities(top_raw[j]["text"])

            share_bridge = len((set(e_i) & set(e_j)) | set(state_anchors)) > 0
            query_linked = any(a in e_i or a in e_j for a in q_anchors)
            answer_like = has_answer_type_signal(q, top_raw[i]["text"]) or has_answer_type_signal(q, top_raw[j]["text"])

            if share_bridge and query_linked and answer_like:
                return True

    return False
```

这里：
- `has_answer_type_signal(q, text)` 用非常粗糙的规则即可，例如：
  - `What river` → 检测 river / creek / stream 等
  - `Which university` → 检测 university / college / school 等

这个模块的目标不是完美 NER/RE，而是便宜地识别“当前 raw 候选里是否存在值得组织成 bridge/verification note 的结构”。

### 8.7 Step 3：保留 recent verification 检查

v2 仍保留：
- 若最近 1–2 步已经加入 `verification_note`，则普通 derived trigger 可适当保守

定义：

\[
has\_recent\_verification_t = has\_recent\_note(S_t, verification\_note)
\]

### 8.8 Step 4：定义 closure-repair-aware derived_need

这是 v2 的核心改动。

v1：
\[
derived\_need_t = composable\_raw_t \land \neg has\_recent\_verification_t
\]

v2 改为：

\[
derived\_need_t =
bridgeable\_raw_t
\land
(
\neg has\_recent\_verification_t
\;\lor\;
recent\_false\_stop_t
\;\lor\;
stagnation_t
\;\lor\;
answer\_focus\_mismatch_t
)
\]

直觉上，这意味着：

- 平常情况下：有 bridgeable raw，且最近没刚做过 verification，就可以 derive
- repair 情况下：即使最近已有 verification，只要已经 false-stop / stagnation / answer-focus mismatch，也允许再次 derive repair

### 8.9 Step 5：最终 trigger

仍保持简单：

\[
T_t^{der}=
\mathbf 1[
s_t^{sem} \ge \tau_{sem}
\land
derived\_need_t = 1
]
\]

推荐默认值不变：

- `tau_sem = 0.5`

### 8.10 derive goal 推断

为了让 proposer 更定向，v2 增加：

\[
derive\_goal_t = infer\_derive\_goal(q, S_t, R_t, failure\_signals)
\]

当前推荐最小枚举值：

- `bridge_query_entity_to_answer_candidate`
- `answer_focus_verification`
- `target_type_disambiguation`
- `generic_bridge_or_verification`

### 一个简单规则版本
- 若 `answer_focus_mismatch = True`  
  → `derive_goal = "answer_focus_verification"`
- 若 `bridgeable_raw = True` 且 query anchors 与答案候选之间缺显式桥接  
  → `derive_goal = "bridge_query_entity_to_answer_candidate"`
- 若问题中答案类型与当前候选指向的实体类型易混淆  
  → `derive_goal = "target_type_disambiguation"`
- 否则  
  → `generic_bridge_or_verification`

### 8.11 伪代码

```python
def NeedDerivedGate_v2(q, T_q_raw, A_t, S_t, R_t, failure_signals, tau_sem=0.5):
    role_scores = get_rolewise_progress(A_t, T_q_raw)
    required_roles = get_required_roles(T_q_raw)

    s_sem = sum(role_scores[r] for r in required_roles) / max(1, len(required_roles))

    bridgeable_raw = has_bridgeable_raw(q, S_t, R_t)
    has_recent_verif = has_recent_note(S_t, note_type="verification_note", window=2)

    recent_false_stop = failure_signals["recent_false_stop"]
    stagnation = failure_signals["stagnation"]
    answer_focus_mismatch = failure_signals["answer_focus_mismatch"]

    derived_need = bridgeable_raw and (
        (not has_recent_verif)
        or recent_false_stop
        or stagnation
        or answer_focus_mismatch
    )

    trigger = (s_sem >= tau_sem) and derived_need
    derive_goal = infer_derive_goal(q, S_t, R_t, failure_signals)

    return {
        "s_sem": s_sem,
        "bridgeable_raw": bridgeable_raw,
        "has_recent_verification": has_recent_verif,
        "recent_false_stop": recent_false_stop,
        "stagnation": stagnation,
        "answer_focus_mismatch": answer_focus_mismatch,
        "derived_need": derived_need,
        "trigger_derived": trigger,
        "derive_goal": derive_goal
    }
```

---

## 9. Gate 组合逻辑（v2）

最终推荐顺序：

### Step 1：CheapStopGate_v2
得到：
- `raw_complete`
- `no_obvious_derived_need`
- `no_recent_closure_failure`
- `stop_candidate`

### Step 2：FalseStopRepairOverride
若 repair override 成立：
- 直接执行 `propose_derived_repair`

### Step 3：若未触发 repair override，且 `stop_candidate = True`
则：
- 执行 stop probe

### Step 4：否则执行 `NeedDerivedGate_v2`
- 若 `trigger_derived = True`：执行 normal `ProposeDerived`
- 否则：当前轮 `raw_only`

### 9.1 统一伪代码

```python
def PreDerivedGates_v2(q, T_q_raw, A_t, S_t, K_t, R_t, failure_signals):
    stop_info = CheapStopGate_v2(q, T_q_raw, A_t, S_t, R_t, failure_signals)

    if FalseStopRepairOverride(failure_signals):
        return {
            "action": "propose_derived_repair",
            "derive_mode": "repair_after_false_stop",
            "derive_goal": infer_derive_goal(q, S_t, R_t, failure_signals),
            "stop_info": stop_info,
            "derived_info": None
        }

    if stop_info["stop_candidate"]:
        return {
            "action": "stop_probe",
            "derive_mode": None,
            "derive_goal": None,
            "stop_info": stop_info,
            "derived_info": None
        }

    derived_info = NeedDerivedGate_v2(q, T_q_raw, A_t, S_t, R_t, failure_signals)

    if derived_info["trigger_derived"]:
        return {
            "action": "propose_derived",
            "derive_mode": "normal",
            "derive_goal": derived_info["derive_goal"],
            "stop_info": stop_info,
            "derived_info": derived_info
        }

    return {
        "action": "raw_only",
        "derive_mode": None,
        "derive_goal": None,
        "stop_info": stop_info,
        "derived_info": derived_info
    }
```

---

## 10. ProposeDerived_v2 的角色

`ProposeDerived_v2` 仍然不是 planner，也不是 compiler。  
它仍然只做一件事：

> **在当前 prefix 下，用一次统一的调用，生成极少量 grounded、goal-conditioned derived notes。**

v2 相比 v1 的核心差别不在 note type，而在：

- proposer 接收 `derive_mode`
- proposer 接收 `derive_goal`
- proposer 可接收 `recent_probe_feedback`
- proposer 可接收 `bridge_anchors`

从而把 generic proposal 升级成 **goal-conditioned repair proposal**。

---

## 11. ProposeDerived_v2 的输入

推荐：

```python
ProposeDerivedV2Inputs = {
    "question": str,
    "state_summary": str,
    "top_raw_candidates": list[RawUnit],
    "gold_answer": str | None,
    "derive_mode": str,           # "normal" | "repair_after_false_stop"
    "derive_goal": str,           # see enum above
    "recent_probe_feedback": dict | None,
    "bridge_anchors": list[str]
}
```

### 11.1 `derive_mode`

允许两种：

- `normal`
- `repair_after_false_stop`

用于告诉 proposer 当前是常规组织，还是在修一个已经暴露的 closure failure。

### 11.2 `derive_goal`

推荐值：

- `bridge_query_entity_to_answer_candidate`
- `answer_focus_verification`
- `target_type_disambiguation`
- `generic_bridge_or_verification`

### 11.3 `recent_probe_feedback`

teacher offline 阶段可选提供：

```python
{
    "pred_answer": "...",
    "answer_correct": bool,
    "error_type": "answer_focus_mismatch" | "unsupported_jump" | "other"
}
```

这不是为了让 proposer“抄答案”，而是为了让它知道当前 repair 方向。

### 11.4 `bridge_anchors`

从：
- `q`
- `S_t`
- `R_t[:J]`

中抽出的关键实体锚点。  
例如：
- query entity
- 最近 notebook 中的高频实体
- raw pair 之间共享的桥接实体

---

## 12. ProposeDerived_v2 的输出

输出结构与 v1 基本一致，仍只允许：

- `bridge_note`
- `verification_note`

推荐 JSON：

```json
{
  "should_derive": true,
  "reason": "closure repair is needed after false stop",
  "derived_candidates": [
    {
      "unit_id": "z1",
      "type": "bridge_note",
      "text": "...",
      "source_unit_ids": ["m17", "m03"],
      "coarse_priority": 1
    },
    {
      "unit_id": "z2",
      "type": "verification_note",
      "text": "...",
      "source_unit_ids": ["m03", "m22"],
      "coarse_priority": 2
    }
  ]
}
```

仍然保持：
- `max_candidates = 4`
- `1 <= len(source_unit_ids) <= 3`

---

## 13. 两类 note 的更新版定义

### 13.1 `bridge_note`
一条短 note，用来显式组织当前 evidence 中隐含但尚未显式形成的桥接关系。

要求：
- 1 句
- grounded
- 不引入 unsupported facts
- 不直接以下结论句口吻回答最终问题
- 更偏向“把桥搭出来”

### 13.2 `verification_note`
一条短 note，用来把当前 evidence 已经较接近答案的关系，做成 answer-facing、可验证的组织。

要求：
- grounded
- 更偏向“告诉系统当前 evidence 支持什么、不支持什么”
- 可在 teacher offline 阶段略微 answer-facing
- 但仍不允许自由延展

---

## 14. ProposeDerived_v2 的 prompt 约束

### 14.1 system instruction

```text
You are proposing a very small number of grounded intermediate notes for a retrieval-augmented reasoning system.

Only propose notes if they are genuinely useful at the current prefix.
Allowed note types:
1. bridge_note
2. verification_note

Each note must:
- be short
- be grounded in the provided source units
- cite only visible source_unit_ids
- not introduce unsupported facts
- not exceed 1 sentence
- use 1 to 3 source units

When derive_mode is repair_after_false_stop, prioritize notes that repair answer-facing organization failures, especially bridge completion, target-type clarification, and answer focus correction.

Return JSON only.
```

### 14.2 user payload 模板（v2）

```text
Question:
{q}

Current notebook summary:
{state_summary}

Derive mode:
{derive_mode}

Derive goal:
{derive_goal}

Recent probe feedback (teacher only):
{recent_probe_feedback_or_none}

Bridge anchors:
{bridge_anchors}

Top raw candidates:
[{unit_id}] {text}
[{unit_id}] {text}
[{unit_id}] {text}

Optional gold answer (teacher only):
{a_q_star_or_none}

Output at most 4 derived candidates.
If no useful derived note is needed, return should_derive=false and an empty list.
```

---

## 15. ProposeDerived_v2 的最小算法

```python
def ProposeDerived_v2(
    q,
    S_t,
    R_t,
    failure_signals,
    gold_answer=None,
    derive_mode="normal",
    derive_goal="generic_bridge_or_verification",
    J=3,
    max_candidates=4
):
    state_summary = BuildProposerStateSummary_v2(S_t, q, failure_signals, derive_goal)
    top_raw = R_t[:J]

    bridge_anchors = extract_bridge_anchors(q, S_t, top_raw)
    recent_probe_feedback = build_recent_probe_feedback(failure_signals)

    prompt = build_propose_prompt_v2(
        question=q,
        state_summary=state_summary,
        top_raw_candidates=top_raw,
        gold_answer=gold_answer,
        derive_mode=derive_mode,
        derive_goal=derive_goal,
        recent_probe_feedback=recent_probe_feedback,
        bridge_anchors=bridge_anchors,
        max_candidates=max_candidates
    )

    raw_output = call_llm(prompt)
    parsed = safe_parse_json(raw_output)

    if parsed is None:
        return {
            "should_derive": False,
            "reason": "invalid_json",
            "derived_candidates": []
        }

    should_derive = bool(parsed.get("should_derive", False))
    candidates = parsed.get("derived_candidates", [])

    validated = []
    for cand in candidates:
        if cand.get("type") not in {"bridge_note", "verification_note"}:
            continue
        if "text" not in cand or not cand["text"].strip():
            continue
        if "source_unit_ids" not in cand:
            continue
        if not (1 <= len(cand["source_unit_ids"]) <= 3):
            continue

        validated.append({
            "unit_id": cand.get("unit_id", make_unit_id()),
            "text": cand["text"].strip(),
            "provenance": "derived",
            "candidate_granularity": "note",
            "type": cand["type"],
            "source_unit_ids": cand["source_unit_ids"],
            "coarse_priority": cand.get("coarse_priority", len(validated) + 1)
        })

    validated = validated[:max_candidates]

    return {
        "should_derive": should_derive and len(validated) > 0,
        "reason": parsed.get("reason", ""),
        "derived_candidates": validated
    }
```

---

## 16. legality filtering 与后续分流：v2 不变

v2 不改变 legality filtering 与后续分流逻辑。  
仍然保持：

1. proposer 输出的是 `G_t^{harvest}`
2. legality filtering 得到：
   - `G_t^{harvest_legal}`
   - `G_t^{illegal}`
3. 按 `coarse_priority` 选出：
   - `G_t^{final}`
   - `G_t^{aux}`

仍然坚持：
- legality 完全规则化
- final retained pool 只保留 top-1 或 top-2
- `G_t^{aux}` 与 `G_t^{illegal}` 保留给训练监督

---

## 17. BuildProposerStateSummary_v2 的建议

为了避免 proposer 输入过重，v2 仍坚持轻量 `state_summary`，但允许比 v1 多两类信息：

### 保留
- 最近 1–2 条 raw evidence
- 最近 1 条 derived note（若有）

### 新增
- 一句 `Need: ...`
- 一句 `Recent issue: ...`

例如：

```text
Recent raw:
- Castle Tioram sits on the tidal island Eilean Tioram.
- River Shiel drains into Loch Moidart near Castle Tioram.

Recent derived:
- None

Need:
Bridge the query entity to the answer candidate and verify the answer focus.

Recent issue:
The last stop probe answered a location rather than the river asked by the question.
```

这样可以显著增强 proposer 的 repair 定向能力，而不会退化成 full `K_t` dump。

---

## 18. 默认参数建议（v2）

### stop gate
- role-wise raw completion threshold: `0.70`

### repair override
- `stagnation_repeat_threshold = 0.6`

### need-derived gate
- `tau_sem = 0.5`

### proposer
- `J = 3`
- `max_candidates = 4`
- allowed types = `{bridge_note, verification_note}`
- max source per note = `3`

### final retained
- top-1 或 top-2

---

## 19. 与 current paper scope 的关系

v2 的改动是：

- 增强 teacher-side gate 对 closure failure 的敏感性
- 增强 derived repair 能力
- 但不改变 current paper 的基本身份

v2 **没有**引入：

- tree search
- beam planner
- richer proposal family
- 更多 note types
- 在线重 verifier
- 大动作空间 agent

因此 v2 仍然符合 current paper 的最小强版本原则。

---

## 20. 一句话总结

v2 的核心变化可以概括为：

> **在保留 teacher-side 低成本 gate + single-call proposer 总体结构不变的前提下，v2 增加了一个 false-stop repair override，并把 derived trigger 从“是否存在可组合 raw”升级为“是否存在 bridgeable raw 且当前是否已暴露出 closure repair need”。同时，proposer 从泛化 note proposal 升级为 goal-conditioned repair proposal，从而更稳地处理“semantic sufficiency 已基本达成但 compiled context 尚未 closure”的 failure cases。**
