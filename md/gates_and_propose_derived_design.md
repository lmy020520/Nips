# Gate + ProposeDerived 设计稿（面向数据集构建）

## 1. 目的与边界

本文档整理当前版本中 **cheap stop gate**、**need-derived gate** 与 **ProposeDerived** 的统一设计，目标是为 **offline teacher 数据集构建** 提供一套清晰、最小、可执行的流程。

本文档首先明确一个关键边界：

> **以下设计默认针对 offline teacher 数据集构建，而不是 online 推理。**

也就是说，这里的目标是：

1. 在 teacher 轨迹构建时，决定当前 prefix 是否应：
   - 优先尝试 stop
   - 继续仅用 raw candidates
   - 还是触发一次有限的 derived proposal
2. 为后续 student 训练准备：
   - teacher trajectory
   - derived positives / negatives
   - stop labels
   - ranking labels
3. 不把当前 paper 的实现扩张成一个复杂在线 agent

---

## 2. 当前新增前提：raw 知识单元需要 role 标注

在最近讨论中，已明确做出如下决定：

> **需要给 raw 知识单元打标，尤其是 atom，需要标记 `support / distinguish / bridge`。**

这意味着：

- `RawTargetUnit.primary_role` 不再只是可有可无的字段
- 对 teacher 数据集构建而言，需要能够识别某个 raw target / raw atom 在三类 raw role 中属于哪一类
- 至少对用于监督和统计的 raw target units，需要保留该标签
- 对 atom 级单元，role 标注尤其重要，因为它们往往承担最细粒度的 raw coverage 信号

因此，本文档以下内容默认建立在一个前提上：

### 2.1 raw role 标签可用
对 raw target units，至少支持：

- `bridge`
- `distinguish`
- `support`

为避免术语歧义，本文中：
- `distinguish` 与之前文档中的 `disambiguation` 视为同一类 role

也就是说，后文中两种记法等价：

- `distinguish`
- `disambiguation`

### 2.2 当前作用范围
当前 role 标签首先用于：

1. raw ledger `A_t` 的 role-wise 统计  
2. role-wise raw progress / raw completion  
3. need-derived gate 中的 raw semantic readiness  
4. 最终 `d_t^*` 的 raw 三维 deficit

### 2.3 当前不展开的问题
本文档不展开：
- raw role 标签究竟如何自动/半自动获得
- human / heuristic / LLM 在其中如何分工

这里只假设：对 teacher 数据构建所需的 raw 监督对象，这些 role 标签已经可用。

---

## 3. 总体流程位置

在每个 prefix `t`，teacher 数据构建流程中与本文相关的核心顺序为：

1. 构造当前检索查询  
2. 执行 raw retrieval，得到 `R_t`
3. 执行 **cheap stop gate**
4. 若当前不该 stop，再执行 **need-derived gate**
5. 若 derived gate 打开，则执行 **ProposeDerived**
6. 得到 `G_t^{harvest}` 后，再进入 legality filtering 与 final retained selection

对应形式化写法：

\[
q_t = \mathrm{Concat}(q, \mathrm{SlotSummary}(S_t))
\]

\[
R_t = \mathrm{TopKRetrieve}(q_t, M_q)
\]

然后：

- 先做 `CheapStopGate`
- 再做 `NeedDerivedGate`
- 若需要，则执行：
\[
G_t^{harvest} = \mathrm{ProposeDerived}(q,S_t,R_t)
\]

这里必须再次强调：

> **stop candidate 的判断优先于 derive。**

也就是说：
- 不是先 propose derived 再看 stop
- 而是先判断当前是否已经足够接近 terminal

---

## 4. 这些模块为何属于数据集构建而不是在线推理

这是当前讨论中必须明确写清楚的一点。

### 4.1 当前设计默认用于 offline teacher 数据集构建
原因：

1. `cheap stop gate` 与 `need-derived gate` 的主要作用是控制 teacher 轨迹生成成本
2. `ProposeDerived` 明确是一个 **single-call LLM proposer**
3. proposer 输出的 harvest pool 后续还要经过：
   - legality filtering
   - final retained selection
   - teacher final scoring
4. 这些步骤的目标，是为了构造训练样本，而不是直接给最终用户在线回答

因此，当前设计服务的是：

- teacher 轨迹生成
- teacher supervision 构建
- offline 样本保存

### 4.2 它们与 online 推理的关系
在线 student 推理时：
- 可以部分蒸馏/近似 stop gate
- 可以保留一个更轻量的 derived trigger
- 但不应直接照搬 offline teacher 的完整逻辑

特别是：
- offline teacher 允许更复杂的验证与判断
- online 需要更轻、更快、更稳

因此：

> **本文档中的 gate + proposer 设计是 teacher-side dataset construction design，而不是 deployment-time inference design。**

---

## 5. Gate 设计总览

当前推荐只保留两个 gate：

### Gate A：cheap stop gate
回答：

> 当前前缀是否已经足够接近 terminal，以至于应优先尝试 stop？

### Gate B：need-derived gate
回答：

> 如果当前不该 stop，那么是否值得触发一次 derived proposal？

二者的优先级严格如下：

1. 先 cheap stop gate
2. 若 stop candidate 成立，则优先 stop probe
3. 若 stop candidate 不成立，再看 need-derived gate
4. 只有此时 trigger 打开，才执行 `ProposeDerived`

---

## 6. Gate A：cheap stop gate

## 6.1 目标

cheap stop gate 的目标是：

- 尽量便宜地识别 terminal candidate state
- 避免系统在已经接近停止时还机械地继续 derive
- 但又不替代真正的 answer-probe stop

因此它本质上是：

> 一个便宜的、保守的 terminal 候选判定器。

---

## 6.2 输入

cheap stop gate 只依赖以下已存在对象：

- `q`
- `T_q^raw`
- `A_t`
- `S_t`
- `R_t`

不调用 LLM。

---

## 6.3 输出

输出为：

```python
CheapStopGateOutput = {
    "raw_complete": bool,
    "no_derived_need": bool,
    "stop_candidate": bool
}
```

其中：

- `raw_complete`：当前 raw 覆盖是否已基本完成
- `no_derived_need`：当前是否看不出明显还需要额外 derived 组织
- `stop_candidate`：二者同时满足时为真

---

## 6.4 条件 1：raw completion

由于当前已经决定为 raw / atom 保留 role 标签，因此推荐使用 role-wise 版本。

设 raw roles 为：

- `bridge`
- `distinguish`
- `support`

对每个 role `r`，定义当前平滑覆盖率：

\[
\tilde s_t^r
=
\frac{k_t^r + \alpha}{N_q^r + 2\alpha}
\]

其中：

- `k_t^r`：当前 prefix 已覆盖的 role-`r` raw target 数量或加权和
- `N_q^r`：问题 `q` 在 role `r` 上的总 raw target 数量
- `\alpha > 0`：平滑项

只对当前问题确实需要的 role 进行检查。记：

\[
m_q^r = 1[N_q^r > 0]
\]

则定义：

\[
RawComplete_t =
\mathbf{1}\Big[
\forall r\; \text{s.t.}\; m_q^r = 1,\;
\tilde s_t^r \ge \tau_r
\Big]
\]

当前最小版推荐统一阈值：

- `\tau_r = 0.70`

也就是说，只要每个需要的 raw role 覆盖率都达到约 0.7，即认为 raw 基本完成。

---

## 6.5 条件 2：no obvious derived need

cheap stop gate 中的第二个条件，不应复杂。  
推荐使用三个便宜信号：

### 信号 A：最近已有 verification note
如果最近 1–2 步已经加入了 `verification_note`，说明 answer-facing organization 已经出现。

### 信号 B：当前 raw pool 高度冗余
如果当前 `R_t` 的 top raw 与 `S_t` 中最近已纳入的 raw evidence 高度重复，则继续扩 raw 的边际收益可能偏低。

### 信号 C：最近已有 bridge note
如果最近已经存在 `bridge_note`，则说明主要桥接组织未必还缺。

据此定义：

\[
NoDerivedNeed_t = 
has\_recent\_verification(S_t)
\;\lor\;
(raw\_redundant(R_t,S_t) \land has\_recent\_bridge(S_t))
\]

其中：
- `has_recent_verification`：最近窗口中是否已有 verification note
- `has_recent_bridge`：最近窗口中是否已有 bridge note
- `raw_redundant`：当前 raw pool 是否明显与已有 raw evidence 重复

---

## 6.6 最终 cheap stop gate

最终定义：

\[
StopCandidate_t = RawComplete_t \land NoDerivedNeed_t
\]

也就是：

- raw 已基本完成
- 当前又没有明显未满足的 derived 需求

此时进入 candidate stop state。

---

## 6.7 伪代码

```python
def CheapStopGate(q, T_q_raw, A_t, S_t, R_t):
    role_scores = get_rolewise_progress(A_t, T_q_raw)   # {"bridge":..., "distinguish":..., "support":...}
    required_roles = get_required_roles(T_q_raw)
    raw_complete = all(role_scores[r] >= 0.70 for r in required_roles)

    has_recent_verif = has_recent_note(S_t, note_type="verification_note", window=2)
    has_recent_bridge = has_recent_note(S_t, note_type="bridge_note", window=2)
    raw_redundant = is_raw_pool_redundant(R_t, S_t)

    no_derived_need = has_recent_verif or (raw_redundant and has_recent_bridge)

    stop_candidate = raw_complete and no_derived_need

    return {
        "raw_complete": raw_complete,
        "no_derived_need": no_derived_need,
        "stop_candidate": stop_candidate
    }
```

---

## 6.8 `is_raw_pool_redundant(R_t, S_t)` 的极简实现

取：

- `R_t` 前 3 个 raw candidates
- `S_t` 中最近 2 条 raw evidence

若前 3 个里有至少 2 个与最近 evidence 高度相似，则记为冗余。

```python
def is_raw_pool_redundant(R_t, S_t, sim_threshold=0.90):
    recent_raw_texts = get_recent_raw_texts(S_t, n=2)
    if not recent_raw_texts:
        return False

    redundant_count = 0
    for cand in R_t[:3]:
        if any(text_similarity(cand["text"], txt) >= sim_threshold for txt in recent_raw_texts):
            redundant_count += 1

    return redundant_count >= 2
```

---

## 7. Gate B：need-derived gate

## 7.1 目标

need-derived gate 用于回答：

> 在当前不该 stop 的前提下，是否值得触发一次 derived proposal？

它不是 stop gate 的替代，而是 stop gate 之后的第二层判定。

---

## 7.2 输入

need-derived gate 依赖：

- `q`
- `T_q^raw`
- `A_t`
- `S_t`
- `R_t`

不调用 LLM。

---

## 7.3 输出

输出结构：

```python
NeedDerivedGateOutput = {
    "s_sem": float,
    "composable_raw": bool,
    "has_recent_verification": bool,
    "derived_need": bool,
    "trigger_derived": bool
}
```

---

## 7.4 Step 1：计算 raw semantic readiness

由于当前已经决定保留 raw role 标签，因此推荐 role-wise 聚合版。

定义：

\[
s_t^{sem}
=
\alpha_{bridge}\, \tilde s_t^{bridge}
+
\alpha_{distinguish}\, \tilde s_t^{distinguish}
+
\alpha_{support}\, \tilde s_t^{support}
\]

当前最小版推荐先用等权：

\[
\alpha_{bridge}
=
\alpha_{distinguish}
=
\alpha_{support}
=
1/3
\]

也就是说：

\[
s_t^{sem}
=
(\tilde s_t^{bridge}
+
\tilde s_t^{distinguish}
+
\tilde s_t^{support})/3
\]

若某个 role 对当前问题不适用，则仅对需要的 role 归一化平均。

---

## 7.5 Step 2：判断是否存在“可组织的 raw”

derived proposal 只有在当前 raw 候选池中确实存在“值得组合、桥接、验证”的信息时才值得运行。

推荐使用一个极简判断：

### `has_composable_raw(R_t)`
若 `R_t` 前 3 个 raw 中：
- 至少有 2 个文本不高度重复
- 且至少来自 2 个不同 `parent_chunk_id`

则认为存在可组合 raw。

形式上：

\[
composable\_raw_t = has\_composable\_raw(R_t)
\]

极简实现：

```python
def has_composable_raw(R_t, sim_threshold=0.85):
    top_raw = R_t[:3]
    if len(top_raw) < 2:
        return False

    usable = []
    for cand in top_raw:
        if all(text_similarity(cand["text"], u["text"]) < sim_threshold for u in usable):
            usable.append(cand)

    distinct_parents = len({u["parent_chunk_id"] for u in usable})
    return len(usable) >= 2 and distinct_parents >= 2
```

---

## 7.6 Step 3：排除“最近刚做过 verification”的情况

如果最近 1–2 步已经加入了 verification note，那么通常不需要立即再次 propose derived。

定义：

\[
has\_recent\_verification_t = has\_recent\_note(S_t, verification\_note)
\]

---

## 7.7 Step 4：定义 derived_need

推荐定义：

\[
derived\_need_t
=
composable\_raw_t
\land
\neg has\_recent\_verification_t
\]

也就是：

- 当前 top raw 里确实有可组织的材料
- 且系统最近还没有刚做过 verification

---

## 7.8 Step 5：最终 derived trigger

只有当：

- raw semantic readiness 足够高
- 当前确实存在 derived need

时，才触发 proposer。

定义：

\[
T_t^{der}
=
\mathbf 1[
s_t^{sem} \ge \tau_{sem}
\land
derived\_need_t = 1
]
\]

推荐默认值：

- `\tau_sem = 0.5`

---

## 7.9 伪代码

```python
def NeedDerivedGate(q, T_q_raw, A_t, S_t, R_t, tau_sem=0.5):
    role_scores = get_rolewise_progress(A_t, T_q_raw)
    required_roles = get_required_roles(T_q_raw)

    s_sem = sum(role_scores[r] for r in required_roles) / max(1, len(required_roles))

    composable_raw = has_composable_raw(R_t)
    has_recent_verif = has_recent_note(S_t, note_type="verification_note", window=2)

    derived_need = composable_raw and (not has_recent_verif)
    trigger = (s_sem >= tau_sem) and derived_need

    return {
        "s_sem": s_sem,
        "composable_raw": composable_raw,
        "has_recent_verification": has_recent_verif,
        "derived_need": derived_need,
        "trigger_derived": trigger
    }
```

---

## 8. Gate 组合逻辑

当前推荐严格采用如下顺序：

### Step 1：cheap stop gate
得到：

- `raw_complete`
- `no_derived_need`
- `stop_candidate`

### Step 2：若 `stop_candidate = True`
则：
- 当前 prefix 进入 candidate stop state
- 优先执行 answer-probe stop
- 不进入 `ProposeDerived`

### Step 3：若 `stop_candidate = False`
则继续执行 need-derived gate

### Step 4：
- 若 `trigger_derived = True`：执行 `ProposeDerived`
- 否则：当前轮只保留 raw candidates，不做 derived proposal

伪代码：

```python
def PreDerivedGates(q, T_q_raw, A_t, S_t, K_t, R_t):
    stop_info = CheapStopGate(q, T_q_raw, A_t, S_t, R_t)

    if stop_info["stop_candidate"]:
        return {
            "action": "stop_probe",
            "stop_info": stop_info,
            "derived_info": None
        }

    derived_info = NeedDerivedGate(q, T_q_raw, A_t, S_t, R_t)

    if derived_info["trigger_derived"]:
        return {
            "action": "propose_derived",
            "stop_info": stop_info,
            "derived_info": derived_info
        }

    return {
        "action": "raw_only",
        "stop_info": stop_info,
        "derived_info": derived_info
    }
```

---

## 9. ProposeDerived 的角色与边界

`ProposeDerived` 的目标是：

> 在当前 prefix 下，用一次统一的生成调用，补出极少量真正有价值的 derived candidates，供 teacher 在当前轮候选池里选择。

它不是：
- 全局 derived 库构造器
- 每轮必跑模块
- planner
- compiler

它只在：
- 当前不该 stop
- 且 `trigger_derived = True`

时触发。

---

## 10. ProposeDerived 的输入

推荐输入如下：

```python
ProposeDerivedInputs = {
    "question": str,
    "state_summary": str,
    "top_raw_candidates": list[RawUnit],
    "gold_answer": str | None
}
```

### 10.1 `question`
原问题 `q`。

### 10.2 `state_summary`
一个短的、轻量的 notebook 摘要，不是全量 `K_t` dump。  
推荐只包含：

- 最近 1–2 条 raw evidence
- 最近 1 条 derived note（若有）
- 一句 `Need: ...`

### 10.3 `top_raw_candidates`
只取 `R_t` 的前 `J` 个 raw candidates。  
当前推荐：

- `J = 3`

### 10.4 `gold_answer`
仅 teacher offline 阶段可选使用；online/student 时应为空。

---

## 11. ProposeDerived 的输出

当前最小版只允许两类：

- `bridge_note`
- `verification_note`

推荐 JSON 输出：

```json
{
  "should_derive": true,
  "reason": "raw evidence is semantically close but not yet answer-organized",
  "derived_candidates": [
    {
      "unit_id": "z1",
      "type": "bridge_note",
      "text": "The author of The Hobbit is J. R. R. Tolkien.",
      "source_unit_ids": ["m17"],
      "coarse_priority": 1
    },
    {
      "unit_id": "z2",
      "type": "verification_note",
      "text": "The evidence suggests Tolkien studied at Exeter College, which is part of the University of Oxford.",
      "source_unit_ids": ["m03", "m22"],
      "coarse_priority": 2
    }
  ]
}
```

### 当前冻结决定
1. 删除 `claimed_role`
2. `type` 已足够表达当前最小版功能
3. harvest pool 最大候选数：
   - `max_candidates = 4`
4. 每个候选允许 source 数：
   - `1 ~ 3`

---

## 12. 两类 derived 的定义

### 12.1 `bridge_note`
一个短 note，用来显式组织当前 evidence 中隐含但尚未显式形成的桥接关系。

要求：
- 1 句
- grounded
- 不引入 unsupported facts
- 不输出最终答案口吻

### 12.2 `verification_note`
一个短 note，用来把当前 evidence 中已经较接近答案的关系做可验证、answer-facing 的组织。

要求：
- grounded
- 强调 evidence 所支持的内容
- 不自由延展

---

## 13. ProposeDerived 的 prompt 约束

### 13.1 system instruction
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

Return JSON only.
```

### 13.2 user payload 模板
```text
Question:
{q}

Current notebook summary:
{state_summary}

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

## 14. ProposeDerived 的最小算法

### 输入
- `q`
- `S_t`
- `R_t`
- `gold_answer=None`
- `J=3`
- `max_candidates=4`

### 步骤

1. 从 `S_t` 构造一个轻量 `state_summary`
2. 从 `R_t` 中取前 `J` 个 raw candidates
3. 调用一次 proposer
4. 解析 JSON
5. 做 schema validation
6. 形成 `G_t^{harvest}`

伪代码：

```python
def ProposeDerived(q, S_t, R_t, gold_answer=None, J=3, max_candidates=4):
    state_summary = BuildProposerStateSummary(S_t, q)
    top_raw = R_t[:J]

    prompt = build_propose_prompt(
        question=q,
        state_summary=state_summary,
        top_raw_candidates=top_raw,
        gold_answer=gold_answer,
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

## 15. ProposeDerived 的输出只是 harvest pool

这一点必须明确：

> `ProposeDerived` 输出的不是最终 `G_t^{final}`，而只是 `G_t^{harvest}`。

后续仍需经过：

1. legality filtering
2. coarse priority selection
3. 分流为：
   - `G_t^{final}`
   - `G_t^{aux}`
   - `G_t^{illegal}`

这使得 proposer 只是“提议”，而不是最后裁决者。

---

## 16. 默认参数建议

当前 paper 推荐固定：

### stop gate
- role-wise raw completion threshold: `0.70`

### need-derived gate
- `tau_sem = 0.5`

### proposer
- `J = 3`
- `max_candidates = 4`
- allowed types = `{bridge_note, verification_note}`
- max source per note = 3

---

## 17. 一句话总结

当前推荐的 teacher-side 设计是：

> **每轮先做 raw retrieval，再先用 cheap stop gate 判断当前是否已足够接近 terminal；若不该 stop，再用 need-derived gate 判断当前 raw evidence 是否已积累到值得做一次有限 derived organization 的程度。只有 stop 不成立且 derived trigger 打开时，才执行一次统一的 `ProposeDerived` 调用，返回最多四个 grounded derived notes，作为后续 legality filtering 与 final retained selection 的输入。**
