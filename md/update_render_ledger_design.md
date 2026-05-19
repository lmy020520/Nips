# Update + Render + Ledger 设计稿（面向数据集构建）

## 1. 目的与边界

本文档整理当前版本中 **Update**、**Render** 与 **Ledger** 三个核心函数的统一设计，目标是为 **offline teacher 数据集构建** 提供一套清晰、最小、可执行的状态递推方案。

本文档首先明确一个关键边界：

> **以下设计默认针对 offline teacher 数据集构建，而不是 online 推理。**

也就是说，这里的目标是：

1. 在 teacher 轨迹构建过程中，给出当前前缀状态的最小递推规则  
2. 保证 `H_t / A_t / S_t / K_t` 之间的关系清晰  
3. 为后续：
   - `SlotSummary`
   - `CheapStopGate`
   - `NeedDerivedGate`
   - `ProposeDerived`
   - teacher stop probe
   提供稳定、轻量的状态接口  
4. 不把 `Update / Render / Ledger` 演化成新的重型模块

---

## 2. 总体位置与时序

在第 `t` 步，当 teacher 已从当前候选池中选出：

\[
u_{t+1}^*
\]

之后，推荐按如下顺序递推状态：

### Step 1：更新 notebook state
\[
S_{t+1} = Update(S_t, u_{t+1}^*, t+1)
\]

### Step 2：更新 raw ledger
\[
A_{t+1} = Ledger(A_t, u_{t+1}^*, T_q^{raw})
\]

### Step 3：渲染 compiled context
\[
K_{t+1} = Render(q, S_{t+1}, UnitRegistry)
\]

然后下一轮才继续：

- `SlotSummary(S_{t+1})`
- `TopKRetrieve(q_{t+1}, M_q)`
- stop gate / need-derived gate
- proposer / final scoring

这里顺序必须固定为：

> **先 Update，再 Ledger，再 Render。**

原因：

1. `Update` 负责 notebook state 的基础递推  
2. `Ledger` 负责 raw 覆盖统计，与 notebook 无直接依赖  
3. `Render` 只依赖最新 `S_{t+1}`

因此，不建议把顺序改成：
- 先 Render 再 Update
- 先 Ledger 再 Update
- 或让三者相互混杂

---

## 3. 当前总体原则

本文档中的三函数都遵循同一个最小化原则：

> **不新增复杂评分系统，不引入新的 LLM 调用点，全部做成确定性、轻量、可复现的函数。**

具体来说：

### Update
- 只做 state 引用更新
- 不做语义抽取
- 不生成新 note

### Render
- 只做模板化 context 拼装
- 不做自由改写
- 不做额外生成

### Ledger
- 只做 raw 覆盖增量统计
- 不做 notebook 更新
- 不做 teacher scoring

---

## 4. Update 的设计

## 4.1 目标

`Update` 的目标是：

> **把当前步选中的知识单元 `u_{t+1}` 写入轻量 notebook state `S_t`，并维护最少量的状态计数。**

当前版本中，`S_t` 已经收敛为：

- 一个 **evidence-centric notebook**
- 只保存 raw / derived 单元在 notebook 中的轻量引用
- 不承担高层语义抽取责任

因此，`Update` 必须尽量“傻”：

- raw unit 进 notebook，只是记录它进来了
- derived unit 进 notebook，只是记录它进来了
- 不再试图在 update 时“顺手”做 semantic lifting

---

## 4.2 输入与输出

### 输入
```python
UpdateInputs = {
    "S_t": State,
    "u_next": RawUnit | DerivedUnit,
    "step_id": int
}
```

### 输出
```python
S_t1: State
```

---

## 4.3 状态结构回顾

当前冻结的最小版 `State` 为：

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

这里的关键原则是：

> `State` 只保存引用，不重复复制 unit payload。  
真正的文本与字段都通过 `UnitRegistry[unit_id]` 访问。

---

## 4.4 更新规则

### Rule 1：按 provenance 分流
若：
- `u_next.provenance == "raw"`  
  则更新 `raw_refs`
- `u_next.provenance == "derived"`  
  则更新 `derived_refs`

### Rule 2：若 unit 已存在，不重复追加
若 `unit_id` 已在对应 ref list 中存在，则：

- 不追加新 ref
- 只做：
  - `selected_count += 1`
  - 更新 `last_added_unit_id`
  - 更新 `last_updated_step`

### Rule 3：若 unit 不存在，则新建一个 `StateRef`
默认初始值：

- `added_step = step_id`
- `used_in_summary_count = 0`
- `selected_count = 1`

### Rule 4：不做任何语义归纳
尤其明确禁止在 `Update` 中做：

- bridge / support / distinguish 自动提炼
- role 预测
- 额外 note 生成
- query rewriting
- compiled context 重写

这是当前版本必须坚持的边界。

---

## 4.5 伪代码

```python
def Update(S_t, u_next, step_id):
    S_next = deepcopy(S_t)

    if u_next["provenance"] == "raw":
        ref_list = S_next["raw_refs"]
    else:
        ref_list = S_next["derived_refs"]

    existing = None
    for ref in ref_list:
        if ref["unit_id"] == u_next["unit_id"]:
            existing = ref
            break

    if existing is None:
        ref_list.append({
            "unit_id": u_next["unit_id"],
            "added_step": step_id,
            "used_in_summary_count": 0,
            "selected_count": 1
        })
    else:
        existing["selected_count"] += 1

    S_next["last_added_unit_id"] = u_next["unit_id"]
    S_next["last_updated_step"] = step_id
    return S_next
```

---

## 4.6 为什么 Update 必须这么轻

前面讨论中已经明确：

- 如果 `Update` 要负责从 `u_{t+1}` 中抽“有用高层信息”
- 那本质上就是把语义结构化成本藏进了 update
- 这会让 raw-side 再次变重

因此，当前 current paper 的最小强实现应坚持：

> **raw 先以 evidence 的形式进入 notebook，derived 才承担高层语义组织。**

---

## 5. Render 的设计

## 5.1 目标

`Render` 的目标是：

> **把当前 notebook state `S_t` 渲染成 answer-facing compiled context `K_t`。**

这个 `K_t` 主要用于：

- candidate stop state 下的 answer-probe stop
- teacher-side answerability / verification 检查
- 部分 analysis / diagnostics

它不是系统内部真实状态，而是一个面向 answerer / verifier 的只读视图。

---

## 5.2 输入与输出

### 输入
```python
RenderInputs = {
    "q": str,
    "S_t": State,
    "UnitRegistry": dict,
    "max_raw": int,
    "max_notes": int,
    "max_chars_per_item": int
}
```

### 输出
```python
K_t: str
```

---

## 5.3 当前版本的极简渲染策略

当前 paper 不应该把 `Render` 做成复杂的内容选择器。  
推荐只保留三步：

### Step 1：选 active notes
从 `S_t.derived_refs` 中最多取：

- 最新 1 条 `bridge_note`
- 最新 1 条 `verification_note`

因此：

- active notes 最多 2 条
- 不做复杂 ranking
- 不取所有旧 note

### Step 2：优先放入 note 的支撑 evidence
若 active notes 存在，则先把其 `source_unit_ids` 对应的 raw units 放入 evidence 区。

原因：

- note 单独出现会显得悬空
- 给 answerer/verifier 看时，source evidence 应优先出现

### Step 3：再补最近 raw evidence
如果 raw evidence 区还没满，再按 `added_step` 倒序补最近 raw refs，直到达到 `max_raw`。

因此 Render 的核心策略是：

> **先 note 的支撑证据，再 recent raw。**

---

## 5.4 默认参数

当前最小版推荐：

- `max_raw = 6`
- `max_notes = 2`
- `max_chars_per_item = 220`

---

## 5.5 模板

推荐固定模板：

```text
Evidence:
[1] ...
[2] ...
[3] ...

Notes:
[bridge] ...
[verification] ...
```

若某一节为空，则省略该节。

### 是否把问题 `q` 也写进 `K_t`？
不需要。  
因为 stop probe 的调用形式本来就是：

\[
Answerer(q, K_t)
\]

问题 `q` 由外部单独传入，不需要重复写进 `K_t`。

---

## 5.6 伪代码

```python
def Render(q, S_t, UnitRegistry, max_raw=6, max_notes=2, max_chars_per_item=220):
    # 1) collect latest notes by type
    bridge_note = get_latest_note_by_type(S_t, UnitRegistry, "bridge_note")
    verification_note = get_latest_note_by_type(S_t, UnitRegistry, "verification_note")

    active_notes = []
    if bridge_note is not None:
        active_notes.append(bridge_note)
    if verification_note is not None:
        active_notes.append(verification_note)

    # 2) collect supporting raw evidence from note sources
    raw_ids = []
    for note in active_notes:
        for sid in note["source_unit_ids"]:
            if sid not in raw_ids:
                raw_ids.append(sid)

    # 3) fill remaining slots with recent raw refs
    for ref in sorted(S_t["raw_refs"], key=lambda x: x["added_step"], reverse=True):
        if len(raw_ids) >= max_raw:
            break
        if ref["unit_id"] not in raw_ids:
            raw_ids.append(ref["unit_id"])

    raw_units = [UnitRegistry[uid] for uid in raw_ids[:max_raw]]

    # 4) render
    parts = []

    if raw_units:
        parts.append("Evidence:")
        for i, u in enumerate(raw_units, start=1):
            txt = shorten(u["text"], max_chars_per_item)
            parts.append(f"[{i}] {txt}")

    if active_notes:
        parts.append("")
        parts.append("Notes:")
        for note in active_notes[:max_notes]:
            label = "bridge" if note["type"] == "bridge_note" else "verification"
            txt = shorten(note["text"], max_chars_per_item)
            parts.append(f"[{label}] {txt}")

    return "\n".join(parts).strip()
```

---

## 5.7 当前版本故意不做的事

Render 当前明确不做：

- note / evidence 的复杂评分
- 句子级重写
- 证据摘要
- query-aware aggressive reordering
- LLM-based compression

原因很简单：

> **Render 是 supporting module，不应自己长成一个新的 context optimization 系统。**

---

## 6. Ledger 的设计

## 6.1 目标

`Ledger` 的目标是：

> **把当前新加入的 raw knowledge unit 对全局 raw target supervision set 的覆盖贡献，增量累加到 `A_t`。**

它是一个 teacher-side supervision bookkeeping function。  
也正因此，它天然属于 **offline teacher 数据集构建**，而不是 online 推理主链。

---

## 6.2 输入与输出

### 输入
```python
LedgerInputs = {
    "A_t": ALedger,
    "u_next": RawUnit | DerivedUnit,
    "T_q_raw": list[RawTargetUnit],
    "tau_overlap": float
}
```

### 输出
```python
A_t1: ALedger
```

---

## 6.3 `A_t` 的推荐结构

推荐在当前版本中使用：

```python
ALedger = {
    "covered_target_ids": set[str],
    "k_bridge": float,
    "k_distinguish": float,
    "k_support": float,
    "coverage_trace": dict   # target_id -> unit_id
}
```

### 字段说明
- `covered_target_ids`：当前 prefix 已覆盖的 raw target ids
- `k_bridge`：bridge 类 raw target 的累计覆盖量
- `k_distinguish`：distinguish 类 raw target 的累计覆盖量
- `k_support`：support 类 raw target 的累计覆盖量
- `coverage_trace`：每个 raw target 是被哪个 raw unit 覆盖的，用于 debug 和分析

---

## 6.4 当前前提：raw role 标签已可用

本函数默认建立在当前新增前提上：

> **raw target units 尤其是 atom 已经具备 `bridge / distinguish / support` role 标签。**

也就是说，`Ledger` 默认使用 role-wise bookkeeping，而不是单维 raw coverage 版本。

---

## 6.5 更新规则总览

### 情况 A：`u_next` 是 derived
则：

\[
A_{t+1} = A_t
\]

也就是说：
- derived 不直接更新 raw ledger

### 情况 B：`u_next` 是 raw
则：

1. 找出所有被 `u_next` 覆盖的 raw targets
2. 对尚未覆盖的 targets：
   - 加入 `covered_target_ids`
   - 根据 `primary_role` 更新 `k_bridge / k_distinguish / k_support`
   - 写入 `coverage_trace[target_id] = u_next.unit_id`

---

## 6.6 默认匹配规则：exact match 优先

当前推荐：

> **`T_q^raw` 的粒度尽量与 retrieval / selection 使用的 raw units 保持一致。**

因此 `Ledger` 的默认主路径应是：

### Rule A：exact id match
若：

\[
u_{next}.unit\_id = y.unit\_id
\]

则认为 raw target `y` 被覆盖。

这应是默认路径。

---

## 6.7 fallback 规则：span overlap

若 raw target 与 raw candidate 粒度不完全一致，则允许 fallback：

### Rule B：same doc + same parent + high span overlap
若满足：

- `doc_id` 相同
- `parent_chunk_id` 相同
- `span_overlap(u_next, y) >= \tau_overlap`

则认为 `y` 被覆盖。

推荐默认：

- `\tau_overlap = 0.8`

但必须强调：

> 这是兼容路径，不应成为默认主路径。

---

## 6.8 伪代码

```python
def Ledger(A_t, u_next, T_q_raw, tau_overlap=0.8):
    A_next = {
        "covered_target_ids": set(A_t["covered_target_ids"]),
        "k_bridge": A_t["k_bridge"],
        "k_distinguish": A_t["k_distinguish"],
        "k_support": A_t["k_support"],
        "coverage_trace": dict(A_t.get("coverage_trace", {}))
    }

    if u_next["provenance"] != "raw":
        return A_next

    matched_targets = []

    for y in T_q_raw:
        if y["target_id"] in A_next["covered_target_ids"]:
            continue

        # exact match preferred
        if u_next["unit_id"] == y["unit_id"]:
            matched_targets.append(y)
            continue

        # fallback: same doc/parent + span overlap
        if (
            u_next.get("doc_id") == y.get("doc_id")
            and u_next.get("parent_chunk_id") == y.get("parent_chunk_id")
            and u_next.get("span_start") is not None
            and y.get("span_start") is not None
        ):
            if span_overlap_ratio(
                (u_next["span_start"], u_next["span_end"]),
                (y["span_start"], y["span_end"])
            ) >= tau_overlap:
                matched_targets.append(y)

    for y in matched_targets:
        A_next["covered_target_ids"].add(y["target_id"])
        A_next["coverage_trace"][y["target_id"]] = u_next["unit_id"]

        role = y.get("primary_role", None)
        w = y.get("weight", 1.0)

        if role == "bridge":
            A_next["k_bridge"] += w
        elif role in {"distinguish", "disambiguation"}:
            A_next["k_distinguish"] += w
        elif role == "support":
            A_next["k_support"] += w

    return A_next
```

---

## 6.9 为什么 Ledger 用增量更新，而不是每次从 `H_t` 全量重算

理论上可以从 `H_t` 全量重算 `A_t`，但当前版本不推荐这么做。

原因：

1. 每轮只新增一个 `u_next`
2. `A_t` 本来就是账本，天然适合增量更新
3. teacher rollout 中频繁重算会增加不必要开销
4. 当前 current paper 不需要把 bookkeeping 做得比必要更重

因此推荐：

- 默认用增量 `Ledger(A_t, u_next, T_q^raw)`
- 仅在 debug / consistency check 时，才允许从 `H_t` 全量回放重算

---

## 7. 三个函数在 teacher-side dataset construction 中的关系

当前推荐的最小状态递推如下：

### 初始化
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

### 每个前缀的递推
给定 `u_{t+1}^*`：

1. 更新 notebook：
\[
S_{t+1} = Update(S_t, u_{t+1}^*, t+1)
\]

2. 更新 raw ledger：
\[
A_{t+1} = Ledger(A_t, u_{t+1}^*, T_q^{raw})
\]

3. 渲染 compiled context：
\[
K_{t+1} = Render(q, S_{t+1}, UnitRegistry)
\]

这里三者职责完全分开：

- `Update`：状态引用层
- `Ledger`：raw 覆盖统计层
- `Render`：answer-facing 视图层

---

## 8. 当前版本的最终冻结建议

### Update
- 只改 `StateRef`
- 不做语义抽取
- 不生成新 note
- 只维护：
  - `raw_refs`
  - `derived_refs`
  - `selected_count`
  - `last_added_unit_id`
  - `last_updated_step`

### Render
- 只保留：
  - 最新 `bridge_note` 1 条
  - 最新 `verification_note` 1 条
  - note 的 source evidence 优先
  - 再补 recent raw evidence
- 固定模板输出字符串
- 不做复杂重写

### Ledger
- derived 不更新 raw ledger
- raw 默认 exact id match
- fallback 才使用 span overlap
- 增量更新：
  - `covered_target_ids`
  - `k_bridge / k_distinguish / k_support`
  - `coverage_trace`

---

## 9. 一句话总结

当前推荐的最小版设计是：

> **`Update` 是纯状态引用更新器，`Render` 是纯模板化 context 拼装器，`Ledger` 是纯 raw 覆盖增量账本更新器；三者都不应再承担额外的语义抽取、打分或生成职责。**
