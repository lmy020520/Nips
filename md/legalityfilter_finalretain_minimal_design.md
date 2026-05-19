# LegalityFilter + FinalRetainSelection 设计方案（当前极简版）

## 1. 设计目标

本文档冻结当前版本中 **LegalityFilter** 与 **FinalRetainSelection** 的最小可执行设计。  
该设计服务于：

- **offline teacher 数据集构建**
- `ProposeDerived` 之后的 derived candidate 清洗与保留
- 为后续 student 训练准备：
  - `G_t^{final}`
  - `G_t^{aux}`
  - `G_t^{illegal}`

本文档采用当前已经收敛的一个关键原则：

> **不要把 legality 与 retained selection 设计成新的复杂评分系统。**

也就是说：

- `LegalityFilter` 只做 **硬约束过滤**
- `FinalRetainSelection` 只做 **极简保留策略**
- 不再引入 legality score / retain score 的复杂加权设计
- 尽量减少阈值和超参数数量

---

## 2. 模块位置

在每个 prefix `t`，当前 teacher 数据构建流程中相关顺序为：

1. `TopKRetrieve` 得到 `R_t`
2. `CheapStopGate`
3. `NeedDerivedGate`
4. 若需要，则执行 `ProposeDerived`
5. 得到 `G_t^{harvest}`
6. 对 `G_t^{harvest}` 执行 `LegalityFilter`
7. 对合法集合执行 `FinalRetainSelection`
8. 得到：
   - `G_t^{final}`
   - `G_t^{aux}`
   - `G_t^{illegal}`
9. teacher 最终只在：
   \
   C_t = R_t \cup G_t^{final}
   \
   上做 planning utility 精算

这里必须明确：

> **`ProposeDerived` 只负责提出少量候选；`LegalityFilter` 负责判定哪些候选在当前前缀下合法可用；`FinalRetainSelection` 负责从合法候选中保留极少量最值得进入 teacher 最终候选池的 note。**

---

## 3. 为什么这一版要极简化

前面讨论中我们已经明确：

- 如果 `LegalityFilter` 里加入太多细粒度规则和软分项
- 如果 `FinalRetainSelection` 再自己计算一套复杂 retain score

那么这两个模块本身就会演化成新的工程系统，并引入大量超参数，例如：

- grounding score 权重
- novelty score 权重
- compactness score 权重
- source spread 权重
- retain threshold
- type-balance margin
- 多种 legality 子阈值

这会带来两个问题：

### 3.1 超参数过多
一旦规则和软分项太多，teacher 数据集构建就会高度依赖调参，稍有不慎就会把偏差写进整个训练集。

### 3.2 主线漂移
current paper 的主线是：

- answer-identifiable compiled context
- closure-aware control
- teacher / student 分工
- retrieval 之后的 sequential decision

而 legality 与 retained selection 只是 supporting module，不应喧宾夺主。

因此当前版本的最终取舍是：

> **把 legality 与 retained selection 从“评分器”降级为“约束器 + 极简选择器”。**

---

## 4. 三个 derived pool 的定义

### 4.1 `G_t^{harvest}`
这是 `ProposeDerived` 的直接输出。

特点：
- 最多 4 个 derived candidates
- 只是 proposer 的 harvest pool
- 尚未经过 legality 或 final retain

---

### 4.2 `G_t^{illegal}`
这是 legality filter 判定为非法的 derived candidates。

特点：
- 不进入 teacher 最终候选池
- 保留为 legality negatives
- 用于后续 student 训练中的 legality / filter supervision

---

### 4.3 `G_t^{aux}`
这是合法但未进入 final retained 的 derived candidates。

特点：
- 不参与当前 teacher 最终精算
- 作为 derived medium negatives 保存

---

### 4.4 `G_t^{final}`
这是最终进入当前 teacher 候选池的极少量 derived candidates。

特点：
- 最多 1–2 个
- 才会进入：
  \
  C_t = R_t \cup G_t^{final}
  \
- 若最终未被 teacher 选中，则构成 hard negatives

---

## 5. LegalityFilter 的设计目标

`LegalityFilter` 只回答一个问题：

> **这个 derived candidate 在当前 prefix 下是否合法、可追溯、可进入当前候选池？**

它不负责：

- usefulness 排序
- teacher planning
- 当前到底更缺 bridge 还是 verification
- legality 的连续打分

因此，当前版本中：

> **LegalityFilter 只做 hard filtering，不做 soft legality scoring。**

---

## 6. LegalityFilter 的输入与输出

### 输入
```python
LegalityFilterInputs = {
    "S_t": State,
    "R_t": list[RawUnit],
    "G_t_harvest": list[DerivedUnit]
}
```

### 输出
```python
LegalityFilterOutput = {
    "G_t_legal": list[DerivedUnit],
    "G_t_illegal": list[dict]   # {"candidate": ..., "reasons": [...]}
}
```

其中：

- `G_t_legal`：通过 legality 的合法 derived 候选
- `G_t_illegal`：被过滤掉的非法 derived candidates 及其原因

---

## 7. LegalityFilter：当前只保留 4 类硬规则

这是本版最重要的冻结决定。

---

### Rule 1：Schema + Type 合法

候选必须至少包含：

- `unit_id`
- `type`
- `text`
- `source_unit_ids`
- `coarse_priority`

并且：

- `type` 只能取：
  - `bridge_note`
  - `verification_note`

否则直接非法。

---

### Rule 2：Source 可见且数量合法

必须满足：

\[
source\_unit\_ids(z) \subseteq S_t^{raw} \cup R_t
\]

也就是：

- source ids 全部必须是当前可见 raw units
- 不能引用未来 source
- 不能引用当前不可见 source
- 当前版本不允许用 derived note 作为 source ids

同时要求：

\[
1 \le |source\_unit\_ids(z)| \le 3
\]

并且 source ids 不可重复。

这条规则是 provenance 的底线，因此必须保留。

---

### Rule 3：句长和句数受控

derived note 必须：

- 只有 1 句
- token 数不超过固定上限

当前推荐统一使用：

- `max_tokens = 45`

不再区分：
- bridge 40
- verification 50

因为那样只会增加超参数。

---

### Rule 4：重复剔除

如果候选：

1. 与 `S_t` 中已有 derived note 重复  
2. 或与当前 `G_t^{harvest}` 中更高优先级候选重复  

则直接非法。

这里重复判定保持极简：

- 先做 normalized exact match
- 再做一个固定的近重复阈值，例如：
  - text similarity > 0.9

不再引入更复杂 novelty 分数。

---

## 8. LegalityFilter 中删除的内容

当前版本明确删除以下内容：

- legality score
- grounding 连续分
- entity support 连续分
- multi-source use 连续分
- compactness 连续分
- unsupported new entity 的复杂判断
- fake multi-source 的复杂判别

原因不是这些内容没价值，而是：

> **它们会引入新的规则项、阈值与权重，使 legality 模块本身变成一个新的调参系统。**

对于 current paper，这不值得。

---

## 9. LegalityFilter 的极简伪代码

```python
def LegalityFilter(S_t, R_t, G_t_harvest):
    visible_raw_ids = set(get_raw_ids_from_state(S_t)) | {u["unit_id"] for u in R_t}

    legal = []
    illegal = []
    accepted_texts = set()

    for z in sorted(G_t_harvest, key=lambda x: x["coarse_priority"]):
        reasons = []

        # 1. schema + type
        if not has_required_fields(z, ["unit_id", "type", "text", "source_unit_ids", "coarse_priority"]):
            reasons.append("invalid_schema")
        if z.get("type") not in {"bridge_note", "verification_note"}:
            reasons.append("invalid_type")

        # 2. visible source ids
        src_ids = z.get("source_unit_ids", [])
        if not (1 <= len(src_ids) <= 3):
            reasons.append("invalid_source_count")
        if len(src_ids) != len(set(src_ids)):
            reasons.append("duplicated_source_ids")
        if not set(src_ids).issubset(visible_raw_ids):
            reasons.append("invisible_source_ids")

        # 3. length / sentence
        if sentence_count(z.get("text", "")) != 1:
            reasons.append("not_single_sentence")
        if token_count(z.get("text", "")) > 45:
            reasons.append("too_long")

        # 4. duplicate
        norm_text = normalize_text(z.get("text", ""))
        if norm_text in accepted_texts:
            reasons.append("duplicate_in_harvest")
        if is_duplicate_with_existing_notes(z, S_t):
            reasons.append("duplicate_with_state")

        if reasons:
            illegal.append({"candidate": z, "reasons": reasons})
            continue

        accepted_texts.add(norm_text)
        legal.append(z)

    return {
        "G_t_legal": legal,
        "G_t_illegal": illegal
    }
```

---

## 10. FinalRetainSelection 的设计目标

`FinalRetainSelection` 的问题不是：

> “哪个候选最合法？”

而是：

> **在已经合法的 derived candidates 中，哪几个最值得进入当前 teacher 最终候选池？**

当前版本中，它的职责极简化为：

1. 复用 proposer 已给出的 `coarse_priority`
2. 保留极少量 derived candidates
3. 尽量保证 type 多样性
4. 其余全部保留到 `G_t^{aux}`

也就是说：

> **FinalRetainSelection 不再自己重新计算复杂 retain score。**

---

## 11. FinalRetainSelection 的输入与输出

### 输入
```python
FinalRetainInputs = {
    "G_t_legal": list[DerivedUnit]
}
```

### 输出
```python
FinalRetainOutput = {
    "G_t_final": list[DerivedUnit],
    "G_t_aux": list[DerivedUnit]
}
```

其中：

- `G_t_final`：进入当前 teacher 最终候选池的 derived
- `G_t_aux`：合法但未进入 final retained 的 derived

---

## 12. FinalRetainSelection 的原则

### 原则 1：按 `coarse_priority` 排序
直接复用 proposer 的顺序，不再另算复杂 usefulness 分数。

---

### 原则 2：第一名一定保留
当前 legal candidates 中，priority 最高者直接进入 `G_t^{final}`。

---

### 原则 3：第二名优先补另一种 type
如果 legal set 中同时有：

- `bridge_note`
- `verification_note`

则第二个优先选择与第一名不同 `type` 的候选。

只有当另一类型不存在时，才选择同类型的下一个。

---

### 原则 4：最终最多 2 个
保持当前版本极简约束：

\[
|G_t^{final}| \le 2
\]

---

### 原则 5：其余全部进入 `G_t^{aux}`
不再做第二轮复杂筛选。

---

## 13. 为什么要保留“类型多样性”规则

即使在极简版中，我仍建议保留这一条。  
原因是：

- 当前 derived 只有两类：
  - `bridge_note`
  - `verification_note`
- 若 legal pool 同时有两类候选，而 final retained 全是同一类，往往会浪费另一类高价值中间产物

因此，“优先补另一种 type”是一个非常低成本、但收益很稳定的规则。

更重要的是：

> 它几乎不引入新的超参数。

---

## 14. FinalRetainSelection 的极简伪代码

```python
def FinalRetainSelection(G_t_legal, max_final=2):
    if not G_t_legal:
        return {"G_t_final": [], "G_t_aux": []}

    candidates = sorted(G_t_legal, key=lambda x: x["coarse_priority"])

    final = [candidates[0]]

    if max_final >= 2 and len(candidates) > 1:
        first_type = candidates[0]["type"]

        second = None
        for cand in candidates[1:]:
            if cand["type"] != first_type:
                second = cand
                break

        if second is None:
            second = candidates[1]

        final.append(second)

    final_ids = {z["unit_id"] for z in final}
    aux = [z for z in candidates if z["unit_id"] not in final_ids]

    return {
        "G_t_final": final[:max_final],
        "G_t_aux": aux
    }
```

---

## 15. 当前版本删除的 FinalRetainSelection 内容

以下内容当前建议全部删除：

- retain score
- legality score 作为 retain 输入
- bridge need / verification need 加权
- source spread 分数
- novelty 分数
- compactness 分数
- retain threshold
- type-balance margin

原因只有一个：

> **这些设计会立刻引入一批新的权重、阈值和耦合行为，使 final retained selection 变成一个新的调参系统。**

对于 current paper，不值得。

---

## 16. 现在还剩多少超参数？

按当前极简版，真正保留下来的超参数非常少：

### LegalityFilter
- `max_source_per_note = 3`
- `max_tokens = 45`
- `duplicate_similarity_threshold = 0.9`
  - 甚至可以更激进，先只做 normalized exact match

### FinalRetainSelection
- `max_final = 2`

相比前一版，绝大多数加权项与软阈值都已删除。

---

## 17. 与 teacher 数据集构建的关系

在当前 teacher-side dataset construction 中：

### `G_t^{illegal}`
作为 legality negatives 保存

### `G_t^{aux}`
作为 legal-but-not-retained 的 derived medium negatives 保存

### `G_t^{final}`
进入当前 teacher 候选池：

\[
C_t = R_t \cup G_t^{final}
\]

若某个 final retained derived 未被 teacher 最终选中，则它构成 hard negatives。

因此，这套设计天然支持后续训练监督构造：

- legality negatives
- medium derived negatives
- hard derived negatives

同时又不需要让 legality 和 retained selection 自己复杂化。

---

## 18. 当前版本的最终冻结建议

### LegalityFilter
只保留 4 类硬规则：

1. schema/type 合法
2. source 可见且数量合法
3. 单句且长度受控
4. 重复剔除

不做 legality score。

---

### FinalRetainSelection
只保留 5 条规则：

1. 按 `coarse_priority` 排序
2. 第一名一定保留
3. 第二名优先补另一种 type
4. 最多 2 个
5. 其余全部进入 `G_t^{aux}`

不做 retain score。

---

## 19. 一句话总结

当前推荐的最小版设计是：

> **`LegalityFilter` 只做硬合法性过滤，`FinalRetainSelection` 只复用 proposer 的 `coarse_priority` 并加一个极简的类型多样性规则，从而把 legality 与 retained selection 这两个模块稳定地降级为“约束器 + 极简选择器”，而不是新的复杂评分系统。**
