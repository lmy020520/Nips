# `c_t^*` 监督信号构建设计稿

## 1. 目的与边界

本文档单独整理当前版本中 **正样本 contribution 监督信号 `c_t^*`** 的定义与构建流程，目标是明确：

1. `c_t^*` 在当前框架中的角色  
2. 概念层的 `c_t^*(u)` 如何定义  
3. 最终推荐存储给 student 的 `c_t^*(u_t^+)` 应如何计算  
4. trajectory 构建完成后，如何离线回放得到 `c_t^*` 标签  
5. `c_t^*` 与 `d_t^*`、`RankingLabel`、failed trajectories 的关系  

本文档首先明确一个关键边界：

> **`c_t^*` 属于 trajectory 构建完成之后的后处理监督标签；它不是 teacher 在线规划时直接读取的真值。**

也就是说：

- teacher 规划层：使用即时 utility `U_t(u)` 做选择
- supervision 层：trajectory 构建完之后，再离线回放得到 `d_t^* / c_t^*`

这两个层面应严格分开。

---

## 2. `c_t^*` 的角色

在当前框架中，`c_t^*` 表示：

> **当前 prefix 下，正样本 `u_t^+` 对 typed residual deficit 的逐维补缺贡献。**

也就是说：

- `d_t^*` 描述：当前还剩多少缺口
- `c_t^*(u_t^+)` 描述：teacher 实际选中的这一步，补掉了多少缺口

因此，`c_t^*` 的作用是：

1. 让 student 学会“某类候选在当前前缀下会带来什么样的缺口下降”
2. 支撑 online control 中的 deficit update
3. 区分：
   - raw 候选主要补 raw role deficit
   - derived 候选主要补 derived/context deficit

---

## 3. 一个关键澄清：概念定义 vs 最终存储标签

`c_t^*` 在当前系统中有两个层次。

---

## 3.1 概念层：对任意候选 `u ∈ C_t`

概念上，对任意当前候选 `u`，都可以定义：

\[
c_t^*(u)
=
[d_t^* - d_{t+1}^{*(u)}]_+
\]

其中：

- `d_t^*`：当前 prefix 的 typed residual deficit
- `d_{t+1}^{*(u)}`：假设把候选 `u` 加入当前 prefix 后得到的新 deficit
- `[x]_+`：逐维非负截断

也就是说：

1. 从当前前缀 `H_t, S_t, A_t, K_t` 出发
2. 模拟把候选 `u` 加入
3. 得到模拟后的状态
4. 重新计算新 deficit
5. 取 deficit 的逐维下降量

这个定义适用于：
- raw 候选
- derived 候选

也是理论上最干净的 contribution 定义。

---

## 3.2 最终存储层：只对正样本保存 `c_t^*(u_t^+)`

当前最小强实现不建议对所有负样本都保存 full role-wise contribution label。  
最终推荐的做法是：

> **只对当前 teacher 选中的正样本 `u_t^+ = u_{t+1}^*` 保存结构化 `c_t^*` 标签。**

理由：

1. contribution 是强 prefix-conditioned 的  
2. 同一个候选在别的 prefix 下可能成为正样本  
3. 对所有负样本做 full structured labeling 成本高、收益小  
4. 负样本主要通过 `RankingLabel` 来学习  

因此：

- `c_t^*(u_t^+)`：保存
- `c_t^*(u^-)`：通常不保存
- 负样本只做 ranking / legality / filter 辅助监督

---

## 4. 当前推荐的 `d_t^*` 背景

`c_t^*` 是由 `d_t^*` 的相邻差值导出的，因此必须先明确当前版本推荐的 `d_t^*`。

当前推荐的 typed residual deficit 为：

\[
d_t^*=
(
d_t^{br*},
d_t^{dis*},
d_t^{sup*},
d_t^{der*}
)
\]

其中：

- `d_t^{br*}`：bridge raw deficit
- `d_t^{dis*}`：distinguish raw deficit
- `d_t^{sup*}`：support raw deficit
- `d_t^{der*}`：derived 剩余工作 deficit

### raw 三维 deficit
raw 三维来自当前 raw ledger `A_t` 与全局 raw target 集 `T_q^{raw}` 的 role-wise 覆盖关系。

### derived 第四维 deficit
当前推荐版本中，derived 第四维采用：

> **remaining derived work**

也就是：

- 看整条 teacher trajectory 最终用了多少个 derived steps
- prefix `t` 后面还剩多少个 derived steps
- 用这个“剩余比例”做 `d_t^{der*}`

这样做的优点是：
- 标签单调
- 不需要全局 GT derived targets
- 与真实 teacher 轨迹对齐

---

## 5. 最终推荐的 `c_t^*(u_t^+)` 定义

当前最推荐的存储版 `c_t^*(u_t^+)` 为：

\[
c_t^*(u_t^+)
=
\frac{
[d_t^* - d_{t+1}^*]_+
}{
d_t^* + \epsilon
}
\]

这里的除法与截断都是逐维进行的。  
也就是：

\[
c_t^{r*}(u_t^+)
=
\frac{
\max(0, d_t^{r*} - d_{t+1}^{r*})
}{
d_t^{r*} + \epsilon
}
\]

其中：
- `r ∈ {br, dis, sup, der}`
- `\epsilon > 0` 为小常数，防止分母为 0

推荐：
- `\epsilon = 1e-6`

---

## 6. 为什么采用“归一化的相邻差值”

当前推荐这一版定义，是因为它同时满足三个要求：

### 6.1 语义清晰
若某一维 deficit 显著下降，则该维 contribution 大。  
若该维 deficit 不变，则 contribution 为 0。

### 6.2 数值稳定
不同 prefix、不同问题中 deficit 规模不同。  
若不归一化，会导致 contribution 在不同问题间难比较。

### 6.3 与 online deficit update 兼容
当前 online student 侧常见的 deficit update 形式可写成：

\[
\bar d_{t+1}=d_t \odot (1-c_t(u_{t+1}))
\]

因此 teacher 监督中的 `c_t^*(u_t^+)` 若是“相对 deficit 降幅”，会与 online control 机制天然对齐。

---

## 7. `c_t^*` 的逐维直觉

对当前正样本 `u_t^+`：

### 若是 raw bridge
通常会带来较大的：
\[
c_t^{br*}(u_t^+)
\]

### 若是 raw support
通常会带来较大的：
\[
c_t^{sup*}(u_t^+)
\]

### 若是 raw distinguish
通常会带来较大的：
\[
c_t^{dis*}(u_t^+)
\]

### 若是有用的 derived step
通常会带来较大的：
\[
c_t^{der*}(u_t^+)
\]

因此，`c_t^*` 能够把 teacher 的下一步行为转成一个“这一步主要补了哪一类缺口”的监督信号。

---

## 8. trajectory 构建完成后的离线计算流程

`c_t^*` 的构建必须在 trajectory 完成后进行，因为它依赖：

- 完整前缀状态
- 相邻 prefix 的 `d_t^*`
- 特别是 derived 第四维 deficit 依赖整条轨迹中真正发生了多少 derived steps

当前推荐流程如下。

---

### Step 0：先得到完整 teacher trajectory

设完整 teacher trajectory 为：

\[
H^*=(u_1^*,u_2^*,\dots,u_T^*)
\]

并且对每个 prefix `t`，已经保存：

- `H_t`
- `A_t`
- `S_t`
- `K_t`
- `C_t`
- `u_t^+ = u_{t+1}^*`（若 `t < T`）

---

### Step 1：计算每个 prefix 的 `d_t^*`

对每个 prefix `t`：

1. 用 `T_q^{raw}` 与 `A_t` 计算 raw 三维 deficit：
   - `d_t^{br*}`
   - `d_t^{dis*}`
   - `d_t^{sup*}`

2. 用整条 trajectory 的 derived 剩余工作计算：
   - `d_t^{der*}`

从而得到完整：

\[
d_t^*=
(
d_t^{br*},
d_t^{dis*},
d_t^{sup*},
d_t^{der*}
)
\]

---

### Step 2：对每个非终止 prefix 计算正样本 `c_t^*(u_t^+)`

对每个 `t = 0,1,\dots,T-1`：

- 正样本为：
\[
u_t^+ = u_{t+1}^*
\]

- 当前 deficit 为：
\[
d_t^*
\]

- 下一前缀 deficit 为：
\[
d_{t+1}^*
\]

于是：

\[
c_t^*(u_t^+)
=
\frac{
[d_t^* - d_{t+1}^*]_+
}{
d_t^* + \epsilon
}
\]

---

### Step 3：终止 prefix 不产生下一步 contribution

在真正 terminal 的 prefix 上：

- 已经没有下一步 `u_t^+`
- 因此不再定义新的 `c_t^*(u_t^+)`

也就是说：

- `c_t^*` 只对非终止 prefix 存在

---

## 9. 伪代码

```python
def compute_c_t_star(d_t_star, d_t1_star, eps=1e-6):
    c = {}
    for k in d_t_star.keys():
        num = max(0.0, d_t_star[k] - d_t1_star[k])
        den = d_t_star[k] + eps
        c[k] = num / den
    return c
```

若采用四维字典形式：

```python
d_t_star = {
    "br": ...,
    "dis": ...,
    "sup": ...,
    "der": ...
}
```

则输出：

```python
c_t_star = {
    "br": ...,
    "dis": ...,
    "sup": ...,
    "der": ...
}
```

---

## 10. 与 `RankingLabel` 的关系

必须明确区分：

### `RankingLabel`
回答的是：

> **在当前 prefix 下，teacher 在当前候选池里选了谁。**

它是排序监督。

### `c_t^*(u_t^+)`
回答的是：

> **teacher 选中的这个正样本，具体补掉了多少 typed residual deficit。**

它是结构化增益监督。

因此：

- `RankingLabel`：训练 student candidate scorer
- `c_t^*(u_t^+)`：训练 student contribution predictor

二者互补，但不等价。

---

## 11. 与 `d_t^*` 的关系

`d_t^*` 和 `c_t^*` 是一组非常紧的监督对象：

- `d_t^*`：当前还剩多少缺口
- `c_t^*(u_t^+)`：这一步补了多少缺口

你可以把它们理解为：

> `d_t^*` 是 **state label**  
> `c_t^*` 是 **action-on-state label**

这也是为什么当前设计中：

- teacher 规划层使用 `U_t(u)`
- supervision 层使用 `d_t^* / c_t^*`

而不把二者混在一起。

---

## 12. raw 与 derived 的 `c_t^*` 差异

### raw 正样本
若 `u_t^+` 是 raw，通常会表现为：

- 在某个或多个 raw role 维上带来明显下降
- `c_t^{br*}` / `c_t^{dis*}` / `c_t^{sup*}` 中至少一项较大
- `c_t^{der*}` 可能较小或为 0

### derived 正样本
若 `u_t^+` 是 derived，通常会表现为：

- raw 三维几乎不变
- `c_t^{der*}` 较大

因此，`c_t^*` 正好把 raw / derived 的功能分工编码进了监督信号里。

---

## 13. failed trajectories 上如何使用 `c_t^*`

当前推荐：

### 13.1 `success`
完整保留所有非终止 prefix 的 `c_t^*`

### 13.2 `failed_but_progressive`
保留：
- 所有高质量进展段 prefix 的 `c_t^*`

不保留：
- 尾部明显停滞段的 `c_t^*`

### 13.3 `failed_stalled`
只保留：
- 最后一个有效进展步之前的 prefix 的 `c_t^*`

丢弃：
- failed 尾部震荡段的 `c_t^*`

原因很简单：

> `c_t^*` 只有在“这一步确实推动了状态前进”时，才是有意义的监督。  
> 对明显停滞和失败震荡段，`c_t^*` 往往只有噪声。

---

## 14. 最终推荐保存格式

对每个保留下来的非终止 prefix `t`，推荐保存：

```python
ContributionLabel = {
    "step_id": int,
    "positive_unit_id": str,
    "c_t_star": {
        "br": float,
        "dis": float,
        "sup": float,
        "der": float
    }
}
```

如果你希望更完整，也可以额外保存：

```python
ContributionRecord = {
    "step_id": int,
    "positive_unit_id": str,
    "d_t_star": {...},
    "d_t1_star": {...},
    "c_t_star": {...}
}
```

这样后续更方便调试与复核。

---

## 15. student 如何使用 `c_t^*`

当前推荐 student 侧引入一个 contribution predictor：

\[
\hat c_t(u)
\]

训练目标是：

\[
\hat c_t(u_t^+) \approx c_t^*(u_t^+)
\]

这可以与 ranking scorer 分工如下：

### ranking scorer
学习：
- 在当前前缀下谁更可能是 teacher 会选的下一步

### contribution predictor
学习：
- 若选中某个候选，它会补掉哪些 deficit 维度

然后在 student 在线推理时，可以结合：

- `\hat d_t`
- `\hat c_t(u)`
- ranking score

进行 closure-aware sequential control。

---

## 16. 当前推荐的实现取舍

如果当前实现要保持最小强版本，建议明确采用下面的取舍：

### 采用
- 只对正样本 `u_t^+` 保存 `c_t^*`
- 用相邻 prefix 的 `d_t^*` 差值来计算
- 用逐维归一化贡献作为标签

### 不采用
- 对所有负样本保存 full `c_t^*(u)`
- 用大 verifier 给所有候选打连续增益分
- 在 trajectory 生成阶段就把 `c_t^*` 当真值使用

---

## 17. 一句话总结

当前推荐的 `c_t^*` 监督信号构建方式是：

> **在完整 teacher trajectory 构建完成之后，先为每个 prefix 计算 typed residual deficit `d_t^*`，再对每个非终止 prefix 用相邻状态的 deficit 差值构造正样本的逐维归一化 contribution 标签 `c_t^*(u_t^+) = [d_t^* - d_{t+1}^*]_+ / (d_t^* + \epsilon)`；其中 raw 正样本主要补 raw 三维 deficit，derived 正样本主要补 derived 第四维 deficit。最终，`c_t^*` 用于训练 student 的 contribution predictor，而负样本仍主要通过 `RankingLabel` 学习。**
