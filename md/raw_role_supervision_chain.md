# Raw 三维监督链设计稿：`N_q^r / m_q^r / k_t^r / \tilde s_t^{r*} / d_t^{r*}`

## 1. 目的与边界

本文档单独整理当前版本中 **raw 三维监督链** 的定义与计算流程，目标是把下面这一组量统一说明清楚：

- `N_q^r`
- `m_q^r`
- `k_t^r`
- `\tilde s_t^{r*}`
- `d_t^{r*}`

本文档首先明确一个关键边界：

> **这套量属于 teacher 数据集构建后的监督链，用于构造 raw 三维 deficit 标签；它们不是 student 在线推理时直接可见的真值。**

也就是说：

- teacher 侧：这些量来自 `T_q^{raw}` 与 `A_t`
- student 侧：这些量是要被学习逼近的监督对象，而不是推理时直接读取的状态

---

## 2. 整体直觉

当前版本里，raw supervision 不再只做一个单维 “raw covered / uncovered”，而是显式拆成三种 raw role：

- `bridge`
- `distinguish`
- `support`

因此，对任意问题 `q`，我们希望知道两件事：

1. **这个问题在每个 raw role 上一共需要多少目标证据？**
2. **到 prefix `t` 为止，这些目标证据已经覆盖了多少？**

于是就得到一条非常自然的监督链：

\[
T_q^{raw}
\;\Rightarrow\;
N_q^r
\;\Rightarrow\;
m_q^r
\;\Rightarrow\;
A_t
\;\Rightarrow\;
k_t^r
\;\Rightarrow\;
\tilde s_t^{r*}
\;\Rightarrow\;
d_t^{r*}
\]

也就是说：

- `T_q^{raw}` 决定这个问题在 raw 三维上“需要什么”
- `A_t` 决定当前 prefix 在 raw 三维上“已经拿到了什么”
- 二者结合，产生当前 prefix 的 raw 三维 deficit 标签

---

## 3. 基础前提

## 3.1 raw target supervision set

对每个问题 `q`，teacher 数据构建阶段应有一个全局 raw 目标监督集：

\[
\mathcal T_q^{raw}
\]

其中每个 raw target `y` 至少应具备：

- `unit_id`
- `text`
- `primary_role`
- 可选 `weight`

当前最小版中，`primary_role` 属于：

- `bridge`
- `distinguish`
- `support`

并且当前假设每个 raw target 只有一个 `primary_role`。

---

## 3.2 role 集合

记 role 集合为：

\[
\mathcal R = \{br, dis, sup\}
\]

并做如下记号对应：

- `br` ↔ `bridge`
- `dis` ↔ `distinguish`
- `sup` ↔ `support`

---

## 4. `N_q^r`：问题级 raw role target 总量

## 4.1 含义

`N_q^r` 表示：

> **对当前问题 `q` 来说，raw role `r` 上总共需要多少目标 raw units。**

它是一个**问题级、全局静态量**，不依赖 prefix `t`。

---

## 4.2 最小版定义

当前最小版中，每个 raw target 只带一个 `primary_role`，并使用单位权重：

\[
\omega_r(y)=
\begin{cases}
1,& r=\text{primary\_role}(y)\\
0,& \text{otherwise}
\end{cases}
\]

\[
w(y)=1
\]

于是：

\[
N_q^r=
\sum_{y\in \mathcal T_q^{raw}}
\omega_r(y)w(y)
\]

在当前最小版下，等价于：

\[
N_q^r = |\mathcal T_q^{raw,r}|
\]

也就是：
- `N_q^{br}`：这个问题有多少个 bridge raw targets
- `N_q^{dis}`：有多少个 distinguish raw targets
- `N_q^{sup}`：有多少个 support raw targets

---

## 4.3 伪代码

```python
def compute_N_q_r(T_q_raw):
    N = {"bridge": 0, "distinguish": 0, "support": 0}
    for y in T_q_raw:
        role = y["primary_role"]
        if role == "disambiguation":
            role = "distinguish"
        if role in N:
            N[role] += y.get("weight", 1.0)
    return N
```

---

## 5. `m_q^r`：问题级 role mask

## 5.1 含义

`m_q^r` 表示：

> **当前问题 `q` 是否真的需要 raw role `r` 这一维监督。**

这是一个**问题级 mask**，不是学习变量。

---

## 5.2 定义

直接由 `N_q^r` 导出：

\[
m_q^r = \mathbf 1[N_q^r > 0]
\]

也就是说：
- 若某个 role 在当前问题的 `T_q^{raw}` 中根本没有 target，则该维关闭
- 若该 role 至少有一个 raw target，则该维开启

---

## 5.3 作用

`m_q^r` 的作用主要有三个：

1. 决定该 role 是否参与 raw deficit 标签  
2. 决定 stop gate 中该 role 是否需要检查  
3. 决定 `s_t^{sem}` 聚合时该 role 是否参与平均

因此，`m_q^r` 本质上是：

> **监督配置掩码，而不是策略变量。**

---

## 5.4 伪代码

```python
def compute_m_q_r(N_q_r):
    return {r: int(v > 0) for r, v in N_q_r.items()}
```

---

## 6. `k_t^r`：prefix 级 raw role coverage

## 6.1 含义

`k_t^r` 表示：

> **截至 prefix `t`，role `r` 上已经覆盖了多少 raw targets。**

它是一个**prefix 级、动态量**，来自当前 raw ledger `A_t`。

---

## 6.2 定义

记 `A_t` 中已经覆盖的 raw target 集为：

\[
\mathcal A_t \subseteq \mathcal T_q^{raw}
\]

则：

\[
k_t^r=
\sum_{y\in \mathcal T_q^{raw}}
\omega_r(y)w(y)\mathbf 1[y\in \mathcal A_t]
\]

当前最小版下，等价于：

> 当前 prefix 已覆盖的 role-`r` raw targets 数量

---

## 6.3 与 `Ledger` 的关系

`k_t^r` 不是直接单独存出来的真值，而是：

- 由 `Ledger(A_t, u_{t+1}, T_q^{raw})` 增量更新
- 或从 `A_t.covered_target_ids` 与 `T_q^{raw}` 回读统计

在当前推荐实现中，`A_t` 本身可以直接保存：

- `k_bridge`
- `k_distinguish`
- `k_support`

因此：

- `k_t^{br}` ↔ `A_t["k_bridge"]`
- `k_t^{dis}` ↔ `A_t["k_distinguish"]`
- `k_t^{sup}` ↔ `A_t["k_support"]`

---

## 6.4 伪代码

```python
def get_k_t_r(A_t, role):
    if role == "bridge":
        return A_t["k_bridge"]
    elif role == "distinguish":
        return A_t["k_distinguish"]
    elif role == "support":
        return A_t["k_support"]
    else:
        raise ValueError(role)
```

---

## 7. `\tilde s_t^{r*}`：平滑后的 raw role coverage ratio

## 7.1 含义

`k_t^r` 是 count，但不同问题的 `N_q^r` 规模不同，所以要归一化成 coverage ratio。  
为了避免小样本 role 下比例跳变太大，当前采用平滑版本：

\[
\tilde s_t^{r*}
\]

它表示：

> **当前 prefix 在 raw role `r` 上的平滑覆盖率。**

---

## 7.2 定义

\[
\tilde s_t^{r*}=
\frac{k_t^r + \alpha}{N_q^r + 2\alpha}
\]

其中：
- `\alpha > 0` 为平滑常数

推荐默认：
- `\alpha = 1`

---

## 7.3 为什么要平滑

若不平滑，当某个问题在某个 role 上只有 1 个 target 时：

- 一旦命中，coverage 会直接从 0 跳到 1
- 这会让 deficit 波动过大，不利于 teacher 规划和 student 监督

加上平滑后：

- 初始值不为 0
- 终值也不直接到 1
- 轨迹上的变化更平稳

---

## 7.4 特殊情况：`N_q^r = 0`

若该 role 对当前问题不适用，即：

\[
N_q^r = 0
\]

则这个 role 维应视为关闭。  
这时：
- 不单独解释 `\tilde s_t^{r*}` 的数值意义
- 最终由 `m_q^r = 0` 把这维 mask 掉

---

## 7.5 伪代码

```python
def compute_s_t_r_star(k_t_r, N_q_r, alpha=1.0):
    return (k_t_r + alpha) / (N_q_r + 2 * alpha)
```

---

## 8. `d_t^{r*}`：raw role-specific deficit label

## 8.1 含义

`d_t^{r*}` 表示：

> **截至 prefix `t`，raw role `r` 这一维还剩多少缺口没有补齐。**

它是当前 student 训练里最重要的 raw 三维监督对象之一。

---

## 8.2 定义

利用 `m_q^r` 只在适用维度上保留 deficit：

\[
d_t^{r*}=
m_q^r(1-\tilde s_t^{r*})
\]

也就是：

- 若该 role 对当前问题不适用，则 `m_q^r = 0`，该维 deficit 直接为 0
- 若适用，则 deficit 等于 1 减去平滑覆盖率

---

## 8.3 展开写法

### bridge
\[
d_t^{br*}=
m_q^{br}(1-\tilde s_t^{br*})
\]

### distinguish
\[
d_t^{dis*}=
m_q^{dis}(1-\tilde s_t^{dis*})
\]

### support
\[
d_t^{sup*}=
m_q^{sup}(1-\tilde s_t^{sup*})
\]

---

## 8.4 关键性质：单调

在当前 teacher 构建中：

- `A_t` 只增不减
- 所以 `k_t^r` 单调增加
- 所以 `\tilde s_t^{r*}` 单调增加
- 所以 `d_t^{r*}` 单调减小

这正是它适合做 student 监督标签的重要原因。

---

## 8.5 伪代码

```python
def compute_d_t_r_star(m_q_r, s_t_r_star):
    return m_q_r * (1.0 - s_t_r_star)
```

---

## 9. 整条 raw 三维监督链的计算顺序

对一个固定问题 `q` 和某个 prefix `t`，推荐按如下顺序计算：

### Step 1：从 `T_q^{raw}` 得到 `N_q^r`
\[
N_q^r = |\mathcal T_q^{raw,r}|
\]

### Step 2：由 `N_q^r` 得到 `m_q^r`
\[
m_q^r = \mathbf 1[N_q^r > 0]
\]

### Step 3：从当前 ledger `A_t` 读出 `k_t^r`
\[
k_t^r = \text{covered count on role } r
\]

### Step 4：计算平滑 coverage
\[
\tilde s_t^{r*}=
\frac{k_t^r + \alpha}{N_q^r + 2\alpha}
\]

### Step 5：计算 raw role deficit
\[
d_t^{r*}=
m_q^r(1-\tilde s_t^{r*})
\]

---

## 10. 一个完整例子

假设某个问题 `q` 的 raw target 集中有：

- 2 个 `bridge`
- 0 个 `distinguish`
- 3 个 `support`

那么：

\[
N_q^{br}=2,\quad
N_q^{dis}=0,\quad
N_q^{sup}=3
\]

于是：

\[
m_q^{br}=1,\quad
m_q^{dis}=0,\quad
m_q^{sup}=1
\]

假设当前 prefix `t` 已覆盖：

- 1 个 `bridge`
- 0 个 `distinguish`
- 2 个 `support`

则：

\[
k_t^{br}=1,\quad
k_t^{dis}=0,\quad
k_t^{sup}=2
\]

取 `\alpha=1`：

\[
\tilde s_t^{br*}=
\frac{1+1}{2+2}=
\frac{2}{4}=0.5
\]

\[
\tilde s_t^{sup*}=\frac{2+1}{3+2}=
\frac{3}{5}=0.6
\]

`distinguish` 维虽然形式上也能代数计算，但因为：

\[
m_q^{dis}=0
\]

所以最终：

\[
d_t^{dis*}=0
\]

其余两维为：

\[
d_t^{br*}=
1(1-0.5)=0.5
\]

\[
d_t^{sup*}=
1(1-0.6)=0.4
\]

因此当前 raw 三维 deficit 为：

\[
(d_t^{br*}, d_t^{dis*}, d_t^{sup*})=
(0.5, 0, 0.4)
\]

---

## 11. 它们分别用在哪里

### 11.1 `N_q^r`
用于：
- 问题级 raw target 总量统计
- role mask 的生成
- raw coverage 归一化

### 11.2 `m_q^r`
用于：
- 关闭不适用 role 维
- stop gate 中决定哪些 role 需要检查
- `s_t^{sem}` 聚合时决定哪些维参与

### 11.3 `k_t^r`
用于：
- 当前 prefix 的 raw role coverage 计数
- 构造 `\tilde s_t^{r*}`

### 11.4 `\tilde s_t^{r*}`
用于：
- teacher 规划中的 raw role coverage 增量
- stop gate / readiness 判断
- 构造 raw deficit

### 11.5 `d_t^{r*}`
用于：
- student 训练中的 raw 三维 deficit 监督
- trajectory 后处理中的 typed residual label

---

## 12. 与 teacher / student 的关系

### teacher 侧
teacher 构建 trajectory 时，主要使用的是：
- `k_t^r`
- `\tilde s_t^r`
- 它们的增量 `\Delta \tilde s_t^r(u)`

也就是说，teacher 规划看的是 **增益**。

### student 侧
student 训练时，主要学习的是：
- `d_t^{r*}`

也就是说，student 学习的是 **当前还剩多少缺口**。

这正体现了：

> **planning-layer gain 和 supervision-layer deficit 应分开。**

---

## 13. 推荐的直接实现接口

若你要在代码里实现当前版本，推荐统一提供如下函数：

```python
def compute_N_and_m(T_q_raw):
    ...
    return N_q_r, m_q_r

def get_k_from_ledger(A_t):
    ...
    return k_t_r

def compute_s_star(k_t_r, N_q_r, alpha=1.0):
    ...
    return s_t_r_star

def compute_d_star(m_q_r, s_t_r_star):
    ...
    return d_t_r_star
```

最终对每个 prefix 可输出：

```python
RawRoleChain = {
    "N_q_r": {...},
    "m_q_r": {...},
    "k_t_r": {...},
    "s_t_r_star": {...},
    "d_t_r_star": {...}
}
```

---

## 14. 一句话总结

当前推荐的 raw 三维监督链是：

> **先从问题的全局 raw 目标监督集 `T_q^{raw}` 统计每个 role 的目标总量 `N_q^r`，再由此得到问题级 role mask `m_q^r`；对任一 prefix `t`，从当前 raw ledger `A_t` 读出已覆盖计数 `k_t^r`，计算平滑覆盖率 `\tilde s_t^{r*}`，最后得到 role-specific deficit `d_t^{r*}=m_q^r(1-\tilde s_t^{r*})`。其中 teacher 规划主要使用 coverage 增量，而 student 训练主要学习 deficit 本身。**
