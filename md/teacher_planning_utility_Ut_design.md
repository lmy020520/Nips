# Teacher Planning Utility `U_t(u)` 设计稿

## 1. 目的与边界

本文档单独整理当前版本中 **teacher planning utility `U_t(u)`** 的设计，目标是明确：

1. teacher 在每个 prefix 上如何对当前候选 `u ∈ C_t` 计算即时 utility  
2. `a_t^{ctx}` 应如何定义与计算  
3. `\Delta a_t^{ctx}(u)` 应如何通过一步模拟得到  
4. `\kappa_q(u \mid H_t)` 应如何以最小复杂度实现  
5. 最终 `u_t^+ = u_{t+1}^*` 是如何从 `C_t` 中被选出来的  

本文档首先明确一个关键边界：

> **`U_t(u)` 只用于 teacher trajectory 构建阶段的规划与选择，不直接作为 student 的训练标签。**

也就是说：

- `U_t(u)` 是 **planning-layer quantity**
- `d_t^* / c_t^*` 是 **supervision-layer quantity**

二者应当分开。

---

## 2. `U_t(u)` 的角色

在第 `t` 步，teacher 并不是从全局所有对象中选下一步，而只在当前前缀可达候选池中选：

\[
u_{t+1}^* = \arg\max_{u \in C_t} U_t(u)
\]

其中：

\[
C_t = R_t \cup G_t^{final}
\]

也就是：

- `R_t`：当前 raw retrieval 候选
- `G_t^{final}`：当前经过 legality / retain 后保留下来的少量 derived candidates

因此，`U_t(u)` 的作用是：

> **衡量：如果现在把候选 `u` 加入当前前缀，它会以多大收益、多少成本，推动 teacher 更接近 answer-identifiable state。**

---

## 3. 当前推荐的 `U_t(u)` 主公式

当前规划层中，推荐使用如下即时 utility：

\[
U_t(u)=
\frac{
\sum_{r\in\{br,dis,sup\}}\eta_r\,\Delta \tilde s_t^r(u)
+
\eta_{ctx}\,\Delta a_t^{ctx}(u)
}{
\kappa_q(u\mid H_t)
}
\]

其中：

- `\Delta \tilde s_t^r(u)`：候选 `u` 带来的 raw role coverage 增量
- `\Delta a_t^{ctx}(u)`：候选 `u` 带来的 context answerability 增量
- `\eta_r`：各 raw role 的收益权重
- `\eta_{ctx}`：context answerability 的收益权重
- `\kappa_q(u\mid H_t)`：候选 `u` 的 planning cost

这一定义的直觉是：

- raw 候选主要通过补 raw evidence 获得分数
- derived 候选主要通过提升 context organization 获得分数
- 最终按成本做轻量归一化

---

## 4. `U_t(u)` 的一步模拟计算流程

对当前 prefix `t`，给定候选 `u \in C_t`，teacher 不直接凭文本猜测其价值，而是做一次 **一步模拟**：

### Step 1：模拟加入当前前缀
\[
H_{t+1}^{(u)} = H_t \oplus u
\]

### Step 2：更新 notebook state
\[
S_{t+1}^{(u)} = Update(S_t, u, t+1)
\]

### Step 3：更新 raw ledger
\[
A_{t+1}^{(u)} = Ledger(A_t, u, T_q^{raw})
\]

### Step 4：渲染 compiled context
\[
K_{t+1}^{(u)} = Render(q, S_{t+1}^{(u)}, UnitRegistry)
\]

### Step 5：计算 raw role coverage 增量
\[
\Delta \tilde s_t^r(u)
\]

### Step 6：计算 context answerability 增量
\[
\Delta a_t^{ctx}(u)
\]

### Step 7：除以成本
\[
\kappa_q(u\mid H_t)
\]

### Step 8：得到最终 utility
\[
U_t(u)
\]

这就是 teacher 在当前 prefix 上评估候选的最小 lookahead 机制。

---

## 5. `\Delta \tilde s_t^r(u)` 的计算

### 5.1 role 集合
当前 raw 监督采用三维 role：

- `bridge`
- `distinguish`
- `support`

分别记为：
- `br`
- `dis`
- `sup`

---

### 5.2 当前 prefix 下的 role coverage

对每个 role `r`，定义：

\[
k_t^r
=
\sum_{y\in \mathcal T_q^{raw}} \omega_r(y) w(y) \mathbf 1[y\in A_t]
\]

\[
N_q^r
=
\sum_{y\in \mathcal T_q^{raw}} \omega_r(y) w(y)
\]

当前最小版中：

\[
\omega_r(y)=
\begin{cases}
1,& r=\text{primary\_role}(y)\\
0,& \text{otherwise}
\end{cases}
\qquad
w(y)=1
\]

因此：
- `k_t^r`：当前已覆盖的 role-`r` raw targets 数量
- `N_q^r`：问题 `q` 在 role `r` 上的总 raw target 数量

---

### 5.3 平滑后的 role coverage

对每个 role 做平滑：

\[
\tilde s_t^r
=
\frac{k_t^r+\alpha}{N_q^r+2\alpha}
\]

其中：
- `\alpha > 0` 为平滑常数

推荐默认：
- `\alpha = 1`

若 `N_q^r = 0`，则该 role 对当前问题不适用，其增量直接视为 0。

---

### 5.4 候选加入后的 role coverage

用模拟后的 `A_{t+1}^{(u)}` 重新计算：

\[
k_{t+1}^{r,(u)}
\]

\[
\tilde s_{t+1}^{r,(u)}
=
\frac{k_{t+1}^{r,(u)}+\alpha}{N_q^r+2\alpha}
\]

于是：

\[
\Delta \tilde s_t^r(u)
=
\tilde s_{t+1}^{r,(u)} - \tilde s_t^r
\]

这就是 raw role coverage 增量。

---

### 5.5 raw 和 derived 在这部分的差异

#### raw 候选
若 `u` 是 raw，通常会直接更新 `A_t`，因此可能带来非零的：
- `\Delta \tilde s_t^{br}(u)`
- `\Delta \tilde s_t^{dis}(u)`
- `\Delta \tilde s_t^{sup}(u)`

#### derived 候选
当前设计中，derived 不直接更新 raw ledger，因此通常：

\[
\Delta \tilde s_t^r(u_{derived}) \approx 0
\]

这意味着：
- raw 的主要得分来自 raw coverage gain
- derived 的主要得分来自 context answerability gain

---

## 6. `a_t^{ctx}` 的设计

这是 `U_t(u)` 中最关键的部分之一。

当前推荐的原则是：

> **`a_t^{ctx}` 不应该由一个 verifier 直接打出不稳定的连续分，而应由若干可解释的离散判断聚合而来。**

---

### 6.1 三个 binary judgments

定义三个二值判断：

\[
A_t^{ans} \in \{0,1\}
\]

\[
A_t^{sup} \in \{0,1\}
\]

\[
A_t^{dis} \in \{0,1\}
\]

其含义如下：

#### 1）`A_t^{ans}`
当前 compiled context `K_t` 是否已经足以让 answerer 输出 gold answer。

也就是：
- 对 `(q, K_t)` 做一次 answer-probe
- 得到预测答案 `\hat a_t`
- 若 `\hat a_t` 与 gold answer 等价，则 `A_t^{ans}=1`
- 否则为 0

这里的“答案等价”采用 teacher stop 文档中的修正版协议：
1. minimal normalization
2. 本地 answer equivalence judge 三分类
3. 只有 `equivalent` 才算正确

---

#### 2）`A_t^{sup}`
当前 context 是否真的支持该答案。

当前最小版建议规则化实现：
- `K_t` 中已有与答案直接相关的 evidence
- 若答案依赖 bridge，则 bridge support / bridge note 已出现

因此：

\[
A_t^{sup} = \mathbf 1[AnsEvidencePresent_t \land BridgeReady_t]
\]

若问题不依赖 bridge，则 `BridgeReady_t = 1`。

---

#### 3）`A_t^{dis}`
当前 context 是否已经足以排除主要歧义或竞争答案。

当前最小版建议与 raw ledger 对齐：

\[
A_t^{dis}=
\begin{cases}
1,& m_q^{dis}=0\\
\mathbf 1[\tilde s_t^{dis} \ge \tau_{dis}^{ctx}],& m_q^{dis}=1
\end{cases}
\]

推荐：
- `\tau_{dis}^{ctx}=0.7`

---

### 6.2 聚合成 `a_t^{ctx}`

当前推荐直接取平均：

\[
a_t^{ctx}
=
\frac{A_t^{ans}+A_t^{sup}+A_t^{dis}}{3}
\]

因此：

\[
a_t^{ctx} \in \{0, \tfrac13, \tfrac23, 1\}
\]

这有两个优点：

1. 结果稳定、可解释  
2. 避免让 verifier 给出主观的连续分数

---

## 7. `\Delta a_t^{ctx}(u)` 的计算

对每个候选 `u`，用模拟后的状态来计算：

### Step 1：生成模拟状态
- `S_{t+1}^{(u)}`
- `A_{t+1}^{(u)}`
- `K_{t+1}^{(u)}`

### Step 2：在 `K_t` 上计算当前 `a_t^{ctx}`
\[
a_t^{ctx}
=
\frac{A_t^{ans}+A_t^{sup}+A_t^{dis}}{3}
\]

### Step 3：在 `K_{t+1}^{(u)}` 上重新计算
\[
a_{t+1}^{ctx}(u)
=
\frac{
A_{t+1}^{ans,(u)}+
A_{t+1}^{sup,(u)}+
A_{t+1}^{dis,(u)}
}{3}
\]

### Step 4：取差值
\[
\Delta a_t^{ctx}(u)
=
a_{t+1}^{ctx}(u)-a_t^{ctx}
\]

这就是 context answerability 增量。

---

## 8. `\kappa_q(u\mid H_t)` 的设计

当前推荐：

> **`\kappa_q(u\mid H_t)` 只做轻量成本归一化，不应演化成新的重型成本模型。**

因此，当前最简可执行版建议直接按 candidate type 赋固定基准成本：

\[
\kappa_q(u\mid H_t)=\kappa(u)
\]

推荐默认值：

\[
\kappa(u)=
\begin{cases}
1.00,& u \text{ is raw atom / sentence}\\
1.10,& u \text{ is raw chunk}\\
1.15,& u \text{ is bridge\_note}\\
1.20,& u \text{ is verification\_note}
\end{cases}
\]

直觉：
- atom / sentence 最便宜
- chunk 稍贵
- derived note 更贵一些
- verification note 略高于 bridge note

### 可选进一步简化
若希望主实验进一步减少工程复杂度，也可令：

\[
\kappa_q(u\mid H_t) \equiv 1
\]

然后把 cost-aware 版本放到补充实验。

---

## 9. `\eta_r` 与 `\eta_{ctx}` 的设定

当前最小实现不建议学习这些权重。  
推荐第一版直接设为常数。

### 方案 A：完全等权
\[
\eta_{br}=\eta_{dis}=\eta_{sup}=1
\]
\[
\eta_{ctx}=1
\]

### 方案 B：轻微强调 context closure（推荐）
\[
\eta_{br}=\eta_{dis}=\eta_{sup}=1
\]
\[
\eta_{ctx}=1.5
\]

推荐从方案 B 起步，因为它更符合当前课题中：

> semantic sufficiency 不等于 context closure

的主线。

---

## 10. raw 与 derived 候选的 utility 直觉

### 10.1 raw 候选
对 raw `u`，通常可能带来：

- 非零 `\Delta \tilde s_t^r(u)`
- 有时也可能带来 `\Delta a_t^{ctx}(u)`

因此：

\[
U_t(u_{raw})
\approx
\frac{
\text{raw coverage gain}
+
\text{some context gain}
}{
\kappa(u)
}
\]

---

### 10.2 derived 候选
对 derived `z`，由于它通常不直接更新 `A_t`，因此：

\[
\Delta \tilde s_t^r(z) \approx 0
\]

它的 utility 主要来自：

\[
U_t(z_{derived})
\approx
\frac{
\eta_{ctx}\,\Delta a_t^{ctx}(z)
}{
\kappa(z)
}
\]

这正体现了当前系统分工：

- raw 负责补 semantic evidence
- derived 负责把 evidence 组织成更接近 answer-identifiable context 的状态

---

## 11. 最终 `u_t^+` 如何得到

把前面都合起来，当前 prefix 的 teacher 正样本 `u_t^+ = u_{t+1}^*` 的计算过程如下：

### 输入
- 当前状态：
  - `H_t`
  - `A_t`
  - `S_t`
  - `K_t`
- 当前候选池：
  - `C_t = R_t ∪ G_t^{final}`

### 对每个候选 `u ∈ C_t`
1. 模拟得到：
   - `H_{t+1}^{(u)}`
   - `S_{t+1}^{(u)}`
   - `A_{t+1}^{(u)}`
   - `K_{t+1}^{(u)}`
2. 计算：
   - `\Delta \tilde s_t^{br}(u)`
   - `\Delta \tilde s_t^{dis}(u)`
   - `\Delta \tilde s_t^{sup}(u)`
3. 计算：
   - `\Delta a_t^{ctx}(u)`
4. 计算：
   - `\kappa_q(u\mid H_t)`
5. 代入主公式得到：
   - `U_t(u)`

### 最终选择
\[
u_t^+ = u_{t+1}^* = \arg\max_{u\in C_t} U_t(u)
\]

---

## 12. tie-break 规则（推荐）

为了保证 trajectory 构建可复现，建议在 `U_t(u)` 并列时固定 tie-break 顺序：

1. 优先选择 raw over derived  
2. 若同类型，则优先更低 `\kappa(u)`  
3. 若 derived 同分，则优先更高 `coarse_priority`  
4. 若仍相同，则按 `unit_id` 字典序  

这样可以显著减少数据构建的不确定性。

---

## 13. 直接可落地的伪代码

```python
def compute_U_t_for_candidate(
    q, H_t, A_t, S_t, K_t, u, T_q_raw, UnitRegistry,
    eta_br=1.0, eta_dis=1.0, eta_sup=1.0, eta_ctx=1.5,
    alpha=1.0
):
    # 1. simulate one-step update
    S_u = Update(S_t, u, step_id=len(H_t) + 1)
    A_u = Ledger(A_t, u, T_q_raw)
    K_u = Render(q, S_u, UnitRegistry)

    # 2. raw role coverage delta
    delta = {}
    for r in ["bridge", "distinguish", "support"]:
        k_t = get_k_r(A_t, r)
        N_r = get_N_q_r(T_q_raw, r)
        s_t = (k_t + alpha) / (N_r + 2 * alpha) if N_r > 0 else 0.0

        k_u = get_k_r(A_u, r)
        s_u = (k_u + alpha) / (N_r + 2 * alpha) if N_r > 0 else 0.0
        delta[r] = s_u - s_t

    # 3. context answerability delta
    a_t = compute_a_ctx(q, K_t, A_t, T_q_raw)
    a_u = compute_a_ctx(q, K_u, A_u, T_q_raw)
    delta_ctx = a_u - a_t

    # 4. cost
    kappa = compute_kappa(u)

    # 5. final utility
    num = (
        eta_br * delta["bridge"] +
        eta_dis * delta["distinguish"] +
        eta_sup * delta["support"] +
        eta_ctx * delta_ctx
    )
    return num / kappa
```

其中 `compute_a_ctx(...)` 推荐写成：

```python
def compute_a_ctx(q, K_t, A_t, T_q_raw):
    A_ans = answerability_probe(q, K_t)          # 0/1
    A_sup = support_sufficient(q, K_t)           # 0/1
    A_dis = distinguish_sufficient(A_t, T_q_raw) # 0/1
    return (A_ans + A_sup + A_dis) / 3.0
```

---

## 14. 与监督层的关系

再次强调：

- `U_t(u)`：用于 **teacher trajectory 生成**
- `d_t^* / c_t^*`：用于 **student 训练监督**

也就是说：

> **teacher 用 `U_t(u)` 做规划；trajectory 构建完成后，再离线回放得到单调的 `d_t^*` 与 `c_t^*(u_t^+)`。**

不要把 planning-layer utility 和 supervision-layer label 混成一个对象。

---

## 15. 一句话总结

当前推荐的 teacher planning utility 设计是：

> **对每个前缀可达候选 `u ∈ C_t`，teacher 通过一步模拟先得到加入该候选后的新状态，再计算其 raw 三维平滑覆盖增量 `\Delta \tilde s_t^r(u)` 与 context answerability 增量 `\Delta a_t^{ctx}(u)`，最后用一个轻量类型成本 `\kappa_q(u\mid H_t)` 做归一化；其中 `a_t^{ctx}` 不应是 verifier 给出的连续主观分，而应由 `answerability / support / distinguish` 三个离散判断聚合得到。最终，`u_t^+ = u_{t+1}^*` 就是使该即时 utility 最大的当前候选。**
