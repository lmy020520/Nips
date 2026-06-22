# KBS True Contribution Head

本文档记录当前 student contribution head 的严格化改动。

## 1. 为什么要补

此前 v17 中的 `candidate_role_aux_weight` 训练的是 candidate role / contribution proxy：

```text
bridge / support / distinguish
```

这有帮助，但它不是严格意义上的 `c_t*`。如果论文中写：

```text
Student learns ranking head + deficit head + contribution head
```

那 contribution head 必须直接学习 teacher 派生的 contribution value，而不能只用 role proxy 代替。

## 2. 当前严格实现

新增模型 head：

```text
contribution_regressor: hidden -> 4
```

输出维度：

```text
[c_br, c_dis, c_sup, c_der]
```

对应 teacher label：

```text
labels.c_t_star = {
  "c_br": ...,
  "c_dis": ...,
  "c_sup": ...,
  "c_der": ...
}
```

## 3. 训练方式

当前 teacher 样本只提供正样本 `u_t+` 的 `c_t*`，所以训练时只对正样本候选计算 contribution regression loss。

```text
positive_candidate = u_t+
pred = contribution_head(q, K_t, u_t+)
loss = MSE(pred, c_t*)
```

这点必须说清楚：我们没有把负样本强行标成 0，因为那不是严格 counterfactual contribution label。

## 4. 当前 loss

训练总目标变为：

```text
L =
  ranking CE
  + role_aux_weight * positive_role_CE
  + candidate_role_aux_weight * candidate_role_CE
  + deficit_aux_weight * deficit_MSE
  + contribution_aux_weight * positive_c_t*_MSE
```

## 5. 对应文件

| 功能 | 文件 |
| --- | --- |
| contribution head | `src/models/ranker.py` |
| 读取 `labels.c_t_star` | `src/datasets/prefix_dataset.py` |
| contribution loss / MAE | `src/train/train_ranker.py` |
| v18 训练配置 | `configs/train_ranker_deberta_v18_true_contribution.yaml` |
| online contribution-aware scoring | `scripts/run_hotpotqa_policy_rag.py` |

## 6. 如何训练

```bash
python3 src/train/train_ranker.py \
  --config configs/train_ranker_deberta_v18_true_contribution.yaml
```

训练日志中应出现：

```text
val_contribution_mae=...
```

这说明 student 正在直接学习 `labels.c_t_star`，而不是只学习 role proxy。

## 7. 如何在 RAG 中启用

训练完成后，可以用 v18 checkpoint 进行 contribution-aware evidence selection：

```bash
python3 scripts/run_hotpotqa_policy_rag.py \
  ... \
  --checkpoint outputs/ranker/deberta_v3_large_v18_true_contribution/best_model.pt \
  --policy-score-mode deficit_contribution \
  --deficit-contribution-weight 0.5
```

这个模式使用：

```text
score(q, K_t, u) + weight * <predicted_c_t(u), predicted_d_t>
```

也就是候选如果被预测为能补当前 deficit，就会得到额外加分。

## 8. 当前边界

当前版本是严格的 **positive c_t* contribution regression**。

还没有实现的是：

```text
candidate-level counterfactual c_t*(u), for every candidate u in C_t
```

那需要 teacher 对每个候选执行 counterfactual deficit update，成本更高，应作为后续增强版本，而不是用假 0 标签糊弄。
