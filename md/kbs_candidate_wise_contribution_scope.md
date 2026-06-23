# Candidate-Wise Contribution Scope

本文档固定当前 contribution supervision 的边界。

## 1. 当前 official v1 的状态

当前 student 已经包含 contribution head，但 official v1 使用的是：

```text
positive-only c_t*
```

也就是只对 teacher 选中的正样本 `u_t+` 监督：

```text
c_t*(u_t+)
```

这可以训练 student 理解“正样本补了哪类 typed deficit”，但不能声称已经有完整的：

```text
c_t*(u), for every candidate u in C_t
```

## 2. 为什么不直接把负样本 contribution 设为 0

不能简单地把所有负样本设为 0，因为负样本中可能存在：

* near-miss evidence
* answer-bearing but unsupported evidence
* same-doc wrong sentence
* partially useful evidence

如果直接置 0，会把 counterfactual contribution 标签做错，反而污染 student。

## 3. 检查脚本

```bash
python3 scripts/check_kbs_contribution_scope.py \
  --data-root data/kbs_teacher_student_v1_hybrid_frontend \
  --splits train,val,test \
  --output outputs/diagnostics/kbs_teacher_student_v1_contribution_scope.json
```

如果输出为：

```text
official_v1_scope = positive_only_contribution
```

说明当前只能主张 positive-only contribution supervision。

如果未来输出为：

```text
official_v1_scope = candidate_wise_contribution_available
```

才可以主张 candidate-wise counterfactual contribution supervision。

## 4. 后续增强方向

要实现真正 candidate-wise `c_t*(u)`，需要 teacher 对每个候选做 counterfactual update：

```text
state S_t
candidate u
simulate Update(S_t, u)
estimate d_{t+1}(u)
c_t*(u) = d_t* - d_{t+1}(u)
```

这一步计算成本高，适合作为后续增强，而不是 official v1 的必要条件。

## 5. 当前写法约束

论文或汇报中可以说：

```text
We train a contribution head using teacher-derived positive contribution labels.
```

不要说：

```text
We supervise every candidate with counterfactual contribution labels.
```

除非数据中真的存在 candidate-wise contribution labels。
