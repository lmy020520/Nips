# KBS Online Policy State

本文档说明当前 RAG 推理阶段如何显式维护在线知识状态，使 offline teacher schema 和 online policy-RAG 执行过程对齐。

## 1. 为什么要补 online state

之前的 policy-RAG 已经能够逐步选择 evidence，但在线推理时主要维护的是：

```text
selected_units
selected_evidence
selected_doc_ids
```

这足够做指标统计，但还不是 KBS 文档中定义的显式知识状态。

为了让流程真正闭合，需要在线推理也维护：

```text
H_t / A_t / S_t / K_t
```

这样每一步都可以被解释为：

```text
当前知识状态 K_t
  ↓
policy 选择下一批 evidence
  ↓
Update / Ledger / Render
  ↓
得到 K_{t+1}
```

## 2. 当前实现

文件：

```text
scripts/run_hotpotqa_policy_rag.py
```

新增三个核心函数：

| 函数 | 作用 |
| --- | --- |
| `init_online_state()` | 初始化空的 `H_t/A_t/S_t/K_t` |
| `update_online_state()` | 对选中的 evidence 做 Update + Ledger + Render |
| `render_online_k_t()` | 把当前 notebook refs 渲染成 compiled context |

当前实现是轻量确定性的，不引入新的 LLM 调用，也不做语义抽取。

## 3. Online State 结构

```json
{
  "H_t": [],
  "A_t": {
    "raw_unit_ids": [],
    "doc_ids": [],
    "unit_doc": {}
  },
  "S_t": {
    "raw_refs": [],
    "derived_refs": [],
    "last_added_unit_id": null,
    "last_updated_step": -1
  },
  "K_t": ""
}
```

## 4. 在线递推顺序

每一步 policy 选中 evidence 后，严格按以下顺序递推：

```text
1. Update:
   写入 H_t 和 S_t.raw_refs

2. Ledger:
   写入 A_t.raw_unit_ids / A_t.doc_ids / A_t.unit_doc

3. Render:
   根据 S_t 渲染新的 K_t
```

这对应 `update_render_ledger_design.md` 中的边界：

```text
先 Update，再 Ledger，再 Render
```

## 5. 如何启用

默认保持旧实验兼容，不改变已有结果。

如果要跑完整 online KBS 闭环，使用：

```bash
python3 scripts/run_hotpotqa_policy_rag.py \
  ... \
  --state-mode policy \
  --policy-context-source online_state \
  --save-online-states
```

其中：

| 参数 | 作用 |
| --- | --- |
| `--policy-context-source online_state` | 下一步 policy 使用 online `K_t` |
| `--save-online-states` | 在输出 JSON 中保存每步 `online_state_before/after` |
| `--online-state-max-raw` | `K_t` 最多渲染多少条 raw evidence |
| `--online-state-max-chars` | 每条 evidence 最大字符数 |

## 6. 当前完成度

这一步完成后，流程中的以下部分已经从“概念/离线字段”推进到“在线可执行”：

```text
构建知识状态 H_t / A_t / S_t / K_t
  ↓
接入 RAG，逐步选择证据
  ↓
每步更新 K_t，用于下一步 evidence selection
```

后续还需要继续补强：

1. `A_t` 中 typed role coverage 的在线更新。
2. `d_t` 的在线预测与更新。
3. 基于 deficit / contribution 的 stop control。
4. derived evidence 的在线写入与渲染。

