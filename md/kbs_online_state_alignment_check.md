# KBS Online State Alignment Check

本文档说明如何验收 policy-RAG 输出中的在线知识状态是否闭环。

## 1. 目的

前一步已经让 online RAG 在推理时维护：

```text
H_t / A_t / S_t / K_t
```

但还需要确认这些状态不是“写在输出里而已”，而是真的按步连续更新。因此新增校验脚本：

```text
scripts/validate_kbs_online_state_alignment.py
```

## 2. 校验内容

脚本检查每个 qid 的轨迹：

| 检查项 | 含义 |
| --- | --- |
| `online_state_before/after` 存在 | 每一步都有前后状态 |
| 第一步 `before.H_t` 为空 | 初始状态正确 |
| 当前步 `before` 等于上一步 `after` | 状态连续，没有断裂 |
| `H_t` 等于累计 selected units | Update 逻辑正确 |
| `A_t.raw_unit_ids` 等于 `H_t` | Ledger 与 trajectory 对齐 |
| `S_t.raw_refs` 等于 `H_t` | Notebook refs 与 trajectory 对齐 |
| 选中 evidence 后 `K_t` 非空 | Render 生效 |
| `final_online_state.H_t` 等于最终累计 selected units | 最终状态一致 |

## 3. 使用命令

```bash
python3 scripts/validate_kbs_online_state_alignment.py \
  --report outputs/rag/kbs_stop_control_smoke5.json \
  --output outputs/diagnostics/kbs_online_state_alignment_smoke5.json
```

期望输出：

```json
{
  "status": "PASS",
  "issue_count": 0
}
```

## 4. 它在流程中的位置

这一步对应：

```text
接入 RAG，逐步选择证据
  ↓
每步 Update / Ledger / Render
  ↓
检查 online state 是否和 selected evidence 对齐
```

如果该校验通过，就说明当前系统已经具备可追踪的 online KBS execution trace。后续再做实验时，失败分析可以直接读取每一步的 state、candidate、policy score、deficit estimate 和 stop decision。

