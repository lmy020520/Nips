# KBS Trajectory Diagnostics

本文档说明如何把 policy-RAG 的完整输出整理成流程级诊断文件。

## 1. 目的

当前流程已经能够输出：

```text
每步候选
policy 分数
selected evidence
online_state_before / online_state_after
deficit_estimate
stop_decision
最终答案
```

但原始 JSON 太长，不适合快速排查和汇报。因此新增：

```text
scripts/build_kbs_trajectory_diagnostics.py
```

它把一个 policy-RAG report 转成：

| 输出 | 作用 |
| --- | --- |
| `.summary.json` | 聚合统计 |
| `.cases.jsonl` | 每个 qid 一条结构化诊断 |
| `.md` | 可直接阅读/汇报的 Markdown 表格 |

## 2. 使用方式

```bash
python3 scripts/build_kbs_trajectory_diagnostics.py \
  --report outputs/rag/kbs_stop_control_smoke5.json \
  --output-prefix outputs/diagnostics/kbs_stop_control_smoke5_diagnostics
```

会生成：

```text
outputs/diagnostics/kbs_stop_control_smoke5_diagnostics.summary.json
outputs/diagnostics/kbs_stop_control_smoke5_diagnostics.cases.jsonl
outputs/diagnostics/kbs_stop_control_smoke5_diagnostics.md
```

## 3. 诊断字段

每个 case 会保留：

| 字段 | 含义 |
| --- | --- |
| `qid` | 问题 ID |
| `question` | 问题文本 |
| `all_steps_correct` | 每一步 top1 是否都选对 |
| `full_gold_unit_coverage` | 最终 selected units 是否覆盖全部 gold units |
| `selected_units` | 最终选中 evidence 数 |
| `gold_units` | gold evidence 数 |
| `stopped_early` | 是否触发 stop-control |
| `first_error_step` | 第一次未选中 gold 的步骤 |
| `last_deficit_mean` | 最后一个可见 step 的 mean deficit |
| `steps` | 每一步的 selected/gold/rank/top5/deficit/stop |

## 4. 它在流程中的位置

这一步对应 KBS 流程搭建的最后一个工程模块：

```text
流程级诊断输出
```

完成后，我们可以正式进入系统实验阶段，因为每次实验都能被统一整理、比较和定位失败原因。

