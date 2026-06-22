# KBS Deficit-Driven Stop Control

本文档说明当前 online policy-RAG 中新增的 deficit-driven stop/control 接口。

## 1. 目的

KBS 流程中，student 不应该只做固定步数 evidence selection。理想情况下，它应根据当前知识状态判断：

```text
当前证据是否已经足够？
继续选证据的边际收益是否还值得？
```

因此，stop/control 不应只依赖一个孤立的 stop classifier，而应主要依赖：

```text
predicted typed deficit
candidate contribution / gain
```

当前实现先补齐第一层接口：用 student 的 deficit head 估计当前 residual deficit，并支持在线提前停止。

## 2. 当前实现位置

文件：

```text
scripts/run_hotpotqa_policy_rag.py
```

新增能力：

| 模块 | 作用 |
| --- | --- |
| `PolicyModel.estimate_deficit()` | 复用模型 deficit head，估计当前 typed deficit |
| `--stop-control deficit` | 开启 deficit-driven stopping |
| `--stop-deficit-threshold` | deficit 低于该阈值时触发停止 |
| `--stop-deficit-mode` | 使用 `mean` 或 `max` deficit 判断 |
| `--stop-min-steps` | 至少选择若干步后才允许停止 |

## 3. Stop 判定

当前规则：

```text
if selected_steps >= stop_min_steps
and deficit_value <= stop_deficit_threshold:
    stop
else:
    continue
```

其中：

```text
deficit_value = mean(d_br, d_dis, d_sup, d_der)
```

或：

```text
deficit_value = max(d_br, d_dis, d_sup, d_der)
```

由 `--stop-deficit-mode` 控制。

## 4. 输出诊断

开启 stop/control 后，每一步会记录：

```json
{
  "deficit_estimate": {
    "d_br": 0.1,
    "d_dis": 0.0,
    "d_sup": 0.2,
    "d_der": 0.0,
    "mean": 0.075,
    "max": 0.2
  },
  "stop_decision": {
    "should_stop": false,
    "reason": "continue"
  }
}
```

如果提前停止，qid 级结果会记录：

```json
{
  "stopped_early": true,
  "stop_record": {
    "t": 1,
    "reason": "deficit_below_threshold"
  }
}
```

summary 中会记录：

```text
stopped_qids
stop_triggered
stop_control
stop_deficit_threshold
stop_deficit_mode
```

## 5. 使用方式

Smoke test：

```bash
python3 scripts/run_hotpotqa_policy_rag.py \
  ... \
  --state-mode policy \
  --policy-context-source online_state \
  --selector policy \
  --select-top-k 5 \
  --save-online-states \
  --stop-control deficit \
  --stop-min-steps 1 \
  --stop-deficit-mode mean \
  --stop-deficit-threshold 0.12
```

## 6. 当前边界

这一步只是把 stop/control 接口接通，还不是最终最优 stop 策略。

后续需要补：

1. 阈值校准曲线。
2. mean vs max deficit 对比。
3. deficit + candidate gain 联合 stop。
4. stop 后答案质量变化分析。
5. 与 teacher `StopLabel` 的一致性分析。

