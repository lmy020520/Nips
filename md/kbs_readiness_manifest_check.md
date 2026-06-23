# KBS Readiness Check and Manifest

本文档说明 official KBS online RAG route 的 readiness check。该检查不跑实验，只确认系统闭环所需资产和配置是否齐全。

## 1. Manifest

机器可读 manifest：

```text
configs/kbs_official_online_rag_v1_manifest.json
```

它固定：

* data root / samples / memory / queries
* student checkpoint
* dense retriever
* official online RAG 参数
* official entrypoint
* required code files

## 2. 基础检查

```bash
python3 scripts/check_kbs_pipeline_readiness.py \
  --manifest configs/kbs_official_online_rag_v1_manifest.json \
  --output outputs/diagnostics/kbs_official_readiness.json
```

需要看到：

```text
"pipeline_ready": true
```

## 3. 检查 RAG Report 是否符合 Official Route

先用 official 脚本跑一个 smoke：

```bash
MAX_QIDS=20 \
OUTPUT=outputs/rag/kbs_official_online_state_v1_smoke20.json \
ANSWER_CACHE_DIR=outputs/rag/cache_kbs_official_online_state_v1_smoke20 \
bash scripts/run_kbs_official_online_rag.sh
```

然后检查 report：

```bash
python3 scripts/check_kbs_pipeline_readiness.py \
  --manifest configs/kbs_official_online_rag_v1_manifest.json \
  --rag-report outputs/rag/kbs_official_online_state_v1_smoke20.json \
  --output outputs/diagnostics/kbs_official_readiness_with_report.json
```

该检查会确认：

* `policy_context_source = online_state`
* `selector = hybrid_policy`
* `policy_score_mode = front_policy_blend`
* `save_online_states = true`
* 每一步包含 `online_state_before`
* 每一步包含 `online_state_after`
* 每个 qid 包含 `final_online_state`

## 4. 作用

readiness check 的作用是保证后续对比实验和消融实验都基于同一条 official route。它不是性能评测，也不替代 answer EM/F1、step acc 或 coverage 指标。
