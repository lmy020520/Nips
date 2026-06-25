# KBS Official Online RAG Route v1

本文档固定当前项目的 official online RAG 路线。它的目的不是做新的实验，而是把系统流程固定下来，避免后续对比实验和消融实验使用不同的隐式配置。

## 1. 官方流程

```text
Question
  -> Hybrid retrieval front-end
  -> Local expansion
  -> MMR candidate compression
  -> Student policy with front-policy score blending
  -> Select top-k evidence
  -> Update online H_t / A_t / S_t / K_t
  -> Render final context
  -> LLM answer
```

## 2. 固定配置

| Component | Official setting |
|---|---|
| `state_mode` | `policy` |
| `policy_context_source` | `online_state` |
| `selector` | `hybrid_policy` |
| `dense_model` | `models/bge-large-en-v1.5` |
| `dense_query_mode` | `state` |
| `front_pool_k` | `30` |
| `front_fusion` | `rrf` |
| `local_expansion_window` | `1` |
| `mmr_lambda` | `0.7` |
| `mmr_same_doc_similarity` | `0.35` |
| `candidate_top_k` | `10` |
| `select_top_k` | `5` |
| `policy_score_mode` | `front_policy_blend` |
| `policy_blend_weight` | `0.35` |
| `answer_mode` | `json` |
| `save_online_states` | enabled |
| `refresh_answer_cache` | enabled for official experiments |

## 3. Student Checkpoint

Default checkpoint:

```text
outputs/ranker/deberta_v3_large_v17_candidate_contribution_lr5e7/best_model.pt
```

该 checkpoint 作为当前 official student policy。后续如果替换 student，必须显式记录 checkpoint 名称和原因。

## 4. 官方入口脚本

统一入口：

```bash
bash scripts/run_kbs_official_online_rag.sh
```

Smoke20:

```bash
MAX_QIDS=20 \
OUTPUT=outputs/rag/kbs_official_online_state_v1_smoke20.json \
ANSWER_CACHE_DIR=outputs/rag/cache_kbs_official_online_state_v1_smoke20 \
bash scripts/run_kbs_official_online_rag.sh
```

Full evaluation:

```bash
OUTPUT=outputs/rag/kbs_official_online_state_v1_full.json \
ANSWER_CACHE_DIR=outputs/rag/cache_kbs_official_online_state_v1_full \
bash scripts/run_kbs_official_online_rag.sh
```

Main experiment wrapper:

```bash
bash scripts/run_kbs_main_experiment.sh
```

主实验默认会重新调用 DeepSeek：

```text
REFRESH_ANSWER_CACHE=1
REQUIRE_DEEPSEEK_API_KEY=1
```

如果环境变量里没有 `DEEPSEEK_API_KEY`，脚本会在终端提示输入。

## 5. 可变项

后续实验中，以下变量可以通过环境变量覆盖：

```text
DATA_ROOT
CHECKPOINT
DENSE_MODEL
CUDA_DEVICE
MAX_QIDS
FRONT_POOL_K
CANDIDATE_TOP_K
SELECT_TOP_K
POLICY_BLEND_WEIGHT
OUTPUT
ANSWER_CACHE_DIR
```

## 6. 与流程图的对应关系

| 流程图节点 | 当前实现 |
|---|---|
| Retrieval front-end 生成 raw candidate pool `R_t` | `hybrid_policy` 中 BM25 + Dense + RRF |
| 构建 / 更新 `H_t / A_t / S_t / K_t` | `--policy-context-source online_state` + `--save-online-states` |
| Student 学习 evidence selection policy | DeBERTa student checkpoint |
| Online RAG 逐步选择 evidence | `front_policy_blend` 排序后选 top-k |
| 更新知识状态并生成 final context | `update_online_state()` 和 rendered `K_t` |
| 最终 context 给 LLM 回答 | `--answer-mode json --generate-answers` |

## 7. 重要约束

Official route 不再使用 legacy notebook 作为主流程。legacy context 只能作为消融项：

```text
official: policy_context_source = online_state
ablation: policy_context_source = legacy
```

这样可以保证系统实现与论文中的知识状态流程一致。
