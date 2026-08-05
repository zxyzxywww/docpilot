# MediDoc —— 医学文献智能问答 Agent

> 仅用于**公开医学文献检索与研究辅助**,不提供任何诊断、治疗或医疗决策建议。
> 本项目是学习与简历项目,所有医学结论均须有真实检索证据与引用。

## MVP 定位红线(贯穿全项目)

| 红线 | 说明 |
|---|---|
| 仅公开文献 | 只使用明确允许复用的 PMC Open Access 文献或明确开放指南,禁止处理真实患者隐私数据 |
| 不提供医疗建议 | 不输出诊断、治疗或用药决策 |
| 证据不足必须拒答 | 检索不到可靠证据时明确回答"证据不足",不猜测 |
| 引用溯源 | 重要医学结论必须附带真实检索到的证据与引用(标题/章节/页码/chunk_id/原文) |

以上红线同时写入 system prompt 与 `config.yaml` 的 `mvp` 段,并由测试守护。

## 项目定位

- **场景**:垂直医学文献 RAG + Agent(英文文献为主,支持中英文提问)
- **技术**:全部通用能力——RAG、双通道混合检索、ReAct Agent、评估体系、容器化部署
- **设计**:纯 API(无本地大模型),手写核心逻辑,不依赖 LangChain

## 技术栈与模型定案(2026-08)

| 能力 | 方案 | 备注 |
|---|---|---|
| 对话/翻译/评估 | DeepSeek API(`deepseek-v4-flash`,可切 `deepseek-v4-pro`) | OpenAI 兼容,`base_url: https://api.deepseek.com` |
| Embedding | SiliconFlow API **`BAAI/bge-m3`**,dimension=**1024** | 固定;更换模型/维度/分块策略必须重建索引 |
| Rerank | SiliconFlow API **`BAAI/bge-reranker-v2-m3`** | 固定 |
| 元数据存储 | SQLite(唯一事实来源) | 文档/chunk 元数据、导入状态 |
| 向量存储 | Qdrant(派生索引) | 开发期 local mode,部署期 Docker mode |
| 检索 | BM25 + 向量双通道 → RRF 融合 → rerank(派生索引) | 参数见 `config.yaml` |

> 所有供应商、模型、参数集中在 `config.yaml`,后续评估调整只改一处。

## 目录结构

```
medidoc/
├── README.md            # 本文件,每阶段结束更新
├── pyproject.toml       # 依赖与元数据 + ruff/mypy/pytest 配置
├── uv.lock              # 版本锁定(uv 管理)
├── .env.example         # DEEPSEEK_API_KEY / SILICONFLOW_API_KEY 模板
├── config.yaml          # 模型/检索/分块/数据库/预算/日志 全部集中
├── src/
│   ├── llm/             # DeepSeek 对话 + SiliconFlow embedding/rerank 客户端
│   ├── ingest/          # (阶段二)数据管道
│   ├── retriever/       # (阶段三)双通道检索
│   ├── agent/           # (阶段四)ReAct Agent
│   ├── evaluator/       # (阶段五)评估体系
│   └── app/             # (阶段六)Streamlit Web 界面
├── data/
│   ├── raw/             # 原始文献 + manifest.jsonl(合规清单,raw 不入库)
│   └── db/              # SQLite + Qdrant 持久化(不入 git,可重建)
├── tests/               # 默认离线 mock,不调用真实 API
└── scripts/             # chat / ingest / query / evaluate 等 CLI
```

## 快速开始

```bash
# 1. 安装 uv(https://docs.astral.sh/uv/)并同步依赖
uv sync --all-groups

# 2. 配置 API key
cp .env.example .env        # 填入 DEEPSEEK_API_KEY(可选 SILICONFLOW_API_KEY)

# 3. 运行测试(离线 mock,零真实 API 调用)
uv run pytest

# 4. 对话测试(需要真实 DEEPSEEK_API_KEY)
uv run python scripts/chat.py
```

## 测试与质量

- `uv run pytest`:离线 mock 测试,默认**不**调用任何真实付费 API
- `uv run pytest -m integration`:真实 API 集成测试,需 `.env` 配置 key 后手动运行
- `uv run ruff check .`:lint
- `uv run mypy src`:类型检查
- GitHub Actions CI(`.github/workflows/ci.yml`):仅运行离线 mock 测试,不调付费 API

## 成本预算(学生友好)

| 项目 | 估算 | 备注 |
|---|---|---|
| DeepSeek 对话 | 约 10–30 元 | 大头;前缀缓存自动命中可降一个数量级 |
| SiliconFlow embedding | 约 0.3–1 元 | 入库 10 篇论文约几十万 token |
| SiliconFlow rerank | 约 1 元以内 | 每查询只重排 top-30 |
| 存储(本地) | 0 元 | SQLite + Qdrant local mode |
| **开发期合计** | **约 20–50 元** | 一次充值可用全程 |

省钱机制:① DeepSeek 前缀缓存默认开启(固定 system + 文档块为共享前缀);② 避开高峰时段(峰谷定价,北京时间 9–12 / 14–18 为 2 倍价);③ 预算护栏 `max_cost_yuan` 在 config.yaml 中集中管理。

## 阶段进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 一 | 脚手架 + 双供应商接入 + MVP 红线 + 质量基线 | ✅ 进行中(待人工验收) |
| 二 | 数据管道:manifest 合规 + 解析 + SQLite/Qdrant/BM25 入库 + reindex.py | ⬜ 未开始 |
| 三 | 双通道检索 + direct_rag + 观测日志 + 注入防护 | ⬜ 未开始 |
| 四 | 手写 ReAct Agent + agentic_rag + 护栏 | ⬜ 未开始 |
| 五 | 评估体系 + direct/agentic 对比 + 调参 | ⬜ 未开始 |
| 六 | Streamlit UI + Docker 部署 + 收尾 | ⬜ 未开始 |

每阶段完成:更新本 README → 输出 Git diff 摘要 → **人工验收通过后**才进入下一阶段。

## 免责声明

MediDoc 仅用于公开医学文献的检索与研究辅助,输出不构成医疗建议。如有医疗问题请咨询专业医生。
