# DocPilot 项目简历素材(Agent 应用工程师方向)

> 说明:项目历经两轮域迁移(医学文献 → Python 后端开发文档)。以下为当前
> DocPilot 叙事;早期医学版本见 git 历史,仅作"架构可插拔、数据域可迁移"的佐证。

## 一句话定位

**DocPilot —— Python 后端开发文档智能助手 Agent**:基于 96 页 FastAPI /
Pydantic / SQLAlchemy / Python 官方文档构建的检索增强问答系统,中文提问 →
检索英文文档 → 输出**带官方文档链接溯源**的回答(含完整可运行代码示例);
复杂需求由手写 ReAct Agent 多步检索后综合。

## 简历条目(STAR 结构)

> 独立开发 **DocPilot:面向开发者的官方文档问答 Agent**。自建 96 页官方开发
> 文档知识库(自研爬虫 + HTML 解析,正文/标题层级/**代码块独立成段**),实现
> 双通道混合检索(手写 BM25 + bge-m3 向量 + RRF 融合 + rerank 精排)与
> 中英跨语言查询;手写 ReAct Agent(不依赖 LangChain),含步数/重复动作/
> 费用三重护栏与 Pydantic 工具参数校验;回答强制带可溯源引用与完整代码
> 示例,证据不足自动拒答。构建 40 条人工开发 QA 评估集(20% 无答案、
> 按文档划分防泄漏)驱动迭代,test 基线:MRR 0.624、Recall@5 0.727、
> 引用完整率 0.955、0.0046 元/问。
> **产品化**:前后端分离 —— Next.js(React/TS/Tailwind/shadcn)三页
> (Chat 三栏 + 引用溯源卡片 + 6 步 RAG 链路可视化、知识库文档管理、评测
> 看板)+ FastAPI 后端(会话持久化/REST API),Docker 三服务
> (qdrant + api + web)一键部署(纯 API、CPU 可跑、模型零权重),CI 全绿。

## 技术栈(简历关键词)

Python / FastAPI / Qdrant / SQLite(手写 BM25 词频索引)/ DeepSeek / SiliconFlow(bge-m3、bge-reranker)/ Pydantic / Next.js / React / TypeScript / Tailwind / shadcn/ui / Docker Compose / GitHub Actions / pytest

## 量化成果(数字先说)

| 项 | 数值 | 说明 |
|---|---|---|
| 语料规模 | 96 页官方开发文档 / 12170 chunks | FastAPI 50 + Pydantic 12 + SQLAlchemy 14 + Python 20 |
| 检索效果 | Recall@5 0.727 / Recall@10 0.818 / MRR 0.624 | 首轮 test 基线(开发文档域) |
| 引用完整率 | 0.955 | 回答全部带可溯源引用 |
| 评估集 | 40 条人工核验(20% 无答案) | 按文档划分 dev/test 防泄漏 |
| 单次成本 | 0.0046 元/问 | direct 路径(全库评估 0.137 元/30 问) |
| 工程 | 100 个离线测试、ruff/mypy 零告警、CI(后端+前端构建) | 全离线 mock,不依赖付费 API |
| 代码检索 | HTML 代码块独立成段 | 中文提问返回完整可运行示例 |

## 面试深挖清单(每个都能展开讲)

1. **为什么换了两轮数据域?** → 知识库与 RAG/Agent 核心解耦,换域只改
   数据接入层与文案(fetch→parser→ingest 全链路可插拔),架构复用度 ~90%
2. **BM25 与 RRF 的原理?** → k1/b 参数、中文/代码 token 化、1/(k+rank) 融合
   为何不直接加权平均(不同分数域不可比)
3. **为什么向量+BM25 双通道?各自失效场景?** → 语义改写 vs 精确 API/参数名;
   代码块里 `OAuth2PasswordBearer` 这种精确词 BM25 强、同义/概念问题向量强
4. **代码块检索怎么做的?** → HTML 解析时 `<pre>/<code>` 独立成段(不混入
   正文)、保留章节上下文;对比普通 HTML-to-text 把代码黏进段落/丢弃
5. **ReAct 循环与护栏?** → Thought/Action/Observation 手写解析 + Pydantic
   校验重试;步数/连续重复动作/单问费用预算/工具超时(线程池真取消)
6. **direct vs agentic 怎么路由?** → 启发式;直接问答走一次生成,复杂
   需求多步工具调用(实测 agentic 成本约为 direct 的 4 倍,靠路由控制)
7. **评估集怎么保证可信?** → 从入库文档反向生成 QA(段落即答案)、20%
   抽样 judge 忠实度核验、按文档划分 dev/test、≥20% 无答案题、引用溯源
   可点击核验
8. **引用怎么防幻觉?** → 只允许引用检索上下文内编号,越界编号丢弃;
   证据不足拒答;rerank 低分预检拒答
9. **前后端分离怎么设计的?** → FastAPI 会话持久化(chat_sessions/messages)、
   6 步 RAG 报告结构化返回、Next.js 三页、Docker 三服务、上传安全
   (类型白名单/大小/路径清洗)
10. **为什么 SQLite 当事实来源而不是 Qdrant?** → 可审计、可迁移;
    Qdrant/BM25 是派生索引可确定性重建(embedding 变更 → reindex)

## 手写能力(差异点,面试官可深挖)

- 手写 BM25(k1/b/中英分词)与 RRF 融合(不依赖库)
- 手写 ReAct 循环(正则解析、Pydantic 工具校验、护栏、引用编号全局分配)
- 自研 HTML 正文/代码块解析器(标准库 html.parser 状态机)
- 自研评估集构建器(反向生成 + LLM judge + 文档级划分)

## 诚实边界(面试主动说明,加分)

- 首轮指标为**开发文档域基线**(MRR 0.624):比医学域低,主因文档段落
  碎片化 + 评估口径(gold 仅 1 块,回答合理引用多块);已做 k6→k8 实验
  并回滚,记录在案
- 96/98 页:SQLAlchemy 两个超大参考页(>800KB)embedding 接口稳定超时,
  已剔除并同步语料声明(体现工程上的取舍与诚实)
- 无答案拒答率 0.375:语义沾边型库外问题仍为难点(医学域同理,已知限制)
- 未做:SSE 流式、用户鉴权、多会话云端同步(一期本地 SQLite)
