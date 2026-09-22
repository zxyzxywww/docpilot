"""问答 CLI:direct_rag 与 agentic_rag 双路径(约束 5)。

- direct 路径:中文问题 → QueryPreprocessor(翻译+术语扩展)→ bge-m3 向量
  → RetrieverPipeline(双通道+RRF+rerank)→ DirectRAG(带引用生成);
- agentic 路径:AgentLoop 多步调用工具(search_docs / summarize_doc /
  get_citation)完成复杂综合;README 只展示结构化工具轨迹,不存思维链;
- auto(默认):按问题复杂度自动路由(router.py)。

用法:
    python scripts/query.py "磁共振到CT合成一般用什么深度学习方法?"
    python scripts/query.py --mode agentic "比较GAN和扩散模型在合成CT上的差异"
    python scripts/query.py --verbose "问句"(显示引用与观测)
需要 .env 配置 DEEPSEEK_API_KEY 与 SILICONFLOW_API_KEY。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import logging  # noqa: E402

from ingest import BM25Index, QdrantStore, SQLiteStore  # noqa: E402
from llm import (  # noqa: E402
    ChatClient,
    EmbeddingClient,
    RerankClient,
    get_api_key,
    load_config,
)
from obs import Tracer  # noqa: E402
from rag import DirectRAG  # noqa: E402
from retriever import BM25Retriever, DenseRetriever, RetrieverPipeline  # noqa: E402
from retriever.query_prep import QueryPreprocessor  # noqa: E402


def build_pipeline(config, embedder, chat, rerank):
    """按 config 组装完整问答管线。"""
    sqlite = SQLiteStore(config.database.sqlite_path)
    qdrant = QdrantStore(config.database.qdrant, config.embedding.dimension)
    bm25 = BM25Index()
    ids, texts = [], []
    for doc in sqlite.all_ready_documents():
        for chunk in sqlite.get_chunks(doc["document_id"]):
            ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])
    bm25.build(ids, texts)

    dense = DenseRetriever(qdrant)
    bm25_retriever = BM25Retriever(bm25, sqlite)
    pipeline = RetrieverPipeline(dense, bm25_retriever, rerank, config.retrieval)
    rag = DirectRAG(chat, sqlite, config.rag.min_relevance_score)
    preprocessor = QueryPreprocessor(chat)
    return sqlite, qdrant, pipeline, rag, preprocessor


def _build_agent(config, chat, embedder, rerank, sqlite, pipeline, preprocessor):
    from agent import AgentLoop, ConversationMemory, ToolContext

    ctx = ToolContext(
        pipeline=pipeline,
        embedder=embedder,
        chat=chat,
        sqlite=sqlite,
        preprocessor=preprocessor,
    )
    return AgentLoop(chat, ctx, config.agent, ConversationMemory())


def _run_direct(config, embedder, pipeline, rag, preprocessor, question, tracer):
    prepared = preprocessor.prepare(question)
    tracer.step(
        "query_prep",
        translated_query=prepared.translated_query,
        expanded_terms=prepared.expanded_terms,
    )
    query_vector = embedder.embed([question])[0]
    tracer.step("embed", dim=len(query_vector))
    retrieval = pipeline.retrieve(query_vector, question, translated_query=prepared.bm25_query)
    tracer.step(
        "retrieve",
        recall_chunk_ids=[c.chunk_id for c in retrieval.context],
        **retrieval.trace,
    )
    answer = rag.answer(prepared, retrieval)
    tracer.step(
        "generate",
        prompt_tokens=answer.trace.get("prompt_tokens"),
        completion_tokens=answer.trace.get("completion_tokens"),
        estimated_cost_yuan=answer.trace.get("estimated_cost_yuan"),
        cited_chunk_ids=[c.chunk_id for c in answer.citations],
    )
    return answer.answer, answer.citations, answer.refused, None


def _run_agentic(config, embedder, pipeline, preprocessor, sqlite, chat, question, tracer):
    loop = _build_agent(config, chat, embedder, None, sqlite, pipeline, preprocessor)
    tracer.step("agentic_route", mode="agentic")
    answer = loop.run(question)
    tracer.step(
        "agent_loop",
        steps=answer.steps,
        stop_reason=answer.stop_reason,
        estimated_cost_yuan=answer.total_cost_yuan,
        tool_trace=answer.tool_trace,  # 结构化轨迹,不含思维链
        cited_chunk_ids=[c.chunk_id for c in answer.citations],
    )
    return answer.answer, answer.citations, False, answer


def main() -> int:
    parser = argparse.ArgumentParser(description="DocPilot 开发文档问答(direct/agentic)")
    parser.add_argument("question", help="问题(中文或英文)")
    parser.add_argument("--mode", choices=["auto", "direct", "agentic"], default="auto")
    parser.add_argument("--verbose", action="store_true", help="显示引用与观测详情")
    args = parser.parse_args()

    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.logging.get("level", "INFO")), format="%(message)s"
    )
    if not config.logging.get("trace", True):
        logging.getLogger("docpilot.trace").setLevel(logging.CRITICAL)

    chat = ChatClient(config.chat, get_api_key("deepseek"))
    embedder = EmbeddingClient(config.embedding, get_api_key("siliconflow"))
    rerank = RerankClient(config.rerank, get_api_key("siliconflow"))
    sqlite, qdrant, pipeline, rag, preprocessor = build_pipeline(
        config, embedder, chat, rerank
    )
    tracer = Tracer(enabled=config.logging.get("trace", True))

    try:
        if args.mode == "auto":
            from agent.router import route

            mode = route(args.question)
        else:
            mode = args.mode
        tracer.step("route", mode=mode)

        if mode == "agentic":
            text, citations, refused, agent_info = _run_agentic(
                config, embedder, pipeline, preprocessor, sqlite, chat, args.question, tracer
            )
        else:
            text, citations, refused, agent_info = _run_direct(
                config, embedder, pipeline, rag, preprocessor, args.question, tracer
            )

        print("\n" + "=" * 60)
        print(f"问题: {args.question}   [路径: {mode}]")
        print("=" * 60)
        print(text)
        print("-" * 60)
        if refused:
            print("[系统] 证据不足,已拒答")
        if agent_info is not None and args.verbose:
            print("[工具轨迹](不含思维链)")
            for item in agent_info.tool_trace:
                print(
                    f"    step {item['step']}: {item['tool']} "
                    f"ok={item['ok']} ({item['elapsed_s']}s)"
                )
        if args.verbose:
            for c in citations:
                print(f"\n[{c.index}] {c.title}({c.source_name})")
                print(f"    章节: {c.section} | 段落: {c.paragraph} | chunk: {c.chunk_id}")
                print(f"    来源: {c.source_url}")
                print(f"    证据: {c.evidence[:120]}...")
            print("\n[观测] trace_id =", tracer.trace_id)
            for s in tracer.summary()["steps"]:
                print(f"    {s}")
        return 0
    except Exception as exc:
        tracer.emit(error=str(exc))
        print(f"\n[错误] {exc}", file=sys.stderr)
        return 1
    finally:
        tracer.emit(question=args.question)
        sqlite.close()
        qdrant.close()


if __name__ == "__main__":
    raise SystemExit(main())
