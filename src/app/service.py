"""app 服务工厂:CLI 与 Streamlit UI 共用的问答服务组装。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ingest import BM25Index, QdrantStore, SQLiteStore
from llm import ChatClient, EmbeddingClient, RerankClient, get_api_key, load_config
from llm.config import AppConfig
from rag import DirectRAG
from retriever import BM25Retriever, DenseRetriever, RetrieverPipeline
from retriever.query_prep import QueryPreprocessor


@dataclass
class Service:
    """一次组装好的完整问答服务(所有依赖)。"""

    config: AppConfig
    sqlite: SQLiteStore
    qdrant: QdrantStore
    bm25: BM25Index
    pipeline: RetrieverPipeline
    rag: DirectRAG
    preprocessor: QueryPreprocessor
    chat: ChatClient
    embedder: EmbeddingClient
    rerank: RerankClient | None


def build_service() -> Service:
    """从 config.yaml + .env 组装服务;关闭由调用方负责(sqlite/qdrant.close)。"""
    config = load_config()
    chat = ChatClient(config.chat, get_api_key("deepseek"))
    embedder = EmbeddingClient(config.embedding, get_api_key("siliconflow"))
    rerank = RerankClient(config.rerank, get_api_key("siliconflow"))

    sqlite = SQLiteStore(config.database.sqlite_path)
    qdrant = QdrantStore(config.database.qdrant, config.embedding.dimension)
    bm25 = BM25Index()
    ids, texts = [], []
    for doc in sqlite.all_ready_documents():
        for chunk in sqlite.get_chunks(doc["document_id"]):
            ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])
    bm25.build(ids, texts)

    pipeline = RetrieverPipeline(
        DenseRetriever(qdrant), BM25Retriever(bm25, sqlite), rerank, config.retrieval
    )
    rag = DirectRAG(chat, sqlite, config.rag.min_relevance_score)
    preprocessor = QueryPreprocessor(chat)
    return Service(
        config, sqlite, qdrant, bm25, pipeline, rag, preprocessor, chat, embedder, rerank
    )


def answer_question(service: Service, question: str, mode: str = "auto") -> dict[str, Any]:
    """执行一次问答,返回 {answer, citations, refused, trace, mode}。"""
    if mode == "auto":
        from agent.router import route

        mode = route(question)

    if mode == "agentic":
        from agent import AgentLoop, ConversationMemory, ToolContext

        ctx = ToolContext(
            pipeline=service.pipeline,
            embedder=service.embedder,
            chat=service.chat,
            sqlite=service.sqlite,
            preprocessor=service.preprocessor,
        )
        loop = AgentLoop(service.chat, ctx, service.config.agent, ConversationMemory())
        ans = loop.run(question)
        return {
            "answer": ans.answer,
            "citations": ans.citations,
            "refused": ans.stop_reason != "final_answer",
            "mode": mode,
            "stop_reason": ans.stop_reason,
            "trace": {
                "steps": ans.steps,
                "stop_reason": ans.stop_reason,
                "cost_yuan": ans.total_cost_yuan,
                "tool_trace": ans.tool_trace,
            },
        }

    prepared = service.preprocessor.prepare(question)
    vector = service.embedder.embed([question])[0]
    retrieval = service.pipeline.retrieve(
        vector, question, translated_query=prepared.bm25_query
    )
    rag_ans = service.rag.answer(prepared, retrieval)
    return {
        "answer": rag_ans.answer,
        "citations": rag_ans.citations,
        "refused": rag_ans.refused,
        "mode": mode,
        "trace": {
            "translated_query": prepared.translated_query,
            "expanded_terms": prepared.expanded_terms,
            "recall_chunk_ids": [c.chunk_id for c in retrieval.context],
            "cost_yuan": rag_ans.trace.get("estimated_cost_yuan", 0.0),
        },
    }


def _silence_trace_logging() -> None:
    logging.getLogger("medidoc.trace").setLevel(logging.CRITICAL)
