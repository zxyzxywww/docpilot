"""运行时组装:RAG 核心(位于 src/)的 Service 单例构建。

原位于 src/app/service.py(Streamlit UI 时代);前端改造为前后端分离后迁至
server 侧,删除 Streamlit UI 时一并迁移到这里。RAG 核心模块(src/retriever、
rag、agent、ingest、llm)未改动,此处只做组装。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ingest import BM25Index, QdrantStore, SQLiteStore  # noqa: E402
from llm import ChatClient, EmbeddingClient, RerankClient, get_api_key, load_config  # noqa: E402
from llm.config import AppConfig  # noqa: E402
from rag import DirectRAG  # noqa: E402
from retriever import BM25Retriever, DenseRetriever, RetrieverPipeline  # noqa: E402
from retriever.query_prep import QueryPreprocessor  # noqa: E402


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
    rag = DirectRAG(
        chat,
        sqlite,
        min_relevance_score=config.rag.min_relevance_score,
        gate_enabled=config.rag.gate_enabled,
        gate_threshold=config.rag.gate_threshold,
    )
    preprocessor = QueryPreprocessor(chat)
    return Service(
        config, sqlite, qdrant, bm25, pipeline, rag, preprocessor, chat, embedder, rerank
    )
