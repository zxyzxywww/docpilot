"""问答 CLI:完整的 direct_rag 路径。

编排(阶段三全链路):
    中文问题 → QueryPreprocessor(翻译+术语扩展)→ bge-m3 向量
    → RetrieverPipeline(双通道+RRF+rerank)→ DirectRAG(带引用生成)
    → Tracer 观测日志(trace_id/延迟/token/费用/chunk_id)

用法:
    python scripts/query.py "磁共振到CT合成一般用什么深度学习方法?"
    python scripts/query.py --verbose "问句"(显示完整引用与观测信息)
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
    # BM25 派生索引:从 SQLite(事实来源)重建
    ids, texts = [], []
    for doc in sqlite.all_ready_documents():
        for chunk in sqlite.get_chunks(doc["document_id"]):
            ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])
    bm25.build(ids, texts)

    dense = DenseRetriever(qdrant)
    bm25_retriever = BM25Retriever(bm25, sqlite)
    pipeline = RetrieverPipeline(dense, bm25_retriever, rerank, config.retrieval)
    rag = DirectRAG(chat, sqlite)
    preprocessor = QueryPreprocessor(chat)
    return sqlite, qdrant, pipeline, rag, preprocessor


def main() -> int:
    parser = argparse.ArgumentParser(description="MediDoc 医学文献问答(direct_rag)")
    parser.add_argument("question", help="问题(中文或英文)")
    parser.add_argument("--verbose", action="store_true", help="显示引用与观测详情")
    args = parser.parse_args()

    config = load_config()
    logging.basicConfig(level=getattr(logging, config.logging.get("level", "INFO")),
                        format="%(message)s")
    if not config.logging.get("trace", True):
        logging.getLogger("medidoc.trace").setLevel(logging.CRITICAL)

    chat = ChatClient(config.chat, get_api_key("deepseek"))
    embedder = EmbeddingClient(config.embedding, get_api_key("siliconflow"))
    rerank = RerankClient(config.rerank, get_api_key("siliconflow"))
    sqlite, qdrant, pipeline, rag, preprocessor = build_pipeline(
        config, embedder, chat, rerank
    )

    tracer = Tracer(enabled=config.logging.get("trace", True))
    try:
        # 1. 查询预处理(翻译 + 术语扩展)
        prepared = preprocessor.prepare(args.question)
        tracer.step(
            "query_prep",
            translated_query=prepared.translated_query,
            expanded_terms=prepared.expanded_terms,
        )

        # 2. 向量化(原始问题 → 稠密通道)
        query_vector = embedder.embed([args.question])[0]
        tracer.step("embed", dim=len(query_vector))

        # 3. 双通道检索 + RRF + rerank
        retrieval = pipeline.retrieve(
            query_vector, args.question, translated_query=prepared.bm25_query
        )
        tracer.step(
            "retrieve",
            recall_chunk_ids=[c.chunk_id for c in retrieval.context],
            **retrieval.trace,
        )

        # 4. 生成(带引用)
        answer = rag.answer(prepared, retrieval)
        tracer.step(
            "generate",
            prompt_tokens=answer.trace.get("prompt_tokens"),
            completion_tokens=answer.trace.get("completion_tokens"),
            estimated_cost_yuan=answer.trace.get("estimated_cost_yuan"),
            cited_chunk_ids=[c.chunk_id for c in answer.citations],
        )

        # 5. 输出
        print("\n" + "=" * 60)
        print("问题:", args.question)
        print("=" * 60)
        print(answer.answer)
        print("-" * 60)
        if answer.refused:
            print("[系统] 证据不足,已拒答")
        if args.verbose:
            for c in answer.citations:
                print(f"\n[{c.index}] {c.title}({c.journal})")
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
