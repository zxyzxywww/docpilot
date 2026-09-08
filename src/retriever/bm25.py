"""BM25 关键词检索:查询 → 手写 BM25 索引 → SQLite 补全 chunk 溯源元数据。"""

from __future__ import annotations

from ingest import BM25Index, SQLiteStore

from .types import RetrievedChunk


class BM25Retriever:
    """稀疏(关键词)通道。

    中英混合场景中,该通道使用查询预处理产出的英文查询(translated_query +
    技术术语扩展)执行字面匹配,而非原始中文问题:BM25 是字面匹配,
    中文词在英文官方文档中不会有命中。
    """

    def __init__(self, bm25: BM25Index, sqlite: SQLiteStore):
        self._bm25 = bm25
        self._sqlite = sqlite

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        hits = self._bm25.search(query, top_k)
        result: list[RetrievedChunk] = []
        for hit in hits:
            meta = self._chunk_meta(hit.chunk_id)
            if meta is None:
                continue
            result.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    document_id=meta["document_id"],
                    section=meta["section"] or "",
                    page=meta["page"] or "",
                    paragraph=int(meta["paragraph"] or 0),
                    text=meta["text"],
                    source_url=meta["source_url"] or "",
                    score=hit.score,
                )
            )
        return result

    def _chunk_meta(self, chunk_id: str) -> dict | None:
        # 直接按 chunk_id 单行查询:SQLite 是事实来源,文档删除/重入库后直接读
        # 最新状态,避免派生索引侧缓存陈旧(检索返回已删除内容)。
        return self._sqlite.get_chunk(chunk_id)
