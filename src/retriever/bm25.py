"""BM25 关键词检索:查询 → 手写 BM25 索引 → SQLite 补全 chunk 溯源元数据。"""

from __future__ import annotations

from ingest import BM25Index, SQLiteStore

from .types import RetrievedChunk


class BM25Retriever:
    """稀疏(关键词)通道。

    跨语言场景中,该通道使用"翻译后的英文查询 + 医学术语扩展"(而非原始
    中文问题),因为 BM25 是字面匹配,中文词在英文文献里不会有命中。
    """

    def __init__(self, bm25: BM25Index, sqlite: SQLiteStore):
        self._bm25 = bm25
        self._sqlite = sqlite
        self._chunk_cache: dict[str, dict] = {}

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
        if chunk_id not in self._chunk_cache:
            doc_id = chunk_id.rsplit("_c", 1)[0]
            for row in self._sqlite.get_chunks(doc_id):
                self._chunk_cache[row["chunk_id"]] = row
        return self._chunk_cache.get(chunk_id)
