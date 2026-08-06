"""稠密检索:查询向量 → Qdrant 余弦相似度 top-k。"""

from __future__ import annotations

from ingest import QdrantStore

from .types import RetrievedChunk


class DenseRetriever:
    """稠密(语义)通道。

    输入必须是查询文本的 embedding 向量(由外部嵌入模型生成,阶段三中
    中文原始问题直接向量化 —— 跨语言检索依赖 bge-m3 的多语言能力)。
    """

    def __init__(self, qdrant: QdrantStore):
        self._qdrant = qdrant

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        payloads = self._qdrant.query(query_vector, top_k)
        return [self._to_chunk(p) for p in payloads]

    @staticmethod
    def _to_chunk(p: dict) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=p["chunk_id"],
            document_id=p["document_id"],
            section=p.get("section", ""),
            page=p.get("page", ""),
            paragraph=int(p.get("paragraph", 0)),
            text=p["text"],
            source_url=p.get("source_url", ""),
        )
