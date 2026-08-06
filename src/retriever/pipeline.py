"""检索编排管线:双通道 → RRF 融合 → rerank → 生成上下文。

链路(config.yaml retrieval 段):
    dense_top_k(30) + bm25_top_k(30) → RRF(rrf_k=60) 取 rrf_top_k(30)
    → SiliconFlow rerank 取 rerank_top_k(20) → 最终注入 final_context_k(6)

学习点 L3:rerank 用交叉编码器(bge-reranker-v2-m3)逐对打分,比双编码器的
余弦相似度更精确,但成本高,所以只对 RRF 后的少量候选重排(top-30 → top-20)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llm import RerankClient
from llm.config import RetrievalConfig

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .rrf import rrf_fuse
from .types import RetrievedChunk


@dataclass
class RetrievalOutput:
    """一次检索的完整输出(观测日志与生成层共用)。"""

    context: list[RetrievedChunk]  # 注入生成的前 final_context_k 条
    candidates: list[RetrievedChunk]  # rerank 后的全部候选(rerank_top_k)
    trace: dict[str, float] = field(default_factory=dict)  # 各阶段耗时(秒)


class RetrieverPipeline:
    """双通道检索编排。"""

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        rerank: RerankClient | None,
        config: RetrievalConfig,
    ):
        self._dense = dense
        self._bm25 = bm25
        self._rerank = rerank
        self._cfg = config

    def retrieve(
        self,
        query_vector: list[float],
        query_text: str,
        translated_query: str | None = None,
    ) -> RetrievalOutput:
        """执行完整检索链路。

        query_vector: 原始问题(中文)的向量 —— 稠密通道用;
        query_text: 中文原始问题;
        translated_query: 英文翻译 + 术语扩展 —— 供 BM25 与 rerank 使用(英文文档更匹配)。
        """
        trace: dict[str, float] = {}

        t0 = time.perf_counter()
        dense_hits = self._dense.search(query_vector, self._cfg.dense_top_k)
        trace["dense_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        bm25_query = translated_query or query_text
        bm25_hits = self._bm25.search(bm25_query, self._cfg.bm25_top_k)
        trace["bm25_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        fused = rrf_fuse(dense_hits, bm25_hits, k=self._cfg.rrf_k, top_k=self._cfg.rrf_top_k)
        trace["rrf_s"] = time.perf_counter() - t0

        if self._rerank and fused:
            t0 = time.perf_counter()
            docs = [c.text for c in fused]
            results = self._rerank.rerank(bm25_query, docs, top_n=self._cfg.rerank_top_k)
            ranked: list[RetrievedChunk] = []
            for r in results:
                if r.index >= len(fused):
                    continue  # rerank 返回越界索引时防御性跳过
                chunk = fused[r.index]
                ranked.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        section=chunk.section,
                        page=chunk.page,
                        paragraph=chunk.paragraph,
                        text=chunk.text,
                        source_url=chunk.source_url,
                        score=r.score,
                    )
                )
            trace["rerank_s"] = time.perf_counter() - t0
            candidates = ranked or fused[: self._cfg.rerank_top_k]
        else:
            candidates = fused[: self._cfg.rerank_top_k]

        return RetrievalOutput(
            context=candidates[: self._cfg.final_context_k],
            candidates=candidates,
            trace=trace,
        )
