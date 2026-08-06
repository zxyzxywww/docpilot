"""RRF(Reciprocal Rank Fusion)融合算法。

为什么需要融合(学习点 L1/L2):
- 向量检索(稠密)擅长"语义相关但用词不同"的匹配;
- BM25(稀疏)擅长"关键词精确命中"的匹配;
- 单一通道各有盲区,融合后召回质量显著更稳。

RRF 公式(不依赖各通道的分数绝对值,只看排名,天然可比较):
    score(chunk) = Σ_{retriever} 1 / (k + rank(chunk))
- rank 从 1 开始(越靠前贡献越大);
- k 为平滑常数(默认 60),减小 k 会放大排名靠前结果的差异;
- 两个通道的分数域不同(余弦相似度 vs BM25 分数),直接用分数相加不可比,
  RRF 用"排名倒数"统一量纲,是工业界最常用的零调参融合法。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .types import RetrievedChunk


def rrf_fuse(
    dense: Sequence[RetrievedChunk],
    bm25: Sequence[RetrievedChunk],
    k: int = 60,
    top_k: int = 30,
) -> list[RetrievedChunk]:
    """融合两个按相关度降序排列的检索结果,返回融合后降序的 top_k。"""
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        first_seen.setdefault(chunk.chunk_id, chunk)
    for rank, chunk in enumerate(bm25, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        first_seen.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        replace(first_seen[cid], score=score) for cid, score in ordered[:top_k]
    ]
