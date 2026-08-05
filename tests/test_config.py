"""配置加载与默认参数测试(离线,不调用任何 API)。"""

from __future__ import annotations

from llm import load_config


def test_retrieval_defaults() -> None:
    cfg = load_config()
    assert cfg.retrieval.dense_top_k == 30
    assert cfg.retrieval.bm25_top_k == 30
    assert cfg.retrieval.rrf_top_k == 30
    assert cfg.retrieval.rerank_top_k == 20
    assert cfg.retrieval.final_context_k == 6


def test_chunking_defaults() -> None:
    cfg = load_config()
    assert cfg.chunking.chunk_size_tokens == 600
    assert cfg.chunking.chunk_overlap_tokens == 100


def test_embedding_model_fixed() -> None:
    """Embedding 定案:BAAI/bge-m3, dimension=1024。"""
    cfg = load_config()
    assert cfg.embedding.model == "BAAI/bge-m3"
    assert cfg.embedding.dimension == 1024
    assert cfg.rerank.model == "BAAI/bge-reranker-v2-m3"


def test_mvp_redlines_enabled() -> None:
    cfg = load_config()
    assert cfg.mvp["no_patient_data"] is True
    assert cfg.mvp["no_medical_advice"] is True
    assert cfg.mvp["refuse_without_evidence"] is True
    assert cfg.mvp["citations_required"] is True
