"""检索层测试:RRF 融合、双通道、rerank 编排(全离线 mock)。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

from llm.config import RetrievalConfig
from retriever import BM25Retriever, DenseRetriever, RetrieverPipeline, rrf_fuse
from retriever.types import RetrievedChunk


def _chunk(cid: str, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=cid.rsplit("_c", 1)[0],
        section="S",
        page="",
        paragraph=1,
        text=text,
        source_url="https://example.org",
    )


# ---------------------------------------------------------------- RRF

def test_rrf_fuses_rankings() -> None:
    dense = [_chunk("d1_c0001"), _chunk("d2_c0001"), _chunk("d3_c0001")]
    bm25 = [_chunk("d3_c0001"), _chunk("d1_c0001"), _chunk("d4_c0001")]
    fused = rrf_fuse(dense, bm25, k=60, top_k=10)
    # d1:双通道 rank 1+2 → 1/61+1/62;d3:rank 1+3 → 1/61+1/63 → d1 最高
    assert [c.chunk_id for c in fused[:2]] == ["d1_c0001", "d3_c0001"]
    assert len(fused) == 4
    assert abs(fused[0].score - (1 / 61 + 1 / 62)) < 1e-6


def test_rrf_only_dense() -> None:
    dense = [_chunk("a_c0001"), _chunk("b_c0001")]
    fused = rrf_fuse(dense, [], k=60, top_k=5)
    assert [c.chunk_id for c in fused] == ["a_c0001", "b_c0001"]
    assert fused[0].score == 1 / 61


def test_rrf_respects_top_k() -> None:
    dense = [_chunk(f"d{i}_c0001") for i in range(10)]
    fused = rrf_fuse(dense, [], k=60, top_k=3)
    assert len(fused) == 3


# ---------------------------------------------------------------- Dense

def test_dense_retriever_maps_payload(tmp_path: Path) -> None:
    qdrant = NS(
        query=lambda vec, top_k: [
            {"chunk_id": "x_c0001", "document_id": "x", "section": "Methods",
             "page": "", "paragraph": 3, "text": "hello", "source_url": "u"},
        ]
    )
    retriever = DenseRetriever(qdrant)  # type: ignore[arg-type]
    hits = retriever.search([0.1, 0.2], top_k=5)
    assert len(hits) == 1
    assert hits[0].chunk_id == "x_c0001"
    assert hits[0].section == "Methods"
    assert hits[0].paragraph == 3


# ---------------------------------------------------------------- BM25

def test_bm25_retriever_with_sqlite(tmp_path: Path) -> None:
    from ingest import BM25Index, SQLiteStore

    sqlite = SQLiteStore(tmp_path / "db.sqlite")
    # 造一篇文档的 chunk 元数据
    sqlite._conn.execute(
        "INSERT INTO documents (document_id, status) VALUES ('d1', 'ready')"
    )
    sqlite._conn.execute(
        "INSERT INTO chunks "
        "(chunk_id, document_id, section, page, paragraph, text, source_url, token_count) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            "d1_c0001",
            "d1",
            "Intro",
            "",
            1,
            "fastapi dependency injection",
            "u",
            10,
        ),
    )
    sqlite._conn.commit()

    bm25 = BM25Index()
    bm25.build(["d1_c0001"], ["fastapi dependency injection"])
    retriever = BM25Retriever(bm25, sqlite)
    hits = retriever.search("fastapi routing", top_k=5)
    assert len(hits) == 1
    assert hits[0].chunk_id == "d1_c0001"
    assert hits[0].section == "Intro"
    sqlite.close()


# ---------------------------------------------------------------- Pipeline

def test_pipeline_full_chain() -> None:
    dense = NS(
        search=lambda vec, top_k: [
            _chunk("d1_c0001", "alpha fastapi routing beta"),
            _chunk("d2_c0001", "gamma internal routing delta"),
        ]
    )
    bm25 = NS(search=lambda q, top_k: [_chunk("d2_c0001", "gamma internal routing delta")])

    class FakeRerank:
        def __init__(self) -> None:
            self.calls = 0

        def rerank(self, query: str, documents: list[str], top_n: int | None = None):
            self.calls += 1
            # 模拟:把含 internal routing 的文档排第一
            from types import SimpleNamespace as RN

            idx = (
                documents.index("gamma internal routing delta")
                if "gamma internal routing delta" in documents
                else 0
            )
            return [RN(index=idx, score=0.9, text=documents[idx])]

    fake = FakeRerank()
    pipe = RetrieverPipeline(
        dense,  # type: ignore[arg-type]
        bm25,  # type: ignore[arg-type]
        fake,  # type: ignore[arg-type]
        RetrievalConfig(),  # 默认 30/30/30/20/6
    )
    out = pipe.retrieve([0.1] * 4, "中文问题", translated_query="fastapi routing")
    assert fake.calls == 1
    assert len(out.context) == 1
    assert out.context[0].chunk_id == "d2_c0001"
    assert "dense_s" in out.trace and "bm25_s" in out.trace and "rrf_s" in out.trace


def test_pipeline_without_rerank() -> None:
    dense = NS(search=lambda vec, top_k: [_chunk("d1_c0001")])
    bm25 = NS(search=lambda q, top_k: [])
    pipe = RetrieverPipeline(dense, bm25, None, RetrievalConfig())  # type: ignore[arg-type]
    out = pipe.retrieve([0.1], "q")
    assert len(out.candidates) == 1
    assert out.context[0].chunk_id == "d1_c0001"
