"""存储层测试:状态机、幂等、数量校验、维度变化重建报错、删除、BM25、IngestService 编排。

全部离线:embedding 用 FakeEmbedder,Qdrant 用 tmp_path local mode。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from ingest import (
    BM25Index,
    IndexRebuildRequiredError,
    IngestService,
    QdrantStore,
    SQLiteStore,
    chunk_paragraphs,
)
from ingest.parser import ParsedParagraph
from ingest.store import _chunk_uuid

SAMPLE_XML = """<articleset><article article-type="research-article">
<front><article-meta><title-group><article-title>Test</article-title></title-group></article-meta></front>
<body><sec><title>INTRO</title><p>Deep learning based fastapi routing usage.</p>
<p>MR to CT synthesis with diffusion models.</p></sec></body>
</article></articleset>"""


class FakeEmbedder:
    """返回固定维度向量的假 embedding 客户端。"""

    def __init__(self, dim: int = 4, model: str = "fake/bge-m3"):
        self.config = NS(model=model, dimension=dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.config.dimension for _ in texts]


def _qdrant_config(tmp_path: Path, dim: int = 4) -> NS:
    return NS(
        mode="local",
        path=str(tmp_path / "qdrant"),
        collection="test_collection",
        docker_url="http://localhost:6333",
    )


def _manifest_rec(tmp_path: Path, doc_id: str = "doc0001", xml: str = SAMPLE_XML) -> dict:
    xml_path = tmp_path / f"{doc_id}.xml"
    xml_path.write_text(xml, encoding="utf-8")
    return {
        "document_id": doc_id,
        "source_id": "DOC123",
        "title": "Test",
        "authors": ["A", "B"],
        "source_name": "Test J",
        "doi": "10.1/test",
        "source_url": "https://example.org",
        "license": "CC BY",
        "publication_date": "2026-01-01",
        "document_type": "research-article",
        "sha256": "x" * 64,
        "local_path": str(xml_path),
    }


def _service(tmp_path: Path, dim: int = 4) -> IngestService:
    sqlite = SQLiteStore(tmp_path / "docpilot.db")
    qdrant = QdrantStore(_qdrant_config(tmp_path, dim), dim)
    return IngestService(sqlite, qdrant, BM25Index(), FakeEmbedder(dim))


# ---------------------------------------------------------------- SQLite

def test_status_machine(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "db.sqlite")
    rec = _manifest_rec(tmp_path)
    store.upsert_document(rec, status="pending")
    store.set_status(rec["document_id"], "indexing")
    store.set_status(rec["document_id"], "ready")
    doc = store.get_document(rec["document_id"])
    assert doc is not None and doc["status"] == "ready"
    with pytest.raises(ValueError):
        store.set_status(rec["document_id"], "bogus")


def test_replace_chunks_idempotent(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "db.sqlite")
    rec = _manifest_rec(tmp_path)
    paras = [ParsedParagraph("doc0001", "S", 1, "some text", "")]
    chunks = chunk_paragraphs("doc0001", paras, 600, 100)
    store.upsert_document(rec)
    store.replace_chunks(chunks)
    store.replace_chunks(chunks)  # 重复写入不产生重复块
    assert store.count_chunks("doc0001") == len(chunks)
    store.close()


# ---------------------------------------------------------------- Qdrant

def test_qdrant_upsert_count_and_delete(tmp_path: Path) -> None:
    qd = QdrantStore(_qdrant_config(tmp_path), 4)
    paras = [ParsedParagraph("doc1", "S", i + 1, f"text {i}", "") for i in range(3)]
    chunks = chunk_paragraphs("doc1", paras, 600, 100)
    qd.upsert(chunks, [[0.1] * 4 for _ in chunks])
    assert qd.count_document("doc1") == 3
    assert qd.count_all() == 3
    qd.delete_document("doc1")
    assert qd.count_document("doc1") == 0
    qd.close()


def test_qdrant_dimension_mismatch_raises(tmp_path: Path) -> None:
    QdrantStore(_qdrant_config(tmp_path), 4)
    with pytest.raises(IndexRebuildRequiredError, match="重建"):
        QdrantStore(_qdrant_config(tmp_path), 8)
    # 同维度重开不报错
    QdrantStore(_qdrant_config(tmp_path), 4)


def test_chunk_uuid_deterministic() -> None:
    assert _chunk_uuid("doc1_c0001") == _chunk_uuid("doc1_c0001")
    assert _chunk_uuid("doc1_c0001") != _chunk_uuid("doc1_c0002")


# ---------------------------------------------------------------- BM25

def test_bm25_search(tmp_path: Path) -> None:
    bm25 = BM25Index()
    bm25.build(["c1", "c2"], ["fastapi routing usage", "cooking recipes"])
    hits = bm25.search("fastapi routing", top_k=2)
    assert hits[0].chunk_id == "c1"
    assert hits[0].score > 0
    assert bm25.search("nothing here", top_k=5) == []


# ---------------------------------------------------------------- IngestService

def test_ingest_ready_and_verify(tmp_path: Path) -> None:
    service = _service(tmp_path)
    rec = _manifest_rec(tmp_path)
    outcome = service.ingest_one(rec)
    assert outcome["ok"] is True
    assert outcome["chunks"] == 2
    # 三存储一致
    assert service.sqlite.count_chunks("doc0001") == 2
    assert service.qdrant.count_document("doc0001") == 2
    assert service.sqlite.get_document("doc0001")["status"] == "ready"
    assert service.bm25.size() == 2


def test_ingest_bad_xml_marks_failed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    rec = _manifest_rec(tmp_path, doc_id="bad001", xml="<broken")
    outcome = service.ingest_one(rec)
    assert outcome["ok"] is False
    assert service.sqlite.get_document("bad001")["status"] == "failed"


def test_ingest_delete_cleans_all_stores(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.ingest_one(_manifest_rec(tmp_path))
    service.delete_document("doc0001")
    assert service.sqlite.get_document("doc0001") is None
    assert service.qdrant.count_document("doc0001") == 0
    assert service.bm25.size() == 0


def test_ingest_manifest_summary(tmp_path: Path) -> None:
    service = _service(tmp_path)
    recs = [_manifest_rec(tmp_path, doc_id="a"), _manifest_rec(tmp_path, doc_id="b", xml="<bad")]
    result = service.ingest_manifest(recs)
    assert result["ok"] == 1
    assert result["failed"] == 1
    assert service.qdrant.count_all() == 2  # 仅 a 成功入库
