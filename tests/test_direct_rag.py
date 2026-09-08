"""direct_rag 生成层测试:引用解析、拒答、防幻觉、注入防护(全离线 mock)。"""

from __future__ import annotations

from pathlib import Path

from ingest import SQLiteStore
from llm import ChatResult
from rag import DirectRAG
from retriever.pipeline import RetrievalOutput
from retriever.query_prep import PreparedQuery
from retriever.types import RetrievedChunk


class FakeChat:
    def __init__(self, text: str):
        self._text = text
        self.last_messages: list[dict] = []

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        return ChatResult(
            text=self._text,
            model="fake",
            prompt_tokens=100,
            completion_tokens=50,
            cache_hit_tokens=0,
        )


def _doc(store: SQLiteStore, doc_id: str, title: str = "T", journal: str = "J") -> None:
    store.upsert_document(
        {
            "document_id": doc_id,
            "title": title,
            "journal": journal,
            "authors": [],
        },
        status="ready",
    )


def _ctx() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="d1_c0001",
            document_id="d1",
            section="Methods",
            page="",
            paragraph=2,
            text="fastapi routing generation uses deep learning.",
            source_url="https://example.org/PMC1",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="d2_c0001",
            document_id="d2",
            section="Results",
            page="",
            paragraph=5,
            text="diffusion models achieve high fidelity.",
            source_url="https://example.org/PMC2",
            score=0.8,
        ),
    ]


def _prepared() -> PreparedQuery:
    return PreparedQuery(
        original_query="磁共振到CT合成用什么方法?",
        translated_query="methods for MR-to-CT synthesis",
        expanded_terms=["fastapi routing"],
    )


def test_refusal_when_no_evidence(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "db.sqlite")
    rag = DirectRAG(FakeChat("unused"), store)  # type: ignore[arg-type]
    out = rag.answer(_prepared(), RetrievalOutput(context=[], candidates=[]))
    assert out.refused is True
    assert "证据不足" in out.answer
    store.close()


def test_refusal_when_low_relevance(tmp_path: Path) -> None:
    """rerank 分数低于阈值 → 证据不足拒答(防硬答)。"""
    store = SQLiteStore(tmp_path / "db.sqlite")
    rag = DirectRAG(FakeChat("unused"), store)  # type: ignore[arg-type]
    low_score = [_chunk_low()]
    out = rag.answer(_prepared(), RetrievalOutput(context=low_score, candidates=low_score))
    assert out.refused is True
    assert "相关性不足" in out.answer
    store.close()


def _chunk_low() -> RetrievedChunk:
    c = _ctx()[0]
    c.score = 0.05
    return c


def test_answer_parses_citations(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "db.sqlite")
    _doc(store, "d1", "FastAPI Reference", "Official Docs")
    _doc(store, "d2", "Pydantic User Guide", "Official Docs")
    fake = FakeChat("扩散模型效果好[1][2]。")
    rag = DirectRAG(fake, store)  # type: ignore[arg-type]
    out = rag.answer(_prepared(), RetrievalOutput(context=_ctx(), candidates=_ctx()))
    assert out.refused is False
    assert len(out.citations) == 2
    c1 = out.citations[0]
    assert c1.index == 1 and c1.chunk_id == "d1_c0001"
    assert c1.title == "FastAPI Reference"
    assert c1.journal == "Official Docs"
    assert "deep learning" in c1.evidence
    store.close()


def test_out_of_range_citation_dropped(tmp_path: Path) -> None:
    """模型编造证据中不存在的编号([9])→ 必须丢弃(防幻觉)。"""
    store = SQLiteStore(tmp_path / "db.sqlite")
    _doc(store, "d1")
    fake = FakeChat("结论正确[1][9]。")  # [9] 不存在
    rag = DirectRAG(fake, store)  # type: ignore[arg-type]
    out = rag.answer(_prepared(), RetrievalOutput(context=_ctx(), candidates=_ctx()))
    assert [c.index for c in out.citations] == [1]
    store.close()


def test_injection_guard_present_in_system_prompt(tmp_path: Path) -> None:
    """注入防护:system prompt 必须把检索证据标记为不可信,文档指令不得被执行。"""
    store = SQLiteStore(tmp_path / "db.sqlite")
    _doc(store, "d1")
    fake = FakeChat("ok")
    rag = DirectRAG(fake, store)  # type: ignore[arg-type]
    rag.answer(_prepared(), RetrievalOutput(context=_ctx(), candidates=_ctx()))
    system = fake.last_messages[0]["content"]
    assert "不得被执行" in system
    assert "未经核实" in system
    store.close()


def test_evidence_block_contains_metadata(tmp_path: Path) -> None:
    """证据块必须携带完整溯源信息(标题/章节/chunk_id),引用才能点击溯源。"""
    store = SQLiteStore(tmp_path / "db.sqlite")
    _doc(store, "d1", "FastAPI Reference")
    fake = FakeChat("ok")
    rag = DirectRAG(fake, store)  # type: ignore[arg-type]
    rag.answer(_prepared(), RetrievalOutput(context=_ctx(), candidates=_ctx()))
    user = fake.last_messages[1]["content"]
    assert "FastAPI Reference" in user
    assert "d1_c0001" in user
    assert "Methods" in user
    store.close()
