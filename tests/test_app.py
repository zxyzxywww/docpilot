"""UI 层可测逻辑测试:上传校验、service 组装、问答分发(全离线 mock)。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.app import MAX_UPLOAD_MB, _validate_upload  # noqa: E402
from app.service import answer_question  # noqa: E402

# ---------------------------------------------------------------- 上传安全

def test_validate_upload_allows_xml_pdf() -> None:
    _validate_upload("paper.xml", 1000)
    _validate_upload("paper.PDF", 1000)  # 大小写不敏感


def test_validate_upload_rejects_bad_type() -> None:
    with pytest.raises(ValueError, match="不支持的文件类型"):
        _validate_upload("malware.exe", 1000)
    with pytest.raises(ValueError, match="不支持的文件类型"):
        _validate_upload("paper.png", 1000)


def test_validate_upload_rejects_oversize() -> None:
    with pytest.raises(ValueError, match="MB 限制"):
        _validate_upload("big.pdf", (MAX_UPLOAD_MB + 1) * 1024 * 1024)


def test_ingest_upload_rejects_path_traversal(monkeypatch) -> None:
    """修复:文件名带路径分隔符必须拒绝(防目录穿越)。"""
    from app.app import _ingest_upload

    with pytest.raises(ValueError, match="非法文件名"):
        _ingest_upload(NS(), "../evil.xml", b"<article/>")
    with pytest.raises(ValueError, match="非法文件名"):
        _ingest_upload(NS(), "..\\evil.xml", b"<article/>")


def test_ingest_upload_doc_id_deterministic(monkeypatch) -> None:
    """修复:同内容文件重复上传得到相同 doc_id(内容 sha256,幂等)。"""
    from app.app import _ingest_upload

    class FakeParsed:
        title = "T"
        paragraphs = [
            {"section": "s", "page": "", "paragraph": 0, "text": "x",
             "source_url": "", "chunk_id": "c1", "document_id": "d", "token_count": 1}
        ]

    class FakeXMLParser:
        def parse(self, xml_bytes, doc_id, source_url=""):
            return FakeParsed()

    class FakeIngest:
        def __init__(self, *a, **kw):
            pass

        def ingest_one(self, rec):
            return {"ok": True}

    monkeypatch.setattr("app.app.XMLParser", FakeXMLParser)
    monkeypatch.setattr("ingest.IngestService", FakeIngest)
    service = _service()
    service.sqlite = NS(upsert_document=lambda rec, status=None: None)
    service.qdrant = NS()
    service.bm25 = NS()
    service.config.chunking = NS(chunk_size_tokens=600, chunk_overlap_tokens=100)
    d1 = _ingest_upload(service, "a.xml", b"same content bytes")
    d2 = _ingest_upload(service, "b.xml", b"same content bytes")
    assert d1 == d2
    assert d1.startswith("upload_")


# ---------------------------------------------------------------- 问答分发

class FakePreprocessor:
    def prepare(self, q):
        from retriever.query_prep import PreparedQuery

        return PreparedQuery(original_query=q, translated_query="synthetic CT", expanded_terms=[])


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 4 for _ in texts]


class FakePipeline:
    def retrieve(self, vector, query, translated_query=None):
        from retriever.pipeline import RetrievalOutput

        return RetrievalOutput(context=[], candidates=[])


class FakeRAG:
    def answer(self, prepared, retrieval):
        from rag.direct import RagAnswer

        return RagAnswer(answer="合成CT常用GAN(证据不足场景示例)。", refused=True)


def _service(mode_override: str | None = None) -> NS:
    return NS(
        preprocessor=FakePreprocessor(),
        embedder=FakeEmbedder(),
        pipeline=FakePipeline(),
        rag=FakeRAG(),
        chat=NS(),
        sqlite=NS(),
        config=NS(agent=NS(max_steps=3, tool_timeout_seconds=5, max_tool_retries=1,
                           max_consecutive_repeat=2, max_cost_yuan_per_query=0.5)),
    )


def test_answer_question_direct(monkeypatch) -> None:
    service = _service()
    out = answer_question(service, "磁共振到CT合成用什么方法?", mode="direct")
    assert out["mode"] == "direct"
    assert out["refused"] is True  # FakeRAG 拒答
    assert "证据不足" in out["answer"]


def test_answer_question_auto_routes_agentic(monkeypatch) -> None:
    """复杂问题 auto 模式应路由到 agentic;agent 依赖缺失时走 loop(此处仅验证路由返回值)。"""
    service = _service()
    # mock AgentLoop 避免真实构造
    calls = {}

    class FakeLoop:
        def __init__(self, *a, **kw):
            calls["created"] = True

        def run(self, question):
            from agent import AgentAnswer

            return AgentAnswer(
                answer="综合多篇文献:GAN 与扩散模型各有优势。",
                stop_reason="final_answer",
                total_cost_yuan=0.01,
                tool_trace=[],
            )

    monkeypatch.setattr("agent.AgentLoop", FakeLoop)
    out = answer_question(service, "比较GAN和扩散模型在合成CT上的差异", mode="auto")
    assert out["mode"] == "agentic"
    assert calls.get("created") is True
    assert out["trace"]["stop_reason"] == "final_answer"
