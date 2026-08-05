"""分块器测试:段落对齐、超长切分、确定性 chunk_id。"""

from __future__ import annotations

from ingest import chunk_paragraphs
from ingest.parser import ParsedParagraph


def _paras(texts: list[str]) -> list[ParsedParagraph]:
    return [
        ParsedParagraph(document_id="doc1", section="S", paragraph=i + 1, text=t, page="")
        for i, t in enumerate(texts)
    ]


def test_short_paragraph_single_chunk() -> None:
    paras = _paras(["short paragraph text"])
    chunks = chunk_paragraphs("doc1", paras, chunk_size_tokens=600, chunk_overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc1_c0001"
    assert chunks[0].section == "S"
    assert chunks[0].paragraph == 1


def test_long_paragraph_split_with_overlap() -> None:
    long_text = "word " * 2000  # 约 10000 字符 ≈ 3300 token
    chunks = chunk_paragraphs(
        "doc1", _paras([long_text]), chunk_size_tokens=100, chunk_overlap_tokens=20
    )
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)
    # 相邻块有重叠内容(滑窗)
    assert chunks[1].text in long_text
    assert chunks[0].paragraph == chunks[1].paragraph == 1
    # chunk_id 确定性且连续
    assert [c.chunk_id for c in chunks] == [
        f"doc1_c{i:04d}" for i in range(1, len(chunks) + 1)
    ]


def test_chunk_ids_deterministic() -> None:
    paras = _paras(["first", "second" * 500, "third"])
    a = chunk_paragraphs("doc1", paras, chunk_size_tokens=100, chunk_overlap_tokens=10)
    b = chunk_paragraphs("doc1", paras, chunk_size_tokens=100, chunk_overlap_tokens=10)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_empty_paragraphs() -> None:
    assert chunk_paragraphs("doc1", [], 600, 100) == []
