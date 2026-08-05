"""XML/PDF 解析器测试:章节路径、参考文献排除、命名空间兼容、扫描 PDF 报错。"""

from __future__ import annotations

import pytest

from ingest import PDFParser, ScannedPDFError, XMLParser
from ingest.parser import JATS_NS

# 无命名空间的 PMC XML 样本
XML_PLAIN = """<pmc-articleset>
<article article-type="research-article">
<front>
<journal-meta><journal-title>Test Journal</journal-title></journal-meta>
<article-meta>
<title-group><article-title>Test Article</article-title></title-group>
<article-id pub-id-type="doi">10.1000/test</article-id>
</article-meta>
</front>
<body>
<sec><title>INTRODUCTION</title>
<p>First paragraph of intro.</p>
<p>Second paragraph of intro.</p>
<sec><title>Sub</title><p>Nested paragraph.</p></sec>
</sec>
<sec><title>METHODS</title><p>Methods paragraph.</p></sec>
</body>
<back><ref-list><ref><label>1</label>
<mixed-citation>Ref one.</mixed-citation></ref></ref-list></back>
</article>
</pmc-articleset>"""

# 带 JATS 命名空间的版本(结构相同)
XML_NS = XML_PLAIN.replace("<pmc-articleset>", '<pmc-articleset xmlns="' + JATS_NS["jats"] + '">')


@pytest.mark.parametrize("xml", [XML_PLAIN, XML_NS])
def test_xml_parser_sections_and_no_references(xml: str) -> None:
    doc = XMLParser().parse(xml.encode(), "doc123", "https://example.org/PMC1")
    assert doc.title == "Test Article"
    # back/ref-list 被排除:4 个正文段落,参考文献不算
    assert len(doc.paragraphs) == 4
    sections = [p.section for p in doc.paragraphs]
    assert sections == ["INTRODUCTION", "INTRODUCTION", "INTRODUCTION / Sub", "METHODS"]
    assert [p.paragraph for p in doc.paragraphs] == [1, 2, 3, 4]
    assert all(p.source_url == "https://example.org/PMC1" for p in doc.paragraphs)


def test_xml_parser_no_body() -> None:
    xml = (
        "<pmc-articleset><article><front><article-meta>"
        "<title-group><article-title>T</article-title></title-group>"
        "</article-meta></front></article></pmc-articleset>"
    )
    doc = XMLParser().parse(xml.encode(), "doc1", "")
    assert doc.paragraphs == []


def test_xml_parser_invalid() -> None:
    from ingest import ParseError

    with pytest.raises(ParseError):
        XMLParser().parse(b"<not-xml", "doc1", "")


def test_pdf_scanned_raises(tmp_path) -> None:
    """空白 PDF(无文本层)必须明确报错,不静默生成空内容。"""
    from pypdf import PdfWriter

    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with blank.open("wb") as f:
        writer.write(f)
    with pytest.raises(ScannedPDFError, match="OCR"):
        PDFParser().parse(str(blank), "doc1", "")
