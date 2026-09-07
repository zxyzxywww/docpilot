"""文档解析器:PMC JATS XML 优先,PDF 作为通用上传格式。

设计:
- XML 解析仅遍历 body(不解析 back/ref-list),天然排除参考文献列表;
  保留 title / section(章节路径)/ paragraph(段落序号)/ source_url。
- PDF 解析按页提取文本,page 记录页码;扫描 PDF(无文本层)明确抛错,
  不静默生成空内容(OCR 暂不支持,属 MVP 红线)。
- page 字段:XML 电子版通常无段落级页码,留空字符串;PDF 填页码。

依赖:仅标准库 + pypdf(XML 解析不依赖任何框架)。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser

from pypdf import PdfReader

JATS_NS = {"jats": "http://www.ncbi.nlm.nih.gov/JATS1"}


class ParseError(Exception):
    """解析失败。"""


class ScannedPDFError(ParseError):
    """PDF 无文本层(扫描件),当前版本暂不支持 OCR。"""


@dataclass
class ParsedParagraph:
    """一个语义段落(引用溯源的最小单位)。"""

    document_id: str
    section: str  # 章节路径,如 "Methods / Data acquisition";无章节则为 ""
    paragraph: int  # 文档内段落序号(从 1 开始,跨章节连续)
    text: str
    page: str  # 页码;不可得时为空字符串
    source_url: str = ""


@dataclass
class ParsedDocument:
    """一篇文档的解析结果。"""

    document_id: str
    title: str
    paragraphs: list[ParsedParagraph] = field(default_factory=list)


# ---------------------------------------------------------------- XML 解析

def _ns_for(root: ET.Element) -> dict[str, str]:
    return JATS_NS if "{" in root.tag else {}


def _find(root: ET.Element, path: str, ns: dict[str, str]) -> ET.Element | None:
    if not ns:
        path = path.replace("jats:", "")
    return root.find(path, ns)


def _findall(root: ET.Element, path: str, ns: dict[str, str]) -> list[ET.Element]:
    if not ns:
        path = path.replace("jats:", "")
    return list(root.findall(path, ns))


def _join_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return " ".join(el.itertext()).strip()


class XMLParser:
    """PMC JATS XML 解析器。"""

    def parse(self, xml_bytes: bytes, document_id: str, source_url: str = "") -> ParsedDocument:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise ParseError(f"XML 解析失败: {exc}") from exc
        ns = _ns_for(root)
        article = _find(root, ".//jats:article", ns)
        if article is None:
            raise ParseError("未找到 article 节点")
        title = _join_text(_find(article, ".//jats:title-group/jats:article-title", ns))
        paras: list[ParsedParagraph] = []
        body = _find(article, ".//jats:body", ns)
        if body is not None:
            counter = 0
            for sec in _findall(body, "jats:sec", ns):
                counter = self._walk_sec(sec, [], counter, document_id, source_url, ns, paras)
            # body 直接子段落(无 sec 包裹的文章)
            for p in _findall(body, "jats:p", ns):
                text = _join_text(p)
                if text:
                    counter += 1
                    paras.append(
                        ParsedParagraph(
                            document_id=document_id,
                            section="",
                            paragraph=counter,
                            text=text,
                            page="",
                            source_url=source_url,
                        )
                    )
        return ParsedDocument(document_id=document_id, title=title, paragraphs=paras)

    def _walk_sec(
        self,
        sec: ET.Element,
        parent_titles: list[str],
        counter: int,
        document_id: str,
        source_url: str,
        ns: dict[str, str],
        out: list[ParsedParagraph],
    ) -> int:
        """递归收集章节标题路径下的段落;嵌套章节标题并入路径,段落不重复。"""
        title = _join_text(_find(sec, "jats:title", ns))
        path = parent_titles + [title] if title else parent_titles
        for p in _findall(sec, "jats:p", ns):
            text = _join_text(p)
            if text:
                counter += 1
                out.append(
                    ParsedParagraph(
                        document_id=document_id,
                        section=" / ".join(path),
                        paragraph=counter,
                        text=text,
                        page="",
                        source_url=source_url,
                    )
                )
        for sub in _findall(sec, "jats:sec", ns):
            counter = self._walk_sec(sub, path, counter, document_id, source_url, ns, out)
        return counter


# ---------------------------------------------------------------- PDF 解析

class PDFParser:
    """PDF 解析器(按页提取;扫描件明确报错)。"""

    def parse(self, pdf_path: str, document_id: str, source_url: str = "") -> ParsedDocument:
        try:
            reader = PdfReader(pdf_path)
        except Exception as exc:
            raise ParseError(f"PDF 读取失败: {exc}") from exc
        paras: list[ParsedParagraph] = []
        counter = 0
        for page_no, page in enumerate(reader.pages, start=1):
            text = _join_text_keep_spaces(page.extract_text() or "")
            if not text.strip():
                continue  # 空白页跳过
            counter += 1
            paras.append(
                ParsedParagraph(
                    document_id=document_id,
                    section="",
                    paragraph=counter,
                    text=text,
                    page=str(page_no),
                    source_url=source_url,
                )
            )
        if not paras:
            raise ScannedPDFError(
                "该 PDF 无文本层(疑似扫描件),当前版本暂不支持 OCR;请提供 PMC XML 或含文本层的 PDF"
            )
        return ParsedDocument(document_id=document_id, title="", paragraphs=paras)


def _join_text_keep_spaces(text: str) -> str:
    """PDF 文本按原样保留,压缩多余空白。"""
    return " ".join(text.split())


# ---------------------------------------------------------------- HTML 解析

_HTML_SKIP_TAGS = {
    "script", "style", "nav", "header", "footer", "noscript",
    "svg", "form", "aside", "iframe",
}


class _HTMLDocHandler(HTMLParser):
    """基于 html.parser 的文档正文提取状态机。

    规则:
    - 跳过导航/脚本/样式等(SKIP 子树不产生内容);
    - 正文容器定位:<article> / <main> / <div class 含 md-content 或 body>;
    - <h1> 作为页面标题;<h2..h6> 维护章节路径(section);
    - <p> 与 <li> 各成一段;<pre>/<code> 作为独立"代码段落"(代码块感知:
      与正文分开,保留完整可运行示例,不做 OCR/混切);
    - paragraph 序号跨全文递增。
    """

    def __init__(self, document_id: str, source_url: str = ""):
        super().__init__(convert_charrefs=True)
        self.document_id = document_id
        self.source_url = source_url
        self.title = ""
        self.paragraphs: list[ParsedParagraph] = []

        self._stack: list[str] = []
        self._skip_depth = 0
        self._in_content = False
        self._content_depth = -1
        self._title_done = False
        self._path: list[str] = []
        self._block: str | None = None  # p | pre | li
        self._heading: int | None = None  # h1..h6
        self._buf: list[str] = []
        self._counter = 0

    # ------------------------------------------------------------ 标签事件

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._stack.append(tag)
        if tag in _HTML_SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if not self._in_content and self._is_content_start(tag, attrs):
            self._in_content = True
            self._content_depth = len(self._stack)
            return
        if not self._in_content:
            return
        if tag in ("p", "li", "pre", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush()
            if tag == "pre":
                self._block = "pre"
            elif tag in ("p", "li"):
                self._block = tag
            else:
                self._heading = int(tag[1])

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # <br/>、<hr/> 等自闭合不影响块结构
        pass

    def handle_endtag(self, tag: str) -> None:
        if tag in _HTML_SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            self._stack.pop()
            return
        if self._skip_depth:
            self._stack.pop()
            return
        # 内容区结束(离开 content 根)
        if self._in_content and len(self._stack) == self._content_depth and tag != "html":
            self._flush()
            self._in_content = False
        self._stack.pop()
        if not self._in_content:
            return
        if tag == "p" or (tag == "pre" and self._block == "pre") or (
            tag == "li" and self._block == "li"
        ):
            self._flush()
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") and self._heading == int(tag[1]):
            self._close_heading(int(tag[1]))

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._in_content:
            return
        if self._block is not None or self._heading is not None:
            self._buf.append(data)

    # ------------------------------------------------------------ 内部

    def _is_content_start(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag == "article" or tag == "main":
            return True
        if tag == "div":
            cls = dict(attrs).get("class") or ""
            return "md-content" in cls or cls.strip() == "body"
        return False

    def _flush(self) -> None:
        text = "".join(self._buf)
        self._buf = []
        if self._block:
            self._block = None
        if self._heading:
            self._heading = None
        text = text.replace("¶", "").strip()
        if not text:
            return
        self._counter += 1
        self.paragraphs.append(
            ParsedParagraph(
                document_id=self.document_id,
                section=" / ".join(self._path),
                paragraph=self._counter,
                text=text,
                page="",
                source_url=self.source_url,
            )
        )

    def _close_heading(self, level: int) -> None:
        text = "".join(self._buf)
        self._buf = []
        self._heading = None
        text = text.replace("¶", "").strip()
        if not text:
            return
        if level == 1:
            if not self._title_done:
                self.title = text
                self._title_done = True
            else:
                # 后续 h1:作为新章节根,清空路径
                self._path = []
            return
        # h2..h6 维护 outline;h(n) 对应路径下标 n-2
        idx = level - 2
        if idx <= 0:
            self._path = [text]
        elif len(self._path) >= idx:
            self._path[idx - 1] = text
            del self._path[idx:]
        else:
            self._path.append(text)


class HTMLDocParser:
    """HTML 文档解析(开发者官方文档正文 → 段落 + 代码块)。"""

    def parse(self, html_bytes: bytes, document_id: str, source_url: str = "") -> ParsedDocument:
        handler = _HTMLDocHandler(document_id, source_url)
        try:
            handler.feed(html_bytes.decode("utf-8", errors="replace"))
            handler.close()
        except Exception as exc:  # noqa: BLE001
            raise ParseError(f"HTML 解析失败: {exc}") from exc
        return ParsedDocument(
            document_id=document_id,
            title=handler.title,
            paragraphs=handler.paragraphs,
        )
