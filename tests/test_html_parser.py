"""HTMLDocParser 单元测试(代码块感知/正文定位/导航忽略)。"""

from __future__ import annotations

from ingest.parser import HTMLDocParser

FASTAPI_LIKE = """<!DOCTYPE html>
<html><head><title>Dependencies - FastAPI</title></head>
<body>
<nav class="md-nav"><p>Sidebar menu item</p></nav>
<article class="md-content md-typeset">
<h1>Dependencies¶</h1>
<p>FastAPI has a very powerful dependency injection system.</p>
<h2>What are Dependencies</h2>
<p>Dependencies let you share common logic.</p>
<div class="highlight"><pre><code>from fastapi import Depends, FastAPI</code></pre></div>
<h2>Recap</h2>
<p>That is the gist of dependencies.</p>
</article>
<footer><p>Copyright</p></footer>
</body></html>"""


def test_html_doc_parses_structure() -> None:
    doc = HTMLDocParser().parse(FASTAPI_LIKE.encode(), "doc1", "https://fastapi.tiangolo.com/")
    assert doc.title == "Dependencies"
    assert len(doc.paragraphs) == 4

    texts = [p.text for p in doc.paragraphs]
    sections = [p.section for p in doc.paragraphs]
    # 正文段落与 section 正确
    assert "powerful dependency injection system" in texts[0]
    assert "What are Dependencies" in sections[1]
    # 代码块作为独立段落保留(代码块感知:不被正文吞并、无 nav 干扰)
    code_paras = [p.text for p in doc.paragraphs if "from fastapi import" in p.text]
    assert code_paras and "from fastapi import Depends" in code_paras[0]
    # 导航与页脚内容必须被忽略
    joined = " ".join(texts)
    assert "Sidebar menu item" not in joined
    assert "Copyright" not in joined
    # 段落内无 pilcrow 残留
    assert "\u00b6" not in joined


def test_html_python_docs_body_container() -> None:
    html = """<html><body>
    <div class="body" role="main">
    <h1>4. More Control Flow Tools</h1>
    <p>Besides the while statement just introduced.</p>
    <h2>4.1 if Statements</h2>
    <p>Perhaps the most well-known statement type is the if statement.</p>
    <div class="highlight"><pre>x = 1</pre></div>
    </div>
    </body></html>"""
    doc = HTMLDocParser().parse(html.encode(), "d2", "")
    assert doc.title == "4. More Control Flow Tools"
    assert [p.text for p in doc.paragraphs][1].startswith("Perhaps")
    assert any("x = 1" in p.text for p in doc.paragraphs)
    # h2 section 生效
    assert any(p.section == "4.1 if Statements" for p in doc.paragraphs)
