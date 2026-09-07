"""fetch_dev_docs.py 抓取器单元测试(离线 mock,不访问网络)。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_dev_docs as fd  # noqa: E402


def _mock_client() -> httpx.Client:
    """MockTransport:按 host/路径返回不同响应。"""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "fastapi.tiangolo.com":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="<html><head><title>First Steps - FastAPI</title></head>"
                "<body><h1>First Steps</h1></body></html>",
            )
        if host == "docs.pydantic.dev":
            return httpx.Response(404, text="not found")
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_extract_title() -> None:
    html = "<html><head><title>  Query Parameters -  FastAPI </title></head></html>"
    assert fd.extract_title(html, "http://x") == "Query Parameters - FastAPI"
    assert fd.extract_title("<html></html>", "https://a.b/c/d/") == "d"


def test_check_domain() -> None:
    assert fd.check_domain("https://fastapi.tiangolo.com/tutorial/") == "fastapi.tiangolo.com"
    with pytest.raises(ValueError, match="非白名单"):
        fd.check_domain("https://evil.example.com/x")


def test_fetch_one_html_ok() -> None:
    with _mock_client() as c:
        html, err, status, size = fd.fetch_one(c, "https://fastapi.tiangolo.com/tutorial/")
    assert status == 200
    assert html and "First Steps" in html
    assert err is None
    assert size > 0


def test_fetch_one_404() -> None:
    with _mock_client() as c:
        html, err, status, _ = fd.fetch_one(c, "https://docs.pydantic.dev/latest/missing/")
    assert html is None
    assert status == 404
    assert "404" in (err or "")


def test_fetch_one_rejects_non_html() -> None:
    with _mock_client() as c:
        html, err, status, _ = fd.fetch_one(c, "https://other.dev/file.pdf")
    assert html is None
    assert "非 HTML" in (err or "")


def test_main_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    """集成:mock 掉网络函数,走完整 main 流程(下载→落盘→manifest)。"""

    def fake_fetch(client, url):  # noqa: ANN001
        if "404" in url:
            return None, "HTTP 404", 404, 0
        return "<html><title>T</title></html>", None, 200, 30

    monkeypatch.setattr(fd, "fetch_one", fake_fetch)
    monkeypatch.setattr(fd, "RATE_SECONDS", 0)

    lst = tmp_path / "list.txt"
    lst.write_text(
        "https://fastapi.tiangolo.com/a/\nhttps://fastapi.tiangolo.com/b-404/\n"
        "https://docs.python.org/3/c/\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["fetch_dev_docs.py", "--list", str(lst), "--out", str(out)],
    )
    rc = fd.main()
    assert rc == 0

    manifest = out / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    ok = [r for r in rows if r["status"] == 200]
    fail = [r for r in rows if r["status"] != 200]
    assert len(ok) == 2
    assert len(fail) == 1
    assert fail[0]["error"] == "HTTP 404"
    # 成功页写入 html 文件,path 指向它
    assert (out / ok[0]["path"]).exists()
