"""server API 端点集成测试(fake service / fake run_chat,不触真实 LLM)。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import server.main as main_mod
from server.main import app
from server.rag_service import RagEngine
from server.schemas import ChatResponse, RagReport
from server.session_store import SessionStore


@pytest.fixture()
def fake_service(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """假的 RagEngine.get_service(只提供端点用到的成员)。"""
    from llm.config import load_config

    cfg = load_config()
    cfg.database.sqlite_path = str(tmp_path / "docpilot_test.db")

    docs = [
        {
            "document_id": "doc1", "title": "测试文档 A", "journal": "Test Journal",
            "publication_date": "2024", "document_type": "xml", "status": "ready",
            "chunk_count": 12, "embedding_model": "bge-m3", "source_url": "http://x",
            "error": None, "updated_at": "2026-01-01",
        }
    ]
    sqlite = SimpleNamespace(
        list_documents=lambda: list(docs),
        get_document=lambda did: docs[0] if did == "doc1" else None,
        get_chunks=lambda did: [{"chunk_id": "doc1_c0001", "text": "x"}],
        upsert_document=lambda rec, status=None: None,
    )
    svc = SimpleNamespace(config=cfg, sqlite=sqlite)

    def _get_service():
        return svc

    monkeypatch.setattr(RagEngine, "get_service", staticmethod(_get_service))
    return svc


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def fake_store(tmp_path: Path, monkeypatch) -> SessionStore:
    store = SessionStore(tmp_path / "chat.db")
    monkeypatch.setattr(main_mod, "_get_store", lambda: store)
    return store


def _fake_chat_response(**kw) -> ChatResponse:
    base = dict(
        session_id="", answer="这是假回答。", citations=[], refused=False,
        mode="direct", stop_reason="final",
        report=RagReport(steps=[], mode="direct", total_s=0.1),
    )
    base.update(kw)
    return ChatResponse(**base)


def _send_and_wait(
    client: TestClient, question: str, mode: str = "direct", session_id: str | None = None
) -> dict:
    """POST /api/chat(立即返回 run)→ 轮询 run 至 done(后台线程执行)。"""
    import time as _t

    body: dict = {"question": question, "mode": mode}
    if session_id:
        body["session_id"] = session_id
    resp = client.post("/api/chat", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "pending" and data["run_id"]
    last = None
    for _ in range(200):
        last = client.get(f"/api/runs/{data['run_id']}").json()
        if last["status"] in ("done", "failed"):
            assert last["status"] == "done", f"run failed: {last.get('error')}"
            return data
        _t.sleep(0.05)
    raise AssertionError(f"run 未在超时内完成,最终状态: {last}")



@pytest.fixture()
def fake_run_chat(monkeypatch):
    """把后端 RAG/LLM 生成替换为固定响应;task_runner 后台线程调用它。"""

    from server import rag_service as rs_mod

    def _run(question: str, mode: str = "auto") -> ChatResponse:
        return _fake_chat_response()

    monkeypatch.setattr(rs_mod, "run_chat", _run)
    return _run


# ---------------------------------------------------------------- 端点


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_chat_creates_session_and_stores_messages(
    client: TestClient, fake_service, fake_store, fake_run_chat
) -> None:
    data = _send_and_wait(client, "什么是 RAG?", mode="direct")
    sid = data["session_id"]
    # 会话标题取自首个问题
    sess = fake_store.get_session(sid)
    assert sess["title"].startswith("什么是 RAG")
    # user 先落库;后台完成后 assistant 写回同一会话
    msgs = fake_store.list_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"], f"msgs={msgs}"
    assert msgs[1]["extra"]["run_id"] == data["run_id"]


def test_chat_reuses_session(
    client: TestClient, fake_service, fake_store, fake_run_chat
) -> None:
    sid = fake_store.create_session()["session_id"]
    data = _send_and_wait(client, "追问", mode="auto", session_id=sid)
    assert data["session_id"] == sid
    msgs = fake_store.list_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_chat_unknown_session_404(
    client: TestClient, fake_service, fake_store, fake_run_chat
) -> None:
    resp = client.post(
        "/api/chat", json={"question": "hi", "mode": "auto", "session_id": "nonexistent"}
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------- 端点


def test_sessions_and_messages_endpoints(
    client: TestClient, fake_service, fake_store
) -> None:
    sid = fake_store.create_session("标题A")["session_id"]
    fake_store.add_message(sid, "user", "问题")
    fake_store.add_message(sid, "assistant", "回答", {"citations": [], "mode": "direct"})

    sessions = client.get("/api/sessions").json()
    assert sessions[0]["session_id"] == sid
    assert sessions[0]["message_count"] == 2

    msgs = client.get(f"/api/sessions/{sid}/messages").json()
    assert len(msgs) == 2
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["mode"] == "direct"

    assert client.delete(f"/api/sessions/{sid}").json() == {"ok": True}
    assert client.get(f"/api/sessions/{sid}/messages").status_code == 404


def test_documents_list(client: TestClient, fake_service) -> None:
    docs = client.get("/api/documents").json()
    assert docs[0]["document_id"] == "doc1"
    assert docs[0]["chunk_count"] == 12


def test_documents_delete_404(client: TestClient, fake_service) -> None:
    assert client.delete("/api/documents/nonexistent").status_code == 404


def test_upload_rejects_bad_type(client: TestClient, fake_service) -> None:
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "不支持的文件类型" in resp.json()["detail"]


def test_stats(client: TestClient, fake_service) -> None:
    body = client.get("/api/stats").json()
    assert body["total_documents"] == 1
    assert body["total_chunks"] == 12
    assert body["retrieval"]["dense_top_k"] > 0


def test_eval_summary(client: TestClient, fake_service) -> None:
    body = client.get("/api/eval/summary").json()
    # 测试运行于项目根(存在真实 eval_report.json)
    assert "available" in body


def test_upload_html_accepted(client: TestClient, fake_service, monkeypatch) -> None:
    """修复 A1:HTML 上传应通过类型白名单并走 HTMLDocParser 路径。"""
    from types import SimpleNamespace as NS

    # fake service 补齐上传路径所需成员
    fake_service.qdrant = NS()
    fake_service.bm25 = NS()
    fake_service.embedder = NS()

    class FakeHTMLParser:
        def parse(self, data, doc_id, source_url=""):
            return NS(title="Test Page", paragraphs=[{"section": "s"}])

    class FakeIngest:
        def __init__(self, *a, **kw):
            pass

        def ingest_one(self, rec):
            return {"ok": True, "document_id": rec["document_id"], "chunks": 1}

    monkeypatch.setattr(main_mod, "HTMLDocParser", FakeHTMLParser)
    monkeypatch.setattr(main_mod, "IngestService", FakeIngest)
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("doc.html", b"<html><body><h1>T</h1><p>x</p></body></html>", "text/html")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is True


# ------------------------------------------------- 会话级状态(问题 1/2 回归)

def test_create_session_endpoint_default_mode(client, fake_service, fake_store) -> None:
    body = client.post("/api/sessions").json()
    assert body["session_id"]
    assert body["mode"] == "auto"
    assert body["title"] == "新对话"
    assert fake_store.get_session(body["session_id"]) is not None


def test_patch_session_mode_persisted(client, fake_service, fake_store) -> None:
    sid = fake_store.create_session()["session_id"]
    r = client.patch(f"/api/sessions/{sid}", json={"mode": "agentic"})
    assert r.status_code == 200
    assert r.json()["mode"] == "agentic"
    # 持久化:重新读库仍为 agentic
    assert fake_store.get_session(sid)["mode"] == "agentic"
    # 会话列表返回 mode
    listed = client.get("/api/sessions").json()
    assert any(s["session_id"] == sid and s["mode"] == "agentic" for s in listed)
    # 未提供 mode → 不改动
    assert client.patch(f"/api/sessions/{sid}", json={}).status_code == 200


def test_patch_session_404(client, fake_service, fake_store) -> None:
    assert client.patch("/api/sessions/nope", json={"mode": "direct"}).status_code == 404
