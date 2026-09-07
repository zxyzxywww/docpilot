"""server API 主入口:FastAPI 应用与全部 REST 端点。

启动(项目根目录下):
    uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any

# 保证可解析 src/(RAG 核心)与 server 内部模块
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src", _PROJECT_ROOT / "server"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi import FastAPI, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.service import Service  # noqa: E402 (src/app 暂留,Phase 5 迁移后改为本地组装)
from ingest import IngestService, PDFParser, XMLParser  # noqa: E402
from ingest.parser import ScannedPDFError  # noqa: E402

from .rag_service import RagEngine, run_chat  # noqa: E402
from .schemas import (  # noqa: E402
    ChatRequest,
    ChatResponse,
    DocumentModel,
    MessageModel,
    SessionInfo,
    StatsModel,
)
from .session_store import SessionStore  # noqa: E402

ALLOWED_SUFFIXES = {".xml", ".pdf"}
MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


_STORE: SessionStore | None = None


def _get_store() -> SessionStore:
    """会话库(lazy,与 service 共用 SQLite 文件)。"""
    global _STORE
    svc = RagEngine.get_service()
    if _STORE is None:
        _STORE = SessionStore(svc.config.database.sqlite_path)
    return _STORE


app = FastAPI(title="MediDoc API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本地开发;生产可收窄为前端地址
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ------------------------------------------------------------ Chat / 会话


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    store = _get_store()
    if req.session_id and store.get_session(req.session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    session_id = req.session_id or store.create_session()["session_id"]

    # 首条问题作为会话标题
    sess = store.get_session(session_id)
    if sess and sess["title"] == "新对话":
        title = req.question.strip().replace("\n", " ")[:24]
        if title:
            store.update_session_title(session_id, title)

    resp = run_chat(req.question, mode=req.mode)
    resp.session_id = session_id

    store.add_message(session_id, "user", req.question, {"mode": resp.mode})
    store.add_message(
        session_id,
        "assistant",
        resp.answer,
        {
            "citations": [c.model_dump() for c in resp.citations],
            "mode": resp.mode,
            "refused": resp.refused,
            "stop_reason": resp.stop_reason,
            "report": resp.report.model_dump() if resp.report else None,
        },
    )
    return resp


@app.get("/api/sessions", response_model=list[SessionInfo])
def list_sessions() -> list[dict[str, Any]]:
    return _get_store().list_sessions()


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageModel])
def get_messages(session_id: str) -> list[dict[str, Any]]:
    store = _get_store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    out: list[dict[str, Any]] = []
    for m in store.list_messages(session_id):
        extra = m.get("extra") or {}
        item: dict[str, Any] = {
            "role": m["role"],
            "content": m["content"],
            "created_at": m["created_at"],
        }
        if m["role"] == "assistant":
            item.update(
                {
                    "citations": extra.get("citations", []),
                    "mode": extra.get("mode"),
                    "refused": extra.get("refused", False),
                    "stop_reason": extra.get("stop_reason"),
                    "report": extra.get("report"),
                }
            )
        out.append(item)
    return out


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    _get_store().delete_session(session_id)
    return {"ok": True}


# ------------------------------------------------------------ 文档库


def _service() -> Service:
    return RagEngine.get_service()


@app.get("/api/documents", response_model=list[DocumentModel])
def list_documents() -> list[dict[str, Any]]:
    svc = _service()
    docs = svc.sqlite.list_documents()
    out = []
    for d in docs:
        out.append(
            {
                "document_id": d["document_id"],
                "title": d.get("title", ""),
                "journal": d.get("journal", ""),
                "publication_date": d.get("publication_date", ""),
                "document_type": d.get("document_type", ""),
                "status": d.get("status", ""),
                "chunk_count": d.get("chunk_count", 0),
                "embedding_model": d.get("embedding_model", ""),
                "source_url": d.get("source_url", ""),
                "error": d.get("error"),
                "updated_at": d.get("updated_at", ""),
            }
        )
    return out


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, bool]:
    svc = _service()
    if svc.sqlite.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    ingest = IngestService(
        svc.sqlite, svc.qdrant, svc.bm25, svc.embedder,
        chunk_size_tokens=svc.config.chunking.chunk_size_tokens,
        chunk_overlap_tokens=svc.config.chunking.chunk_overlap_tokens,
    )
    ingest.delete_document(document_id)
    return {"ok": True}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile) -> dict[str, Any]:
    name = file.filename or "upload"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix},仅支持 {sorted(ALLOWED_SUFFIXES)}",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"文件超过 {MAX_UPLOAD_MB}MB 限制")

    # 安全:只取 basename,拒绝路径分隔符
    safe_name = Path(name).name
    if not safe_name or safe_name != name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="非法文件名")

    svc = _service()
    raw_sha = hashlib.sha256(data).hexdigest()
    doc_id = f"upload_{raw_sha[:12]}"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / safe_name
            path.write_bytes(data)
            if suffix == ".xml":
                parsed = XMLParser().parse(path.read_bytes(), doc_id, source_url="")
            else:
                try:
                    parsed = PDFParser().parse(str(path), doc_id, source_url="")
                except ScannedPDFError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not parsed.paragraphs:
                raise HTTPException(status_code=400, detail="文档未解析出有效内容")

            rec = {
                "document_id": doc_id,
                "pmcid": "",
                "title": parsed.title or safe_name,
                "authors": [],
                "journal": "用户上传",
                "doi": "",
                "source_url": "",
                "license": "",
                "publication_date": "",
                "document_type": "upload",
                "sha256": raw_sha,
                "local_path": str(path),
            }
            svc.sqlite.upsert_document(rec, status="pending")
            ingest = IngestService(
                svc.sqlite, svc.qdrant, svc.bm25, svc.embedder,
                chunk_size_tokens=svc.config.chunking.chunk_size_tokens,
                chunk_overlap_tokens=svc.config.chunking.chunk_overlap_tokens,
            )
            outcome = ingest.ingest_one(rec)
            if not outcome["ok"]:
                raise HTTPException(status_code=500, detail=f"入库失败: {outcome.get('error')}")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc

    return {"document_id": doc_id, "title": safe_name, "ok": True}


# ------------------------------------------------------------ 统计 / 评估


@app.get("/api/stats", response_model=StatsModel)
def stats() -> dict[str, Any]:
    svc = _service()
    cfg = svc.config
    docs = svc.sqlite.list_documents()
    ready = [d for d in docs if d.get("status") == "ready"]
    return {
        "total_documents": len(ready),
        "total_chunks": sum(d.get("chunk_count", 0) for d in ready),
        "embedding_model": cfg.embedding.model,
        "embedding_dimension": cfg.embedding.dimension,
        "rerank_model": cfg.rerank.model,
        "retrieval": {
            "dense_top_k": cfg.retrieval.dense_top_k,
            "bm25_top_k": cfg.retrieval.bm25_top_k,
            "rrf_k": cfg.retrieval.rrf_k,
            "rrf_top_k": cfg.retrieval.rrf_top_k,
            "rerank_top_k": cfg.retrieval.rerank_top_k,
            "final_context_k": cfg.retrieval.final_context_k,
        },
        "chunking": {
            "chunk_size_tokens": cfg.chunking.chunk_size_tokens,
            "chunk_overlap_tokens": cfg.chunking.chunk_overlap_tokens,
        },
        "qdrant": {"mode": cfg.database.qdrant.mode, "collection": cfg.database.qdrant.collection},
    }


@app.get("/api/eval/summary")
def eval_summary() -> dict[str, Any]:
    """读取预计算的评估报告(data/eval/eval_report.json);不存在则返回提示。"""
    report_path = _PROJECT_ROOT / "data" / "eval" / "eval_report.json"
    if not report_path.exists():
        return {"available": False, "message": "评估报告尚未生成:python scripts/eval_report.py"}
    import json

    return {"available": True, **json.loads(report_path.read_text(encoding="utf-8"))}
