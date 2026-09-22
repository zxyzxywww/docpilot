"""DocPilot:重建开发文档语料(清空旧语料后,重新入库
FastAPI/Pydantic/SQLAlchemy/Python 官方开发文档 HTML)。

适用:DocPilot 数据域重建/换库;早期数据域迁移的演进说明见 README 历史章节。

流程:
    1) wipe:清空 SQLite documents/chunks + Qdrant collection(事实来源与派生索引同步);
    2) ingest:逐页解析(HTMLDocParser,正文+代码块感知分块)→ embedding → 三存储 → ready。

用法:
    python scripts/rebuild_docs.py
需要 .env 配置 SILICONFLOW_API_KEY(embedding);Qdrant 存储模式随 config.yaml。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingest import BM25Index, IngestService, QdrantStore, SQLiteStore  # noqa: E402
from llm import EmbeddingClient, get_api_key, load_config  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "devdocs"
MANIFEST = RAW_DIR / "manifest.jsonl"


def normalize_rec(row: dict, raw_dir: Path) -> dict:
    """manifest 行 → ingest rec(确定性 document_id=url sha1 前 12)。"""
    url = row["url"]
    doc_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return {
        "document_id": doc_id,
        "source_id": "",
        "title": row.get("title", ""),
        "authors": [],
        "source_name": row["domain"],
        "doi": "",
        "source_url": url,
        "license": "",
        "publication_date": "",
        "document_type": "web",
        "sha256": row.get("sha256", ""),
        "local_path": str(raw_dir / row["path"]),
    }


def main() -> int:
    config = load_config()
    embedder = EmbeddingClient(config.embedding, get_api_key("siliconflow"))
    sqlite = SQLiteStore(config.database.sqlite_path)
    qdrant = QdrantStore(config.database.qdrant, config.embedding.dimension)
    bm25 = BM25Index()
    service = IngestService(
        sqlite,
        qdrant,
        bm25,
        embedder,
        chunk_size_tokens=config.chunking.chunk_size_tokens,
        chunk_overlap_tokens=config.chunking.chunk_overlap_tokens,
    )

    recs = [
        json.loads(ln)
        for ln in MANIFEST.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    print(f"[rebuild_docs] manifest 共 {len(recs)} 页,开始清空旧语料…")
    sqlite.wipe()
    qdrant.clear()
    print("[rebuild_docs] 已清空 SQLite + Qdrant")

    normalized = [normalize_rec(r, RAW_DIR) for r in recs]
    result = service.ingest_manifest(normalized)
    print(f"[rebuild_docs] 入库完成:成功 {result['ok']} / 失败 {result['failed']}")
    for d in result["detail"]:
        if not d["ok"]:
            print(f"  FAIL {d.get('document_id')}: {d.get('error', '')[:120]}")
    ready = [d for d in sqlite.list_documents(status="ready")]
    total_chunks = sum(d.get("chunk_count", 0) for d in ready)
    print(f"[rebuild_docs] SQLite ready 文档 {len(ready)},chunks {total_chunks}")
    print(f"[rebuild_docs] Qdrant 点数 {qdrant.count_all()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
