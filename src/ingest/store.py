"""存储层:SQLite(唯一事实来源)+ Qdrant/BM25(可重建派生索引)+ 导入状态机。

设计要点(约束 2/3):
- SQLite 是文档与 chunk 元数据的唯一事实来源;Qdrant 与 BM25 均为可重建派生索引。
- 导入状态机:pending → indexing → ready / failed / deleted。
- 三存储全部写入并通过数量校验后才标记 ready;失败记录 error 可重试。
- embedding 模型或维度变化时,Qdrant 集合尺寸不匹配 → 强制要求重建。
- chunk_id 确定性;Qdrant 内部使用 uuid5(chunk_id) 保证幂等 upsert。
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from .chunker import Chunk
from .parser import ParsedDocument, ParseError, PDFParser, XMLParser

DOC_STATUS = ("pending", "indexing", "ready", "failed", "deleted")

# 固定命名空间,保证 chunk_id -> uuid 确定性
_CHUNK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class IndexRebuildRequiredError(RuntimeError):
    """embedding 模型/维度/分块策略变化,须重建索引。"""


def _chunk_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(_CHUNK_NS, chunk_id))


def _tokenize(text: str) -> list[str]:
    """BM25 分词:英文单词 + 中文单字(大小写归一)。"""
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


# ------------------------------------------------------------------ SQLite

class SQLiteStore:
    """文档与 chunk 元数据的唯一事实来源。"""

    def __init__(self, path: str | Path):
        self._path = str(path)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False:Agent 工具在线程池中执行,需允许跨线程读;
        # 本项目中同一时刻仅一个工具在跑(ThreadPoolExecutor max_workers=1),无并发写风险。
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id     TEXT PRIMARY KEY,
                pmcid           TEXT,
                title           TEXT,
                authors         TEXT,          -- JSON array
                journal         TEXT,
                doi             TEXT,
                source_url      TEXT,
                license         TEXT,
                publication_date TEXT,
                document_type   TEXT,
                sha256          TEXT,
                local_path      TEXT,
                status          TEXT NOT NULL,
                error           TEXT,
                embedding_model TEXT,
                embedding_dim   INTEGER,
                chunk_count     INTEGER DEFAULT 0,
                updated_at      TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                section     TEXT,
                page        TEXT,
                paragraph   INTEGER,
                text        TEXT NOT NULL,
                source_url  TEXT,
                token_count INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            """
        )
        self._conn.commit()

    def upsert_document(self, rec: dict[str, Any], status: str = "pending") -> None:
        self._conn.execute(
            """
            INSERT INTO documents (
                document_id, pmcid, title, authors, journal, doi, source_url, license,
                publication_date, document_type, sha256, local_path, status, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(document_id) DO UPDATE SET
                title=excluded.title, authors=excluded.authors, journal=excluded.journal,
                doi=excluded.doi, source_url=excluded.source_url, license=excluded.license,
                publication_date=excluded.publication_date, document_type=excluded.document_type,
                sha256=excluded.sha256, local_path=excluded.local_path,
                status=excluded.status, error=NULL, updated_at=excluded.updated_at
            """,
            (
                rec["document_id"],
                rec.get("pmcid", ""),
                rec.get("title", ""),
                json.dumps(rec.get("authors", []), ensure_ascii=False),
                rec.get("journal", ""),
                rec.get("doi", ""),
                rec.get("source_url", ""),
                rec.get("license", ""),
                rec.get("publication_date", ""),
                rec.get("document_type", ""),
                rec.get("sha256", ""),
                rec.get("local_path", ""),
                status,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        self._conn.commit()

    def set_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        if status not in DOC_STATUS:
            raise ValueError(f"非法状态: {status}")
        self._conn.execute(
            "UPDATE documents SET status=?, error=?, updated_at=? WHERE document_id=?",
            (status, error, time.strftime("%Y-%m-%d %H:%M:%S"), doc_id),
        )
        self._conn.commit()

    def set_embedding_meta(self, doc_id: str, model: str, dim: int) -> None:
        self._conn.execute(
            "UPDATE documents SET embedding_model=?, embedding_dim=? WHERE document_id=?",
            (model, dim, doc_id),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE document_id=?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_documents(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE status=?", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM documents").fetchall()
        return [dict(r) for r in rows]

    def replace_chunks(self, chunks: list[Chunk]) -> None:
        """先删后插(事务),保证幂等;SQLite 是事实来源,chunk 元数据必须准确。"""
        if not chunks:
            return
        doc_id = chunks[0].document_id
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            self._conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, section, page, paragraph, text, source_url, token_count
                )
                VALUES (?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.section,
                        c.page,
                        c.paragraph,
                        c.text,
                        c.source_url,
                        c.token_count,
                    )
                    for c in chunks
                ],
            )
            self._conn.execute(
                "UPDATE documents SET chunk_count=? WHERE document_id=?",
                (len(chunks), doc_id),
            )

    def count_chunks(self, doc_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE document_id=?", (doc_id,)
        ).fetchone()
        return int(row["n"])

    def get_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id=? ORDER BY paragraph", (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            self._conn.execute("DELETE FROM documents WHERE document_id=?", (doc_id,))

    def all_ready_documents(self) -> list[dict[str, Any]]:
        return self.list_documents(status="ready")

    def embedding_signature(self, doc_id: str) -> tuple[str | None, int | None]:
        doc = self.get_document(doc_id)
        if not doc:
            return None, None
        return doc.get("embedding_model"), doc.get("embedding_dim")

    def close(self) -> None:
        self._conn.close()


# ------------------------------------------------------------------ Qdrant

class QdrantStore:
    """Qdrant 向量索引(派生索引,可重建)。"""

    def __init__(self, config: Any, dimension: int):
        self._collection = config.collection
        self._dimension = dimension
        if config.mode == "local":
            Path(config.path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=config.path)
        elif config.mode == "docker":
            self._client = QdrantClient(url=config.docker_url)
        else:
            raise ValueError(f"未知 Qdrant mode: {config.mode}")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimension, distance=models.Distance.COSINE
                ),
            )
            return
        info = self._client.get_collection(self._collection)
        vectors_cfg = info.config.params.vectors
        if vectors_cfg is None:
            raise IndexRebuildRequiredError("Qdrant 集合无向量配置,须重建索引")
        if isinstance(vectors_cfg, dict):  # 命名向量配置,取首个
            size = next(iter(vectors_cfg.values())).size
        else:
            size = vectors_cfg.size
        if size != self._dimension:
            raise IndexRebuildRequiredError(
                f"Qdrant 集合维度 {size} != 配置维度 {self._dimension};"
                "embedding 模型/维度变化必须重建索引(reindex.py)"
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        points = [
            models.PointStruct(
                id=_chunk_uuid(c.chunk_id),
                vector=v,
                payload={
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "section": c.section,
                    "page": c.page,
                    "paragraph": c.paragraph,
                    "text": c.text,
                    "source_url": c.source_url,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self._collection, points=points)

    def query(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """向量查询:返回 top_k 个点的 payload(按相似度降序)。"""
        hits = self._client.query_points(
            self._collection, query=vector, limit=top_k, with_payload=True
        )
        payloads: list[dict[str, Any]] = []
        for h in hits.points:
            if h.payload is not None:
                payloads.append(h.payload)
        return payloads

    def count_document(self, doc_id: str) -> int:
        result = self._client.count(
            self._collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=doc_id)
                    )
                ]
            ),
            exact=True,
        )
        return int(result.count)

    def count_all(self) -> int:
        return int(self._client.count(self._collection, exact=True).count)

    def delete_document(self, doc_id: str) -> None:
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
        )

    def drop_and_recreate(self) -> None:
        """重建派生索引前清空集合(维度不变)。"""
        self._client.delete_collection(self._collection)
        self._ensure_collection()

    def close(self) -> None:
        self._client.close()


# ------------------------------------------------------------------ BM25

@dataclass
class BM25Hit:
    chunk_id: str
    score: float


class BM25Index:
    """手写 BM25 关键词索引(派生索引,全库重建,无第三方依赖)。

    标准 BM25 公式:score = Σ idf·(f·(k1+1)) / (f + k1·(1−b+b·dl/avgdl))
    idf = ln((N − df + 0.5) / (df + 0.5) + 1);k1=1.5, b=0.75。
    分词:_tokenize(英文单词 + 中文单字)。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._ids: list[str] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("ids 与 texts 数量不一致")
        self._ids = list(chunk_ids)
        n = len(texts)
        nd: Counter[str] = Counter()
        total_len = 0
        self._doc_freqs = []
        self._doc_len = []
        for text in texts:
            freq = Counter(_tokenize(text))
            self._doc_freqs.append(freq)
            self._doc_len.append(sum(freq.values()))
            total_len += sum(freq.values())
            nd.update(freq.keys())
        self._avgdl = total_len / n if n else 0.0
        self._idf = {
            w: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for w, df in nd.items()
        }

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        if not self._ids:
            return []
        q_tokens = set(_tokenize(query))
        scores: list[float] = []
        for i, freq in enumerate(self._doc_freqs):
            dl = self._doc_len[i]
            score = 0.0
            for token in q_tokens:
                f = freq.get(token, 0)
                if f == 0:
                    continue
                idf = self._idf.get(token, 0.0)
                score += idf * (f * (self._k1 + 1)) / (
                    f + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                )
            scores.append(score)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            BM25Hit(chunk_id=self._ids[i], score=scores[i])
            for i in order
            if scores[i] > 0
        ]

    def size(self) -> int:
        return len(self._ids)


# ------------------------------------------------------------------ IngestService

class IngestService:
    """入库编排:解析 → 分块 → embedding → 三存储 → 状态机 → 数量校验。"""

    def __init__(
        self,
        sqlite: SQLiteStore,
        qdrant: QdrantStore,
        bm25: BM25Index,
        embedder: Any,
        chunk_size_tokens: int = 600,
        chunk_overlap_tokens: int = 100,
    ):
        self.sqlite = sqlite
        self.qdrant = qdrant
        self.bm25 = bm25
        self.embedder = embedder
        self._chunk_size = chunk_size_tokens
        self._chunk_overlap = chunk_overlap_tokens
        self._xml_parser = XMLParser()
        self._pdf_parser = PDFParser()

    def ingest_manifest(
        self, manifest_recs: list[dict[str, Any]], only_ids: set[str] | None = None
    ) -> dict[str, Any]:
        results: dict[str, Any] = {"ok": 0, "failed": 0, "skipped": 0, "detail": []}
        for rec in manifest_recs:
            doc_id = rec["document_id"]
            if only_ids and doc_id not in only_ids:
                results["skipped"] += 1
                continue
            outcome = self.ingest_one(rec)
            results["detail"].append(outcome)
            if outcome["ok"]:
                results["ok"] += 1
            else:
                results["failed"] += 1
        return results

    def ingest_one(self, rec: dict[str, Any]) -> dict[str, Any]:
        doc_id = rec["document_id"]
        self.sqlite.upsert_document(rec, status="indexing")
        try:
            parsed = self._parse(rec)
            chunks = self._chunk(rec, parsed)
            if not chunks:
                raise ParseError(f"文档 {doc_id} 未解析出任何段落")
            vectors = self.embedder.embed([c.text for c in chunks])
            self.qdrant.upsert(chunks, vectors)
            self.sqlite.replace_chunks(chunks)
            self.sqlite.set_embedding_meta(
                doc_id, self.embedder.config.model, self.embedder.config.dimension
            )
            self._verify(doc_id, len(chunks))
            self.sqlite.set_status(doc_id, "ready")
            self.rebuild_bm25()
            return {"document_id": doc_id, "ok": True, "chunks": len(chunks)}
        except Exception as exc:
            self.sqlite.set_status(doc_id, "failed", error=str(exc))
            return {"document_id": doc_id, "ok": False, "error": str(exc)}

    def _parse(self, rec: dict[str, Any]) -> ParsedDocument:
        path = Path(rec["local_path"])
        doc_id = rec["document_id"]
        source_url = rec.get("source_url", "")
        if path.suffix.lower() == ".xml":
            return self._xml_parser.parse(path.read_bytes(), doc_id, source_url)
        if path.suffix.lower() == ".pdf":
            return self._pdf_parser.parse(str(path), doc_id, source_url)
        raise ParseError(f"不支持的文档类型: {path.suffix}(仅支持 .xml / .pdf)")

    def _chunk(self, rec: dict[str, Any], parsed: ParsedDocument) -> list[Chunk]:
        from .chunker import chunk_paragraphs

        return chunk_paragraphs(
            rec["document_id"],
            parsed.paragraphs,
            self._chunk_size,
            self._chunk_overlap,
        )

    def _verify(self, doc_id: str, expected: int) -> None:
        """三存储数量校验:SQLite == 期望 == Qdrant。"""
        n_sqlite = self.sqlite.count_chunks(doc_id)
        n_qdrant = self.qdrant.count_document(doc_id)
        if n_sqlite != expected:
            raise RuntimeError(f"SQLite chunk 数 {n_sqlite} != 期望 {expected}")
        if n_qdrant != expected:
            raise RuntimeError(f"Qdrant 点数 {n_qdrant} != 期望 {expected}")

    def delete_document(self, doc_id: str) -> None:
        self.sqlite.set_status(doc_id, "deleted")
        self.qdrant.delete_document(doc_id)
        self.sqlite.delete_document(doc_id)
        self.rebuild_bm25()

    def rebuild_bm25(self) -> None:
        """从 SQLite(事实来源)重建 BM25 派生索引。"""
        ids: list[str] = []
        texts: list[str] = []
        for doc in self.sqlite.all_ready_documents():
            for chunk in self.sqlite.get_chunks(doc["document_id"]):
                ids.append(chunk["chunk_id"])
                texts.append(chunk["text"])
        self.bm25.build(ids, texts)

    def rebuild_indexes(self) -> dict[str, Any]:
        """reindex:清空并重建 Qdrant + BM25,全部从 SQLite 事实来源派生。

        Qdrant local → docker 切换时,不复用本地存储目录,用本方法确定性重建。
        """
        docs = self.sqlite.all_ready_documents()
        self.qdrant.drop_and_recreate()
        total_chunks = 0
        for doc in docs:
            doc_id = doc["document_id"]
            chunks = self.sqlite.get_chunks(doc_id)
            if not chunks:
                continue
            from .chunker import Chunk

            chunk_objs = [
                Chunk(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    section=c["section"] or "",
                    page=c["page"] or "",
                    paragraph=c["paragraph"] or 0,
                    text=c["text"],
                    source_url=c["source_url"] or "",
                    token_count=c["token_count"] or 0,
                )
                for c in chunks
            ]
            vectors = self.embedder.embed([c.text for c in chunk_objs])
            self.qdrant.upsert(chunk_objs, vectors)
            total_chunks += len(chunk_objs)
        self.rebuild_bm25()
        return {"documents": len(docs), "chunks": total_chunks}
