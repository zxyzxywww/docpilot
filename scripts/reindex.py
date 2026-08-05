"""重建派生索引(Qdrant + BM25),全部从 SQLite 事实来源 + 原始文档派生。

适用场景:
- embedding 模型 / 维度 / 分块策略变化后(必须重建,见 store.py 校验);
- Qdrant local mode 切换到 Docker server 时(不复用本地存储目录,
  依据 SQLite + manifest + 原始文档确定性重建,约束 3)。

用法:
    python scripts/reindex.py
需要 .env 已配置 SILICONFLOW_API_KEY(embedding 重新调用)。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingest import BM25Index, IngestService, QdrantStore, SQLiteStore  # noqa: E402
from llm import EmbeddingClient, get_api_key, load_config  # noqa: E402


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
    try:
        result = service.rebuild_indexes()
        print(f"重建完成: {result['documents']} 篇 / {result['chunks']} chunks")
        print(f"Qdrant 总点数: {qdrant.count_all()} | BM25 文档数: {bm25.size()}")
        return 0
    finally:
        sqlite.close()
        qdrant.close()


if __name__ == "__main__":
    raise SystemExit(main())
