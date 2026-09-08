"""入库 CLI:解析 → 分块 → embedding → SQLite/Qdrant/BM25 三存储入库。

用法:
    python scripts/ingest.py                     # 处理 manifest 中全部文档(幂等)
    python scripts/ingest.py --only doc1,doc2    # 只处理指定 document_id
    python scripts/ingest.py --delete <doc_id>   # 删除文档(三存储清理)
    python scripts/ingest.py --rebuild           # 仅重建派生索引(Qdrant + BM25)

默认读取 data/raw/manifest.jsonl(合规清单,唯一入库来源)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingest import BM25Index, IngestService, QdrantStore, SQLiteStore  # noqa: E402
from llm import EmbeddingClient, get_api_key, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="DocPilot 文档入库")
    parser.add_argument("--manifest", default="data/raw/manifest.jsonl")
    parser.add_argument("--only", default="", help="逗号分隔的 document_id 白名单")
    parser.add_argument("--delete", default="", help="删除指定 document_id")
    parser.add_argument("--rebuild", action="store_true", help="仅重建派生索引")
    args = parser.parse_args()

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
        if args.delete:
            service.delete_document(args.delete)
            print(f"已删除: {args.delete}")
            return 0

        if args.rebuild:
            result = service.rebuild_indexes()
            print(f"重建完成: {result['documents']} 篇 / {result['chunks']} chunks")
            return 0

        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"manifest 不存在: {manifest_path}")
            return 1
        recs = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        only_ids = {x.strip() for x in args.only.split(",") if x.strip()}
        result = service.ingest_manifest(recs, only_ids or None)
        print(f"入库结果: 成功 {result['ok']} / 失败 {result['failed']} / 跳过 {result['skipped']}")
        for detail in result["detail"]:
            status = "OK " if detail["ok"] else "FAIL"
            extra = f"chunks={detail.get('chunks')}" if detail["ok"] else detail.get("error", "")
            print(f"  [{status}] {detail['document_id']} {extra}")
        qdrant_total = qdrant.count_all()
        print(f"Qdrant 总点数: {qdrant_total} | BM25 文档数: {bm25.size()}")
        return 0 if result["failed"] == 0 else 1
    finally:
        sqlite.close()
        qdrant.close()


if __name__ == "__main__":
    raise SystemExit(main())
