"""ingest 包:文档解析 + 分块 + 入库编排(解析 → 分块 → embedding → 三存储)。"""

from .chunker import Chunk, chunk_paragraphs
from .parser import (
    ParsedDocument,
    ParsedParagraph,
    ParseError,
    PDFParser,
    ScannedPDFError,
    XMLParser,
)
from .store import (
    DOC_STATUS,
    BM25Hit,
    BM25Index,
    IndexRebuildRequiredError,
    IngestService,
    QdrantStore,
    SQLiteStore,
)

__all__ = [
    "BM25Hit",
    "BM25Index",
    "Chunk",
    "DOC_STATUS",
    "IndexRebuildRequiredError",
    "IngestService",
    "ParseError",
    "ParsedDocument",
    "ParsedParagraph",
    "PDFParser",
    "QdrantStore",
    "ScannedPDFError",
    "SQLiteStore",
    "XMLParser",
    "chunk_paragraphs",
]
