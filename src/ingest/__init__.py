"""ingest 包:文档解析 + 分块 + 入库编排(解析 → 分块 → embedding → 三存储)。"""

from .chunker import Chunk, chunk_paragraphs
from .parser import (
    HTMLDocParser,
    PDFParser,
    ParseError,
    ParsedDocument,
    ParsedParagraph,
    ScannedPDFError,
    XMLParser,
)
from .store import (
    BM25Hit,
    BM25Index,
    DOC_STATUS,
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
    "HTMLDocParser",
    "IndexRebuildRequiredError",
    "IngestService",
    "PDFParser",
    "ParseError",
    "ParsedDocument",
    "ParsedParagraph",
    "QdrantStore",
    "ScannedPDFError",
    "SQLiteStore",
    "XMLParser",
    "chunk_paragraphs",
]
