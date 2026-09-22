"""server API 数据模型(Pydantic v2,与前端共享的结构)。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CitationModel(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    title: str = ""
    source_name: str = ""
    section: str = ""
    page: str = ""
    paragraph: int = 0
    source_url: str = ""
    evidence: str = ""
    score: float | None = None


class TraceStep(BaseModel):
    """右侧 RAG 执行链路的一步。"""

    # 链路步骤标识:query_understanding / query_rewrite / hybrid_retrieval
    # / rerank / context / final_generation / tool_call
    key: str
    label: str
    status: str = "done"
    elapsed_s: float | None = None
    detail: dict = Field(default_factory=dict)


class RagReport(BaseModel):
    steps: list[TraceStep] = Field(default_factory=list)
    translated_query: str | None = None
    expanded_terms: list[str] = Field(default_factory=list)
    recall_chunk_ids: list[str] = Field(default_factory=list)
    context_count: int = 0
    total_cost_yuan: float = 0.0
    total_s: float = 0.0
    mode: str = "direct"




class RunInfo(BaseModel):
    run_id: str
    session_id: str
    status: str
    mode: str = "auto"
    question: str = ""
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""


class RunCreated(BaseModel):
    session_id: str
    run_id: str
    status: str = "pending"
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str = "auto"          # auto | direct | agentic
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    refused: bool = False
    mode: str = "direct"
    stop_reason: str | None = None
    report: RagReport | None = None


class SessionUpdate(BaseModel):
    """会话元数据更新(当前仅模式)。"""

    mode: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    title: str
    mode: str = "auto"
    created_at: str
    updated_at: str
    message_count: int = 0


class MessageModel(BaseModel):
    role: str                       # user | assistant
    content: str
    citations: list[CitationModel] = Field(default_factory=list)
    mode: str | None = None
    refused: bool = False
    stop_reason: str | None = None
    report: RagReport | None = None
    created_at: str = ""


class DocumentModel(BaseModel):
    document_id: str
    title: str = ""
    source_name: str = ""
    publication_date: str = ""
    document_type: str = ""
    status: str = ""
    chunk_count: int = 0
    embedding_model: str = ""
    source_url: str = ""
    error: str | None = None
    updated_at: str = ""


class StatsModel(BaseModel):
    total_documents: int
    total_chunks: int
    embedding_model: str
    embedding_dimension: int
    rerank_model: str
    retrieval: dict
    chunking: dict
    qdrant: dict
