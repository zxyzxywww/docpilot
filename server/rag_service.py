"""server 问答编排:组装 service(现有 RAG 核心)并把执行过程整理为 6 步
RagReport 供前端右侧链路面板展示。RAG 核心模块不改动,只调用公开接口。"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from .schemas import ChatResponse, CitationModel, RagReport, TraceStep  # noqa: E402
from .service_runtime import Service, build_service  # noqa: E402


def _citations_to_models(citations: list[Any]) -> list[CitationModel]:
    """把 Citation dataclass 转为 API 模型。"""
    return [
        CitationModel(
            index=getattr(c, "index", 0),
            chunk_id=getattr(c, "chunk_id", ""),
            document_id=getattr(c, "document_id", ""),
            title=getattr(c, "title", ""),
            source_name=getattr(c, "source_name", ""),
            section=getattr(c, "section", ""),
            page=getattr(c, "page", ""),
            paragraph=getattr(c, "paragraph", 0),
            source_url=getattr(c, "source_url", ""),
            evidence=getattr(c, "evidence", ""),
        )
        for c in citations
    ]


class RagEngine:
    """服务单例 + run_chat。懒加载(测试时便于替换)。"""

    _service: Service | None = None

    @classmethod
    def get_service(cls) -> Service:
        if cls._service is None:
            cls._service = build_service()
        return cls._service

    @classmethod
    def reset(cls) -> None:
        cls._service = None


def run_chat(question: str, mode: str = "auto") -> ChatResponse:
    """执行一次问答并组装 6 步 RagReport。session 由调用方负责落库。"""
    svc = RagEngine.get_service()
    t_start = time.perf_counter()

    if mode == "agentic" or (mode == "auto" and _route_agentic(question)):
        return _run_agentic(svc, question, t_start)
    return _run_direct(svc, question, t_start)


def _route_agentic(question: str) -> bool:
    from agent.router import route

    return route(question) == "agentic"


def _run_direct(svc: Service, question: str, t_start: float) -> ChatResponse:
    steps: list[TraceStep] = []

    # 1+2. Query 理解与改写(翻译 + 术语扩展)
    t0 = time.perf_counter()
    prepared = svc.preprocessor.prepare(question)
    t_prep = time.perf_counter() - t0
    steps.append(
        TraceStep(
            key="query_understanding",
            label="Query 理解",
            elapsed_s=round(t_prep, 3),
            detail={"original_query": question},
        )
    )
    steps.append(
        TraceStep(
            key="query_rewrite",
            label="Query 改写",
            elapsed_s=round(t_prep, 3),
            detail={
                "translated_query": prepared.translated_query or question,
                "expanded_terms": prepared.expanded_terms,
            },
        )
    )

    # 3+4. 混合检索与重排
    t0 = time.perf_counter()
    vector = svc.embedder.embed([question])[0]
    t_embed = time.perf_counter() - t0
    retrieval = svc.pipeline.retrieve(
        vector, question, translated_query=prepared.bm25_query
    )
    trace = retrieval.trace
    cfg = svc.config.retrieval
    steps.append(
        TraceStep(
            key="hybrid_retrieval",
            label="混合检索",
            elapsed_s=round(
                trace.get("dense_s", 0) + trace.get("bm25_s", 0)
                + trace.get("rrf_s", 0) + t_embed,
                3,
            ),
            detail={
                "dense_top_k": cfg.dense_top_k,
                "bm25_top_k": cfg.bm25_top_k,
                "rrf_k": cfg.rrf_k,
                "rrf_top_k": cfg.rrf_top_k,
            },
        )
    )
    steps.append(
        TraceStep(
            key="rerank",
            label="重排",
            elapsed_s=round(trace.get("rerank_s", 0), 3),
            detail={
                "model": svc.config.rerank.model,
                "candidates": len(retrieval.candidates),
                "top_scores": [
                    round(c.score, 4)
                    for c in retrieval.candidates[:5]
                    if c.score is not None
                ],
            },
        )
    )

    # 5. Context 构建
    steps.append(
        TraceStep(
            key="context",
            label="Context 构建",
            elapsed_s=None,
            detail={
                "count": len(retrieval.context),
                "chunk_ids": [c.chunk_id for c in retrieval.context],
            },
        )
    )

    # 6. Final Generation
    rag_ans = svc.rag.answer(prepared, retrieval)
    steps.append(
        TraceStep(
            key="final_generation",
            label="生成回答",
            elapsed_s=None,
            detail={
                "prompt_tokens": rag_ans.trace.get("prompt_tokens", 0),
                "completion_tokens": rag_ans.trace.get("completion_tokens", 0),
                "cost_yuan": rag_ans.trace.get("estimated_cost_yuan", 0.0),
            },
        )
    )

    report = RagReport(
        steps=steps,
        translated_query=prepared.translated_query,
        expanded_terms=prepared.expanded_terms,
        recall_chunk_ids=[c.chunk_id for c in retrieval.context],
        context_count=len(retrieval.context),
        total_cost_yuan=round(rag_ans.trace.get("estimated_cost_yuan", 0.0), 6),
        total_s=round(time.perf_counter() - t_start, 3),
        mode="direct",
    )
    return ChatResponse(
        session_id="",
        answer=rag_ans.answer,
        citations=_citations_to_models(rag_ans.citations),
        refused=rag_ans.refused,
        mode="direct",
        stop_reason="refused" if rag_ans.refused else "final",
        report=report,
    )


def _run_agentic(svc: Service, question: str, t_start: float) -> ChatResponse:
    from agent import AgentLoop, ConversationMemory, ToolContext

    ctx = ToolContext(
        pipeline=svc.pipeline,
        embedder=svc.embedder,
        chat=svc.chat,
        sqlite=svc.sqlite,
        preprocessor=svc.preprocessor,
    )
    loop = AgentLoop(svc.chat, ctx, svc.config.agent, ConversationMemory())
    ans = loop.run(question)

    steps: list[TraceStep] = []
    for step in ans.tool_trace:
        steps.append(
            TraceStep(
                key="tool_call",
                label=f"工具 {step.get('tool', '')}",
                status="ok" if step.get("ok") else "error",
                elapsed_s=step.get("elapsed_s"),
                detail={
                    "step": step.get("step", 0),
                    "ok": step.get("ok"),
                },
            )
        )
    steps.append(
        TraceStep(
            key="final_generation",
            label="最终回答",
            detail={
                "stop_reason": ans.stop_reason,
                "cost_yuan": ans.total_cost_yuan,
            },
        )
    )
    report = RagReport(
        steps=steps,
        recall_chunk_ids=[c.chunk_id for c in ctx.gathered.values()],
        context_count=len(ctx.gathered),
        total_cost_yuan=round(ans.total_cost_yuan, 6),
        total_s=round(time.perf_counter() - t_start, 3),
        mode="agentic",
    )
    return ChatResponse(
        session_id="",
        answer=ans.answer,
        citations=_citations_to_models(ans.citations),
        refused=ans.stop_reason != "final_answer",
        mode="agentic",
        stop_reason=ans.stop_reason,
        report=report,
    )


logging.getLogger("docpilot.trace").setLevel(logging.CRITICAL)
