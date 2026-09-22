"""Agent 测试:护栏、记忆、ReAct 循环(工具/重复/步数)、路由(全离线 mock)。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

from agent import (
    AgentLoop,
    ConversationMemory,
    Guardrails,
    StopReason,
    ToolContext,
)
from agent.router import route
from ingest import SQLiteStore
from llm import ChatResult
from llm.config import AgentConfig
from retriever.pipeline import RetrievalOutput
from retriever.query_prep import PreparedQuery
from retriever.types import RetrievedChunk

# ---------------------------------------------------------------- 假依赖

class FakeChat:
    """按序返回 ReAct 响应,最后一个复用。"""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls = 0

    def chat(self, messages, **kwargs):
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ChatResult(text=text, model="fake", prompt_tokens=10, completion_tokens=10)


class FakePipeline:
    def retrieve(self, vector, query, translated_query=None):
        return RetrievalOutput(
            context=[
                RetrievedChunk(
                    chunk_id="d1_c0001",
                    document_id="d1",
                    section="Intro",
                    page="",
                    paragraph=1,
                    text="fastapi routing generation uses deep learning.",
                    source_url="https://example.org/docs/1",
                )
            ],
            candidates=[],
        )


class FakeEmbedder:
    config = NS(model="fake/bge-m3", dimension=4)

    def embed(self, texts):
        return [[0.1] * 4 for _ in texts]


class FakePreprocessor:
    def prepare(self, q):
        return PreparedQuery(
            original_query=q, translated_query="fastapi routing", expanded_terms=[]
        )


def _ctx(tmp_path: Path) -> ToolContext:
    sqlite = SQLiteStore(tmp_path / "db.sqlite")
    sqlite.upsert_document(
        {"document_id": "d1", "title": "FastAPI Reference", "source_name": "Official Docs"},
        status="ready",
    )
    return ToolContext(
        pipeline=FakePipeline(),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        chat=FakeChat([]),  # type: ignore[arg-type]
        sqlite=sqlite,
        preprocessor=FakePreprocessor(),  # type: ignore[arg-type]
    )


def _cfg(**overrides) -> AgentConfig:
    defaults = dict(
        max_steps=8,
        tool_timeout_seconds=30,
        max_tool_retries=2,
        max_consecutive_repeat=2,
        max_cost_yuan_per_query=0.5,
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


# ---------------------------------------------------------------- 护栏

def test_guardrails_repeat_detection() -> None:
    g = Guardrails(_cfg(max_consecutive_repeat=2))
    g.record_action("search_docs", '{"question": "a"}')
    assert g.repeated_action() is False
    g.record_action("search_docs", '{"question": "a"}')
    assert g.repeated_action() is True


def test_guardrails_steps_and_cost() -> None:
    g = Guardrails(_cfg(max_steps=3, max_cost_yuan_per_query=0.1))
    assert g.over_steps(3) is True
    assert g.over_steps(2) is False
    assert g.over_cost(0.2) is True


# ---------------------------------------------------------------- 记忆

def test_memory_trims_old_rounds() -> None:
    m = ConversationMemory(max_rounds=2)
    for i in range(5):
        m.add_user(f"q{i}")
        m.add_assistant(f"a{i}")
    hist = m.history()
    assert len(hist) == 4  # 最近 2 轮
    assert hist[0]["content"] == "q3"


# ---------------------------------------------------------------- ReAct 循环

def test_loop_requires_evidence_before_final(tmp_path: Path) -> None:
    """零检索就想直接回答 → 先被要求检索一次(红线:不输出未经证据支持的结论)。"""
    ctx = _ctx(tmp_path)
    chat = FakeChat(["Thought: 直接回答\nFinal Answer: 凭记忆给出的答案。"])
    loop = AgentLoop(chat, ctx, _cfg(), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("合成CT用什么方法?")
    assert answer.stop_reason == StopReason.FINAL
    assert "凭记忆给出的答案" in answer.answer
    assert answer.steps == 1  # 被提醒"先检索"的那一步计入步数
    assert chat.calls == 2  # 提醒后再次请求模型


def test_loop_max_steps_summarizes_with_evidence(tmp_path: Path) -> None:
    """撞步数上限但已有证据 → 用证据收尾成答(不再只回一句模板话)。"""
    ctx = _ctx(tmp_path)
    action = (
        "Thought: 继续\nAction: search_docs\n"
        'Action Input: {"question": "fastapi routing"}'
    )
    chat = FakeChat(
        [action, action, action, "Thought: 收尾\nFinal Answer: 基于已检索证据的结论[1]。"]
    )
    loop = AgentLoop(chat, ctx, _cfg(max_steps=3, max_consecutive_repeat=10), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("问题")
    assert answer.stop_reason == StopReason.MAX_STEPS
    assert "基于已检索证据的结论" in answer.answer  # 来自收尾生成,而非模板话
    assert answer.citations  # 收尾答案的 [1] 映射到真实证据
    assert answer.citations[0].chunk_id == "d1_c0001"


def test_loop_tool_then_final(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    chat = FakeChat(
        [
            "Thought: 需要检索\nAction: search_docs\n"
            'Action Input: {"question": "fastapi routing methods", "top_k": 3}',
            "Thought: 已有证据\nFinal Answer: 合成CT用深度学习[1]。",
        ]
    )
    loop = AgentLoop(chat, ctx, _cfg(), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("合成CT用什么方法?")
    assert answer.stop_reason == StopReason.FINAL
    assert answer.steps == 1
    assert len(answer.tool_trace) == 1
    assert answer.tool_trace[0]["tool"] == "search_docs"
    assert answer.citations  # 从检索证据构建引用
    assert answer.citations[0].title == "FastAPI Reference"


def test_loop_unknown_tool_reports(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    chat = FakeChat(
        [
            "Thought: 用工具\nAction: no_such_tool\nAction Input: {}",
            "Thought: 结束\nFinal Answer: 完成。",
        ]
    )
    loop = AgentLoop(chat, ctx, _cfg(), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("问题")
    assert answer.stop_reason == StopReason.FINAL
    assert answer.tool_trace[0]["ok"] is False


def test_loop_repeat_action_stops(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    repeat = (
        "Thought: 再查一次\nAction: search_docs\n"
        'Action Input: {"question": "fastapi routing"}'
    )
    chat = FakeChat([repeat, repeat, "Thought: 结束\nFinal Answer: 完成。"])
    loop = AgentLoop(chat, ctx, _cfg(max_consecutive_repeat=2), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("问题")
    assert answer.stop_reason == StopReason.REPEAT_ACTION


def test_loop_max_steps_stops(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    action = (
        "Thought: 继续\nAction: search_docs\n"
        'Action Input: {"question": "fastapi routing"}'
    )
    chat = FakeChat([action, action, action, action])
    # 提高重复阈值,让"步数上限"先于"重复检测"触发
    loop = AgentLoop(chat, ctx, _cfg(max_steps=3, max_consecutive_repeat=10), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("问题")
    assert answer.stop_reason == StopReason.MAX_STEPS


def test_loop_invalid_params_reported(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    chat = FakeChat(
        [
            "Thought: 检索\nAction: search_docs\n"
            'Action Input: {"top_k": 999}',  # top_k 超界(1-10)
            "Thought: 结束\nFinal Answer: 完成。",
        ]
    )
    loop = AgentLoop(chat, ctx, _cfg(), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("问题")
    assert answer.stop_reason == StopReason.FINAL
    assert answer.tool_trace[0]["ok"] is False  # Pydantic 拦截非法参数


# ---------------------------------------------------------------- 引用溯源(修复:只保留被引用证据)

def test_build_citations_only_keeps_referenced(tmp_path: Path) -> None:
    """final answer 只引用 [2] 时,citations 只保留第 2 条证据(防罗列全部)。"""
    ctx = _ctx(tmp_path)
    ctx.gathered[1] = RetrievedChunk(
        chunk_id="d1_c0001", document_id="d1", section="Intro", page="",
        paragraph=1, text="chunk one text", source_url="u1", score=1.0,
    )
    ctx.gathered[2] = RetrievedChunk(
        chunk_id="d1_c0002", document_id="d1", section="Methods", page="",
        paragraph=2, text="chunk two text", source_url="u1", score=1.0,
    )
    loop = AgentLoop(FakeChat([]), ctx, _cfg(), ConversationMemory())  # type: ignore[arg-type]
    citations = loop._build_citations("综合 [2] 的证据,扩散模型更优。")
    assert len(citations) == 1
    assert citations[0].chunk_id == "d1_c0002"
    assert citations[0].index == 2


def test_build_citations_drops_out_of_range(tmp_path: Path) -> None:
    """模型编造不存在的编号 [5] 应被丢弃(防幻觉)。"""
    ctx = _ctx(tmp_path)
    ctx.gathered[1] = RetrievedChunk(
        chunk_id="d1_c0001", document_id="d1", section="Intro", page="",
        paragraph=1, text="t", source_url="u", score=1.0,
    )
    loop = AgentLoop(FakeChat([]), ctx, _cfg(), ConversationMemory())  # type: ignore[arg-type]
    assert loop._build_citations("编造的编号 [5] 应被丢弃。") == []


def test_evidence_text_global_numbering(tmp_path: Path) -> None:
    """修复:多次检索时证据编号全局递增,与 gathered 顺序一致(引用映射正确)。"""
    from agent.tools import _evidence_text

    ctx = _ctx(tmp_path)
    c1 = RetrievedChunk(
        chunk_id="d1_c0001", document_id="d1", section="Intro", page="",
        paragraph=1, text="first chunk", source_url="u", score=1.0,
    )
    c2 = RetrievedChunk(
        chunk_id="d1_c0002", document_id="d1", section="Methods", page="",
        paragraph=2, text="second chunk", source_url="u", score=1.0,
    )
    # 第一次检索:编号从 1 开始,并写入 gathered
    text1 = _evidence_text(ctx, [c1])
    assert "[1]" in text1
    assert ctx.gathered[1].chunk_id == "d1_c0001"
    # 第二次检索:编号从 2 开始(不重置)
    text2 = _evidence_text(ctx, [c2])
    assert "[2]" in text2
    assert "[1]" not in text2
    assert ctx.gathered[2].chunk_id == "d1_c0002"


# ---------------------------------------------------------------- 路由

def test_router_direct_vs_agentic() -> None:
    assert route("磁共振到CT合成用什么方法?") == "direct"
    assert route("比较GAN和扩散模型在合成CT上的差异") == "agentic"
    assert (
        route(
            "磁共振到CT合成中,生成对抗网络与扩散模型在图像质量、"
            "训练稳定性和临床可用性上的差异是什么?"
        )
        == "agentic"
    )
