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
                    text="synthetic CT generation uses deep learning.",
                    source_url="https://example.org/PMC1",
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
        return PreparedQuery(original_query=q, translated_query="synthetic CT", expanded_terms=[])


def _ctx(tmp_path: Path) -> ToolContext:
    sqlite = SQLiteStore(tmp_path / "db.sqlite")
    sqlite.upsert_document(
        {"document_id": "d1", "title": "Synthetic CT Paper", "journal": "Medical Physics"},
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
    g.record_action("search_literature", '{"question": "a"}')
    assert g.repeated_action() is False
    g.record_action("search_literature", '{"question": "a"}')
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

def test_loop_direct_final_answer(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    chat = FakeChat(["Thought: 直接回答\nFinal Answer: 合成CT常用GAN[1]。"])
    loop = AgentLoop(chat, ctx, _cfg(), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("合成CT用什么方法?")
    assert answer.stop_reason == StopReason.FINAL
    assert "GAN" in answer.answer
    assert answer.steps == 0


def test_loop_tool_then_final(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    chat = FakeChat(
        [
            "Thought: 需要检索\nAction: search_literature\n"
            'Action Input: {"question": "synthetic CT methods", "top_k": 3}',
            "Thought: 已有证据\nFinal Answer: 合成CT用深度学习[1]。",
        ]
    )
    loop = AgentLoop(chat, ctx, _cfg(), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("合成CT用什么方法?")
    assert answer.stop_reason == StopReason.FINAL
    assert answer.steps == 1
    assert len(answer.tool_trace) == 1
    assert answer.tool_trace[0]["tool"] == "search_literature"
    assert answer.citations  # 从检索证据构建引用
    assert answer.citations[0].title == "Synthetic CT Paper"


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
        "Thought: 再查一次\nAction: search_literature\n"
        'Action Input: {"question": "synthetic CT"}'
    )
    chat = FakeChat([repeat, repeat, "Thought: 结束\nFinal Answer: 完成。"])
    loop = AgentLoop(chat, ctx, _cfg(max_consecutive_repeat=2), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("问题")
    assert answer.stop_reason == StopReason.REPEAT_ACTION


def test_loop_max_steps_stops(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    action = (
        "Thought: 继续\nAction: search_literature\n"
        'Action Input: {"question": "synthetic CT"}'
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
            "Thought: 检索\nAction: search_literature\n"
            'Action Input: {"top_k": 999}',  # top_k 超界(1-10)
            "Thought: 结束\nFinal Answer: 完成。",
        ]
    )
    loop = AgentLoop(chat, ctx, _cfg(), ConversationMemory())  # type: ignore[arg-type]
    answer = loop.run("问题")
    assert answer.stop_reason == StopReason.FINAL
    assert answer.tool_trace[0]["ok"] is False  # Pydantic 拦截非法参数


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
