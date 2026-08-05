"""ChatClient 测试:mock openai,验证调用、429 重试、异常映射与费用统计。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from llm import ChatClient
from llm.base import APIRateLimitError
from llm.config import ChatLLMConfig
from tests.fake_llm import FakeOpenAI, fake_completion


def _make_client(monkeypatch, behaviour: Callable[[dict], Any]) -> ChatClient:
    monkeypatch.setattr("llm.chat.OpenAI", lambda **kw: FakeOpenAI(chat_behaviour=behaviour))
    return ChatClient(ChatLLMConfig(), api_key="test-key")


def test_chat_returns_text_and_usage(monkeypatch) -> None:
    client = _make_client(monkeypatch, lambda kwargs: fake_completion("你好"))
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result.text == "你好"
    assert result.model == "deepseek-v4-flash"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.cache_hit_tokens == 0
    assert result.estimated_cost_yuan > 0


def test_chat_retries_on_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr("llm.base.time.sleep", lambda s: None)
    calls = {"n": 0}

    def behaviour(kwargs: dict) -> object:
        del kwargs
        calls["n"] += 1
        if calls["n"] == 1:
            raise APIRateLimitError("simulated 429")
        return fake_completion("ok")

    client = _make_client(monkeypatch, behaviour)
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert calls["n"] == 2


def test_openai_rate_limit_mapped_and_retried(monkeypatch) -> None:
    """openai.RateLimitError 应被映射为 APIRateLimitError 并触发重试。"""
    monkeypatch.setattr("llm.base.time.sleep", lambda s: None)

    class FakeRateLimit(Exception):
        pass

    monkeypatch.setattr("llm.chat.RateLimitError", FakeRateLimit)
    calls = {"n": 0}

    def behaviour(kwargs: dict) -> object:
        del kwargs
        calls["n"] += 1
        if calls["n"] == 1:
            raise FakeRateLimit("429 from openai")
        return fake_completion("recovered")

    client = _make_client(monkeypatch, behaviour)
    result = client.chat([{"role": "user", "content": "hi"}])
    assert result.text == "recovered"
    assert calls["n"] == 2


def _always_429(kwargs: dict) -> object:
    del kwargs
    raise APIRateLimitError("always 429")


def test_chat_gives_up_after_max_retries(monkeypatch) -> None:
    monkeypatch.setattr("llm.base.time.sleep", lambda s: None)
    client = _make_client(monkeypatch, _always_429)
    with pytest.raises(APIRateLimitError):
        client.chat([{"role": "user", "content": "hi"}])
