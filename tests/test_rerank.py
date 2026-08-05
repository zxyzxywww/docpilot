"""RerankClient 测试:mock httpx,验证解析、429 重试、503 处理。"""

from __future__ import annotations

import pytest

from llm import RerankClient
from llm.base import APIServerError
from llm.config import RerankLLMConfig
from tests.fake_llm import FakeHttpxClient, FakeResponse


def _make_client(monkeypatch, behaviours) -> RerankClient:
    monkeypatch.setattr("llm.rerank.httpx.Client", lambda **kw: FakeHttpxClient(behaviours))
    return RerankClient(RerankLLMConfig(), api_key="test-key")


def test_rerank_parses_results(monkeypatch) -> None:
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.7},
        ]
    }
    client = _make_client(monkeypatch, [FakeResponse(200, payload)])
    results = client.rerank("query", ["a", "b", "c"], top_n=2)
    assert [r.index for r in results] == [2, 0]
    assert results[0].text == "c"
    assert results[0].score == pytest.approx(0.9)


def test_rerank_empty_documents(monkeypatch) -> None:
    client = _make_client(monkeypatch, [])
    assert client.rerank("query", []) == []


def test_rerank_retries_on_429(monkeypatch) -> None:
    """429 后应触发重试;用 callable 行为使状态跨重试共享(每次重试会新建 httpx.Client)。"""
    monkeypatch.setattr("llm.base.time.sleep", lambda s: None)
    payload = {"results": [{"index": 0, "relevance_score": 0.8}]}
    calls = {"n": 0}

    def behaviour() -> FakeResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(429)
        return FakeResponse(200, payload)

    client = _make_client(monkeypatch, behaviour)
    results = client.rerank("query", ["a"])
    assert len(results) == 1
    assert calls["n"] == 2


def test_rerank_gives_up_on_503(monkeypatch) -> None:
    monkeypatch.setattr("llm.base.time.sleep", lambda s: None)
    client = _make_client(monkeypatch, [FakeResponse(503), FakeResponse(503), FakeResponse(503)])
    with pytest.raises(APIServerError):
        client.rerank("query", ["a"])  # max_retries=3,仍全部失败
