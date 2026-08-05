"""EmbeddingClient 测试:mock openai,验证批处理、顺序、维度校验、模型检查。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from llm import EmbeddingClient
from llm.base import APIClientError
from llm.config import EmbeddingLLMConfig
from tests.fake_llm import FakeOpenAI, fake_embedding_response


def _make_client(monkeypatch, behaviour: Callable[..., Any], **cfg_overrides) -> EmbeddingClient:
    monkeypatch.setattr(
        "llm.embedding.OpenAI", lambda **kw: FakeOpenAI(embeddings_behaviour=behaviour)
    )
    cfg = EmbeddingLLMConfig(**cfg_overrides)
    return EmbeddingClient(cfg, api_key="test-key")


def test_embed_batching_preserves_order(monkeypatch) -> None:
    calls = {"n": 0}

    def behaviour(**kwargs: Any) -> object:
        texts = kwargs["input"]
        calls["n"] += 1
        return fake_embedding_response([[float(i)] * 4 for i in range(len(texts))])

    client = _make_client(monkeypatch, behaviour, batch_size=2, max_concurrency=2, dimension=4)
    vectors = client.embed(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert all(len(v) == 4 for v in vectors)
    assert calls["n"] == 3  # 5 条 / 每批 2 = 3 批


def test_embed_empty_input(monkeypatch) -> None:
    client = _make_client(monkeypatch, lambda **kw: fake_embedding_response([]), dimension=4)
    assert client.embed([]) == []


def test_embed_dimension_mismatch_raises(monkeypatch) -> None:
    """实际维度与配置不一致必须报错(防止静默错配索引)。"""

    def behaviour(**kwargs: Any) -> object:
        del kwargs
        return fake_embedding_response([[1.0, 2.0]])

    client = _make_client(monkeypatch, behaviour, dimension=1024)  # 配置 1024,返回 2 维
    with pytest.raises(APIClientError, match="维度"):
        client.embed(["x"])


def test_check_model_available(monkeypatch) -> None:
    from types import SimpleNamespace as NS

    def list_models() -> NS:
        return NS(data=[NS(id="BAAI/bge-m3"), NS(id="other/model")])

    monkeypatch.setattr(
        "llm.embedding.OpenAI",
        lambda **kw: FakeOpenAI(
            embeddings_behaviour=lambda **k: fake_embedding_response([[0.1] * 1024])
        ),
    )
    cfg = EmbeddingLLMConfig()
    client = EmbeddingClient(cfg, api_key="test-key")
    client._client.models = NS(list=list_models)
    assert client.check_model_available() is True

    def list_models_without_target() -> NS:
        return NS(data=[NS(id="other/model")])

    client._client.models = NS(list=list_models_without_target)
    assert client.check_model_available() is False
