"""真实 API 集成测试(integration marker)。

默认跳过:只有 .env 中配置了对应 API key 时才运行。
运行方式:uv run pytest -m integration
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from llm import ChatClient, get_api_key, load_config

load_dotenv()

pytestmark = pytest.mark.integration

HAS_DEEPSEEK_KEY = bool(os.getenv("DEEPSEEK_API_KEY"))
HAS_SILICONFLOW_KEY = bool(os.getenv("SILICONFLOW_API_KEY"))


@pytest.mark.skipif(not HAS_DEEPSEEK_KEY, reason="需要真实 DEEPSEEK_API_KEY")
def test_chat_real_api() -> None:
    config = load_config()
    client = ChatClient(config.chat, get_api_key("deepseek"))
    result = client.chat([{"role": "user", "content": "请只回复:ok"}])
    assert result.text
    assert result.model


@pytest.mark.skipif(not HAS_SILICONFLOW_KEY, reason="需要真实 SILICONFLOW_API_KEY")
def test_embedding_real_api() -> None:
    config = load_config()
    client = _embedding_client(config, get_api_key("siliconflow"))
    vectors = client.embed(["breast cancer", "radiology"])
    assert len(vectors) == 2
    assert len(vectors[0]) == config.embedding.dimension


@pytest.mark.skipif(not HAS_SILICONFLOW_KEY, reason="需要真实 SILICONFLOW_API_KEY")
def test_rerank_real_api() -> None:
    config = load_config()
    client = _rerank_client(config, get_api_key("siliconflow"))
    results = client.rerank("breast cancer imaging", ["radiology", "cooking recipes"], top_n=2)
    assert len(results) == 2


def _embedding_client(config, api_key: str):
    from llm import EmbeddingClient

    return EmbeddingClient(config.embedding, api_key)


def _rerank_client(config, api_key: str):
    from llm import RerankClient

    return RerankClient(config.rerank, api_key)
