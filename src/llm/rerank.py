"""SiliconFlow rerank 客户端(非 OpenAI 兼容端点,直接 HTTP 调用)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .base import (
    APIClientError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    retry_call,
)
from .config import RerankLLMConfig


@dataclass
class RerankResult:
    """重排结果:index 为在输入 documents 中的下标。"""

    index: int
    score: float
    text: str


class RerankClient:
    """SiliconFlow rerank 客户端。

    端点:POST {base_url}/rerank(OpenAI 兼容路径前缀 /v1)。
    统一行为:超时、指数退避重试(429/503/5xx)。
    """

    def __init__(self, config: RerankLLMConfig, api_key: str):
        self.config = config
        self._url = f"{config.base_url.rstrip('/')}/rerank"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """对 documents 按与 query 的相关性重排,返回按分数降序的 top_n 条。"""
        if not documents:
            return []
        payload: dict[str, Any] = {
            "model": self.config.model,
            "query": query,
            "documents": documents,
            "top_n": top_n if top_n is not None else len(documents),
        }

        def _call() -> dict[str, Any]:
            try:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    resp = client.post(self._url, headers=self._headers, json=payload)
            except httpx.TimeoutException as exc:
                raise APITimeoutError(str(exc)) from exc
            if resp.status_code == 429:
                raise APIRateLimitError("rerank HTTP 429 限流")
            if 500 <= resp.status_code < 600:
                raise APIServerError(f"rerank HTTP {resp.status_code}")
            if resp.status_code != 200:
                raise APIClientError(f"rerank HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()

        data = retry_call(
            _call,
            max_retries=self.config.max_retries,
            backoff_base_seconds=1.0,
        )
        results: list[RerankResult] = []
        for item in data.get("results", []):
            idx = int(item["index"])
            results.append(
                RerankResult(index=idx, score=float(item["relevance_score"]), text=documents[idx])
            )
        return results
