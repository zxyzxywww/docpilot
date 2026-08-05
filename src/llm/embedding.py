"""SiliconFlow embedding 客户端(OpenAI 兼容 /embeddings 端点)。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import APIStatusError, OpenAI, RateLimitError
from openai import APITimeoutError as OpenAIAPITimeoutError

from .base import (
    APIClientError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    ModelUnavailableError,
    retry_call,
)
from .config import EmbeddingLLMConfig


class EmbeddingClient:
    """SiliconFlow embedding 客户端。

    统一行为:批处理(batch_size)、并发限制(max_concurrency)、
    指数退避重试(429/503/5xx)、超时、可用模型检查、维度一致性校验。
    返回的向量顺序与输入文本顺序一致。
    """

    def __init__(self, config: EmbeddingLLMConfig, api_key: str):
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。自动按 batch_size 分批,并以 max_concurrency 并发。"""
        if not texts:
            return []
        batches = [
            texts[i : i + self.config.batch_size]
            for i in range(0, len(texts), self.config.batch_size)
        ]
        results: list[tuple[int, list[list[float]]]] = []
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as pool:
            futures = {
                pool.submit(self._embed_batch, batch): idx for idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                results.append((futures[future], future.result()))
        results.sort(key=lambda item: item[0])
        return [vec for _, batch_vectors in results for vec in batch_vectors]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        def _call() -> Any:
            try:
                return self._client.embeddings.create(model=self.config.model, input=batch)
            except RateLimitError as exc:
                raise APIRateLimitError(str(exc)) from exc
            except OpenAIAPITimeoutError as exc:
                raise APITimeoutError(str(exc)) from exc
            except APIStatusError as exc:
                if 500 <= exc.status_code < 600:
                    raise APIServerError(str(exc)) from exc
                raise APIClientError(str(exc)) from exc

        resp = retry_call(
            _call,
            max_retries=self.config.max_retries,
            backoff_base_seconds=1.0,
        )
        # 按 index 排序保证与输入顺序一致
        ordered = sorted(resp.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]
        if len(vectors) != len(batch):
            raise APIClientError(f"embedding 返回数量 {len(vectors)} != 输入数量 {len(batch)}")
        dim = len(vectors[0]) if vectors else 0
        if dim != self.config.dimension:
            raise APIClientError(
                f"embedding 实际维度 {dim} != 配置维度 {self.config.dimension},须更换配置并重建索引"
            )
        return vectors

    def check_model_available(self) -> bool:
        """可用模型检查:配置的模型是否在供应商模型列表中。"""
        try:
            models = self._client.models.list()
        except Exception as exc:  # noqa: BLE001 - 网络/鉴权异常统一包装
            raise ModelUnavailableError(f"模型列表获取失败: {exc}") from exc
        ids = {m.id for m in models.data}
        return self.config.model in ids
