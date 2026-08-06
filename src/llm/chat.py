"""DeepSeek 对话客户端(OpenAI 兼容)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import APIStatusError, OpenAI, RateLimitError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai.types.chat import ChatCompletionMessageParam

from .base import (
    APIClientError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    retry_call,
)
from .config import ChatLLMConfig
from .usage import estimate_cost


@dataclass
class ChatResult:
    """一次对话的结果与用量信息。"""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0

    @property
    def estimated_cost_yuan(self) -> float:
        """按估算单价计算本次调用费用(元)。"""
        return estimate_cost("chat", self.prompt_tokens, self.completion_tokens)


class ChatClient:
    """DeepSeek 对话客户端。

    统一行为:超时、指数退避重试(429/503/5xx)、token 与费用统计。
    openai SDK 的 max_retries 置 0,重试统一由本层 retry_call 控制,
    便于测试时用 mock 模拟 429/503 验证重试逻辑。
    """

    def __init__(self, config: ChatLLMConfig, api_key: str):
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """多轮对话。messages 形如 [{"role": "system"|"user"|"assistant", "content": "..."}]。"""

        def _call() -> Any:
            try:
                return self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.config.temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                )
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
            backoff_base_seconds=self.config.backoff_base_seconds,
        )
        usage = resp.usage
        return ChatResult(
            text=resp.choices[0].message.content or "",
            model=resp.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_hit_tokens=getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
            cache_miss_tokens=getattr(usage, "prompt_cache_miss_tokens", 0) or 0,
        )

    def stream_chat(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """流式对话:逐个产出文本增量(供 Web 界面实时展示)。

        用法:for delta in client.stream_chat(messages): ...
        token 统计在流结束后不完整,费用估算请用非流式 chat。
        """
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
