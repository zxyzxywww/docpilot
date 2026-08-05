"""统一客户端基础设施:异常层次 + 指数退避重试。"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """LLM API 调用失败基类。"""


class APIRateLimitError(APIClientError):
    """HTTP 429:触发限流,可重试。"""


class APIServerError(APIClientError):
    """HTTP 5xx(含 503):服务端错误,可重试。"""


class APITimeoutError(APIClientError):
    """请求超时,可重试。"""


class ModelUnavailableError(APIClientError):
    """可用模型检查失败或模型不在供应商列表中。"""


# 可重试的异常集合(供 retry_call 使用)
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
)

T = TypeVar("T")


def retry_call(
    func: Callable[..., T],
    *,
    max_retries: int,
    backoff_base_seconds: float,
    **kwargs: Any,
) -> T:
    """执行 func 并做指数退避重试(带随机抖动),处理 429/503/5xx/超时。

    规则:第 n 次重试前等待 backoff_base * 2^(n-1) + 抖动(0~0.5s)。
    超过 max_retries 次失败后抛出原始异常。
    """
    attempt = 0
    while True:
        try:
            return func(**kwargs)
        except RETRYABLE_EXCEPTIONS as exc:
            attempt += 1
            if attempt > max_retries:
                logger.error("%s 重试 %d 次后仍失败: %s", func.__name__, max_retries, exc)
                raise
            delay = backoff_base_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "%s 失败(%s),第 %d/%d 次重试,%.1fs 后重试",
                func.__name__,
                exc,
                attempt,
                max_retries,
                delay,
            )
            time.sleep(delay)
