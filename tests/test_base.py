"""重试机制测试:指数退避、429/503 处理、超限放弃。"""

from __future__ import annotations

import pytest

from llm.base import APIRateLimitError, APIServerError, retry_call


class Flaky:
    """前 failures 次抛异常,之后返回 ok。"""

    def __init__(self, failures: int, exc: Exception) -> None:
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def run(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "ok"


def test_retry_succeeds_after_transient_429(monkeypatch) -> None:
    monkeypatch.setattr("llm.base.time.sleep", lambda s: None)
    flaky = Flaky(2, APIRateLimitError("simulated 429"))
    assert retry_call(flaky.run, max_retries=3, backoff_base_seconds=1.0) == "ok"
    assert flaky.calls == 3


def test_retry_succeeds_after_503(monkeypatch) -> None:
    monkeypatch.setattr("llm.base.time.sleep", lambda s: None)
    flaky = Flaky(1, APIServerError("simulated 503"))
    assert retry_call(flaky.run, max_retries=3, backoff_base_seconds=1.0) == "ok"
    assert flaky.calls == 2


def test_retry_gives_up_after_max_retries(monkeypatch) -> None:
    monkeypatch.setattr("llm.base.time.sleep", lambda s: None)
    flaky = Flaky(99, APIServerError("always 503"))
    with pytest.raises(APIServerError):
        retry_call(flaky.run, max_retries=2, backoff_base_seconds=1.0)
    assert flaky.calls == 3  # 1 次原调用 + 2 次重试


def test_non_retryable_error_not_retried(monkeypatch) -> None:
    monkeypatch.setattr("llm.base.time.sleep", lambda s: None)
    calls = {"n": 0}

    def boom() -> str:
        calls["n"] += 1
        raise ValueError("不可重试")

    with pytest.raises(ValueError):
        retry_call(boom, max_retries=3, backoff_base_seconds=1.0)
    assert calls["n"] == 1
