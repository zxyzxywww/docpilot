"""共享测试工具:fake OpenAI / httpx 响应对象。"""

from __future__ import annotations

from types import SimpleNamespace as NS


def fake_completion(
    text: str = "测试回答",
    model: str = "deepseek-v4-flash",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cache_hit: int = 0,
    cache_miss: int = 10,
) -> NS:
    """构造一次 chat.completions.create 的响应。"""
    usage = NS(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cache_hit_tokens=cache_hit,
        prompt_cache_miss_tokens=cache_miss,
    )
    return NS(
        choices=[NS(message=NS(content=text))],
        model=model,
        usage=usage,
    )


def fake_embedding_response(vectors: list[list[float]]) -> NS:
    """构造 /embeddings 的响应(data 按输入顺序,index 升序)。"""
    data = [NS(index=i, embedding=vec) for i, vec in enumerate(vectors)]
    return NS(data=data)


class FakeOpenAI:
    """替换 openai.OpenAI:chat / embeddings 子模块可注入行为。"""

    def __init__(
        self,
        *,
        chat_behaviour=None,
        embeddings_behaviour=None,
        **kwargs: object,
    ) -> None:
        del kwargs  # 忽略 base_url / api_key / timeout / max_retries
        if chat_behaviour is None:
            chat_behaviour = lambda **kw: fake_completion()  # noqa: E731
        self.chat = NS(completions=FakeCompletions(chat_behaviour))
        if embeddings_behaviour is None:
            embeddings_behaviour = lambda **kw: fake_embedding_response([])  # noqa: E731
        self.embeddings = NS(create=embeddings_behaviour)


class FakeCompletions:
    def __init__(self, behaviour) -> None:
        # behaviour: callable(kwargs) -> 响应;或可调用列表(按序弹出,最后一个复用)
        self._behaviour = behaviour if callable(behaviour) else list(behaviour)

    def create(self, **kwargs) -> object:
        if callable(self._behaviour):
            return self._behaviour(kwargs)
        if len(self._behaviour) > 1:
            return self._behaviour.pop(0)(kwargs)
        return self._behaviour[0](kwargs)


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class FakeHttpxClient:
    """替换 httpx.Client:post 行为可注入。

    behaviours 可以是:
      - callable():每次调用返回一个 FakeResponse(用于跨重试共享状态,推荐)
      - list[FakeResponse]:按序弹出,最后一个复用(每次新建 client 时重置)
    """

    def __init__(self, behaviours) -> None:
        self._behaviours = behaviours

    def __enter__(self) -> FakeHttpxClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
        del url, headers, json
        if callable(self._behaviours):
            return self._behaviours()
        if len(self._behaviours) > 1:
            return self._behaviours.pop(0)
        return self._behaviours[0]
