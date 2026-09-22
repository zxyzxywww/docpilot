"""查询预处理测试:三字段保留、JSON 解析(含 markdown 围栏)、失败回退。"""

from __future__ import annotations

from llm import ChatResult
from retriever.query_prep import PreparedQuery, QueryPreprocessor, _extract_json


class FakeChat:
    def __init__(self, text: str):
        self._text = text
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return ChatResult(text=self._text, model="fake")


def test_prepare_records_three_fields() -> None:
    fake = FakeChat(
        '{"translated_query": "fastapi dependency injection", '
        '"expanded_terms": ["fastapi routing", "internal routing", "request pipeline"]}'
    )
    prep = QueryPreprocessor(fake)  # type: ignore[arg-type]
    out = prep.prepare("磁共振到CT的深度学习合成方法有哪些?")
    assert isinstance(out, PreparedQuery)
    assert "磁共振" in out.original_query
    assert out.translated_query == "fastapi dependency injection"
    assert len(out.expanded_terms) == 3
    assert "fastapi routing" in out.bm25_query
    assert fake.calls == 1


def test_extract_json_with_markdown_fence() -> None:
    text = '```json\n{"a": 1, "b": [2, 3]}\n```'
    assert _extract_json(text) == {"a": 1, "b": [2, 3]}


def test_extract_json_with_noise() -> None:
    text = '好的,这是结果:{"translated_query": "fastapi routing"} 完毕'
    assert _extract_json(text) == {"translated_query": "fastapi routing"}


def test_prepare_falls_back_on_bad_json() -> None:
    fake = FakeChat("抱歉我无法输出 JSON")
    prep = QueryPreprocessor(fake)  # type: ignore[arg-type]
    out = prep.prepare("如何评估合成CT质量")
    assert out.translated_query == "如何评估合成CT质量"  # 回退为原问题
    assert out.expanded_terms == []
    assert out.bm25_query == "如何评估合成CT质量"
