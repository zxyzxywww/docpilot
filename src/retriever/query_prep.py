"""查询预处理:中文问题 → 英文检索查询 + 技术术语扩展。

跨语言场景(学习点 L4):
- 文档是英文,BM25 是字面匹配 —— 中文词在英文段落里不会有命中;
- 因此 BM25 通道必须用"英文翻译 + 技术术语扩展"后的查询;
- 稠密通道则直接用原始中文问题的向量(bge-m3 多语言,中文可直接检索英文)。

输出同时保留 original_query / translated_query / expanded_terms 三个字段,
供检索、评估与观测日志使用(约束 2)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from llm import ChatClient

SYSTEM_PROMPT = (
    "你是 Python 后端开发文档检索助手,负责把用户的中文开发问题转换为英文检索查询。"
    "只输出 JSON,不要任何其他文字。JSON 格式:"
    '{"translated_query": "英文检索查询(忠实翻译,保留技术术语)", '
    '"expanded_terms": ["3-8 个技术术语扩展词(FastAPI/Pydantic 等 API 名、参数名)"]}。'
    "扩展词可以是同义词、缩写、上位/下位概念,用于提升关键词检索召回。"
)


@dataclass
class PreparedQuery:
    """预处理后的检索查询(三字段全部保留,约束 2)。"""

    original_query: str
    translated_query: str
    expanded_terms: list[str]

    @property
    def bm25_query(self) -> str:
        """BM25 通道使用的查询:英文翻译 + 术语扩展。"""
        return (self.translated_query + " " + " ".join(self.expanded_terms)).strip()


def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON(容忍 ```json 围栏与多余文字)。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"输出中未找到 JSON: {text[:100]}")
    return json.loads(text[start : end + 1])


class QueryPreprocessor:
    """调用 DeepSeek 完成翻译与术语扩展。"""

    def __init__(self, chat: ChatClient):
        self._chat = chat

    def prepare(self, question: str) -> PreparedQuery:
        result = self._chat.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_tokens=512,
        )
        try:
            payload = _extract_json(result.text)
            translated = str(payload.get("translated_query", "")).strip()
            terms = payload.get("expanded_terms", [])
            if isinstance(terms, str):
                terms = [terms]
            terms = [str(t).strip() for t in terms if str(t).strip()]
        except (ValueError, json.JSONDecodeError):
            # 解析失败时优雅回退:原问题当翻译,无扩展词(不阻断问答)
            translated = question
            terms = []
        if not translated:
            translated = question
        return PreparedQuery(
            original_query=question,
            translated_query=translated,
            expanded_terms=terms,
        )
