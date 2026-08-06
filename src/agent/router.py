"""direct_rag vs agentic_rag 路由(约束 5)。

- 单跳简单问题 → direct_rag(快、省);
- 多文档比较、证据冲突、复杂综合问题 → agentic_rag(Agent 多步检索综合)。
判断基于问题长度与语义线索关键词(可扩展为 LLM 判断,当前启发式足够)。
"""

from __future__ import annotations

COMPLEX_KEYWORDS = (
    "比较",
    "对比",
    "差异",
    "区别",
    "综合",
    "总结",
    "多篇",
    "综述",
    "冲突",
    "矛盾",
    "一致性",
    "compare",
    "contrast",
    "difference",
    "synthesize",
    "conflict",
    "consistency",
    "multi",
)


def route(question: str, length_threshold: int = 60) -> str:
    """返回 'direct' 或 'agentic'。"""
    if len(question) > length_threshold:
        return "agentic"
    q = question.lower()
    if any(k in q for k in COMPLEX_KEYWORDS):
        return "agentic"
    return "direct"
