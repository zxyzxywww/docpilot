"""检索结果的数据结构(引用溯源的最小载体)。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """一个被检索命中的 chunk,携带完整溯源信息。

    这是生成层引用的依据:回答里的 [1][2] 索引对应这里的一条记录,
    引用内容(标题/章节/页码/chunk_id/证据原文)全部来自真实检索结果,
    禁止模型凭空生成不存在的来源(MVP 红线)。
    """

    chunk_id: str
    document_id: str
    section: str
    page: str
    paragraph: int
    text: str
    source_url: str
    score: float = 0.0
