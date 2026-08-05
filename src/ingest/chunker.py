"""文本分块:与段落对齐,超长段落按 token 滑窗切分。

策略(为保证引用溯源到段落):
- 段落 ≤ chunk_size_tokens:整段一个 chunk(引用精确到段落)
- 段落 > chunk_size_tokens:按 token 近似(字符滑窗)切分为多块,带 overlap
- chunk_id 确定性:docid_c0001,文档内按序递增(幂等 upsert 的基础)

token 估算复用 llm.usage.estimate_tokens(约 3 字符/token,量级足够)。
"""

from __future__ import annotations

from dataclasses import dataclass

from llm.usage import estimate_tokens

from .parser import ParsedParagraph

CHARS_PER_TOKEN = 3  # 与 llm.usage.estimate_tokens 保持一致


@dataclass
class Chunk:
    """检索与引用的最小单位。"""

    chunk_id: str
    document_id: str
    section: str
    page: str
    paragraph: int
    text: str
    source_url: str
    token_count: int


def chunk_paragraphs(
    document_id: str,
    paragraphs: list[ParsedParagraph],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Chunk]:
    """将解析出的段落分块,生成确定性 chunk_id。"""
    if chunk_overlap_tokens >= chunk_size_tokens:
        chunk_overlap_tokens = max(0, chunk_size_tokens // 2)
    chunks: list[Chunk] = []
    for para in paragraphs:
        text = para.text
        tokens = estimate_tokens(text)
        if tokens <= chunk_size_tokens:
            chunks.append(
                Chunk(
                    chunk_id="",  # 稍后统一编号
                    document_id=document_id,
                    section=para.section,
                    page=para.page,
                    paragraph=para.paragraph,
                    text=text,
                    source_url=para.source_url,
                    token_count=tokens,
                )
            )
        else:
            for piece in _split_text(text, chunk_size_tokens, chunk_overlap_tokens):
                chunks.append(
                    Chunk(
                        chunk_id="",
                        document_id=document_id,
                        section=para.section,
                        page=para.page,
                        paragraph=para.paragraph,
                        text=piece,
                        source_url=para.source_url,
                        token_count=estimate_tokens(piece),
                    )
                )
    for idx, chunk in enumerate(chunks, start=1):
        chunk.chunk_id = f"{document_id}_c{idx:04d}"
    return chunks


def _split_text(text: str, chunk_size_tokens: int, chunk_overlap_tokens: int) -> list[str]:
    """按字符滑窗切分超长文本;窗口 = size*3 字符,步长 = (size-overlap)*3 字符。"""
    window_chars = chunk_size_tokens * CHARS_PER_TOKEN
    step_chars = max(1, (chunk_size_tokens - chunk_overlap_tokens) * CHARS_PER_TOKEN)
    if step_chars >= window_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + window_chars])
        start += step_chars
    # 去掉与上一块几乎重复的尾块
    while len(pieces) > 1 and len(pieces[-1]) <= step_chars * 0.2:
        pieces.pop()
    return [p for p in pieces if p.strip()]
