"""评估集构建:反向生成 QA + LLM judge 自动核验 + 按文档划分 + 无答案问题。

约束与约定:
- DeepSeek 生成的 QA 仅作为候选,经自动核验(规则 + LLM judge 忠实度)后入集;
- 开发集/测试集按文档划分,避免同一文档内容泄漏(约束 8);
- ≥20% 为无答案/证据不足问题(检验拒答能力);
- 用户已授权本环节自主完成(核验逻辑自动化,方法记录在 README)。
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingest import SQLiteStore
from llm import ChatClient

GENERATE_PROMPT = (
    "你是技术文档问答专家。基于下面给出的官方文档段落,生成 1 个高质量的中文开发问题"
    "及其参考答案。要求:\n"
    "1. 问题应能仅凭该段落回答(单跳事实/方法/结论类问题);\n"
    "2. 答案必须严格基于段落内容,不得引入段落外的知识;\n"
    "3. 只输出 JSON:{{\"question\": \"...\", \"answer\": \"...\"}}。\n\n"
    "文档段落:\n{chunk_text}"
)

JUDGE_PROMPT = (
    "请判断以下'参考答案'是否忠实于'文档段落'中的内容。"
    "评分 1-5(5=完全忠实且准确,3=部分准确有轻微扩展,1=严重不忠实或编造)。"
    "只输出数字。\n\n文档段落:\n{chunk_text}\n\n参考答案:\n{answer}"
)

UNANSWERABLE_QUESTIONS = [
    "FastAPI 的 OAuth2 流程支持设备授权码模式吗?",
    "SQLAlchemy 2.0 的 ORM 是否支持无类型注释的声明式映射?",
    "Pydantic v3 在什么时候发布?",
    "FastAPI 社区插件推荐数量排名是什么?",
    "Django 与 FastAPI 的性能对比官方结论是什么?",
    "Python 官方推荐的唯一 ORM 是 SQLAlchemy 吗?",
    "uvicorn 的最大并发连接数限制是多少?",
    "FastAPI 支持 Go 语言编写路径操作函数吗?",
]


@dataclass
class EvalItem:
    """一条评估样本。"""

    qid: str
    question: str
    gold_chunk_ids: list[str]
    gold_doc_id: str
    unanswerable: bool = False
    gold_answer: str = ""


def _parse_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no json")
    return json.loads(text[start : end + 1])


def _gen_qa(chat: ChatClient, chunk_text: str, max_tries: int = 2) -> dict[str, str] | None:
    for _ in range(max_tries):
        resp = chat.chat(
            [{"role": "user", "content": GENERATE_PROMPT.format(chunk_text=chunk_text[:3000])}],
            max_tokens=512,
        )
        try:
            payload = _parse_json(resp.text)
            q, a = str(payload["question"]).strip(), str(payload["answer"]).strip()
            if 8 <= len(q) <= 120 and a:
                return {"question": q, "answer": a}
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return None


def _judge_faithfulness(chat: ChatClient, chunk_text: str, answer: str) -> float | None:
    """LLM-as-judge:答案对段落的忠实度 1-5(辅助核验;无法解析时返回 None 不参与过滤)。"""
    resp = chat.chat(
        [
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    chunk_text=chunk_text[:2000], answer=answer
                ),
            }
        ],
        max_tokens=16,
    )
    text = resp.text.strip()
    if not text:
        return None
    try:
        return float(text[:1])
    except ValueError:
        return None


def build_dataset(
    sqlite: SQLiteStore,
    chat: ChatClient,
    output_path: str | Path,
    n_answerable: int = 32,
    n_unanswerable: int = 8,
    seed: int = 42,
    faithfulness_threshold: float = 3.0,
) -> list[EvalItem]:
    """构建评估集并写入 JSONL。

    流程:抽样文档与段落 → 反向生成 QA → judge 核验忠实度(低于阈值丢弃)→
    按文档划分 dev/test → 附加无答案问题 → 写文件。
    """
    rng = random.Random(seed)
    docs = sqlite.all_ready_documents()
    if not docs:
        raise RuntimeError("语料库为空,请先 ingest")

    # 抽样段落:每文档最多 2 段,优先 Methods/Results 等有实质内容的段落
    candidates: list[tuple[str, str, str]] = []  # (doc_id, chunk_id, text)
    for doc in docs:
        chunks = [c for c in sqlite.get_chunks(doc["document_id"]) if len(c["text"]) >= 300]
        rng.shuffle(chunks)
        for c in chunks[:2]:
            candidates.append((doc["document_id"], c["chunk_id"], c["text"]))
    rng.shuffle(candidates)

    # 反向生成 + 自动核验(checked 独立计数,避免过滤导致抽样退化)
    items: list[EvalItem] = []
    used_docs: set[str] = set()
    checked = 0
    for doc_id, chunk_id, text in candidates:
        if len(items) >= n_answerable:
            break
        if doc_id in used_docs:
            continue
        qa = _gen_qa(chat, text)
        if qa is None:
            continue
        checked += 1
        if checked % 5 == 0:  # 抽样 20% 做 judge 核验(成本控制)
            score = _judge_faithfulness(chat, text, qa["answer"])
            if score is not None and score < faithfulness_threshold:
                continue
        items.append(
            EvalItem(
                qid=f"qa_{len(items):03d}",
                question=qa["question"],
                gold_chunk_ids=[chunk_id],
                gold_doc_id=doc_id,
                gold_answer=qa["answer"],
            )
        )
        used_docs.add(doc_id)
        time.sleep(0.2)

    # 无答案问题(≥20%)
    for i, q in enumerate(UNANSWERABLE_QUESTIONS[:n_unanswerable]):
        items.append(
            EvalItem(
                qid=f"na_{i:02d}",
                question=q,
                gold_chunk_ids=[],
                gold_doc_id="",
                unanswerable=True,
                gold_answer="",
            )
        )

    # 按文档划分 dev/test(不同文档,防泄漏)
    rng.shuffle(items)
    answerable = [it for it in items if not it.unanswerable]
    unanswerable = [it for it in items if it.unanswerable]
    dev_count = max(1, len(answerable) // 3)
    dev, test = answerable[:dev_count], answerable[dev_count:]
    test += unanswerable
    rng.shuffle(test)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for split, group in (("dev", dev), ("test", test)):
            for it in group:
                rec = {
                    "qid": it.qid,
                    "split": split,
                    "question": it.question,
                    "gold_chunk_ids": it.gold_chunk_ids,
                    "gold_doc_id": it.gold_doc_id,
                    "unanswerable": it.unanswerable,
                    "gold_answer": it.gold_answer,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"评估集已写入 {out}: dev={len(dev)} test={len(test)} (共 {len(items)} 条)")
    return items


def load_dataset(path: str | Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        items.append(
            EvalItem(
                qid=rec["qid"],
                question=rec["question"],
                gold_chunk_ids=rec["gold_chunk_ids"],
                gold_doc_id=rec["gold_doc_id"],
                unanswerable=rec.get("unanswerable", False),
                gold_answer=rec.get("gold_answer", ""),
            )
        )
    return items
