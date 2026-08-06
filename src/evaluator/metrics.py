"""评估指标:检索(Recall@k / MRR / nDCG@10)+ 生成(正确性/忠实度/引用/拒答)。

约束 8:LLM-as-judge 仅作辅助评价,不作为唯一标准;报告同时给出规则指标。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from llm import EmbeddingClient
from rag import DirectRAG
from retriever.pipeline import RetrieverPipeline
from retriever.query_prep import QueryPreprocessor

from .dataset import EvalItem

# ---------------------------------------------------------------- 检索指标

def _recall_at(gold: list[str], retrieved: list[str], k: int) -> float:
    top = set(retrieved[:k])
    hits = [g for g in gold if g in top]
    return len(hits) / len(gold) if gold else 0.0


def _mrr(gold: list[str], retrieved: list[str]) -> float:
    gold_set = set(gold)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def _ndcg_at10(gold: list[str], retrieved: list[str]) -> float:
    gold_set = set(gold)
    k = 10
    dcg = 0.0
    for i in range(min(k, len(retrieved))):
        rel = 1.0 if retrieved[i] in gold_set else 0.0
        dcg += rel / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal > 0 else 0.0


def compute_retrieval_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, float]:
    """从预测结果计算检索指标(仅对有答案样本)。"""
    items = [p for p in predictions if not p["unanswerable"]]
    if not items:
        return {}
    n = len(items)
    return {
        "Recall@5": round(
            sum(_recall_at(p["gold_chunk_ids"], p["retrieved_chunk_ids"], 5) for p in items) / n, 4
        ),
        "Recall@10": round(
            sum(_recall_at(p["gold_chunk_ids"], p["retrieved_chunk_ids"], 10) for p in items) / n, 4
        ),
        "MRR": round(
            sum(_mrr(p["gold_chunk_ids"], p["retrieved_chunk_ids"]) for p in items) / n, 4
        ),
        "nDCG@10": round(
            sum(_ndcg_at10(p["gold_chunk_ids"], p["retrieved_chunk_ids"]) for p in items) / n, 4
        ),
    }


# ---------------------------------------------------------------- 生成指标

def _citation_accuracy(p: dict[str, Any]) -> float | None:
    """引用准确率:引用的 chunk 中有多少属于该问题的 gold 证据集。

    无引用样本返回 None(由"引用完整率"负责统计),不拉低准确率。
    """
    cited = {c.chunk_id for c in p.get("citations", [])}
    gold = set(p["gold_chunk_ids"])
    if not cited:
        return None
    return len(cited & gold) / len(cited)


def compute_generation_metrics(
    predictions: list[dict[str, Any]],
    judge_fn: Callable[[str, str, str], tuple[float, float]] | None = None,
) -> dict[str, float]:
    """生成指标。judge_fn 可选:对正确性/忠实度/完整性打分(辅助,非唯一标准)。"""
    answerable = [p for p in predictions if not p["unanswerable"]]
    unanswerable = [p for p in predictions if p["unanswerable"]]

    # 无证据拒答率:无答案样本中正确拒答的比例
    refuse_rate = (
        sum(1 for p in unanswerable if p["refused"]) / len(unanswerable)
        if unanswerable
        else 0.0
    )
    # 引用完整率:有答案样本中至少带 1 条有效引用的比例
    citation_complete = (
        sum(1 for p in answerable if p.get("citations")) / len(answerable)
        if answerable
        else 0.0
    )
    # 引用准确率(平均,仅统计有引用的样本)
    acc_scores: list[float] = []
    for p in answerable:
        score = _citation_accuracy(p)
        if score is not None:
            acc_scores.append(score)
    citation_accuracy = sum(acc_scores) / len(acc_scores) if acc_scores else 0.0
    # 完整性:答案非空且长度合理(规则;judge 存在时用平均分)
    completeness = (
        sum(
            1
            for p in answerable
            if len(p.get("answer", "")) >= 30 and len(p.get("answer", "")) <= 3000
        )
        / len(answerable)
        if answerable
        else 0.0
    )

    metrics: dict[str, float] = {
        "无证据拒答率": round(refuse_rate, 4),
        "引用完整率": round(citation_complete, 4),
        "引用准确率": round(citation_accuracy, 4),
        "完整性(规则)": round(completeness, 4),
    }

    if judge_fn is not None and answerable:
        scores = [
            judge_fn(p["question"], p.get("answer", ""), p.get("gold_answer", ""))
            for p in answerable
        ]
        metrics["正确性(judge)"] = round(sum(s[0] for s in scores) / len(scores), 4)
        metrics["忠实度(judge)"] = round(sum(s[1] for s in scores) / len(scores), 4)
    return metrics


# ---------------------------------------------------------------- direct 评估

def evaluate_direct(
    items: list[EvalItem],
    pipeline: RetrieverPipeline,
    preprocessor: QueryPreprocessor,
    embedder: EmbeddingClient,
    rag: DirectRAG,
) -> list[dict[str, Any]]:
    """在评估集上运行 direct_rag 路径,返回逐条预测。"""
    predictions: list[dict[str, Any]] = []
    for item in items:
        t0 = time.perf_counter()
        prepared = preprocessor.prepare(item.question)
        vector = embedder.embed([item.question])[0]
        retrieval = pipeline.retrieve(vector, item.question, translated_query=prepared.bm25_query)
        ans = rag.answer(prepared, retrieval)
        predictions.append(
            {
                "qid": item.qid,
                "question": item.question,
                "unanswerable": item.unanswerable,
                "gold_chunk_ids": item.gold_chunk_ids,
                "retrieved_chunk_ids": [c.chunk_id for c in retrieval.candidates[:10]],
                "answer": ans.answer,
                "citations": ans.citations,
                "refused": ans.refused,
                "cost_yuan": ans.trace.get("estimated_cost_yuan", 0.0),
                "latency_s": round(time.perf_counter() - t0, 2),
            }
        )
    return predictions


def summarize(predictions: list[dict[str, Any]]) -> dict[str, float]:
    """成本与延迟汇总。"""
    n = len(predictions) or 1
    return {
        "总费用(元)": round(sum(p["cost_yuan"] for p in predictions), 4),
        "平均费用(元/问)": round(sum(p["cost_yuan"] for p in predictions) / n, 6),
        "平均延迟(秒)": round(sum(p["latency_s"] for p in predictions) / n, 2),
    }
