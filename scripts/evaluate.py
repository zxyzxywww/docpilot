"""评估 CLI:构建评估集 → 运行 direct 评估 → 输出指标报告。

用法:
    python scripts/evaluate.py --build                 # 构建评估集(DeepSeek 反向生成+核验)
    python scripts/evaluate.py --split test            # 在测试集上跑 direct 评估
    python scripts/evaluate.py --split dev             # 开发集(调参迭代用)
    python scripts/evaluate.py --split test --judge    # 附加 LLM-as-judge 打分
    python scripts/evaluate.py --compare-agentic       # direct vs agentic 小样本对比

输出:data/eval/predictions_<split>.jsonl + 指标报告(stdout)。
需要 .env 配置 DEEPSEEK_API_KEY 与 SILICONFLOW_API_KEY。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import logging  # noqa: E402

from evaluator.dataset import build_dataset, load_dataset  # noqa: E402
from evaluator.metrics import (  # noqa: E402
    compute_generation_metrics,
    compute_retrieval_metrics,
    evaluate_direct,
    summarize,
)
from ingest import BM25Index, QdrantStore, SQLiteStore  # noqa: E402
from llm import (  # noqa: E402
    ChatClient,
    EmbeddingClient,
    RerankClient,
    get_api_key,
    load_config,
)
from rag import DirectRAG  # noqa: E402
from retriever import BM25Retriever, DenseRetriever, RetrieverPipeline  # noqa: E402
from retriever.query_prep import QueryPreprocessor  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DATASET_PATH = EVAL_DIR / "dataset.jsonl"


def _build_all(config, chat, embedder, rerank):
    sqlite = SQLiteStore(config.database.sqlite_path)
    qdrant = QdrantStore(config.database.qdrant, config.embedding.dimension)
    bm25 = BM25Index()
    ids, texts = [], []
    for doc in sqlite.all_ready_documents():
        for chunk in sqlite.get_chunks(doc["document_id"]):
            ids.append(chunk["chunk_id"])
            texts.append(chunk["text"])
    bm25.build(ids, texts)
    pipeline = RetrieverPipeline(
        DenseRetriever(qdrant), BM25Retriever(bm25, sqlite), rerank, config.retrieval
    )
    rag = DirectRAG(chat, sqlite, config.rag.min_relevance_score)
    preprocessor = QueryPreprocessor(chat)
    return sqlite, qdrant, pipeline, rag, preprocessor


def _judge(chat: ChatClient) -> callable:
    def judge(question: str, answer: str, gold: str) -> tuple[float, float]:
        """正确性 + 忠实度打分(1-5),LLM-as-judge 辅助。"""
        resp = chat.chat(
            [
                {
                    "role": "user",
                    "content": (
                        f"标准答案:{gold}\n模型答案:{answer}\n\n"
                        "请打分(只输出两个数字,空格分隔,各 1-5):\n"
                        "1. 正确性(答案内容是否准确一致)\n2. 忠实度(是否引入标准答案之外的信息)"
                    ),
                }
            ],
            max_tokens=16,
        )
        parts = resp.text.strip().split()
        try:
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return 1.0, 1.0

    return judge


def _print_report(title: str, retrieval: dict, generation: dict, cost: dict) -> None:
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)
    print("[检索指标]")
    for k, v in retrieval.items():
        print(f"  {k:<12} {v}")
    print("[生成指标]")
    for k, v in generation.items():
        print(f"  {k:<16} {v}")
    print("[成本与延迟]")
    for k, v in cost.items():
        print(f"  {k:<16} {v}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MediDoc 评估")
    parser.add_argument("--build", action="store_true", help="构建评估集")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--judge", action="store_true", help="附加 LLM-as-judge 打分")
    parser.add_argument("--compare-agentic", action="store_true", help="direct vs agentic 对比")
    args = parser.parse_args()

    config = load_config()
    logging.getLogger("medidoc.trace").setLevel(logging.CRITICAL)  # 评估时静默观测日志
    chat = ChatClient(config.chat, get_api_key("deepseek"))
    embedder = EmbeddingClient(config.embedding, get_api_key("siliconflow"))
    rerank = RerankClient(config.rerank, get_api_key("siliconflow"))
    sqlite, qdrant, pipeline, rag, preprocessor = _build_all(
        config, chat, embedder, rerank
    )
    try:
        if args.build:
            build_dataset(sqlite, chat, DATASET_PATH)
            return 0

        if not DATASET_PATH.exists():
            print("评估集不存在,请先运行: python scripts/evaluate.py --build")
            return 1

        items = load_dataset(DATASET_PATH)
        split_items = [it for it in items if it.qid in _split_qids(DATASET_PATH, args.split)]
        if not split_items:
            print(f"{args.split} 集为空")
            return 1

        if args.compare_agentic:
            return _compare_agentic(config, split_items, sqlite, pipeline, preprocessor, embedder)

        preds = evaluate_direct(split_items, pipeline, preprocessor, embedder, rag)
        _save(preds, args.split)
        retrieval = compute_retrieval_metrics(preds)
        generation = compute_generation_metrics(preds)
        if args.judge:
            generation = compute_generation_metrics(preds, judge_fn=_judge(chat))
        cost = summarize(preds)
        _print_report(
            f"direct_rag 评估({args.split} 集, {len(preds)} 条)", retrieval, generation, cost
        )
        return 0
    finally:
        sqlite.close()
        qdrant.close()


def _split_qids(path: Path, split: str) -> set[str]:
    return {
        json.loads(line)["qid"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("split") == split
    }


def _save(preds: list[dict], split: str) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"predictions_{split}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
    print(f"预测已保存: {out}")


def _compare_agentic(config, items, sqlite, pipeline, preprocessor, embedder):
    """direct vs agentic 小样本对比(准确性/引用完整率/延迟/费用)。"""
    from agent import AgentLoop, ConversationMemory, ToolContext

    sample = items[:5]  # agent 成本高,取前 5 条对比
    direct_preds = evaluate_direct(sample, pipeline, preprocessor, embedder, DirectRAG(  # type: ignore[arg-type]
        ChatClient(config.chat, get_api_key("deepseek")), sqlite
    ))

    chat = ChatClient(config.chat, get_api_key("deepseek"))
    agent_preds = []
    for it in sample:
        ctx = ToolContext(
            pipeline=pipeline,
            embedder=embedder,
            chat=chat,
            sqlite=sqlite,
            preprocessor=preprocessor,
        )
        loop = AgentLoop(chat, ctx, config.agent, ConversationMemory())
        ans = loop.run(it.question)
        agent_preds.append(
            {
                "qid": it.qid,
                "question": it.question,
                "unanswerable": it.unanswerable,
                "gold_chunk_ids": it.gold_chunk_ids,
                "retrieved_chunk_ids": [],
                "answer": ans.answer,
                "citations": ans.citations,
                "refused": ans.stop_reason != "final_answer",
                "cost_yuan": ans.total_cost_yuan,
                "latency_s": 0.0,
            }
        )

    print(f"\n===== direct vs agentic 对比({len(sample)} 条)=====")
    for name, preds in (("direct", direct_preds), ("agentic", agent_preds)):
        gen = compute_generation_metrics(preds)
        cost = summarize(preds)
        cite = gen.get("引用完整率", 0)
        print(f"\n[{name}] 引用完整率={cite} 平均费用={cost['平均费用(元/问)']} 元/问")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
