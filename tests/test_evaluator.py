"""评估模块测试:检索指标公式、生成指标、数据集加载(离线)。"""

from __future__ import annotations

import json
from pathlib import Path

from evaluator.dataset import load_dataset
from evaluator.metrics import (
    _mrr,
    _ndcg_at10,
    _recall_at,
    compute_generation_metrics,
    compute_retrieval_metrics,
)


def test_recall_mrr_ndcg_formulas() -> None:
    gold = ["a_c0001"]
    retrieved = ["x_c0001", "a_c0001", "y_c0001"]
    assert _recall_at(gold, retrieved, 5) == 1.0
    assert _recall_at(gold, retrieved, 1) == 0.0
    assert _mrr(gold, retrieved) == 0.5  # 第 2 位命中 → 1/2
    # nDCG@10:唯一命中在第 2 位(0-based 1)→ dcg=1/log2(3);idcg=1/log2(2)
    expected = (1 / 1.58496) / 1.0
    assert abs(_ndcg_at10(gold, retrieved) - expected) < 0.01


def test_retrieval_metrics_aggregate() -> None:
    preds = [
        {
            "unanswerable": False,
            "gold_chunk_ids": ["a_c0001"],
            "retrieved_chunk_ids": ["a_c0001", "b_c0001"],
        },
        {
            "unanswerable": False,
            "gold_chunk_ids": ["c_c0001"],
            "retrieved_chunk_ids": ["d_c0001"],
        },
    ]
    m = compute_retrieval_metrics(preds)
    assert m["Recall@5"] == 0.5
    assert m["MRR"] == (1.0 + 0.0) / 2
    assert m["nDCG@10"] > 0


def test_generation_metrics() -> None:
    preds = [
        # 有答案 + 有引用(gold 内)
        {
            "unanswerable": False,
            "gold_chunk_ids": ["a_c0001"],
            "citations": [_citation("a_c0001")],
            "answer": "x" * 60,
            "refused": False,
        },
        # 有答案但无引用
        {
            "unanswerable": False,
            "gold_chunk_ids": ["b_c0001"],
            "citations": [],
            "answer": "y" * 60,
            "refused": False,
        },
        # 无答案样本:正确拒答 + 未拒答
        {"unanswerable": True, "citations": [], "answer": "证据不足", "refused": True},
        {"unanswerable": True, "citations": [], "answer": "随便答", "refused": False},
    ]
    m = compute_generation_metrics(preds)
    assert m["无证据拒答率"] == 0.5
    assert m["引用完整率"] == 0.5
    assert m["引用准确率"] == 1.0  # 唯一引用命中 gold
    assert m["完整性(规则)"] == 1.0


def _citation(chunk_id: str) -> object:
    from rag.direct import Citation

    return Citation(
        index=1, chunk_id=chunk_id, document_id="d", title="t", source_name="j",
        section="s", page="", paragraph=1, source_url="u", evidence="e",
    )


def test_dataset_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    rows = [
        {"qid": "qa_000", "split": "dev", "question": "q1", "gold_chunk_ids": ["a_c0001"],
         "gold_doc_id": "a", "unanswerable": False, "gold_answer": "A"},
        {"qid": "na_00", "split": "test", "question": "q2", "gold_chunk_ids": [],
         "gold_doc_id": "", "unanswerable": True, "gold_answer": ""},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    items = load_dataset(str(path))
    assert len(items) == 2
    assert items[0].qid == "qa_000" and items[0].unanswerable is False
    assert items[1].unanswerable is True
