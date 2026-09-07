"""生成 data/eval/eval_report.json —— Evaluation 页展示的指标数据。

数据来源:
- 当前指标:重新计算 data/eval/predictions_test.jsonl(direct 路径真实评测结果);
- baseline/调参后对比:固化自阶段五调参记录(scripts/evaluate.py + README 量化表)。

用法:
    python scripts/eval_report.py
输出:data/eval/eval_report.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluator.metrics import (  # noqa: E402
    compute_retrieval_metrics,
    summarize,
)

EVAL_DIR = PROJECT_ROOT / "data" / "eval"

# 阶段五调参记录(README 量化表;来源 scripts/evaluate.py --split dev 两轮)
BASELINE = {
    "Recall@5": 0.9545,
    "MRR": 0.8500,
    "nDCG@10": 0.8866,
    "citation_accuracy": 0.63,
    "refusal_rate": 0.50,
}
TUNED = {
    "Recall@5": 0.9545,
    "MRR": 0.9030,
    "nDCG@10": 0.9261,
    "citation_accuracy": 0.77,
    "refusal_rate": 0.50,
}
TUNING_NOTES = (
    "引入证据相关性预检(config.rag.min_relevance_score=0.3:rerank 分数低于阈值直接拒答)"
    " + 精简检索上下文,将引用准确率从 0.63 提升至 0.77、MRR 从 0.85 提升至 0.90;"
    "语义沾边型库外问题仍是拒答难点(README 如实记录)。"
)


def _load_predictions() -> list[dict] | None:
    for name in ("predictions_test.jsonl", "predictions_dev.jsonl"):
        p = EVAL_DIR / name
        if p.exists():
            return [
                json.loads(line)
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    return None


def main() -> int:
    preds = _load_predictions()
    report: dict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": (
            "predictions_test.jsonl"
            if (EVAL_DIR / "predictions_test.jsonl").exists()
            else "predictions_dev.jsonl"
        ),
        "comparison": {"baseline": BASELINE, "tuned": TUNED},
        "tuning_notes": TUNING_NOTES,
    }
    if not preds:
        report["available_current"] = False
        report["message"] = (
            "未找到 predictions 文件,请先运行 python scripts/evaluate.py --split test"
        )
    else:
        n = len(preds)
        unans = sum(1 for p in preds if p.get("unanswerable"))
        current = compute_retrieval_metrics(preds)
        summ = summarize(preds)
        # 生成侧指标从字段直接计算(无答案应拒答;有答案应给出带引用回答)
        n = len(preds)
        ans = [p for p in preds if not p.get("unanswerable")]
        unans = [p for p in preds if p.get("unanswerable")]
        gen = {
            "unanswerable_refusal_rate": round(
                sum(1 for p in unans if p.get("refused")) / len(unans), 4
            ) if unans else None,
            "answerable_answer_rate": round(
                sum(1 for p in ans if not p.get("refused")) / len(ans), 4
            ) if ans else None,
            "citation_rate": round(
                sum(1 for p in ans if p.get("citations")) / len(ans), 4
            ) if ans else None,
        }
        # 生成指标 key 汉化(compute_generation_metrics 输出中文 key)
        report.update(
            {
                "available_current": True,
                "dataset": {
                    "size": n,
                    "unanswerable": len(unans),
                    "unanswerable_ratio": round(len(unans) / n, 3) if n else 0,
                },
                "current": {
                    "Recall@5": current.get("Recall@5"),
                    "Recall@10": current.get("Recall@10"),
                    "MRR": current.get("MRR"),
                    "nDCG@10": current.get("nDCG@10"),
                    **gen,
                    **summ,
                },
            }
        )

    out = EVAL_DIR / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"eval_report 已写入: {out}")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
