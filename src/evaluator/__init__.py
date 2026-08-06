"""evaluator 包:评估集构建 + 指标计算(阶段五)。"""

from .dataset import EvalItem, build_dataset, load_dataset
from .metrics import (
    compute_generation_metrics,
    compute_retrieval_metrics,
    evaluate_direct,
)

__all__ = [
    "EvalItem",
    "build_dataset",
    "compute_generation_metrics",
    "compute_retrieval_metrics",
    "evaluate_direct",
    "load_dataset",
]
