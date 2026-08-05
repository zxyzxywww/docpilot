"""token 与费用估算(粗略,以各厂商官网实时价格为准)。"""

from __future__ import annotations

# 估算单价(元 / 百万 token)。
# 注意:DeepSeek 有峰谷定价与自动前缀缓存(命中价低约两个数量级),
# 此处为量级估算,实际费用以官方账单为准。
PRICES: dict[str, dict[str, float]] = {
    "chat": {
        "input": 2.0,  # 输入(缓存未命中)
        "output": 8.0,  # 输出
    },
    "embedding": {"input": 0.3},
    "rerank": {"input": 0.3},
}


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数:英文约 4 字符/token,中文约 1~2 字/token。

    混合文本用 len//3 作为量级估算即可(评估与护栏用,不追求精确)。
    """
    return max(1, len(text) // 3)


def estimate_cost(kind: str, input_tokens: int, output_tokens: int = 0) -> float:
    """按估算单价计算费用(元),保留 6 位小数。"""
    price = PRICES.get(kind, {})
    cost = input_tokens / 1_000_000 * price.get("input", 0.0)
    cost += output_tokens / 1_000_000 * price.get("output", 0.0)
    return round(cost, 6)
