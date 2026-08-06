"""Agent 护栏:步数 / 连续重复 / 费用预算(约束 9 全部可配置)。

停止条件:
- max_steps:ReAct 循环最大迭代步数;
- max_consecutive_repeat:连续相同(工具,参数)动作次数上限,防死循环;
- max_cost_yuan_per_query:单次查询累计费用上限。
tool_timeout 与 max_tool_retries 由执行器(loop.py)按 config 应用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm.config import AgentConfig


class StopReason:
    MAX_STEPS = "max_steps"
    REPEAT_ACTION = "repeat_action"
    COST_BUDGET = "cost_budget"
    FINAL = "final_answer"
    FORMAT_ERROR = "format_error"


@dataclass
class Guardrails:
    """每次 Agent 运行一个实例(状态不跨查询共享)。"""

    config: AgentConfig
    _last_actions: list[tuple[str, str]] = field(default_factory=list)  # (tool, args)

    def record_action(self, tool: str, args: str) -> None:
        self._last_actions.append((tool, args))
        # 只保留最近 N 个用于重复检测
        self._last_actions = self._last_actions[-(self.config.max_consecutive_repeat + 1):]

    def repeated_action(self) -> bool:
        """连续 max_consecutive_repeat 次相同动作 → 判定死循环。"""
        n = self.config.max_consecutive_repeat
        if len(self._last_actions) < n:
            return False
        return len(set(self._last_actions[-n:])) == 1

    def over_steps(self, steps: int) -> bool:
        return steps >= self.config.max_steps

    def over_cost(self, cost_yuan: float) -> bool:
        return cost_yuan > self.config.max_cost_yuan_per_query
