"""会话级多轮记忆:保留最近 N 轮对话,注入 Agent 上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai.types.chat import ChatCompletionMessageParam


@dataclass
class ConversationMemory:
    """多轮记忆(约束 9:历史注入;只保留最近 max_rounds 轮控制 token)。"""

    max_rounds: int = 5
    _messages: list[ChatCompletionMessageParam] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._trim()

    def history(self) -> list[ChatCompletionMessageParam]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        # 每轮 = 1 user + 1 assistant,保留最近 max_rounds 轮
        keep = self.max_rounds * 2
        if len(self._messages) > keep:
            self._messages = self._messages[-keep:]
