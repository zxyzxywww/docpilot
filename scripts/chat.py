"""最小对话脚本:验证 DeepSeek API 连通与多轮对话。

用法:
    uv run python scripts/chat.py
需要 .env 中已配置 DEEPSEEK_API_KEY(参考 .env.example)。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 使 src 可导入(scripts 目录不在包内,直接运行脚本时生效)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm import ChatClient, get_api_key, load_config  # noqa: E402

SYSTEM_PROMPT = (
    "你是 MediDoc,一个医学文献研究助手。"
    "仅用于公开医学文献检索与研究辅助,不提供诊断或治疗建议。"
    "对没有依据的问题,请明确说明证据不足。"
)


def main() -> None:
    config = load_config()
    api_key = get_api_key("deepseek")
    client = ChatClient(config.chat, api_key)

    print("MediDoc 对话测试(输入 exit 退出)")
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            user_input = input("你: ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user_input})
        result = client.chat(messages)
        print(f"MediDoc: {result.text}")
        print(
            f"[model={result.model} tokens={result.prompt_tokens}+{result.completion_tokens}"
            f" 约¥{result.estimated_cost_yuan}]"
        )
        messages.append({"role": "assistant", "content": result.text})


if __name__ == "__main__":
    main()
