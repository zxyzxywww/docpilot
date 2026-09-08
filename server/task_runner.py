"""后台任务执行器:把「RAG/Agent 生成」从 HTTP 请求中剥离。

生命周期:POST /api/chat 只负责「写 user 消息 + 建 run(pending)」后立即返回;
真正的 direct/agentic 生成在线程池中执行,完成后再把 assistant 消息写回同一
session 并把 run 置为 done/failed。这样:
- 浏览器刷新/切走不丢任务(任务状态在 SQLite chat_runs);
- 前端轮询 GET /api/runs/{id} 即可恢复「生成中」并最终拿到结果;
- 会话存在、消息、任务三件事彻底解耦。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session_store import SessionStore

logger = logging.getLogger("docpilot.runs")

# 单用户本地服务:2 个并行 run 足够;真正的并发安全由 SQLite RLock 与只读检索保证
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="docpilot-run")


def start_run(store: SessionStore, run_id: str) -> None:
    _executor.submit(_execute, store, run_id)


def _execute(store: SessionStore, run_id: str) -> None:
    run = store.get_run(run_id)
    if run is None:
        return
    try:
        store.update_run(run_id, "running")
        # 动态取 run_chat,便于测试 patch(server.rag_service.run_chat)
        from server import rag_service as rs

        resp = rs.run_chat(run["question"], mode=run["mode"])
        store.add_message(
            run["session_id"],
            "assistant",
            resp.answer,
            {
                "run_id": run_id,
                "citations": [c.model_dump() for c in resp.citations],
                "mode": resp.mode,
                "refused": resp.refused,
                "stop_reason": resp.stop_reason,
                "report": resp.report.model_dump() if resp.report else None,
            },
        )
        store.update_run(run_id, "done")
    except Exception as exc:  # noqa: BLE001 - 后台任务统一记失败
        logger.exception("run %s failed", run_id)
        store.update_run(run_id, "failed", error=str(exc)[:500])
