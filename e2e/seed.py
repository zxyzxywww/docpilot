"""E2E 测试数据播种(直连 SQLite,不经 RAG):
- setup:清空全部会话 → 造 S2/S3(切走目标,含 1 轮历史)+ S_fail(user 消息 + failed run)
- history:造 E1..E5,每会话 2 轮 user/assistant(第二轮带引用)
用 docpilot 环境 python 运行:python seed.py setup|history|all
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # medidoc 仓库根(e2e/ 的上一级)
sys.path.insert(0, str(ROOT))

from server.session_store import SessionStore  # noqa: E402

DB = ROOT / "data" / "db" / "docpilot.db"


def _store() -> SessionStore:
    return SessionStore(DB)


def _clean(store: SessionStore) -> None:
    for s in store.list_sessions():
        store.delete_session(s["session_id"])


def _inject_round(store, sid, q, a, citations=()):
    store.add_message(sid, "user", q, {"mode": "direct"})
    store.add_message(
        sid,
        "assistant",
        a,
        {
            "mode": "direct",
            "refused": False,
            "stop_reason": "final",
            "citations": list(citations),
            "report": None,
        },
    )


def setup() -> None:
    store = _store()
    _clean(store)
    # S2 / S3:预置真实会话,供「回答中切换」目标
    s2 = store.create_session()
    store.update_session_title(s2["session_id"], "第二个会话-切走目标")
    _inject_round(store, s2["session_id"], "SQLAlchemy session 怎么管理?",
                  "使用 SessionLocal 工厂,每次请求创建、用完关闭。")
    s3 = store.create_session()
    store.update_session_title(s3["session_id"], "第三个会话-快速问答")
    _inject_round(store, s3["session_id"], "Pydantic v2 的 Field 校验?",
                  "Field 用于描述字段默认值与约束,如 min_length。")
    # S_fail:user 在 + failed run(刷新后应显示失败而非消失)
    sf = store.create_session()
    store.update_session_title(sf["session_id"], "失败恢复示例")
    store.add_message(sf["session_id"], "user", "这个任务会失败", {"mode": "direct"})
    run = store.create_run(sf["session_id"], "direct", "这个任务会失败")
    store.update_run(run["run_id"], "failed", error="模拟网络故障(测试注入)")
    print(
        f"setup done: S2={s2['session_id'][:8]} S3={s3['session_id'][:8]} "
        f"SF={sf['session_id'][:8]}"
    )


def history() -> None:
    store = _store()
    for i in range(1, 6):
        s = store.create_session()
        store.update_session_title(s["session_id"], f"历史完整性 E{i}")
        _inject_round(
            store, s["session_id"],
            f"E{i} 第一个问题:FastAPI 路由怎么注册?",
            f"E{i}-A1:使用 @app.get(\"/x\") 装饰器注册路由。",
        )
        _inject_round(
            store, s["session_id"],
            f"E{i} 第二个问题:依赖注入 Depends?",
            f"E{i}-A2:Depends 允许声明依赖,可复用与测试。",
            citations=[
                {"chunk_id": f"chunk-{i}-1", "document_id": f"doc-e{i}",
                 "index": 0, "score": 0.92,
                 "title": f"E{i} 文档A", "content": "Depends 依赖注入说明", "url": "https://example.com/e"},
            ],
        )
    print("history done: E1..E5 (5 会话 × 2 轮)")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("setup", "all"):
        setup()
    if phase in ("history", "all"):
        history()
