"""SessionStore 会话持久化单元测试。"""

from __future__ import annotations

from pathlib import Path

from server.session_store import SessionStore


def test_session_crud(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "chat.db")

    # 创建会话
    s = store.create_session()
    sid = s["session_id"]
    assert store.get_session(sid) is not None
    assert store.list_sessions()[0]["session_id"] == sid

    # 消息与标题
    store.add_message(sid, "user", "你好")
    store.add_message(sid, "assistant", "你好!", {"mode": "direct"})
    msgs = store.list_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["extra"]["mode"] == "direct"

    store.update_session_title(sid, "测试标题")
    assert store.get_session(sid)["title"] == "测试标题"

    # 删除
    store.delete_session(sid)
    assert store.get_session(sid) is None
    assert store.list_messages(sid) == []
    store.close()


def test_session_order_by_updated(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "chat.db")
    a = store.create_session()["session_id"]
    b = store.create_session()["session_id"]
    store.touch_session(a)  # a 最新
    sessions = store.list_sessions()
    assert sessions[0]["session_id"] == a
    assert sessions[1]["session_id"] == b
    store.close()
