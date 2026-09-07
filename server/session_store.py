"""server 会话持久化:chat_sessions / chat_messages 表(独立连接,与 ingest 的
documents/chunks 表同库不同表,互不干扰)。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class SessionStore:
    """会话与消息存取(供 Chat 页"历史会话列表"使用)。"""

    def __init__(self, db_path: str | Path):
        self._path = str(db_path)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id   TEXT PRIMARY KEY,
                title        TEXT NOT NULL DEFAULT '新对话',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,          -- user | assistant
                content     TEXT NOT NULL,
                extra       TEXT NOT NULL DEFAULT '{}',  -- JSON: citations/mode/stop_reason
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON chat_messages(session_id, id);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------ 会话

    def create_session(self, title: str = "新对话") -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_sessions(session_id, title, created_at, updated_at)"
                " VALUES(?,?,?,?)",
                (session_id, title, ts, ts),
            )
            self._conn.commit()
        return {"session_id": session_id, "title": title, "created_at": ts, "updated_at": ts}

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM chat_messages m
                        WHERE m.session_id = s.session_id) AS message_count
                FROM chat_sessions s
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chat_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_session_title(self, session_id: str, title: str) -> None:
        ts = _now()
        with self._lock:
            self._conn.execute(
                "UPDATE chat_sessions SET title=?, updated_at=? WHERE session_id=?",
                (title, ts, session_id),
            )
            self._conn.commit()

    def touch_session(self, session_id: str) -> None:
        ts = _now()
        with self._lock:
            self._conn.execute(
                "UPDATE chat_sessions SET updated_at=? WHERE session_id=?", (ts, session_id)
            )
            self._conn.commit()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM chat_messages WHERE session_id=?", (session_id,)
                )
                self._conn.execute(
                    "DELETE FROM chat_sessions WHERE session_id=?", (session_id,)
                )

    # ------------------------------------------------------------ 消息

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_messages(session_id, role, content, extra, created_at)"
                " VALUES(?,?,?,?,?)",
                (session_id, role, content, json.dumps(extra or {}, ensure_ascii=False), ts),
            )
            self._conn.execute(
                "UPDATE chat_sessions SET updated_at=? WHERE session_id=?", (ts, session_id)
            )
            self._conn.commit()

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, extra, created_at FROM chat_messages"
                " WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            try:
                item["extra"] = json.loads(item["extra"])
            except (ValueError, TypeError):
                item["extra"] = {}
            out.append(item)
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
