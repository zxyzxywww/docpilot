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
                mode         TEXT NOT NULL DEFAULT 'auto',
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,          -- user | assistant
                content     TEXT NOT NULL,
                extra       TEXT NOT NULL DEFAULT '{}',  -- JSON: citations/stop_reason
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON chat_messages(session_id, id);
            CREATE TABLE IF NOT EXISTS chat_runs (
                run_id       TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',  -- 任务状态
                mode         TEXT NOT NULL DEFAULT 'auto',
                question     TEXT NOT NULL,
                error        TEXT,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_runs_session
                ON chat_runs(session_id, status);
            """
        )
        # 旧库迁移:chat_sessions 增加 mode 列(历史会话默认 auto)
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(chat_sessions)")}
        if "mode" not in cols:
            self._conn.execute(
                "ALTER TABLE chat_sessions ADD COLUMN mode TEXT NOT NULL DEFAULT 'auto'"
            )
        self._conn.commit()

    # ------------------------------------------------------------ 会话

    def create_session(self, title: str = "新对话", mode: str = "auto") -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_sessions(session_id, title, mode, created_at, updated_at)"
                " VALUES(?,?,?,?,?)",
                (session_id, title, mode, ts, ts),
            )
            self._conn.commit()
        return {
            "session_id": session_id,
            "title": title,
            "mode": mode,
            "created_at": ts,
            "updated_at": ts,
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.session_id, s.title, s.mode, s.created_at, s.updated_at,
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

    def update_mode(self, session_id: str, mode: str) -> None:
        ts = _now()
        with self._lock:
            self._conn.execute(
                "UPDATE chat_sessions SET mode=?, updated_at=? WHERE session_id=?",
                (mode, ts, session_id),
            )
            self._conn.commit()

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
                    "DELETE FROM chat_runs WHERE session_id=?", (session_id,)
                )
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


    # ------------------------------------------------------------ Runs / Task

    def create_run(
        self, session_id: str, mode: str, question: str
    ) -> dict[str, Any]:
        run_id = uuid.uuid4().hex
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_runs(run_id, session_id, status, mode, question,"
                " created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
                (run_id, session_id, "pending", mode, question, ts, ts),
            )
            self._conn.commit()
        return {
            "run_id": run_id,
            "session_id": session_id,
            "status": "pending",
            "mode": mode,
            "question": question,
            "error": None,
            "created_at": ts,
            "updated_at": ts,
        }

    def update_run(
        self, run_id: str, status: str, error: str | None = None
    ) -> None:
        ts = _now()
        with self._lock:
            self._conn.execute(
                "UPDATE chat_runs SET status=?, error=?, updated_at=? WHERE run_id=?",
                (status, error, ts, run_id),
            )
            self._conn.execute(
                "UPDATE chat_sessions SET updated_at=? WHERE session_id="
                "(SELECT session_id FROM chat_runs WHERE run_id=?)",
                (ts, run_id),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chat_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_runs(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chat_runs WHERE session_id=? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_active_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chat_runs WHERE status IN ('pending','running')"
                " ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
