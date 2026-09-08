"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, BookOpen, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  chat,
  createSession,
  deleteSession,
  getMessages,
  listSessions,
  patchSessionMode,
} from "@/lib/api";
import type { ChatMessage, SessionInfo } from "@/lib/types";
import { Markdown } from "./markdown";
import {
  CitationCard,
  EXAMPLE_QUESTIONS,
  ModeSelect,
  SessionList,
  TracePanel,
} from "./panels";

/**
 * 会话级状态管理:
 * - bySession: 每个会话独立消息缓存(切走不丢进行中内容)
 * - modeBySession: 模式是会话属性(draft 用 draftMode),持久化到后端 chat_sessions.mode
 * - running: 按 session_id 标记执行中,切换会话不取消任务
 * - 生命周期: draft(未发送,不落库)→ 发送瞬间 POST /api/sessions 建会话 →
 *   user 立即写入(前端缓存+后端)→ 回答完成后 assistant 写回同一会话
 */
export function ChatUI() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [bySession, setBySession] = useState<Record<string, ChatMessage[]>>({});
  const [draftMsgs, setDraftMsgs] = useState<ChatMessage[]>([]);
  const [modeBySession, setModeBySession] = useState<Record<string, string>>({});
  const [draftMode, setDraftMode] = useState("auto");
  const [running, setRunning] = useState<Record<string, boolean>>({});
  const [errBySession, setErrBySession] = useState<Record<string, string>>({});
  const [draftErr, setDraftErr] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // 派生:当前渲染内容与模式
  const msgs = activeId ? (bySession[activeId] ?? []) : draftMsgs;
  const curMode = activeId ? (modeBySession[activeId] ?? "auto") : draftMode;
  const isRunning = activeId ? !!running[activeId] : false;
  const curErr = activeId ? (errBySession[activeId] ?? null) : draftErr;

  const refreshSessions = useCallback(async () => {
    try {
      const list = await listSessions();
      setSessions(list);
      // 同步会话级 mode(用于切换/刷新后恢复)
      setModeBySession((prev) => {
        const next = { ...prev };
        for (const s of list) next[s.session_id] = s.mode;
        return next;
      });
    } catch {
      /* 后端不可达时保留旧列表 */
    }
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [msgs, isRunning]);

  const focusInput = () => {
    // 让出当前事件循环,等渲染后再聚焦
    requestAnimationFrame(() => inputRef.current?.focus());
  };

  /** 新建/回到空草稿:只重置本地状态,不创建任何数据库记录(场景 E) */
  const newChat = () => {
    setActiveId(null);
    setDraftMsgs([]);
    setDraftMode("auto");
    setDraftErr(null);
    setInput("");
    focusInput();
  };

  const selectSession = async (id: string) => {
    setActiveId(id);
    setInput("");
    setDraftErr(null);
    try {
      const history = await getMessages(id);
      setBySession((prev) => ({
        ...prev,
        [id]: history.map((m) => ({
          role: m.role,
          content: m.content,
          citations: m.citations ?? [],
          refused: m.refused ?? false,
          mode: m.mode ?? null,
          report: m.report ?? null,
          created_at: m.created_at,
        })),
      }));
    } catch (e) {
      setErrBySession((prev) => ({
        ...prev,
        [id]: e instanceof Error ? e.message : "加载会话失败",
      }));
    }
  };

  const removeSession = async (id: string) => {
    try {
      await deleteSession(id);
      if (id === activeId) {
        newChat();
      } else {
        refreshSessions();
      }
    } catch (e) {
      setErrBySession((prev) => ({
        ...prev,
        [id]: e instanceof Error ? e.message : "删除失败",
      }));
    }
  };

  /** 更新当前会话/草稿的模式(会话级,持久化到后端) */
  const updateMode = (next: string) => {
    if (activeId) {
      setModeBySession((prev) => ({ ...prev, [activeId]: next }));
      patchSessionMode(activeId, next).catch(() => {
        setErrBySession((prev) => ({
          ...prev,
          [activeId]: "模式保存失败,请重试",
        }));
      });
    } else {
      setDraftMode(next);
    }
  };

  const send = async (text?: string) => {
    const q = (text ?? input).trim();
    if (!q || isRunning) return;
    setInput("");
    if (activeId) setErrBySession((prev) => ({ ...prev, [activeId]: "" }));
    else setDraftErr(null);

    // 1) 发送瞬间:若无会话先建会话(draft → 真实 session),会话存在 ≠ 回答完成
    let sid = activeId;
    if (!sid) {
      try {
        const created = await createSession();
        sid = created.session_id;
        setActiveId(sid);
        setDraftMsgs([]);
      } catch {
        setDraftErr("创建会话失败,请确认后端已启动");
        return;
      }
    }
    const mode = modeBySession[sid] ?? "auto";
    const userMsg: ChatMessage = { role: "user", content: q, created_at: "" };

    // 2) user 立即可见(前端缓存 + 后端落库),标记该会话 running
    setBySession((prev) => ({
      ...prev,
      [sid]: [...(prev[sid] ?? []), userMsg],
    }));
    setRunning((prev) => ({ ...prev, [sid]: true }));
    refreshSessions(); // 让左侧立即出现新会话(标题随后台更新)

    // 3) 后台执行回答(同步 API 阻塞本请求,但 UI 可自由切换其他会话)
    try {
      const resp = await chat(q, mode, sid);
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: resp.answer,
        citations: resp.citations,
        refused: resp.refused,
        mode: resp.mode,
        report: resp.report,
        created_at: "",
      };
      // 4) 完成:assistant 写回同一会话(即使当前已切到别处,也按 sid 归位)
      setBySession((prev) => ({
        ...prev,
        [sid]: [...(prev[sid] ?? []), assistantMsg],
      }));
    } catch {
      setErrBySession((prev) => ({
        ...prev,
        [sid]: "回答失败,请重试(会话与问题已保留)",
      }));
    } finally {
      setRunning((prev) => ({ ...prev, [sid]: false }));
      refreshSessions();
    }
  };

  const lastAssistant = [...msgs].reverse().find((m) => m.role === "assistant");
  const showWelcome = msgs.length === 0;

  return (
    <div className="flex h-full">
      {/* 左:会话列表 */}
      <div className="w-64 shrink-0 border-r border-neutral-200 bg-neutral-50/60">
        <SessionList
          sessions={sessions}
          activeId={activeId}
          running={running}
          onSelect={selectSession}
          onNew={newChat}
          onDelete={removeSession}
        />
      </div>

      {/* 中:对话区 */}
      <div className="flex min-w-0 flex-1 flex-col">
        {curErr && (
          <div className="border-b border-amber-200 bg-amber-50 px-6 py-2 text-[13px] text-amber-700">
            {curErr}
          </div>
        )}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {showWelcome && !isRunning ? (
            <Welcome onAsk={(q) => send(q)} />
          ) : (
            <div className="mx-auto max-w-3xl px-6 py-6">
              {msgs.map((m, i) => (
                <MessageBubble key={i} msg={m} />
              ))}
              {isRunning && <LoadingBubble />}
            </div>
          )}
        </div>

        {/* 输入区 */}
        <div className="border-t border-neutral-100 bg-white px-6 py-3">
          <div className="mx-auto max-w-3xl">
            <div className="flex items-end gap-2 rounded-xl border border-neutral-200 bg-white p-2 shadow-sm focus-within:border-neutral-400">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                rows={Math.min(4, Math.max(1, input.split("\n").length))}
                placeholder="输入你的问题,例如:FastAPI 怎么给接口加 JWT 认证?"
                className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-2 py-1.5 text-[14px] outline-none placeholder:text-neutral-400"
              />
              <Button
                size="icon"
                disabled={isRunning || !input.trim()}
                onClick={() => send()}
                className="h-9 w-9 shrink-0 rounded-lg"
              >
                <ArrowUp className="h-4 w-4" />
              </Button>
            </div>
            <div className="mt-2 flex items-center justify-between">
              <ModeSelect value={curMode} onChange={updateMode} />
              <span className="text-[11px] text-neutral-400">
                Enter 发送 · Shift+Enter 换行 · 模式与会话绑定
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 右:RAG 链路 */}
      <div className="w-80 shrink-0 border-l border-neutral-200 bg-neutral-50/60">
        <TracePanel report={lastAssistant?.report ?? null} />
        {!lastAssistant && (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-8 text-center text-[12px] text-neutral-400">
            <BookOpen className="h-5 w-5" />
            <p>发起问答后,这里会展示<br />每一步 RAG 执行链路</p>
          </div>
        )}
      </div>
    </div>
  );
}

function Welcome({ onAsk }: { onAsk: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6">
      <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-2xl bg-neutral-900 text-white shadow">
        <Sparkles className="h-6 w-6" />
      </div>
      <h2 className="text-xl font-semibold tracking-tight text-neutral-900">
        Python 后端开发文档助手
      </h2>
      <p className="mt-1.5 max-w-md text-center text-[13px] leading-relaxed text-neutral-500">
        基于 FastAPI / Pydantic / SQLAlchemy / Python 官方文档回答开发问题,回答含完整代码示例与文档链接溯源;复杂需求自动切换深度调研模式。
      </p>
      <div className="mt-6 grid w-full max-w-xl gap-2">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => onAsk(q)}
            className="rounded-xl border border-neutral-200 bg-white px-4 py-2.5 text-left text-[13px] text-neutral-600 transition-colors hover:border-neutral-300 hover:bg-neutral-50"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function LoadingBubble() {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-1.5 pb-2 text-[12px] text-neutral-400">
        <Skeleton className="h-3 w-3 rounded-full" />
        <span>检索证据并生成回答…(可先切换其他会话,结果会回到本会话)</span>
      </div>
      <div className="space-y-2">
        <Skeleton className="h-4 w-full max-w-xl" />
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
      </div>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === "user") {
    return (
      <div className="mb-5 flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-neutral-900 px-4 py-2.5 text-[14px] leading-relaxed text-white">
          {msg.content}
        </div>
      </div>
    );
  }
  return (
    <div className="mb-5">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="rounded bg-neutral-100 px-1.5 py-0.5 text-[11px] font-medium text-neutral-500">
          {msg.mode === "agentic" ? "深度调研" : "快速问答"}
        </span>
        {msg.refused && (
          <span className="rounded bg-amber-50 px-1.5 py-0.5 text-[11px] text-amber-600">
            证据不足,已拒答
          </span>
        )}
      </div>
      <div className="rounded-2xl border border-neutral-200 bg-white p-4 shadow-sm">
        <Markdown>{msg.content}</Markdown>
        {msg.citations && msg.citations.length > 0 && (
          <div className="mt-3 border-t border-neutral-100 pt-3">
            <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-neutral-400">
              来源引用
            </div>
            <div className="space-y-1.5">
              {msg.citations.map((c) => (
                <CitationCard key={`${c.chunk_id}-${c.index}`} citation={c} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
