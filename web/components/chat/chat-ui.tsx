"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, BookOpen, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  chat,
  deleteSession,
  getMessages,
  listSessions,
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

type Role = ChatMessage["role"];

export function ChatUI() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await listSessions());
    } catch {
      setSessions([]);
    }
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, loading]);

  const newChat = () => {
    setActiveId(null);
    setMessages([]);
    setError(null);
  };

  const selectSession = async (id: string) => {
    setActiveId(id);
    setError(null);
    try {
      const msgs = await getMessages(id);
      // user 消息 content 原样;assistant 消息恢复 citations/report
      setMessages(
        msgs.map((m) => ({
          role: m.role,
          content: m.content,
          citations: m.citations ?? [],
          refused: m.refused ?? false,
          mode: m.mode ?? null,
          report: m.report ?? null,
          created_at: m.created_at,
        })),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载会话失败");
    }
  };

  const removeSession = async (id: string) => {
    try {
      await deleteSession(id);
      if (id === activeId) newChat();
      refreshSessions();
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除失败");
    }
  };

  const submit = async (text?: string) => {
    const q = (text ?? input).trim();
    if (!q || loading) return;
    setInput("");
    setError(null);
    setLoading(true);

    const userMsg: ChatMessage = { role: "user", content: q, created_at: "" };
    const prev = messages;
    setMessages([...prev, userMsg]);

    try {
      const resp = await chat(q, mode, activeId ?? undefined);
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: resp.answer,
        citations: resp.citations,
        refused: resp.refused,
        mode: resp.mode,
        report: resp.report,
        created_at: "",
      };
      setMessages([...prev, userMsg, assistantMsg]);
      setActiveId(resp.session_id);
      refreshSessions();
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败,请确认后端已启动 (python -m uvicorn server.main:app --port 8000)");
      setMessages(prev);
    } finally {
      setLoading(false);
    }
  };

  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const showWelcome = !loading && messages.length === 0;

  return (
    <div className="flex h-full">
      {/* 左:会话列表 */}
      <div className="w-64 shrink-0 border-r border-neutral-200 bg-neutral-50/60">
        <SessionList
          sessions={sessions}
          activeId={activeId}
          onSelect={selectSession}
          onNew={newChat}
          onDelete={removeSession}
        />
      </div>

      {/* 中:对话区 */}
      <div className="flex min-w-0 flex-1 flex-col">
        {error && (
          <div className="border-b border-amber-200 bg-amber-50 px-6 py-2 text-[13px] text-amber-700">
            {error}
          </div>
        )}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {showWelcome ? (
            <Welcome onAsk={(q) => submit(q)} />
          ) : (
            <div className="mx-auto max-w-3xl px-6 py-6">
              {messages.map((m, i) => (
                <MessageBubble key={i} msg={m} />
              ))}
              {loading && <LoadingBubble />}
            </div>
          )}
        </div>

        {/* 输入区 */}
        <div className="border-t border-neutral-100 bg-white px-6 py-3">
          <div className="mx-auto max-w-3xl">
            <div className="flex items-end gap-2 rounded-xl border border-neutral-200 bg-white p-2 shadow-sm focus-within:border-neutral-400">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submit();
                  }
                }}
                rows={Math.min(4, Math.max(1, input.split("\n").length))}
                placeholder="输入你的问题,例如:CycleGAN 在合成 CT 上有哪些局限?"
                className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-2 py-1.5 text-[14px] outline-none placeholder:text-neutral-400"
              />
              <Button
                size="icon"
                disabled={loading || !input.trim()}
                onClick={() => submit()}
                className="h-9 w-9 shrink-0 rounded-lg"
              >
                <ArrowUp className="h-4 w-4" />
              </Button>
            </div>
            <div className="mt-2 flex items-center justify-between">
              <ModeSelect value={mode} onChange={setMode} />
              <span className="text-[11px] text-neutral-400">
                Enter 发送 · Shift+Enter 换行
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 右:RAG 链路 */}
      <div className="w-80 shrink-0 border-l border-neutral-200 bg-neutral-50/60">
        <TracePanel
          report={lastAssistant?.report ?? null}
        />
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
        医学文献智能问答
      </h2>
      <p className="mt-1.5 max-w-md text-center text-[13px] leading-relaxed text-neutral-500">
        基于已入库顶刊文献回答医学影像问题,每个结论都可溯源到原文;复杂问题自动切换深度调研模式。
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
        <span>检索证据并生成回答…</span>
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
