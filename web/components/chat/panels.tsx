"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileText,
  Plus,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Citation, RagReport, SessionInfo, TraceStep } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Markdown } from "./markdown";

// ---------------------------------------------------------------- 会话列表

export function SessionList({
  sessions,
  activeId,
  running = {},
  draftActive = false,
  onSelect,
  onNew,
  onDelete,
}: {
  sessions: SessionInfo[];
  activeId: string | null;
  running?: Record<string, boolean>;
  draftActive?: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div className="flex h-full flex-col">
      <div className="px-3 pt-3">
        <Button
          variant="outline"
          className="w-full justify-start gap-2 border-neutral-200 text-neutral-700 hover:bg-neutral-50"
          onClick={onNew}
        >
          <Plus className="h-4 w-4" /> 新对话
        </Button>
      </div>
      <div className="mt-2 flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
        {/* 进行中的草稿会话(未发送首条前即立即可见、高亮) */}
        {draftActive && (
          <div className="flex cursor-pointer items-center gap-2 rounded-lg bg-neutral-900 px-2.5 py-2 text-left text-[13px] text-white">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <span className="truncate font-medium">新对话</span>
                <span className="shrink-0 rounded bg-white/15 px-1 py-0.5 text-[10px]">
                  草稿
                </span>
              </div>
              <div className="text-[11px] text-white/50">尚未发送</div>
            </div>
          </div>
        )}
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={cn(
              "group flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[13px] transition-colors",
              s.session_id === activeId
                ? "bg-neutral-100 text-neutral-900"
                : "text-neutral-600 hover:bg-neutral-50",
            )}
            onClick={() => onSelect(s.session_id)}
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <span className="truncate font-medium">{s.title}</span>
                {running[s.session_id] && (
                  <span className="flex shrink-0 items-center gap-1 text-[10px] font-medium text-sky-600">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-500" />
                    生成中
                  </span>
                )}
              </div>
              <div className="text-[11px] text-neutral-400">
                {s.message_count} 条消息 · {s.updated_at.slice(5, 16)}
              </div>
            </div>
            <button
              className="hidden shrink-0 text-neutral-400 hover:text-red-500 group-hover:block"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(s.session_id);
              }}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
        {sessions.length === 0 && (
          <div className="px-3 py-6 text-center text-[12px] text-neutral-400">
            暂无历史对话
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- 引用卡片

export function CitationCard({ citation }: { citation: Citation }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-neutral-200 bg-white">
      <button
        className="flex w-full items-start gap-2 px-3 py-2 text-left"
        onClick={() => setOpen(!open)}
      >
        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-neutral-900 text-[11px] font-medium text-white">
          {citation.index}
        </span>
        <span className="min-w-0 flex-1 text-[13px]">
          <span className="font-medium text-neutral-900">
            {citation.title || "文档"}
          </span>
          <span className="ml-2 text-[11px] text-neutral-400">
            {citation.journal && `${citation.journal} · `}
            {citation.section}
          </span>
        </span>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-neutral-400" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-neutral-400" />
        )}
      </button>
      {open && (
        <div className="border-t border-neutral-100 px-3 py-2">
          <div className="mb-1.5 flex flex-wrap gap-x-3 text-[11px] text-neutral-400">
            <span className="font-mono">{citation.chunk_id}</span>
            {citation.source_url && (
              <a
                href={citation.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-neutral-500 hover:text-neutral-900"
              >
                原文 <ExternalLink className="h-3 w-3" />
              </a>
            )}
          </div>
          <p className="text-[12.5px] leading-relaxed text-neutral-600">
            {citation.evidence}
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- RAG 链路面板

function StepRow({ step, index }: { step: TraceStep; index: number }) {
  const [open, setOpen] = useState(false);
  const failed = step.status !== "done" && step.status !== "ok";
  return (
    <div className="border-b border-neutral-100 last:border-0">
      <button
        className="flex w-full items-center gap-2.5 px-3 py-2 text-left hover:bg-neutral-50"
        onClick={() => setOpen(!open)}
      >
        <span
          className={cn(
            "h-2 w-2 shrink-0 rounded-full",
            failed ? "bg-red-400" : "bg-emerald-400",
          )}
        />
        <span className="flex-1 text-[13px] text-neutral-800">
          <span className="mr-1.5 text-[11px] text-neutral-300">
            {String(index + 1).padStart(2, "0")}
          </span>
          {step.label}
        </span>
        {typeof step.elapsed_s === "number" && (
          <span className="shrink-0 font-mono text-[11px] text-neutral-400">
            {(step.elapsed_s * 1000).toFixed(0)}ms
          </span>
        )}
      </button>
      {open && (
        <div className="px-3 pb-2.5 pt-0.5">
          <pre className="whitespace-pre-wrap break-all rounded-md bg-neutral-50 p-2 font-mono text-[11px] leading-relaxed text-neutral-500">
            {JSON.stringify(step.detail ?? {}, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export function TracePanel({ report }: { report: RagReport | null }) {
  if (!report) return null;
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-neutral-100 px-3 py-2.5">
        <span className="text-[12px] font-medium text-neutral-500">
          RAG 执行链路
        </span>
        <span className="flex items-center gap-2 text-[11px] text-neutral-400">
          <span className="font-mono">{report.total_s}s</span>
          <span className="rounded bg-neutral-100 px-1.5 py-0.5 font-mono">
            ¥{report.total_cost_yuan.toFixed(4)}
          </span>
        </span>
      </div>
      <div className="flex-1 overflow-y-auto py-1">
        {report.steps.map((s, i) => (
          <StepRow key={`${s.key}-${i}`} step={s} index={i} />
        ))}
      </div>
      {report.translated_query && (
        <div className="border-t border-neutral-100 px-3 py-2 text-[11px] text-neutral-400">
          <div className="mb-0.5 font-medium text-neutral-500">改写后查询</div>
          <div className="break-all">{report.translated_query}</div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- 模式与示例

export const EXAMPLE_QUESTIONS = [
  "FastAPI 怎么给接口加 OAuth2 密码流认证?",
  "FastAPI 中 Depends 与子依赖的执行顺序是什么?",
  "FastAPI 里如何限制 UploadFile 的大小与类型?",
  "Pydantic v2 中 model_config 的 extra 选项怎么用?",
];

export function ModeSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex shrink-0 items-center gap-1 rounded-lg border border-neutral-200 bg-white p-0.5 text-[12px]">
      {[
        { k: "auto", label: "自动" },
        { k: "direct", label: "快速问答" },
        { k: "agentic", label: "深度调研" },
      ].map((m) => (
        <button
          key={m.k}
          className={cn(
            "rounded-md px-2.5 py-1 transition-colors",
            value === m.k
              ? "bg-neutral-900 text-white"
              : "text-neutral-500 hover:text-neutral-900",
          )}
          onClick={() => onChange(m.k)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

// 文档图标(供回答头部复用)
export function DocGlyph({ className }: { className?: string }) {
  return <FileText className={cn("h-3.5 w-3.5", className)} />;
}
