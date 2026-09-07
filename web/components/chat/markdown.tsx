"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Markdown 渲染(回答区):支持表格/代码/链接,外链新窗口打开
export function Markdown({ children }: { children: string }) {
  return (
    <div className="prose prose-neutral max-w-none prose-sm">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children: c }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-neutral-900 underline decoration-neutral-300 underline-offset-2 hover:decoration-neutral-900"
            >
              {c}
            </a>
          ),
          code: ({ className, children: c }) => {
            const inline = !className;
            return inline ? (
              <code className="rounded bg-neutral-100 px-1 py-0.5 text-[13px] text-neutral-900">
                {c}
              </code>
            ) : (
              <code
                className={`block overflow-x-auto rounded-lg bg-neutral-950 p-3 text-[13px] text-neutral-100 ${className ?? ""}`}
              >
                {c}
              </code>
            );
          },
          table: ({ children: c }) => (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-[13px]">{c}</table>
            </div>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
