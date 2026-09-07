import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { AppNav } from "@/components/shell/nav";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DocPilot — Python 后端开发文档助手",
  description:
    "基于 FastAPI/Pydantic/SQLAlchemy/Python 官方文档的 RAG Agent:代码检索、引用溯源与执行链路可视化。",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="zh-CN"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-white text-neutral-900">
        <div className="flex h-screen overflow-hidden">
          {/* 左侧品牌 + 导航 */}
          <aside className="flex w-60 shrink-0 flex-col border-r border-neutral-200 bg-white">
            <div className="flex items-center gap-2.5 px-5 py-5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-neutral-900 text-sm font-semibold text-white">
                D
              </div>
              <div className="leading-tight">
                <div className="text-[15px] font-semibold tracking-tight">
                  DocPilot
                </div>
                <div className="text-[11px] text-neutral-400">
                  Python Docs Agent
                </div>
              </div>
            </div>
            <AppNav />
            <div className="mt-auto px-5 py-4 text-[11px] leading-relaxed text-neutral-400">
              官方技术文档问答
              <br />
              FastAPI · Pydantic · SQLAlchemy · Python
            </div>
          </aside>

          {/* 主内容区 */}
          <main className="flex-1 overflow-hidden">{children}</main>
        </div>
      </body>
    </html>
  );
}
