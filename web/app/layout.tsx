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
  title: "MediDoc — 医学文献智能知识库",
  description:
    "基于顶刊文献的检索增强问答 Agent:带引用溯源、执行链路可视化与量化评测。",
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
                M
              </div>
              <div className="leading-tight">
                <div className="text-[15px] font-semibold tracking-tight">
                  MediDoc
                </div>
                <div className="text-[11px] text-neutral-400">
                  Literature QA Agent
                </div>
              </div>
            </div>
            <AppNav />
            <div className="mt-auto px-5 py-4 text-[11px] leading-relaxed text-neutral-400">
              医学文献研究辅助
              <br />
              仅基于已入库文献作答
            </div>
          </aside>

          {/* 主内容区 */}
          <main className="flex-1 overflow-hidden">{children}</main>
        </div>
      </body>
    </html>
  );
}
