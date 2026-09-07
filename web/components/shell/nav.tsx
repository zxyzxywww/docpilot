"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Database, Gauge, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/chat", label: "智能问答", desc: "检索增强对话", icon: MessageSquare },
  { href: "/kb", label: "知识库", desc: "文档与索引", icon: Database },
  { href: "/eval", label: "评测", desc: "指标与对比", icon: Gauge },
];

export function AppNav() {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-1 px-3 pt-2">
      {NAV_ITEMS.map((item) => {
        const active =
          pathname === item.href || pathname.startsWith(`${item.href}/`);
        const Icon = item.icon;
        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
              active
                ? "bg-neutral-900 text-white"
                : "text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900",
            )}
          >
            <Icon className="h-4 w-4 shrink-0" />
            <span className="flex flex-col leading-tight">
              <span className="font-medium">{item.label}</span>
              <span
                className={cn(
                  "text-[11px]",
                  active ? "text-neutral-300" : "text-neutral-400",
                )}
              >
                {item.desc}
              </span>
            </span>
          </Link>
        );
      })}
    </nav>
  );
}
