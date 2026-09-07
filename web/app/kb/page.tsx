"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Database, FileUp, RefreshCw, Trash2, UploadCloud } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { deleteDocument, getStats, listDocuments, uploadDocument } from "@/lib/api";
import type { KnowledgeDocument, Stats } from "@/lib/types";
import { cn } from "@/lib/utils";

const STATUS_BADGE: Record<string, string> = {
  ready: "bg-emerald-50 text-emerald-600 border-emerald-200",
  pending: "bg-neutral-100 text-neutral-500 border-neutral-200",
  indexing: "bg-sky-50 text-sky-600 border-sky-200",
  failed: "bg-red-50 text-red-600 border-red-200",
  deleted: "bg-neutral-100 text-neutral-400 border-neutral-200",
};

function StatusBadge({ status }: { status: string }) {
  return (
    <Badge variant="outline" className={cn("border", STATUS_BADGE[status] ?? "")}>
      {status}
    </Badge>
  );
}

export default function KnowledgeBasePage() {
  const [docs, setDocs] = useState<KnowledgeDocument[] | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const [d, s] = await Promise.all([listDocuments(), getStats()]);
      setDocs(d);
      setStats(s);
    } catch {
      setDocs([]);
      setStats(null);
      setMessage({
        ok: false,
        text: "无法连接后端,请确认 python -m uvicorn server.main:app --port 8000",
      });
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const onUpload = async (file: File | undefined) => {
    if (!file || uploading) return;
    setUploading(true);
    setMessage(null);
    try {
      const r = await uploadDocument(file);
      setMessage({ ok: true, text: `已入库:${r.title || file.name}` });
      load();
    } catch (e) {
      setMessage({ ok: false, text: e instanceof Error ? e.message : "上传失败" });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const onDelete = async (id: string, title: string) => {
    if (!window.confirm(`删除文档「${title.slice(0, 40)}」?将从索引与库中移除。`)) return;
    try {
      await deleteDocument(id);
      setMessage({ ok: true, text: "已删除" });
      load();
    } catch (e) {
      setMessage({ ok: false, text: e instanceof Error ? e.message : "删除失败" });
    }
  };

  const readyCount = docs?.filter((d) => d.status === "ready").length ?? 0;
  const totalChunks = docs?.reduce((s, d) => s + (d.status === "ready" ? d.chunk_count : 0), 0) ?? 0;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-8 py-6">
        {/* 头部 */}
        <div className="mb-5 flex items-end justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-neutral-900">
              知识库
            </h1>
            <p className="mt-0.5 text-[13px] text-neutral-500">
              管理入库文献与检索索引
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={load} className="text-neutral-500">
            <RefreshCw className="mr-1.5 h-3.5 w-3.5" /> 刷新
          </Button>
        </div>

        {message && (
          <div
            className={cn(
              "mb-4 rounded-lg border px-4 py-2 text-[13px]",
              message.ok
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-red-200 bg-red-50 text-red-600",
            )}
          >
            {message.text}
          </div>
        )}

        {/* 统计卡 */}
        <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatCard label="已入库文献" value={stats ? String(readyCount) : undefined} sub="ready" />
          <StatCard label="文本块 (chunk)" value={stats ? String(totalChunks) : undefined} sub="已向量化" />
          <StatCard label="Embedding 模型" value={stats?.embedding_model ?? undefined} sub={`${stats?.embedding_dimension ?? "–"} 维`} />
          <StatCard label="重排模型" value={stats?.rerank_model ?? undefined} sub="Rerank" />
        </div>

        {/* 上传 */}
        <Card className="mb-4">
          <CardContent className="flex items-center justify-between gap-4 py-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-neutral-100 text-neutral-500">
                <FileUp className="h-5 w-5" />
              </div>
              <div>
                <div className="text-[14px] font-medium text-neutral-800">
                  上传文献
                </div>
                <div className="text-[12px] text-neutral-400">
                  支持 XML / PDF,≤20MB;XML 解析优先,扫描版 PDF 将提示不支持
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <input
                ref={fileRef}
                type="file"
                accept=".xml,.pdf"
                className="hidden"
                onChange={(e) => onUpload(e.target.files?.[0])}
              />
              <Button onClick={() => fileRef.current?.click()} disabled={uploading}>
                <UploadCloud className="mr-1.5 h-4 w-4" />
                {uploading ? "上传中…" : "选择文件"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 文档表 */}
        <Card>
          <CardContent className="pt-4">
            {docs === null ? (
              <div className="space-y-2 py-2">
                <Skeleton className="h-9 w-full" />
                <Skeleton className="h-9 w-full" />
                <Skeleton className="h-9 w-2/3" />
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow className="text-[12px]">
                    <TableHead className="w-[40%]">文献</TableHead>
                    <TableHead>类型</TableHead>
                    <TableHead>状态</TableHead>
                    <TableHead className="text-right">chunk</TableHead>
                    <TableHead className="w-[70px]"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {docs.map((d) => (
                    <TableRow key={d.document_id} className="group">
                      <TableCell className="align-top">
                        <div className="text-[13px] font-medium leading-snug text-neutral-800">
                          {d.title || d.document_id}
                        </div>
                        <div className="mt-0.5 text-[11px] text-neutral-400">
                          {d.journal}
                          {d.publication_date && ` · ${d.publication_date}`}
                          {d.embedding_model && ` · ${d.embedding_model}`}
                        </div>
                        {d.error && (
                          <div className="mt-0.5 text-[11px] text-red-400">{d.error}</div>
                        )}
                      </TableCell>
                      <TableCell className="text-[12px] text-neutral-500">
                        {d.document_type}
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={d.status} />
                      </TableCell>
                      <TableCell className="text-right font-mono text-[12px] text-neutral-500">
                        {d.chunk_count}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-neutral-300 hover:text-red-500 group-hover:text-neutral-400"
                          onClick={() => onDelete(d.document_id, d.title)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
            {docs !== null && docs.length === 0 && (
              <div className="flex flex-col items-center gap-2 py-12 text-neutral-400">
                <Database className="h-8 w-8" />
                <p className="text-[13px]">知识库为空,上传第一份文献开始</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 检索配置 */}
        {stats && (
          <div className="mt-4 flex flex-wrap gap-2 text-[11px] text-neutral-400">
            <span className="mr-1 font-medium text-neutral-500">检索配置</span>
            {Object.entries(stats.retrieval)
              .filter(([, v]) => typeof v === "number")
              .map(([k, v]) => (
                <span key={k} className="rounded bg-neutral-100 px-1.5 py-0.5 font-mono">
                  {k}={v}
                </span>
              ))}
            <span className="rounded bg-neutral-100 px-1.5 py-0.5 font-mono">
              chunk={stats.chunking.chunk_size_tokens}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value?: string; sub?: string }) {
  return (
    <Card>
      <CardContent className="py-3.5">
        <div className="text-[12px] text-neutral-400">{label}</div>
        <div className="mt-0.5 truncate font-mono text-[20px] font-medium tracking-tight text-neutral-900">
          {value ?? "—"}
        </div>
        {sub && <div className="text-[11px] text-neutral-300">{sub}</div>}
      </CardContent>
    </Card>
  );
}
