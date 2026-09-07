"use client";

import { useEffect, useState } from "react";
import { BarChart3, CircleCheck, Info, Timer } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { getEvalSummary } from "@/lib/api";
import type { EvalSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

const METRIC_META: Record<string, { label: string; tip: string; higher: boolean }> = {
  "Recall@5": { label: "Recall@5", tip: "相关证据在前 5 命中率", higher: true },
  "Recall@10": { label: "Recall@10", tip: "相关证据在前 10 命中率", higher: true },
  MRR: { label: "MRR", tip: "首个相关证据的排名倒数", higher: true },
  "nDCG@10": { label: "nDCG@10", tip: "排序质量(折损累积增益)", higher: true },
  unanswerable_refusal_rate: { label: "无答案拒答率", tip: "库外问题正确拒答比例", higher: true },
  answerable_answer_rate: { label: "有答案答出率", tip: "库内问题给出回答比例", higher: true },
  citation_rate: { label: "引用完整率", tip: "回答带引用溯源的比例", higher: true },
};

const COMPARE_KEYS = [
  ["Recall@5", "Recall@5"],
  ["MRR", "MRR"],
  ["nDCG@10", "nDCG@10"],
] as const;

export default function EvaluationPage() {
  const [data, setData] = useState<EvalSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getEvalSummary()
      .then(setData)
      .catch((e) => setErr(e instanceof Error ? e.message : "加载失败"));
  }, []);

  if (err) {
    return (
      <div className="flex h-full items-center justify-center px-8 text-[13px] text-red-500">
        {err}(请确认后端已启动)
      </div>
    );
  }
  if (!data) {
    return (
      <div className="mx-auto max-w-5xl space-y-4 px-8 py-6">
        <Skeleton className="h-6 w-40" />
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-24 rounded-xl" />
          ))}
        </div>
        <Skeleton className="h-64 rounded-xl" />
      </div>
    );
  }

  const cur = data.current ?? {};
  const baseline = data.comparison?.baseline ?? {};
  const tuned = data.comparison?.tuned ?? {};
  const ds = data.dataset;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-8 py-6">
        <div className="mb-5 flex items-end justify-between">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-neutral-900">
              评测
            </h1>
            <p className="mt-0.5 text-[13px] text-neutral-500">
              RAG 检索与生成质量量化评估
            </p>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-neutral-400">
            <CircleCheck className="h-3.5 w-3.5" />
            {data.available_current ? `基于 ${data.source ?? "predictions"} 实时计算` : "无当前评测"}
          </div>
        </div>

        {/* 数据集信息 */}
        {ds && (
          <Card className="mb-4">
            <CardContent className="flex items-center gap-6 py-3 text-[13px]">
              <Info className="h-4 w-4 text-neutral-300" />
              <span>
                固定测试集 <b className="text-neutral-800">{ds.size}</b> 条
              </span>
              <span>
                无答案 <b className="text-neutral-800">{ds.unanswerable}</b> 条
                ({Math.round(ds.unanswerable_ratio * 100)}%,按文档划分防泄漏)
              </span>
              <span className="text-neutral-300">生成于 {data.generated_at}</span>
            </CardContent>
          </Card>
        )}

        {/* 当前指标 */}
        <h2 className="mb-2 text-[13px] font-medium text-neutral-500">当前指标</h2>
        <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
          {Object.entries(METRIC_META)
            .filter(([k]) => typeof cur[k] === "number")
            .map(([k, meta]) => (
              <Card key={k}>
                <CardContent className="py-3">
                  <div className="text-[12px] text-neutral-400" title={meta.tip}>
                    {meta.label}
                  </div>
                  <div className="mt-0.5 font-mono text-[22px] font-medium tracking-tight text-neutral-900">
                    {(cur[k] as number).toFixed(3)}
                  </div>
                </CardContent>
              </Card>
            ))}
          {typeof cur["平均费用(元/问)"] === "number" && (
            <Card>
              <CardContent className="py-3">
                <div className="text-[12px] text-neutral-400">平均费用(元/问)</div>
                <div className="mt-0.5 font-mono text-[22px] font-medium text-neutral-900">
                  ¥{cur["平均费用(元/问)"] as number}
                </div>
              </CardContent>
            </Card>
          )}
          {typeof cur["平均延迟(秒)"] === "number" && (
            <Card>
              <CardContent className="py-3">
                <div className="text-[12px] text-neutral-400">平均延迟</div>
                <div className="mt-0.5 flex items-center gap-1 font-mono text-[22px] font-medium text-neutral-900">
                  <Timer className="h-4 w-4 text-neutral-300" />
                  {(cur["平均延迟(秒)"] as number).toFixed(1)}s
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* 优化前后对比 */}
        {Object.keys(tuned).length > 0 && (
          <>
            <h2 className="mb-2 text-[13px] font-medium text-neutral-500">
              调参前后对比
            </h2>
            <Card className="mb-4">
              <CardContent className="space-y-3 py-4">
                {COMPARE_KEYS.map(([bKey, tKey]) => {
                  const b = baseline[bKey];
                  const t = tuned[tKey];
                  if (typeof b !== "number" || typeof t !== "number") return null;
                  const max = Math.max(b, t, 0.0001);
                  return (
                    <CompareRow
                      key={bKey}
                      label={bKey}
                      baseline={b}
                      tuned={t}
                      max={max}
                    />
                  );
                })}
                {typeof baseline.citation_accuracy === "number" &&
                  typeof tuned.citation_accuracy === "number" && (
                    <CompareRow
                      label="引用准确率"
                      baseline={baseline.citation_accuracy}
                      tuned={tuned.citation_accuracy}
                      max={1}
                    />
                  )}
                <div className="border-t border-neutral-100 pt-3 text-[12px] leading-relaxed text-neutral-500">
                  {data.tuning_notes}
                </div>
              </CardContent>
            </Card>
          </>
        )}

        {!data.available_current && (
          <Card>
            <CardContent className="py-8 text-center text-[13px] text-neutral-400">
              <BarChart3 className="mx-auto mb-2 h-8 w-8 text-neutral-200" />
              {data.message ?? "暂无评测数据"}
              <div className="mt-1 text-[12px]">
                运行 <code className="rounded bg-neutral-100 px-1">python scripts/evaluate.py --split test</code> 生成
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

function CompareRow({
  label,
  baseline,
  tuned,
  max,
}: {
  label: string;
  baseline: number;
  tuned: number;
  max: number;
}) {
  const up = tuned >= baseline;
  return (
    <div className="flex items-center gap-3">
      <div className="w-24 shrink-0 text-[12px] text-neutral-500">{label}</div>
      <div className="flex-1 space-y-1">
        <Bar label="调参前" value={baseline} max={max} tone="bg-neutral-200" text="text-neutral-500" />
        <Bar label="调参后" value={tuned} max={max} tone="bg-neutral-900" text="text-neutral-900" />
      </div>
      <div
        className={cn(
          "w-12 shrink-0 text-right font-mono text-[12px]",
          up ? "text-emerald-600" : "text-red-500",
        )}
        title="相对变化"
      >
        {up ? "+" : ""}
        {((tuned - baseline) / (baseline || 1)) * 100 < 0.01
          ? "0%"
          : `${(((tuned - baseline) / (baseline || 1)) * 100).toFixed(0)}%`}
      </div>
    </div>
  );
}

function Bar({
  label,
  value,
  max,
  tone,
  text,
}: {
  label: string;
  value: number;
  max: number;
  tone: string;
  text: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-12 shrink-0 text-right font-mono text-[10px] text-neutral-400">
        {label}
      </span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-neutral-50">
        <div
          className={cn("h-full rounded-full transition-all", tone)}
          style={{ width: `${(value / max) * 100}%` }}
        />
      </div>
      <span className={cn("w-12 shrink-0 font-mono text-[11px]", text)}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}
