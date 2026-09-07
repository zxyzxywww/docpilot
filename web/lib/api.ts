// API 客户端:浏览器直连后端(server/,端口 8000)
import type {
  ChatMessage,
  ChatResponse,
  EvalSummary,
  KnowledgeDocument,
  SessionInfo,
  Stats,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// ------------------------------------------------------------ Chat / 会话

export function chat(
  question: string,
  mode: string,
  sessionId?: string,
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ question, mode, session_id: sessionId }),
  });
}

export function listSessions(): Promise<SessionInfo[]> {
  return request<SessionInfo[]>("/api/sessions");
}

export function getMessages(sessionId: string): Promise<ChatMessage[]> {
  return request<ChatMessage[]>(`/api/sessions/${sessionId}/messages`);
}

export function deleteSession(sessionId: string): Promise<{ ok: boolean }> {
  return request(`/api/sessions/${sessionId}`, { method: "DELETE" });
}

// ------------------------------------------------------------ 文档库

export function listDocuments(): Promise<KnowledgeDocument[]> {
  return request<KnowledgeDocument[]>("/api/documents");
}

export function deleteDocument(documentId: string): Promise<{ ok: boolean }> {
  return request(`/api/documents/${documentId}`, { method: "DELETE" });
}

export async function uploadDocument(file: File): Promise<{ ok: boolean; document_id: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

// ------------------------------------------------------------ 统计 / 评估

export function getStats(): Promise<Stats> {
  return request<Stats>("/api/stats");
}

export function getEvalSummary(): Promise<EvalSummary> {
  return request<EvalSummary>("/api/eval/summary");
}
