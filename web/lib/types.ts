// 与 server/schemas.py 对齐的 API 类型

export interface Citation {
  index: number;
  chunk_id: string;
  document_id: string;
  title: string;
  source_name: string;
  section: string;
  page: string;
  paragraph: number;
  source_url: string;
  evidence: string;
  score?: number | null;
}

export interface TraceStep {
  key:
    | "query_understanding"
    | "query_rewrite"
    | "hybrid_retrieval"
    | "rerank"
    | "context"
    | "final_generation"
    | "tool_call";
  label: string;
  status: string;
  elapsed_s?: number | null;
  detail: Record<string, unknown>;
}

export interface RagReport {
  steps: TraceStep[];
  translated_query?: string | null;
  expanded_terms: string[];
  recall_chunk_ids: string[];
  context_count: number;
  total_cost_yuan: number;
  total_s: number;
  mode: string;
}


export interface RunCreated {
  session_id: string;
  run_id: string;
  status: string;
}

export interface RunInfo {
  run_id: string;
  session_id: string;
  status: string; // pending | running | done | failed | stopped
  mode: string;
  question: string;
  error?: string | null;
  created_at: string;
  updated_at: string;
}
export interface ChatResponse {
  session_id: string;
  answer: string;
  citations: Citation[];
  refused: boolean;
  mode: string;
  stop_reason?: string | null;
  report: RagReport | null;
}

export interface SessionInfo {
  session_id: string;
  title: string;
  mode: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  mode?: string | null;
  refused?: boolean;
  stop_reason?: string | null;
  report?: RagReport | null;
  created_at: string;
}

export interface KnowledgeDocument {
  document_id: string;
  title: string;
  source_name: string;
  publication_date: string;
  document_type: string;
  status: string;
  chunk_count: number;
  embedding_model: string;
  source_url: string;
  error?: string | null;
  updated_at: string;
}

export interface Stats {
  total_documents: number;
  total_chunks: number;
  embedding_model: string;
  embedding_dimension: number;
  rerank_model: string;
  retrieval: Record<string, number>;
  chunking: Record<string, number>;
  qdrant: Record<string, string>;
}

export interface EvalSummary {
  available?: boolean;
  available_current?: boolean;
  generated_at?: string;
  source?: string;
  dataset?: { size: number; unanswerable: number; unanswerable_ratio: number };
  current?: Record<string, number>;
  comparison?: {
    baseline: Record<string, number>;
    tuned: Record<string, number>;
  };
  tuning_notes?: string;
  message?: string;
}
