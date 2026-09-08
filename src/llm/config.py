"""配置加载:config.yaml + .env,统一入口。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 项目根目录 = src/llm/ 向上两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# provider 与 .env 环境变量名的映射
API_KEY_ENV: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
}


class ChatLLMConfig(BaseModel):
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: float = 60.0
    max_retries: int = 3
    backoff_base_seconds: float = 1.0


class EmbeddingLLMConfig(BaseModel):
    provider: str = "siliconflow"
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "BAAI/bge-m3"  # 固定;变更模型/维度/分块策略必须重建索引
    dimension: int = 1024
    batch_size: int = 32
    max_concurrency: int = 4
    timeout_seconds: float = 60.0
    max_retries: int = 3


class RerankLLMConfig(BaseModel):
    provider: str = "siliconflow"
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    timeout_seconds: float = 60.0
    max_retries: int = 3


class RetrievalConfig(BaseModel):
    dense_top_k: int = 30
    bm25_top_k: int = 30
    rrf_top_k: int = 30
    rerank_top_k: int = 20
    final_context_k: int = 6
    rrf_k: int = 60


class ChunkingConfig(BaseModel):
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 100


class QdrantConfig(BaseModel):
    mode: str = "local"  # local(开发) | docker(部署)
    path: str = "data/db/qdrant"
    collection: str = "docpilot_chunks"
    docker_url: str = "http://localhost:6333"


class DatabaseConfig(BaseModel):
    sqlite_path: str = "data/db/docpilot.db"
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)


class BudgetConfig(BaseModel):
    max_cost_yuan: float = 50.0
    max_tokens_per_query: int = 8000
    track_costs: bool = True


class AgentConfig(BaseModel):
    max_steps: int = 8
    tool_timeout_seconds: float = 30.0
    max_tool_retries: int = 2
    max_consecutive_repeat: int = 2
    max_cost_yuan_per_query: float = 0.5


class RagConfig(BaseModel):
    min_relevance_score: float = 0.3  # 检索证据相关性下限(rerank 分数),低于则拒答
    gate_enabled: bool = True  # 灰区语义门控开关(rerank 分数落在灰区时 LLM 判定是否可答)
    gate_threshold: float = 0.6  # 灰区上界:分数低于此才触发门控;高于直接生成(零额外成本)


class AppConfig(BaseModel):
    mvp: dict[str, Any] = Field(default_factory=dict)
    llm: dict[str, Any] = Field(default_factory=dict)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    logging: dict[str, Any] = Field(default_factory=dict)

    @property
    def chat(self) -> ChatLLMConfig:
        return ChatLLMConfig.model_validate(self.llm.get("chat", {}))

    @property
    def embedding(self) -> EmbeddingLLMConfig:
        return EmbeddingLLMConfig.model_validate(self.llm.get("embedding", {}))

    @property
    def rerank(self) -> RerankLLMConfig:
        return RerankLLMConfig.model_validate(self.llm.get("rerank", {}))


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载 config.yaml,并读取 .env 到环境变量(不覆盖已存在的值)。"""
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)


def get_api_key(provider: str) -> str:
    """按 provider 读取 API key;未配置则报错(指向 .env.example)。"""
    env_name = API_KEY_ENV[provider]
    key = os.getenv(env_name, "")
    if not key:
        raise RuntimeError(f"缺少环境变量 {env_name}:请在 .env 中配置(参考 .env.example)")
    return key
