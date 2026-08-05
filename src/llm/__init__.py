"""llm 包:统一对外接口(DeepSeek 对话 + SiliconFlow embedding/rerank)。"""

from .base import (
    APIClientError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    ModelUnavailableError,
)
from .chat import ChatClient, ChatResult
from .config import AppConfig, get_api_key, load_config
from .embedding import EmbeddingClient
from .rerank import RerankClient, RerankResult

__all__ = [
    "APIClientError",
    "APIRateLimitError",
    "APIServerError",
    "APITimeoutError",
    "AppConfig",
    "ChatClient",
    "ChatResult",
    "EmbeddingClient",
    "ModelUnavailableError",
    "RerankClient",
    "RerankResult",
    "get_api_key",
    "load_config",
]
