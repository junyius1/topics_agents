"""Configuration for the topic summarizer agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the topic summarizer agent.

    All values default to environment variables with sensible fallbacks.
    """

    # API keys
    gemini_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        )
    )
    exa_api_key: str = field(default_factory=lambda: os.environ.get("EXA_API_KEY", ""))

    # Local LLM settings (used for per-article summarization)
    local_llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL_ID", "local:Qwen3.6-35B-A3B-LM-Q8_0")
    )
    local_llm_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "LOCAL_LLM_BASE_URL", "http://localhost:8080/v1"
        )
    )

    # Model settings
    gemini_model: str = field(default="gemini-2.0-flash")
    gemini_embedding_model: str = field(default="text-embedding-004")

    # Search settings
    search_count: int = field(default=30)  # candidate URLs to fetch
    target_article_count: int = field(default=20)
    search_timeout: int = field(default=30)

    # Download settings
    download_timeout: int = field(default=15)
    max_content_length: int = field(default=8000)  # max characters to keep from a page

    # Summary settings
    single_summary_max_tokens: int = field(default=300)
    merge_summary_max_tokens: int = field(default=2000)

    # Archive settings
    archive_root: str = field(default="./archive")

    # Concurrency
    max_concurrent_downloads: int = field(default=5)
    max_concurrent_summaries: int = field(default=5)

    def __post_init__(self) -> None:
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Set it or pass it via AgentConfig."
            )
        if not self.exa_api_key:
            raise ValueError(
                "EXA_API_KEY environment variable is required. Set it or pass it via AgentConfig."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """Create an AgentConfig from a dict, filtering out LangGraph internal keys."""
        known_fields = {
            "gemini_api_key",
            "exa_api_key",
            "gemini_model",
            "gemini_embedding_model",
            "local_llm_model",
            "local_llm_base_url",
            "search_count",
            "target_article_count",
            "search_timeout",
            "download_timeout",
            "max_content_length",
            "single_summary_max_tokens",
            "merge_summary_max_tokens",
            "archive_root",
            "max_concurrent_downloads",
            "max_concurrent_summaries",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
