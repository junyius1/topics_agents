"""Pydantic models for the topic summarizer agent state and data structures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Article(BaseModel):
    """A single article with its title, URL, and content."""

    title: str
    url: str
    content: str
    excerpt: str = ""


class Summary(BaseModel):
    """A single article summary."""

    title: str
    url: str
    summary: str


class AgentState(BaseModel):
    """LangGraph state for the topic summarizer agent.

    Attributes:
        topic: The topic being researched.
        urls: Candidate URLs from search.
        articles: Successfully downloaded and extracted articles.
        summaries: Individual summaries for each article.
        final_summary: The merged final summary.
        error: Any error that occurred during execution.
    """

    topic: str
    urls: list[str] = Field(default_factory=list)
    articles: list[Article] = Field(default_factory=list)
    summaries: list[Summary] = Field(default_factory=list)
    final_summary: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a serializable dictionary."""
        return {
            "topic": self.topic,
            "urls": self.urls,
            "articles": [article.model_dump() for article in self.articles],
            "summaries": [summary.model_dump() for summary in self.summaries],
            "final_summary": self.final_summary,
            "error": self.error,
        }
