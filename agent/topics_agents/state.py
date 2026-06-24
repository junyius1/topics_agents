"""Pydantic State definition for the Topic Agent workflow."""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Article(BaseModel):
    """A single fetched article."""

    title: str = Field(description="Article title")
    url: str = Field(description="Article URL")
    content: str = Field(description="Clean text content extracted from the article")


class Summary(BaseModel):
    """A single article summary."""

    title: str = Field(description="Article title")
    url: str = Field(description="Article URL")
    summary: str = Field(description="Summary of the article, max 300 words")


class TopicState(BaseModel):
    """State for the topic research workflow."""

    topic: str = Field(description="The research topic")
    urls: list[str] = Field(
        default_factory=list,
        description="Candidate article URLs discovered via search",
    )
    articles: list[Article] = Field(
        default_factory=list,
        description="Successfully fetched articles with clean text content",
    )
    summaries: list[Summary] = Field(
        default_factory=list,
        description="Individual summaries for each article",
    )
    final_summary: str = Field(
        default="",
        description="Merged final summary combining all individual summaries",
    )

    # Internal: message history for the LLM
    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Message history for LLM interaction",
    )

    # Extra metadata for tracking
    _metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Internal metadata (download attempts, errors, etc.)",
    )
