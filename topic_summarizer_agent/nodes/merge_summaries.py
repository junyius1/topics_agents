"""Merge summaries node: consolidates individual summaries into a final summary."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from topic_summarizer_agent.config import AgentConfig
from topic_summarizer_agent.models import AgentState, Summary

logger = logging.getLogger(__name__)

MERGE_PROMPT = """You are a senior research analyst. You have been given {num_summaries} individual article summaries on the topic "{topic}".

Your task is to create a comprehensive, well-structured final summary that:
1. Synthesizes the key themes and insights across ALL summaries
2. Identifies common patterns, agreements, and disagreements among the sources
3. Organizes the content into logical sections with clear headings
4. Maintains factual accuracy — only include information present in the source summaries
5. Provides a balanced overview of the topic

Format the final summary in Markdown with:
- A brief introduction (1-2 paragraphs)
- Thematic sections with headings (###)
- A conclusion with key takeaways
- A reference list at the end with all {num_summaries} articles numbered, showing title and URL

Here are the individual summaries:

{summaries_text}

Generate the final summary now:
"""


def _create_llm(agent_config: AgentConfig) -> ChatGoogleGenerativeAI:
    """Create a Gemini LLM instance for merging."""
    return ChatGoogleGenerativeAI(
        model=agent_config.gemini_model,
        google_api_key=agent_config.gemini_api_key,
        temperature=0.1,
        max_tokens=agent_config.merge_summary_max_tokens,
    )


def _build_summaries_text(summaries: list[Summary]) -> str:
    """Format summaries as numbered text for the merge prompt."""
    parts = []
    for i, s in enumerate(summaries, 1):
        parts.append(f"--- Article {i}: {s.title} ---\nURL: {s.url}\nSummary:\n{s.summary}\n")
    return "\n".join(parts)


def merge_summaries(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Merge individual article summaries into a single comprehensive summary.

    Uses Gemini API to synthesize all individual summaries into a structured
    final summary with thematic sections and a reference list.
    """
    agent_config = AgentConfig.from_dict(config["configurable"])
    summaries = state.summaries
    if not summaries:
        return {"error": "No summaries to merge"}

    logger.info("Merging %d summaries for topic '%s'", len(summaries), state.topic)

    summaries_text = _build_summaries_text(summaries)

    # Truncate if too long
    if len(summaries_text) > 50000:
        summaries_text = summaries_text[:50000] + "\n\n[... truncated ...]"

    llm = _create_llm(agent_config)
    prompt = MERGE_PROMPT.format(
        num_summaries=len(summaries),
        topic=state.topic,
        summaries_text=summaries_text,
    )

    try:
        response = llm.invoke(prompt)
        final_summary = response.content.strip()
    except Exception as exc:
        logger.error("Failed to merge summaries: %s", exc)
        # Fallback: concatenate summaries manually
        final_summary = _fallback_merge(summaries, state.topic)

    logger.info("Generated final summary (%d characters)", len(final_summary))
    return {"final_summary": final_summary}


def _fallback_merge(summaries: list[Summary], topic: str) -> str:
    """Fallback merge when LLM call fails: simple concatenation."""
    lines = [f"# Final Summary: {topic}\n"]
    lines.append("## Overview\n")
    for i, s in enumerate(summaries, 1):
        lines.append(f"### {i}. {s.title}\n")
        lines.append(f"[Source]({s.url})\n")
        lines.append(f"{s.summary}\n")
    return "\n".join(lines)
