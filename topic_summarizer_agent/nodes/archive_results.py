"""Archive results node: persists articles, summaries, and final summary to disk."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from topic_summarizer_agent.config import AgentConfig
from topic_summarizer_agent.models import AgentState

logger = logging.getLogger(__name__)


def _normalize_topic(topic: str) -> str:
    """Normalize a topic string into a filesystem-safe directory name."""
    # Replace spaces and special chars with hyphens, lowercase
    name = re.sub(r"[^\w\s-]", "", topic)
    name = re.sub(r"[\s]+", "-", name).lower().strip("-")
    return name or "untitled"


def archive_results(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Archive articles, summaries, and final summary to local filesystem.

    Creates the following structure under ``./archive/{normalized_topic}/``:
        articles/   - 20 article files (01_title.md ... 20_title.md)
        summaries/  - 20 summary files (01_title.md ... 20_title.md)
        final_summary.md - merged final summary with references
    """
    agent_config = AgentConfig.from_dict(config["configurable"])
    if state.error:
        logger.warning("Skipping archive due to error: %s", state.error)
        return {}

    topic = state.topic
    normalized = _normalize_topic(topic)
    base_dir = Path(agent_config.archive_root) / normalized

    # Ensure directories exist
    articles_dir = base_dir / "articles"
    summaries_dir = base_dir / "summaries"
    articles_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Archive articles
    for i, article in enumerate(state.articles, 1):
        slug = _make_slug(article.title or f"article-{i}")
        filename = f"{i:02d}_{slug}.md"
        filepath = articles_dir / filename
        filepath.write_text(
            f"# {article.title or f'Article {i}'}\n\n"
            f"**URL:** [{article.url}]({article.url})\n\n"
            f"---\n\n"
            f"{article.content}\n",
            encoding="utf-8",
        )
        logger.debug("Archived article %d: %s", i, filename)

    # Archive summaries
    for i, summary in enumerate(state.summaries, 1):
        slug = _make_slug(summary.title or f"summary-{i}")
        filename = f"{i:02d}_{slug}.md"
        filepath = summaries_dir / filename
        filepath.write_text(
            f"# {summary.title or f'Summary {i}'}\n\n"
            f"**URL:** [{summary.url}]({summary.url})\n\n"
            f"---\n\n"
            f"{summary.summary}\n",
            encoding="utf-8",
        )
        logger.debug("Archived summary %d: %s", i, filename)

    # Archive final summary
    # Prepend reference list
    ref_lines = ["## References\n\n"]
    for i, s in enumerate(state.summaries, 1):
        ref_lines.append(f"{i}. **{s.title}** — [{s.url}]({s.url})\n")

    final_content = f"# Final Summary: {topic}\n\n"
    final_content += f'> Comprehensive summary of {len(state.summaries)} articles on "{topic}"\n\n'
    final_content += "---\n\n"
    final_content += state.final_summary + "\n\n"
    final_content += "---\n\n"
    final_content += "\n".join(ref_lines)

    (base_dir / "final_summary.md").write_text(final_content, encoding="utf-8")
    logger.info("Archived results to %s", base_dir)

    return {
        "archive_path": str(base_dir),
        "articles_count": len(state.articles),
        "summaries_count": len(state.summaries),
    }


def _make_slug(title: str) -> str:
    """Create a filesystem-safe slug from a title."""
    slug = re.sub(r"[^\w\s-]", "", title)
    slug = re.sub(r"[\s]+", "-", slug).lower().strip("-")
    # Limit length
    return slug[:50] if slug else "untitled"
