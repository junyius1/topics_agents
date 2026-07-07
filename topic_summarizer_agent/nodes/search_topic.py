"""Search node: find candidate URLs and content for a topic using Exa API."""

from __future__ import annotations

import logging
from typing import Any

from exa_py import Exa
from langchain_core.runnables import RunnableConfig

from topic_summarizer_agent.config import AgentConfig
from topic_summarizer_agent.models import AgentState, Article

logger = logging.getLogger(__name__)


def search_topic(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Search for candidate URLs related to the topic.

    Queries Exa for up to ``config.search_count`` results (default 30),
    deduplicates by normalised URL, and builds articles directly from
    Exa's full-text response (no extra download step).
    """
    topic = state.topic
    agent_config = AgentConfig.from_dict(config["configurable"])
    logger.info("Searching for topic: %s (target=%d)", topic, agent_config.search_count)

    client = Exa(api_key=agent_config.exa_api_key)

    try:
        results = client.search(
            topic,
            num_results=agent_config.search_count,
            contents={"text": True, "summary": False},
        )
    except Exception as exc:
        logger.error("Exa search failed: %s", exc)
        return {"error": f"Search failed: {exc}"}

    # Deduplicate by normalised URL, preserving titles from Exa
    seen: set[str] = set()
    articles: list[Article] = []
    for item in results.results:
        url = str(item.url)
        normalised = url.rstrip("/")
        if normalised in seen:
            continue
        seen.add(normalised)

        title = item.title or ""
        content = item.text or ""
        if len(content) < 100:
            continue

        articles.append(Article(title=title, url=url, content=content))

    logger.info("Found %d unique articles for topic '%s'", len(articles), topic)
    return {
        "articles": articles,
        "urls": [a.url for a in articles],
    }
