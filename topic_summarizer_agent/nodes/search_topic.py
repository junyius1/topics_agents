"""Search node: finds candidate URLs for a given topic using Exa API."""

from __future__ import annotations

import logging
from typing import Any

from exa_py import Exa
from langchain_core.runnables import RunnableConfig

from topic_summarizer_agent.config import AgentConfig
from topic_summarizer_agent.models import AgentState

logger = logging.getLogger(__name__)


def search_topic(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Search for candidate URLs related to the topic.

    Queries Exa for up to ``config.search_count`` results (default 30),
    deduplicates by normalised URL, and returns the updated state fragment.
    """
    topic = state.topic
    agent_config = AgentConfig.from_dict(config["configurable"])
    logger.info("Searching for topic: %s (target=%d)", topic, agent_config.search_count)

    client = Exa(api_key=agent_config.exa_api_key)

    try:
        results = client.search(
            topic,
            num_results=agent_config.search_count,
            contents={"text": True, "summary": True},
        )
    except Exception as exc:
        logger.error("Exa search failed: %s", exc)
        return {"error": f"Search failed: {exc}"}

    # Deduplicate by normalised URL
    seen: set[str] = set()
    urls: list[str] = []
    for item in results.results:
        url = str(item.url)
        normalised = url.rstrip("/")
        if normalised not in seen:
            seen.add(normalised)
            urls.append(url)

    logger.info("Found %d unique URLs for topic '%s'", len(urls), topic)
    return {"urls": urls}
