"""Summarize individual node: generates summaries for each article."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from topic_summarizer_agent.config import AgentConfig
from topic_summarizer_agent.llm_client import summarize_with_local_llm
from topic_summarizer_agent.models import AgentState, Article, Summary

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """You are a research assistant. Summarize the following article in at most {max_tokens} words.
Focus on the key points, main arguments, and conclusions. Write in clear, concise language.
Do NOT include the title or URL in your summary. Just the summary content.

Article title: {title}
Article URL: {url}

Article content:
{content}

Summary:
"""


async def _summarize_one(article: Article, agent_config: AgentConfig) -> Summary:
    """Summarize a single article using the local LLM."""
    content = article.content
    if len(content) > 12000:
        content = content[:12000] + "\n\n[... content truncated for summarization ...]"

    prompt = SUMMARIZE_PROMPT.format(
        max_tokens=agent_config.single_summary_max_tokens,
        title=article.title or "Untitled",
        url=article.url,
        content=content,
    )

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            summary_text = await summarize_with_local_llm(prompt, agent_config)
            if summary_text:
                return Summary(
                    title=article.title or "Untitled",
                    url=article.url,
                    summary=summary_text,
                )
            logger.warning("Empty summary for '%s' (attempt %d/%d)", article.url, attempt + 1, max_retries)
        except Exception as exc:
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)
            if is_rate_limit and attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "Rate limited on '%s' (attempt %d/%d), retrying in %ds...",
                    article.url,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error("Failed to summarize '%s': %s", article.url, exc)

    return Summary(
        title=article.title or "Untitled",
        url=article.url,
        summary=f"[Summary failed after {max_retries + 1} attempts]",
    )


async def _summarize_batch(
    articles: list[Article],
    agent_config: AgentConfig,
    max_concurrent: int,
) -> list[Summary]:
    """Summarize articles with concurrency control."""
    logger.info(
        "Starting batch summarize: %d articles, concurrency=%d",
        len(articles),
        max_concurrent,
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(article: Article) -> Summary:
        async with semaphore:
            return await _summarize_one(article, agent_config)

    tasks = [_bounded(article) for article in articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries: list[Summary] = []
    for r in results:
        if isinstance(r, Summary):
            summaries.append(r)
        elif isinstance(r, Exception):
            logger.error("Summary task raised: %s", r)
        else:
            logger.error("Unexpected result type: %s", type(r))
    logger.info("Batch complete: %d/%d summaries generated", len(summaries), len(articles))
    return summaries


def summarize_individual(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Generate individual summaries for all articles using the local LLM.

    Uses the configured local LLM (e.g. Qwen) for per-article summarization.
    """
    agent_config = AgentConfig.from_dict(config["configurable"])
    articles = state.articles
    if not articles:
        return {"error": "No articles to summarize"}

    logger.info("Summarizing %d articles with local LLM", len(articles))

    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if running:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(_run_summarize, articles, agent_config)
            summaries = future.result()
    else:
        summaries = _run_summarize(articles, agent_config)

    logger.info("Generated %d/%d summaries", len(summaries), len(articles))
    return {"summaries": summaries}


def _run_summarize(articles: list[Article], agent_config: AgentConfig) -> list[Summary]:
    """Synchronous wrapper around async summarization."""
    return asyncio.run(
        _summarize_batch(articles, agent_config, agent_config.max_concurrent_summaries)
    )
