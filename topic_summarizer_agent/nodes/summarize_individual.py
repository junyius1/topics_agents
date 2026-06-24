"""Summarize individual node: generates summaries for each article."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from topic_summarizer_agent.config import AgentConfig
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


def _create_llm(agent_config: AgentConfig) -> ChatGoogleGenerativeAI:
    """Create a Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model=agent_config.gemini_model,
        google_api_key=agent_config.gemini_api_key,
        temperature=0.2,
        max_tokens=agent_config.single_summary_max_tokens,
    )


def _summarize_one(article: Article, agent_config: AgentConfig) -> Summary:
    """Summarize a single article using Gemini."""
    llm = _create_llm(agent_config)

    # Truncate content if too long for the prompt
    content = article.content
    if len(content) > 12000:
        content = content[:12000] + "\n\n[... content truncated for summarization ...]"

    prompt = SUMMARIZE_PROMPT.format(
        max_tokens=agent_config.single_summary_max_tokens,
        title=article.title or "Untitled",
        url=article.url,
        content=content,
    )

    try:
        response = llm.invoke(prompt)
        summary_text = response.content.strip()
    except Exception as exc:
        logger.error("Failed to summarize '%s': %s", article.url, exc)
        summary_text = f"[Summary generation failed: {exc}]"

    return Summary(
        title=article.title or "Untitled",
        url=article.url,
        summary=summary_text,
    )


async def _summarize_batch(
    articles: list[Article],
    config: AgentConfig,
    max_concurrent: int,
) -> list[Summary]:
    """Summarize articles with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(article: Article) -> Summary:
        async with semaphore:
            return _summarize_one(article, config)

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
    return summaries


def summarize_individual(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Generate individual summaries for all articles.

    Uses Gemini API to summarize each article in parallel.
    """
    agent_config = AgentConfig.from_dict(config["configurable"])
    articles = state.articles
    if not articles:
        return {"error": "No articles to summarize"}

    logger.info("Summarizing %d articles", len(articles))

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


def _run_summarize(articles: list[Article], config: AgentConfig) -> list[Summary]:
    """Synchronous wrapper around async summarization."""
    return asyncio.run(_summarize_batch(articles, config, config.max_concurrent_summaries))
