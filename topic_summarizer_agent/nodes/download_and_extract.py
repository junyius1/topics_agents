"""Download and extract node: fetches web pages and extracts clean text content."""

from __future__ import annotations

import asyncio
import logging
import re
from html import unescape
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from topic_summarizer_agent.config import AgentConfig
from topic_summarizer_agent.models import AgentState, Article

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags whose direct text content we want to preserve
_CONTENT_TAGS = {
    "article",
    "main",
    "div",
    "section",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "span",
    "blockquote",
}

# Tags to skip entirely
_SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "form"}


def _strip_tags(html: str) -> str:
    """Minimal HTML-to-text: strip tags, collapse whitespace, decode entities."""
    text = html

    # Remove skip tags and their content (naive but covers common cases)
    for tag in _SKIP_TAGS:
        text = re.sub(rf"<{tag}(?:\s[^>]*)?>.*?</{tag}>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Replace block-level tags with newlines
    for tag in _CONTENT_TAGS:
        text = re.sub(rf"</?{tag}(?:\s[^>]*)?>", "\n", text, flags=re.IGNORECASE)

    # Remove any remaining tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    text = unescape(text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


async def _fetch_one(url: str, timeout: int) -> tuple[str, str | None]:
    """Fetch a single URL and return (url, clean_text) or (url, None) on failure."""
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            headers=HEADERS,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = _strip_tags(resp.text)
            # Filter out very short results (< 100 chars likely junk)
            if len(text) < 100:
                return url, None
            return url, text
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return url, None
    except Exception as exc:
        logger.warning("Unexpected error fetching %s: %s", url, exc)
        return url, None


async def _download_with_retry(
    url: str, timeout: int, max_retries: int = 2
) -> tuple[str, str | None]:
    """Retry fetching a URL on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await _fetch_one(url, timeout)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.info("Retrying %s (attempt %d/%d): %s", url, attempt + 1, max_retries, exc)
                await asyncio.sleep(1 * (attempt + 1))
    logger.warning("All retries exhausted for %s: %s", url, last_exc)
    return url, None


async def _download_all(
    urls: list[str],
    timeout: int,
    max_concurrent: int,
) -> list[tuple[str, str | None]]:
    """Download multiple URLs with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded(url: str) -> tuple[str, str | None]:
        async with semaphore:
            return await _download_with_retry(url, timeout)

    tasks = [_bounded(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final: list[tuple[str, str | None]] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Download task raised exception: %s", r)
        else:
            final.append(r)
    return final


def download_and_extract(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Download and extract article content from candidate URLs.

    Returns up to ``config.target_article_count`` successful downloads,
    sorted by content length (longest first) as a quality signal.
    """
    agent_config = AgentConfig.from_dict(config["configurable"])
    urls = state.urls
    if not urls:
        return {"error": "No URLs to download"}

    logger.info(
        "Downloading %d URLs (target=%d, concurrency=%d)",
        len(urls),
        agent_config.target_article_count,
        agent_config.max_concurrent_downloads,
    )

    # Detect if we're already in an async context (LangGraph node)
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if running:
        # We're in an async context (LangGraph node); run in executor
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(_run_download, urls, agent_config)
            articles = future.result()
    else:
        articles = _run_download(urls, agent_config)

    # Sort by content length descending and take top N
    articles.sort(key=lambda a: len(a.content), reverse=True)
    articles = articles[: agent_config.target_article_count]

    logger.info("Successfully extracted %d/%d articles", len(articles), len(urls))
    return {"articles": articles}


def _run_download(urls: list[str], agent_config: AgentConfig) -> list[Article]:
    """Synchronous wrapper around async download logic."""
    results = asyncio.run(
        _download_all(urls, agent_config.download_timeout, agent_config.max_concurrent_downloads)
    )

    articles: list[Article] = []
    for url, text in results:
        if text is None:
            continue
        # Truncate to max content length
        if len(text) > agent_config.max_content_length:
            text = text[: agent_config.max_content_length] + "\n\n[... truncated ...]"
        articles.append(Article(title="", url=url, content=text))

    return articles
