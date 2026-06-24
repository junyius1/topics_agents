"""Nodes for the Topic Agent workflow."""

from __future__ import annotations

import asyncio
import logging
import re
import textwrap
from typing import Any

import httpx
import markdownify
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import Article, Summary, TopicState

logger = logging.getLogger(__name__)

# Default timeout for HTTP requests
HTTP_TIMEOUT = 30
# Max retries for HTTP requests
MAX_RETRIES = 3


def _get_llm() -> ChatGoogleGenerativeAI:
    """Get a Gemini LLM instance."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


def _sanitize_title(title: str) -> str:
    """Sanitize a title to be used as a filename."""
    # Remove special characters and limit length
    title = re.sub(r'[<>:"/\\|?*]', "", title)
    title = re.sub(r"\s+", "_", title)
    title = title[:80]  # Limit filename length
    return title or "untitled"


async def search_topics(state: TopicState) -> dict[str, Any]:
    """Search for articles about the given topic.

    Uses web_search to find candidate URLs (25-30 URLs to have buffer for failures).
    """
    from exa_py import Exa

    logger.info(f"Searching for articles about: {state.topic}")

    # Use Exa search (already in dependencies) for topic search
    try:
        exa = Exa()
        results = exa.search(
            state.topic,
            type="auto",
            num_results=30,
            use_autoprompt=True,
        )

        urls = list({result.url for result in results.results if result.url})
        # Remove duplicates and limit to 30
        urls = list(dict.fromkeys(urls))[:30]

        logger.info(f"Found {len(urls)} candidate URLs")

        return {
            "urls": urls,
            "_metadata": {
                "search_provider": "exa",
                "total_found": len(urls),
            },
        }
    except Exception as e:
        logger.warning(f"Exa search failed: {e}. Using fetch_url-based search.")
        # Fallback: try to extract URLs from a web search
        llm = _get_llm()
        response = llm.invoke(
            [
                SystemMessage(content="You are a web search assistant. Return URLs only."),
                HumanMessage(
                    content=f"Find 30 URLs about: {state.topic}. Return only URLs, one per line."
                ),
            ]
        )
        urls = [
            line.strip() for line in response.content.split("\n") if line.strip() and "http" in line
        ]
        urls = list(dict.fromkeys(urls))[:30]
        return {
            "urls": urls,
            "_metadata": {"search_provider": "gemini_fallback", "total_found": len(urls)},
        }


async def download_and_extract(state: TopicState) -> dict[str, Any]:
    """Download and extract article content from URLs.

    Tries to download up to 20 clean articles. Has retry logic and fallbacks.
    """
    logger.info(f"Downloading {len(state.urls)} articles...")

    articles: list[Article] = []
    errors: list[str] = []
    seen_titles: set[str] = set()

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        # Process URLs in batches to avoid overwhelming the network
        batch_size = 5
        for i in range(0, len(state.urls), batch_size):
            batch = state.urls[i : i + batch_size]
            tasks = []
            for url in batch:
                tasks.append(_fetch_single_article(client, url, seen_titles))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                elif result:
                    articles.append(result)
                    seen_titles.add(result.title.lower())

            logger.info(f"Downloaded {len(articles)} articles so far...")

            # Stop if we have enough
            if len(articles) >= 20:
                break

    # If we don't have enough, try remaining URLs
    if len(articles) < 20:
        remaining = [u for u in state.urls if u not in [a.url for a in articles]]
        for url in remaining[: 20 - len(articles)]:
            try:
                article = await _fetch_single_article(client, url, seen_titles)
                if article:
                    articles.append(article)
                    seen_titles.add(article.title.lower())
            except Exception as e:
                errors.append(str(e))
            if len(articles) >= 20:
                break

    logger.info(f"Successfully downloaded {len(articles)} articles, {len(errors)} failures")

    return {
        "articles": articles[:20],  # Ensure max 20
        "_metadata": {
            **state._metadata,
            "download_errors": errors[:10],  # Keep first 10 errors
            "downloaded_count": len(articles),
        },
    }


async def _fetch_single_article(
    client: httpx.AsyncClient, url: str, seen_titles: set[str]
) -> Article | None:
    """Fetch a single article from a URL."""
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url)
            response.raise_for_status()

            # Try to extract clean text using markdownify
            html_content = response.text

            # Simple approach: convert HTML to markdown/text
            text_content = markdownify.markdownify(
                html_content,
                heading_style="ATX",
                list_items=True,
                strip=["img", "script", "style", "nav", "footer", "header"],
            )

            # Clean up whitespace
            text_content = re.sub(r"\n{3,}", "\n\n", text_content)
            text_content = text_content.strip()

            # Extract title from HTML if possible
            title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else url.split("/")[-1]

            # Remove duplicate titles
            if title.lower() in seen_titles:
                return None

            # Ensure content is reasonable length
            if len(text_content) < 100:
                raise ValueError(f"Content too short ({len(text_content)} chars)")

            return Article(
                title=title, url=url, content=text_content[:50000]
            )  # Limit content length

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.warning(f"Failed to fetch {url} after {MAX_RETRIES} attempts: {e}")
                raise
            await asyncio.sleep(2**attempt)  # Exponential backoff

    return None


async def summarize_individual(state: TopicState) -> dict[str, Any]:
    """Generate individual summaries for each article (Map node)."""
    logger.info(f"Summarizing {len(state.articles)} articles...")

    llm = _get_llm()
    summaries: list[Summary] = []

    # Process articles in batches to manage token usage
    batch_size = 4
    for i in range(0, len(state.articles), batch_size):
        batch = state.articles[i : i + batch_size]
        tasks = []

        for article in batch:
            # Truncate content if too long
            content = article.content[:8000]  # Limit to ~8000 chars for summarization

            prompt = textwrap.dedent(f"""\
            Summarize the following article in Chinese, keeping it under 300 words.
            Focus on key points, findings, and conclusions.

            Title: {article.title}
            URL: {article.url}

            Content:
            {content}

            Return your summary in this exact format:
            SUMMARY: <your summary here>
            """)

            tasks.append(_summarize_single(llm, article, prompt))

        batch_summaries = await asyncio.gather(*tasks, return_exceptions=True)
        for summary in batch_summaries:
            if isinstance(summary, Exception):
                logger.warning(f"Failed to summarize article: {summary}")
            else:
                summaries.append(summary)

        logger.info(f"Summarized {len(summaries)} articles so far...")

    return {
        "summaries": summaries,
    }


async def _summarize_single(llm: ChatGoogleGenerativeAI, article: Article, prompt: str) -> Summary:
    """Summarize a single article."""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    summary_text = response.content

    # Extract summary from response
    summary_match = re.search(r"SUMMARY:\s*(.*?)(?:\n|$)", summary_text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else summary_text[:500]

    return Summary(
        title=article.title,
        url=article.url,
        summary=summary,
    )


async def merge_summaries(state: TopicState) -> dict[str, Any]:
    """Merge individual summaries into a final comprehensive summary (Reduce node)."""
    logger.info(f"Merging {len(state.summaries)} summaries...")

    llm = _get_llm()

    # Build combined summaries text
    summaries_text = "\n\n".join(
        [
            f"## {i + 1}. {summary.title}\n{summary.summary}"
            for i, summary in enumerate(state.summaries)
        ]
    )

    prompt = textwrap.dedent(f"""\
    You are a research synthesizer. You have {len(state.summaries)} article summaries about "{state.topic}".

    Your task is to:
    1. Identify key themes and patterns across all summaries
    2. Create a comprehensive, well-structured final summary in Chinese
    3. Group related ideas together
    4. Remove redundancies
    5. Provide an index of all articles at the end

    Here are the individual summaries:

    {summaries_text}

    Return your response in this format:
    # 综合摘要：{state.topic}

    [Your comprehensive summary here, organized by themes]

    ## 文章索引

    [List all {len(state.summaries)} articles with title and URL]
    """)

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    final_summary = response.content

    return {
        "final_summary": final_summary,
    }


async def archive_results(state: TopicState) -> dict[str, Any]:
    """Archive all results to local filesystem."""
    import os

    logger.info(f"Archiving results for: {state.topic}")

    # Create normalized topic directory name
    topic_dir = re.sub(r"[^\w\s-]", "", state.topic).strip().lower()
    topic_dir = re.sub(r"[-\s]+", "-", topic_dir)
    base_dir = f"./archive/{topic_dir}"

    articles_dir = f"{base_dir}/articles"
    summaries_dir = f"{base_dir}/summaries"

    os.makedirs(articles_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    # Save articles
    for i, article in enumerate(state.articles[:20], 1):
        filename = f"{i:02d}_{_sanitize_title(article.title)}.md"
        filepath = f"{articles_dir}/{filename}"
        content = f"# {article.title}\n\n**来源**: {article.url}\n\n---\n\n{article.content}"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved article: {filepath}")

    # Save summaries
    for i, summary in enumerate(state.summaries[:20], 1):
        filename = f"{i:02d}_{_sanitize_title(summary.title)}.md"
        filepath = f"{summaries_dir}/{filename}"
        content = f"# {summary.title}\n\n**来源**: {summary.url}\n\n---\n\n{summary.summary}"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved summary: {filepath}")

    # Save final summary
    final_filepath = f"{base_dir}/final_summary.md"
    with open(final_filepath, "w", encoding="utf-8") as f:
        f.write(state.final_summary)
    logger.info(f"Saved final summary: {final_filepath}")

    return {
        "_metadata": {
            **state._metadata,
            "archive_path": base_dir,
            "articles_saved": len(state.articles[:20]),
            "summaries_saved": len(state.summaries[:20]),
        },
    }
