"""CLI entrypoint for the topic summarizer agent.

Usage:
    python run_agent.py --topic "Generative AI in Software Engineering"
    python run_agent.py --topic "Rust vs Go" --search-count 40 --target-count 15
"""

from __future__ import annotations

import argparse
import logging
import sys

from topic_summarizer_agent.config import AgentConfig
from topic_summarizer_agent.graph import get_graph
from topic_summarizer_agent.models import AgentState


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Topic Summarizer Agent — search, download, summarize, and archive articles on a topic.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help='Topic to research, e.g. "Generative AI in Software Engineering"',
    )
    parser.add_argument(
        "--search-count",
        type=int,
        default=30,
        help="Number of candidate URLs to search for (default: 30)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
        help="Number of articles to download and summarize (default: 20)",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default=None,
        help="Gemini model to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--archive-root",
        type=str,
        default=None,
        help="Root directory for archives (default: ./archive)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the topic summarizer agent from the command line."""
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("topic_summarizer")

    # Build config from args
    config = AgentConfig(
        search_count=args.search_count,
        target_article_count=args.target_count,
    )
    if args.gemini_model:
        object.__setattr__(config, "gemini_model", args.gemini_model)
    if args.archive_root:
        object.__setattr__(config, "archive_root", args.archive_root)

    # Create initial state
    initial_state = AgentState(topic=args.topic)

    logger.info("=" * 60)
    logger.info("Topic Summarizer Agent")
    logger.info("=" * 60)
    logger.info("Topic : %s", args.topic)
    logger.info("Search count : %d", args.search_count)
    logger.info("Target count : %d", args.target_count)
    logger.info("Gemini model : %s", config.gemini_model)
    logger.info("Archive path : %s", config.archive_root)
    logger.info("=" * 60)

    # Get and run the graph
    graph = get_graph(config)

    logger.info("Starting pipeline...")
    try:
        result = graph.invoke(initial_state, config={"configurable": config.__dict__})
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Print results
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)

    if result.get("error"):
        logger.warning("Pipeline completed with error: %s", result["error"])

    article_count = len(result.get("articles", []))
    summary_count = len(result.get("summaries", []))
    archive_path = result.get("archive_path", "N/A")

    print("\nResults:")
    print(f"  Articles downloaded : {article_count}")
    print(f"  Summaries generated : {summary_count}")
    print(f"  Archive location    : {archive_path}")

    if result.get("final_summary"):
        summary_preview = result["final_summary"][:300]
        if len(result["final_summary"]) > 300:
            summary_preview += "\n[...]"
        print("\nFinal summary preview:")
        print(f"  {summary_preview}")

    print("\nDone.")


if __name__ == "__main__":
    main()
