"""CLI entry point for the Topic Agent."""

from __future__ import annotations

import argparse
import asyncio
import logging

from .graph import build_graph
from .state import TopicState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def run(topic: str) -> None:
    """Run the Topic Agent workflow for a given topic."""
    # Initialize state
    initial_state = TopicState(topic=topic)

    # Get compiled graph
    app = build_graph()

    print(f"Starting research on: {topic}")
    print("-" * 60)

    # Run the workflow
    result = await app.ainvoke(initial_state)

    print("-" * 60)
    print(
        f"Research complete! Results archived to: {result.get('_metadata', {}).get('archive_path', 'unknown')}"
    )
    print(f"Topics covered: {result.get('_metadata', {}).get('articles_saved', 0)} articles")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Topic Agent - Research any topic")
    parser.add_argument("--topic", type=str, required=True, help="The topic to research")
    args = parser.parse_args()

    asyncio.run(run(args.topic))


if __name__ == "__main__":
    main()
