"""LangGraph state graph for the topic summarizer agent."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from topic_summarizer_agent.config import AgentConfig
from topic_summarizer_agent.models import AgentState
from topic_summarizer_agent.nodes import (
    archive_results,
    download_and_extract,
    merge_summaries,
    search_topic,
    summarize_individual,
)


def _create_graph() -> StateGraph:
    """Build and compile the topic summarizer graph.

    Pipeline:
        search_topic → download_and_extract → summarize_individual
        → merge_summaries → archive_results → END
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("search_topic", search_topic)
    builder.add_node("download_and_extract", download_and_extract)
    builder.add_node("summarize_individual", summarize_individual)
    builder.add_node("merge_summaries", merge_summaries)
    builder.add_node("archive_results", archive_results)

    # Set entry point
    builder.set_entry_point("search_topic")

    # Chain edges
    builder.add_edge("search_topic", "download_and_extract")
    builder.add_edge("download_and_extract", "summarize_individual")
    builder.add_edge("summarize_individual", "merge_summaries")
    builder.add_edge("merge_summaries", "archive_results")
    builder.add_edge("archive_results", END)

    return builder.compile()


def get_graph(config: AgentConfig | None = None) -> StateGraph:
    """Get the compiled topic summarizer graph.

    Args:
        config: Optional AgentConfig. If not provided, uses default settings.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    if config is None:
        config = AgentConfig()
    return _create_graph()
