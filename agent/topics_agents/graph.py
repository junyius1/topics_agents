"""LangGraph graph construction for the Topic Agent workflow."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    archive_results,
    download_and_extract,
    merge_summaries,
    search_topics,
    summarize_individual,
)
from .state import TopicState


def build_graph() -> StateGraph:
    """Build and compile the Topic Agent workflow graph."""
    workflow = StateGraph(TopicState)

    # Add nodes
    workflow.add_node("search_topics", search_topics)
    workflow.add_node("download_and_extract", download_and_extract)
    workflow.add_node("summarize_individual", summarize_individual)
    workflow.add_node("merge_summaries", merge_summaries)
    workflow.add_node("archive_results", archive_results)

    # Define edges
    workflow.set_entry_point("search_topics")
    workflow.add_edge("search_topics", "download_and_extract")
    workflow.add_edge("download_and_extract", "summarize_individual")
    workflow.add_edge("summarize_individual", "merge_summaries")
    workflow.add_edge("merge_summaries", "archive_results")
    workflow.add_edge("archive_results", END)

    return workflow.compile()
