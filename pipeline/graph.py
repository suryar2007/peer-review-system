"""LangGraph pipeline definition for peer-review verification."""

from __future__ import annotations

import asyncio
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from pipeline.nodes.citation_resolver import citation_resolver_node
from pipeline.nodes.extractor import extractor_node
from pipeline.nodes.reasoner import reasoner_node
from pipeline.nodes.reporter import reporter_node
from pipeline.state import PipelineState, initial_state


def _route_after_extract(state: PipelineState) -> str:
    """Conditional edge: skip to END if extraction produced no citations."""
    citations = state.get("citations") or []
    if not citations:
        return "end"
    return "resolve"


def build_graph():
    """Compile the verification workflow as a LangGraph runnable."""
    graph = StateGraph(PipelineState)
    graph.add_node("extract", extractor_node)
    graph.add_node("resolve", citation_resolver_node)
    graph.add_node("reason", reasoner_node)
    graph.add_node("report", reporter_node)

    graph.set_entry_point("extract")
    graph.add_conditional_edges(
        "extract",
        _route_after_extract,
        {"resolve": "resolve", "end": END},
    )
    graph.add_edge("resolve", "reason")
    graph.add_edge("reason", "report")
    graph.add_edge("report", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_pipeline(paper_path: str) -> PipelineState:
    """
    Run the full peer-review verification pipeline on a PDF.

    Creates initial state, generates a unique thread_id for checkpointing,
    invokes the compiled graph, and returns the final state.
    """
    app = build_graph()
    start = initial_state(paper_path)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"Starting pipeline for: {paper_path}")
    print(f"  Thread ID: {thread_id}")
    final_state = app.invoke(start, config=config)
    return final_state


async def async_run_pipeline(paper_path: str) -> PipelineState:
    """
    Async version of run_pipeline for async contexts.

    Creates initial state, generates a unique thread_id for checkpointing,
    invokes the compiled graph asynchronously, and returns the final state.
    """
    app = build_graph()
    start = initial_state(paper_path)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    final_state = await app.ainvoke(start, config=config)
    return final_state
