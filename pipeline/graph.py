"""LangGraph pipeline definition for peer-review verification."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from pipeline.nodes.citation_resolver import citation_resolver_node
from pipeline.nodes.extractor import extractor_node
from pipeline.nodes.reasoner import reasoner_node
from pipeline.nodes.reporter import reporter_node
from pipeline.state import PipelineState


def build_graph():
    """Compile the verification workflow as a LangGraph runnable."""
    graph = StateGraph(PipelineState)
    graph.add_node("extract", extractor_node)
    graph.add_node("resolve", citation_resolver_node)
    graph.add_node("reason", reasoner_node)
    graph.add_node("report", reporter_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "resolve")
    graph.add_edge("resolve", "reason")
    graph.add_edge("reason", "report")
    graph.add_edge("report", END)

    return graph.compile()
