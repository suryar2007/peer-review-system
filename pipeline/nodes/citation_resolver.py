"""Lava MCP retrieval node: resolve citations against academic databases."""

from __future__ import annotations

from typing import Any

from agents.lava_tools import LavaRetrievalClient
from pipeline.state import PipelineState


def citation_resolver_node(state: PipelineState) -> dict[str, Any]:
    citations = state.get("citations") or []
    if not citations:
        return {
            "resolved_citations": [],
            "errors": ["citation_resolver_node: no citations to resolve"],
            "current_phase": "retrieval_complete",
        }

    client = LavaRetrievalClient.from_config()
    resolved = client.resolve_citations(list(citations))
    return {"resolved_citations": resolved, "current_phase": "retrieval_complete"}
