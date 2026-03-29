"""Lava MCP retrieval node: resolve citations against academic databases."""

from __future__ import annotations

import logging
from typing import Any

from agents.lava_tools import LavaKnowledgeTools
from pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def citation_resolver_node(state: PipelineState) -> dict[str, Any]:
    """
    Phase 2: Resolve all extracted citations against real databases.

    Uses LavaKnowledgeTools.resolve_citations_batch for concurrent resolution
    with a concurrency limit of 5 to avoid rate limits.
    Tracks counts: how many resolved, how many confirmed to exist, how many failed.
    """
    citations = state.get("citations") or []
    if not citations:
        return {
            "resolved_citations": [],
            "total_citations": 0,
            "errors": ["citation_resolver_node: no citations to resolve"],
            "current_phase": "resolution_complete",
        }

    client = LavaKnowledgeTools.from_config()
    errors: list[str] = []

    try:
        resolved = client.resolve_citations_batch(list(citations), max_concurrency=5)
    except Exception as exc:
        logger.error("Citation resolution failed: %s", exc)
        errors.append(f"citation_resolver_node: resolution failed: {exc}")
        resolved = list(citations)

    resolved_count = sum(1 for c in resolved if c.resolved)
    total = len(resolved)
    logger.info(
        "Citation resolution: %d/%d resolved", resolved_count, total
    )

    result: dict[str, Any] = {
        "resolved_citations": resolved,
        "total_citations": total,
        "current_phase": "resolution_complete",
    }
    if errors:
        result["errors"] = errors
    return result
