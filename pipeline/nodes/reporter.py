"""Hex dashboard trigger node: push results and compute summary statistics."""

from __future__ import annotations

from typing import Any

from pipeline.state import PipelineState
from utils.hex_client import HexClient


def _hex_payload(state: PipelineState) -> dict[str, Any]:
    citations = state.get("citations") or []
    claims = state.get("claims") or []
    resolved = state.get("resolved_citations") or []
    verifications = state.get("verification_results") or []
    audits = state.get("statistical_audit_results") or []
    stats = state.get("statistical_assertions") or []
    return {
        "citations": [c.model_dump(mode="json") for c in citations],
        "claims": [c.model_dump(mode="json") for c in claims],
        "statistical_assertions": [s.model_dump(mode="json") for s in stats],
        "resolved_citations": [c.model_dump(mode="json") for c in resolved],
        "verification_results": [v.model_dump(mode="json") for v in verifications],
        "statistical_audit_results": [a.model_dump(mode="json") for a in audits],
        "errors": list(state.get("errors") or []),
    }


def reporter_node(state: PipelineState) -> dict[str, Any]:
    citations = state.get("citations") or []
    resolved_citations = state.get("resolved_citations") or []
    verification_results = state.get("verification_results") or []

    total_citations = len(citations)
    resolved_count = sum(1 for c in resolved_citations if c.resolved)
    hallucinated_count = sum(1 for c in resolved_citations if c.exists is False)
    supported_claims = sum(1 for v in verification_results if v.verdict == "supported")
    flagged_claims = sum(
        1
        for v in verification_results
        if v.verdict in ("overstated", "contradicted", "unverifiable")
    )

    payload = _hex_payload(state)
    client = HexClient.from_config()
    run_id = client.trigger_dashboard_update(payload)
    dashboard_url = None

    return {
        "hex_run_id": run_id,
        "dashboard_url": dashboard_url,
        "total_citations": total_citations,
        "resolved_count": resolved_count,
        "hallucinated_count": hallucinated_count,
        "supported_claims": supported_claims,
        "flagged_claims": flagged_claims,
        "current_phase": "reporting_complete",
    }
