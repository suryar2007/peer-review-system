"""K2 Think V2 reasoning node: claim verification and statistical audits."""

from __future__ import annotations

from typing import Any

from agents.k2 import K2ThinkClient
from pipeline.state import PipelineState


def reasoner_node(state: PipelineState) -> dict[str, Any]:
    claims = state.get("claims") or []
    resolved_citations = state.get("resolved_citations") or []
    statistical_assertions = state.get("statistical_assertions") or []

    if not claims and not statistical_assertions:
        return {
            "verification_results": [],
            "statistical_audit_results": [],
            "errors": ["reasoner_node: no claims or statistical assertions to verify"],
            "current_phase": "reasoning_complete",
        }

    client = K2ThinkClient.from_config()
    verification_results: list[Any] = []
    statistical_audit_results: list[Any] = []

    if claims:
        verification_results = client.verify_claims(
            claims=list(claims),
            resolved_citations=list(resolved_citations),
        )
    if statistical_assertions:
        statistical_audit_results = client.audit_statistical_assertions(
            list(statistical_assertions),
        )

    return {
        "verification_results": verification_results,
        "statistical_audit_results": statistical_audit_results,
        "current_phase": "reasoning_complete",
    }
