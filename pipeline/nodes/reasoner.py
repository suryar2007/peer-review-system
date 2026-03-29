"""Reasoning node: claim verification and statistical audits via Hermes/K2."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents.k2 import K2ReasoningAgent
from pipeline.state import PipelineState, VerificationResult

logger = logging.getLogger(__name__)

_MAX_CLAIM_CONCURRENCY = 5


def reasoner_node(state: PipelineState) -> dict[str, Any]:
    """
    Phase 3: Claim verification and statistical audit.

    For each claim, gathers available evidence (resolved abstracts OR raw citation
    text as fallback) and asks the reasoning model to assess support. Claims with
    zero evidence of any kind are marked unverifiable.

    Runs claim verification concurrently, then batches statistical assertions.
    """
    claims = state.get("claims") or []
    resolved_citations = state.get("resolved_citations") or []
    statistical_assertions = state.get("statistical_assertions") or []
    paper_text = state.get("paper_text") or ""

    if not claims and not statistical_assertions:
        return {
            "verification_results": [],
            "statistical_audit_results": [],
            "errors": ["reasoner_node: no claims or statistical assertions to verify"],
            "current_phase": "reasoning_complete",
        }

    client = K2ReasoningAgent.from_config()
    errors: list[str] = []

    verification_results: list[VerificationResult] = []
    if claims:
        try:
            verification_results = _verify_claims_concurrent(
                client, claims, resolved_citations, paper_text,
            )
        except Exception as exc:
            logger.error("Claim verification failed: %s", exc)
            errors.append(f"reasoner_node: claim verification failed: {exc}")

    statistical_audit_results: list[Any] = []
    if statistical_assertions:
        try:
            statistical_audit_results = client.audit_statistical_assertions(
                list(statistical_assertions),
            )
        except Exception as exc:
            logger.error("Statistical audit failed: %s", exc)
            errors.append(f"reasoner_node: statistical audit failed: {exc}")

    supported_claims = sum(1 for v in verification_results if v.verdict == "supported")
    flagged_claims = sum(
        1 for v in verification_results
        if v.verdict in ("overstated", "contradicted", "out_of_scope", "paper_mill_journal")
    )

    result: dict[str, Any] = {
        "verification_results": verification_results,
        "statistical_audit_results": statistical_audit_results,
        "supported_claims": supported_claims,
        "flagged_claims": flagged_claims,
        "current_phase": "reasoning_complete",
    }
    if errors:
        result["errors"] = errors
    return result


def _extract_claim_context(claim_text: str, paper_text: str, window: int = 600) -> str:
    """Find the claim in the paper text and return surrounding context."""
    if not paper_text or not claim_text:
        return ""
    needle = claim_text[:120]
    idx = paper_text.find(needle)
    if idx == -1:
        words = claim_text.split()[:8]
        for n in range(len(words), 3, -1):
            fragment = " ".join(words[:n])
            idx = paper_text.find(fragment)
            if idx != -1:
                break
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(paper_text), idx + len(claim_text) + window)
    return paper_text[start:end]


_PAPER_MILL_JOURNALS: set[str] = {
    "evidence-based complementary and alternative medicine",
    "journal of healthcare engineering",
    "bioengineered",
    "international wound journal",
    "international journal of environmental research and public health",
    "journal of personalized medicine",
    "oncology reports",
    "molecular medicine reports",
    "foundations and trends\u00ae in privacy and security",
    "foundations and trends in privacy and security",
    "siam j. comput.",
    "siam journal on computing",
}


def _check_paper_mill_journals(claim, resolved_citations: list) -> str | None:
    """Return the matched journal name if any cited source is from a flagged venue."""
    for idx in claim.supporting_citation_indices:
        if 0 <= idx < len(resolved_citations):
            journal = (resolved_citations[idx].journal or "").strip().lower()
            if journal in _PAPER_MILL_JOURNALS:
                return resolved_citations[idx].journal
    return None


def _build_sources_for_claim(claim, resolved_citations: list) -> tuple[list[dict], bool]:
    """
    Build cited_sources for a claim. Returns (sources_list, has_resolved_source).

    If resolved citations with abstracts exist, use those (high quality).
    Otherwise, fall back to raw citation text so the model can still reason
    about the claim rather than returning "unverifiable" by default.
    """
    resolved_sources: list[dict] = []
    fallback_sources: list[dict] = []

    for idx in claim.supporting_citation_indices:
        if 0 <= idx < len(resolved_citations):
            cit = resolved_citations[idx]
            if cit.resolved and cit.source_text:
                resolved_sources.append({
                    "title": cit.title or cit.raw_text[:100],
                    "source_text": cit.source_text,
                    "journal": cit.journal,
                    "year": cit.year,
                    "resolved": True,
                })
            else:
                raw = cit.raw_text or ""
                title = cit.title or ""
                if raw or title:
                    fallback_sources.append({
                        "title": title or raw[:100],
                        "source_text": f"[Unresolved citation — only bibliographic info available] {raw}",
                        "journal": cit.journal,
                        "year": cit.year,
                        "resolved": False,
                    })

    if resolved_sources:
        return resolved_sources + fallback_sources, True
    if fallback_sources:
        return fallback_sources, False
    return [], False


def _verify_claims_concurrent(
    client: K2ReasoningAgent,
    claims: list,
    resolved_citations: list,
    paper_text: str,
) -> list[VerificationResult]:
    """Run claim verification with concurrency limit."""

    def _verify_one(claim) -> VerificationResult:
        # Deterministic paper mill journal check
        flagged_journal = _check_paper_mill_journals(claim, resolved_citations)
        if flagged_journal:
            return VerificationResult(
                claim_text=claim.text,
                verdict="paper_mill_journal",
                confidence=1.0,
                explanation=f"Cited source published in '{flagged_journal}', "
                "a journal commonly flagged as a paper mill venue.",
                relevant_passage=None,
                citation_indices=list(claim.supporting_citation_indices),
            )

        cited_sources, has_resolved = _build_sources_for_claim(claim, resolved_citations)
        context = _extract_claim_context(claim.text, paper_text)

        if not cited_sources:
            return VerificationResult(
                claim_text=claim.text,
                verdict="unverifiable",
                confidence=0.0,
                explanation="No citations (resolved or raw) available for this claim.",
                relevant_passage=None,
                citation_indices=list(claim.supporting_citation_indices),
            )

        vr = client.verify_claim(claim.text, cited_sources, paper_context=context)
        verdict = vr.verdict
        confidence = vr.confidence
        if not has_resolved and verdict == "supported":
            confidence = min(confidence, 0.5)

        return VerificationResult(
            claim_text=vr.claim_text,
            verdict=verdict,
            confidence=confidence,
            explanation=vr.explanation,
            relevant_passage=vr.relevant_passage,
            citation_indices=list(claim.supporting_citation_indices),
        )

    def _run_all() -> list[VerificationResult]:
        with ThreadPoolExecutor(max_workers=_MAX_CLAIM_CONCURRENCY) as executor:
            return list(executor.map(_verify_one, claims))

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return [_verify_one(c) for c in claims]
    else:
        return _run_all()
