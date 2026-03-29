"""Reasoning node: claim verification and statistical audits via Hermes/K2."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents.k2 import K2ReasoningAgent
from config import get_settings
from pipeline.state import PipelineState, VerificationResult

logger = logging.getLogger(__name__)

_MAX_CLAIM_CONCURRENCY = 5


def _build_rag_index(claims: list, resolved_citations: list) -> Any | None:
    """Create and populate a PaperRAGIndex for claim-referenced citations.

    Returns None when Lava is not configured or RAG setup fails entirely.
    """
    settings = get_settings()
    lava_key = settings.lava_api_key
    if not lava_key or lava_key == "not-set":
        logger.info("RAG: Lava API key not configured, skipping RAG")
        return None

    try:
        from agents.lava_gateway import LavaGateway
        from agents.rag import GeminiEmbedder, PaperRAGIndex

        lava_gw = LavaGateway(
            secret_key=lava_key,
            customer_id=settings.lava_customer_id,
            meter_slug=settings.lava_meter_slug,
        )
        embedder = GeminiEmbedder(lava_gw)

        # Collect unique citation indices actually referenced by claims
        needed: set[int] = set()
        total_claim_refs = 0
        for claim in claims:
            for idx in claim.supporting_citation_indices:
                total_claim_refs += 1
                if 0 <= idx < len(resolved_citations):
                    cit = resolved_citations[idx]
                    has_oa = bool(cit.open_access_pdf_url)
                    has_arxiv = bool(cit.arxiv_id)
                    if has_oa or has_arxiv:
                        needed.add(idx)
                    else:
                        logger.debug(
                            "RAG: citation %d (%s) has no PDF URL or arXiv ID",
                            idx, (cit.title or cit.raw_text[:60]),
                        )

        logger.info(
            "RAG: %d claims reference %d unique citations, %d have downloadable PDFs",
            len(claims), total_claim_refs, len(needed),
        )

        if not needed:
            logger.info("RAG: no citations have downloadable PDFs, skipping")
            return None

        rag_index = PaperRAGIndex(embedder)
        logger.info("RAG: indexing %d cited papers...", len(needed))

        def _index_one(idx: int) -> None:
            cit = resolved_citations[idx]
            src = cit.open_access_pdf_url or f"arXiv:{cit.arxiv_id}"
            logger.info("RAG: downloading citation %d from %s", idx, src)
            success = rag_index.index_paper(idx, cit)
            if success:
                logger.info("RAG: citation %d indexed successfully", idx)
            else:
                logger.warning("RAG: citation %d failed to index", idx)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(_index_one, needed))

        indexed = sum(1 for idx in needed if rag_index.has_store(idx))
        logger.info("RAG: indexed %d/%d papers successfully", indexed, len(needed))
        return rag_index if indexed > 0 else None
    except Exception as exc:
        logger.warning("RAG setup failed, falling back to abstract-only: %s", exc)
        return None


def reasoner_node(state: PipelineState) -> dict[str, Any]:
    """
    Phase 3: Claim verification and statistical audit.

    For each claim, gathers available evidence (RAG chunks from full paper text
    when available, resolved abstracts, or raw citation text as fallback) and
    asks the reasoning model to assess support.

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

    # Build RAG index for citations with available full-text PDFs
    rag_index = None
    if claims:
        rag_index = _build_rag_index(claims, resolved_citations)

    verification_results: list[VerificationResult] = []
    if claims:
        try:
            verification_results = _verify_claims_concurrent(
                client, claims, resolved_citations, paper_text,
                rag_index=rag_index,
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
        if v.verdict in ("overstated", "contradicted", "out_of_scope")
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


def _build_sources_for_claim(
    claim, resolved_citations: list, rag_index=None,
) -> tuple[list[dict], bool]:
    """
    Build cited_sources for a claim. Returns (sources_list, has_resolved_source).

    When *rag_index* is provided and has indexed the cited paper, retrieves
    semantically relevant full-text passages instead of just the abstract.
    Otherwise falls back to abstract or raw citation text.
    """
    resolved_sources: list[dict] = []
    fallback_sources: list[dict] = []

    for idx in claim.supporting_citation_indices:
        if 0 <= idx < len(resolved_citations):
            cit = resolved_citations[idx]
            if cit.resolved and cit.source_text:
                # Use RAG-retrieved passages when available
                if rag_index is not None and rag_index.has_store(idx):
                    source_text = rag_index.retrieve(
                        idx, claim.text, cit.source_text,
                    )
                else:
                    source_text = cit.source_text
                resolved_sources.append({
                    "title": cit.title or cit.raw_text[:100],
                    "source_text": source_text,
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
    rag_index=None,
) -> list[VerificationResult]:
    """Run claim verification with concurrency limit."""

    def _verify_one(claim) -> VerificationResult:
        cited_sources, has_resolved = _build_sources_for_claim(
            claim, resolved_citations, rag_index=rag_index,
        )
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
