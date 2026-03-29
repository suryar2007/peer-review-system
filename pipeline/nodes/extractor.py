"""Hermes-backed extraction: citations from references, claims, and statistical assertions."""

from __future__ import annotations

import logging
from typing import Any

from agents.hermes import HermesAgent
from pipeline.state import Citation, Claim, PipelineState, StatisticalAssertion
from utils.citation_detector import CitationMention, detect_all_citations
from utils.pdf_parser import PaperParser

logger = logging.getLogger(__name__)


def _coerce_citation(obj: dict[str, Any]) -> Citation | None:
    try:
        raw = str(obj.get("raw_text") or obj.get("text") or "").strip()
        authors = obj.get("authors")
        if not isinstance(authors, list):
            authors = []
        data = {
            **obj,
            "raw_text": raw,
            "authors": [str(a) for a in authors],
        }
        return Citation.model_validate(data)
    except Exception:
        return None


def _coerce_claim(obj: dict[str, Any]) -> Claim | None:
    try:
        indices = obj.get("supporting_citation_indices")
        if not isinstance(indices, list):
            indices = []
        data = {
            "text": str(obj.get("text") or ""),
            "paper_section": str(obj.get("paper_section") or obj.get("section") or "unknown"),
            "supporting_citation_indices": [int(i) for i in indices],
            "claim_type": str(obj.get("claim_type") or "empirical"),
        }
        return Claim.model_validate(data)
    except Exception:
        return None


def _coerce_statistical(obj: dict[str, Any]) -> StatisticalAssertion | None:
    try:
        data = {
            "text": str(obj.get("text") or ""),
            "p_value": obj.get("p_value"),
            "effect_size": obj.get("effect_size"),
            "sample_size": obj.get("sample_size"),
            "confidence_interval": obj.get("confidence_interval"),
            "section": str(obj.get("section") or obj.get("paper_section") or "unknown"),
        }
        return StatisticalAssertion.model_validate(data)
    except Exception:
        return None


def _mentions_to_claim_dicts(
    mentions: list[CitationMention],
    num_bib: int,
) -> list[dict]:
    """Convert detector output into the dict format expected by classify/coerce.

    Keeps mentions even when no bibliography index could be resolved — the
    sentence still references *something* and should be analysed.
    """
    out: list[dict] = []
    for m in mentions:
        valid_indices = [i for i in m.citation_indices if 0 <= i < num_bib]
        out.append({
            "text": m.sentence,
            "section": m.section,
            "sentence": m.sentence,
            "supporting_citation_indices": valid_indices,
            "claim_type": "empirical",
        })
    return out


def extractor_node(state: PipelineState) -> dict[str, Any]:
    paper_path = state.get("paper_path")
    if not paper_path:
        return {"errors": ["extractor_node: paper_path is required"], "current_phase": "extraction_failed"}

    parsed = PaperParser(paper_path).parse()
    paper_text = parsed.get("full_text") or ""
    sections = parsed.get("sections") or {}
    references_raw = parsed.get("references_raw") or ""

    agent = HermesAgent()
    errors: list[str] = []

    # ── Phase 1a: Extract bibliography from references section (LLM) ──
    try:
        raw_citations = agent.extract_citations(references_raw)
    except Exception as exc:
        raw_citations = []
        errors.append(f"extractor_node: citation extraction failed: {exc}")

    citations: list[Citation] = []
    for item in raw_citations:
        if isinstance(item, dict):
            c = _coerce_citation(item)
            if c is not None:
                citations.append(c)

    # ── Phase 1b: Detect in-text citations deterministically ──
    try:
        mentions = detect_all_citations(sections, raw_citations)
    except Exception as exc:
        mentions = []
        errors.append(f"extractor_node: citation detection failed: {exc}")

    mention_dicts = _mentions_to_claim_dicts(mentions, len(citations))
    logger.info("Deterministic detection found %d citation mentions", len(mention_dicts))

    # ── Phase 1c: Classify mentions + extract stats (concurrent) ──
    from concurrent.futures import Future, ThreadPoolExecutor

    classified_claims: list[dict] = []
    raw_stats: list[dict] = []

    def _classify() -> list[dict]:
        return agent.classify_citation_mentions(mention_dicts)

    def _extract_stats() -> list[dict]:
        return agent.extract_statistical_assertions(paper_text)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claim_future: Future = pool.submit(_classify)
        stats_future: Future = pool.submit(_extract_stats)

        try:
            classified_claims = claim_future.result(timeout=180)
        except Exception as exc:
            errors.append(f"extractor_node: claim classification failed: {exc}")

        try:
            raw_stats = stats_future.result(timeout=180)
        except Exception as exc:
            errors.append(f"extractor_node: statistical assertion extraction failed: {exc}")

    # ── Coerce and deduplicate claims ──
    claims: list[Claim] = []
    seen_claim_texts: set[str] = set()
    for item in classified_claims:
        if isinstance(item, dict):
            c = _coerce_claim(item)
            if c is not None and c.text.strip():
                normalised = c.text.strip().lower()
                if normalised not in seen_claim_texts:
                    seen_claim_texts.add(normalised)
                    claims.append(c)

    statistical_assertions: list[StatisticalAssertion] = []
    for item in raw_stats:
        if isinstance(item, dict):
            s = _coerce_statistical(item)
            if s is not None and s.text.strip():
                statistical_assertions.append(s)

    logger.info(
        "Extractor: %d citations, %d claims (from %d mentions), %d stats",
        len(citations), len(claims), len(mentions), len(statistical_assertions),
    )

    result = {
        "paper_text": paper_text,
        "paper_title": parsed.get("title"),
        "paper_abstract": parsed.get("abstract"),
        "citations": citations,
        "claims": claims,
        "statistical_assertions": statistical_assertions,
        "current_phase": "extraction_complete",
    }
    if errors:
        result["errors"] = errors
    return result
