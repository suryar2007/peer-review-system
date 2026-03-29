"""Map pipeline results to precise PDF page coordinates using PyMuPDF.

Each annotation gets a list of rectangles (page, x0, y0, x1, y1) that
correspond to the exact positions of the claim/assertion text in the PDF.
The frontend uses these to draw highlight overlays on the rendered PDF pages.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import fitz

logger = logging.getLogger(__name__)

_CITATION_PREFIX_RE = re.compile(
    r"^(?:[A-Z][a-zA-Z\-]+(?:\s+(?:et\s+al\.|and\s+[A-Z][a-zA-Z\-]+))?"
    r"(?:\s*\([^)]*\))?[\s,;]+(?:and\s+)?)+",
)
_VERBAL_CONNECTOR_RE = re.compile(
    r"^(?:showed|demonstrated|found|noted|concluded|discovered|reported|"
    r"argued|identified|observed|indicated|revealed|established|determined)"
    r"\s+that\s+",
    re.IGNORECASE,
)


def compute_score(data: dict[str, Any]) -> int:
    """Compute an integrity score from the full pipeline results.

    Uses ALL verification results (not just PDF-matched annotations) so the
    score accurately reflects the paper's citation integrity.
    """
    vr = data.get("verification_results", [])
    total = len(vr)
    if total == 0:
        return 50

    verdict_scores = {
        "supported": 100,
        "overstated": 30,
        "unverifiable": 50,
        "out_of_scope": 20,
        "paper_mill_journal": 40,
        "contradicted": 0,
    }
    claim_score = sum(
        verdict_scores.get(v.get("verdict", "unverifiable"), 50) for v in vr
    ) / total

    audits = data.get("statistical_audit_results", [])
    inconsistent = sum(
        1 for a in audits if not a.get("is_internally_consistent", True)
    )
    stat_penalty = min(20, inconsistent * 5)

    hallucinated = data.get("summary", {}).get("hallucinated_count", 0)
    hall_penalty = min(15, hallucinated * 5)

    return max(0, min(100, round(claim_score - stat_penalty - hall_penalty)))


def _try_search(doc: fitz.Document, query: str) -> list[dict]:
    """Search every page for *query*, return rect dicts on first hit."""
    for page_num in range(len(doc)):
        rects = doc[page_num].search_for(query)
        if rects:
            return [
                {
                    "page": page_num,
                    "x0": round(r.x0, 1),
                    "y0": round(r.y0, 1),
                    "x1": round(r.x1, 1),
                    "y1": round(r.y1, 1),
                }
                for r in rects
            ]
    return []


def _search_pdf(doc: fitz.Document, text: str) -> list[dict]:
    """Search for *text* across all pages, returning rect dicts.

    Strategy:
      1. Try full text, then progressively shorter prefixes.
      2. If the text looks like an LLM-paraphrased claim starting with
         citation info (e.g. "Author et al. (Year) showed that …"), strip
         that prefix and re-try with the substantive content.
      3. As a last resort, try 5-word sliding windows from the claim to
         locate any matching fragment.
    """
    text = text.strip()
    if not text:
        return []

    # --- Pass 1: verbatim prefixes (original strategy) ---
    for attempt_len in (len(text), 100, 60, 40):
        query = text[:attempt_len]
        results = _try_search(doc, query)
        if results:
            if attempt_len < len(text):
                tail = text[-min(40, len(text) - attempt_len):]
                results.extend(_try_search(doc, tail))
            return results

    # --- Pass 2: strip citation prefix and search substantive content ---
    stripped = _CITATION_PREFIX_RE.sub("", text)
    stripped = _VERBAL_CONNECTOR_RE.sub("", stripped).strip()
    if stripped and len(stripped) > 20 and stripped != text:
        for attempt_len in (min(len(stripped), 100), 60, 40, 30):
            if attempt_len > len(stripped):
                continue
            results = _try_search(doc, stripped[:attempt_len])
            if results:
                return results

    # --- Pass 3: sliding 5-word fragment windows ---
    words = text.split()
    for start in range(0, max(1, len(words) - 5), 3):
        frag = " ".join(words[start : start + 6])
        if len(frag) < 20:
            continue
        results = _try_search(doc, frag)
        if results:
            return results

    return []


def build_pdf_annotations(pdf_path: str, data: dict[str, Any]) -> list[dict]:
    """Build annotation list with PDF coordinates from pipeline results.

    Every verification result gets an annotation entry. If the claim text
    can be located in the PDF, the annotation includes ``rects`` for
    highlighting; otherwise ``rects`` is empty but the annotation still
    appears in the sidebar so no findings are hidden.
    """
    doc = fitz.open(pdf_path)
    annotations: list[dict] = []

    try:
        for i, vr in enumerate(data.get("verification_results", [])):
            claim = vr.get("claim_text", "")
            if not claim:
                continue
            rects = _search_pdf(doc, claim)
            annotations.append({
                "id": f"claim-{i}",
                "text": claim[:200],
                "verdict": vr.get("verdict", "unverifiable"),
                "confidence": vr.get("confidence", 0),
                "explanation": vr.get("explanation", ""),
                "passage": vr.get("relevant_passage"),
                "rects": rects,
            })

        for i, ar in enumerate(data.get("statistical_audit_results", [])):
            if ar.get("is_internally_consistent", True):
                continue
            issues = ar.get("issues")
            if not issues:
                continue
            text = ar.get("assertion_text", "")
            if not text:
                continue
            rects = _search_pdf(doc, text)
            annotations.append({
                "id": f"stat-{i}",
                "text": text[:200],
                "verdict": "statistical_issue",
                "confidence": 0.8,
                "explanation": "; ".join(issues),
                "passage": None,
                "rects": rects,
            })
    finally:
        doc.close()

    return annotations
