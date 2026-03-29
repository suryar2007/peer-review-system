"""Map pipeline results to precise PDF page coordinates using PyMuPDF.

Each annotation gets a list of rectangles (page, x0, y0, x1, y1) that
correspond to the exact positions of the claim/assertion text in the PDF.
The frontend uses these to draw highlight overlays on the rendered PDF pages.
"""

from __future__ import annotations

import logging
from typing import Any

import fitz

logger = logging.getLogger(__name__)

_VERDICT_PENALTY = {
    "contradicted": 15,
    "overstated": 8,
    "out_of_scope": 5,
    "unverifiable": 3,
    "statistical_issue": 8,
}


def compute_score(annotations: list[dict]) -> int:
    score = 100
    for ann in annotations:
        score -= _VERDICT_PENALTY.get(ann.get("verdict", ""), 0)
    return max(0, min(100, score))


def _search_pdf(doc: fitz.Document, text: str) -> list[dict]:
    """Search for *text* across all pages, returning rect dicts.

    Tries the full text first, then progressively shorter prefixes.
    PyMuPDF's ``search_for`` handles hyphenated line breaks internally.
    """
    text = text.strip()
    if not text:
        return []

    for attempt_len in (len(text), 100, 60, 40):
        query = text[:attempt_len]
        for page_num in range(len(doc)):
            page = doc[page_num]
            rects = page.search_for(query)
            if not rects:
                continue

            results = []
            for r in rects:
                results.append({
                    "page": page_num,
                    "x0": round(r.x0, 1),
                    "y0": round(r.y0, 1),
                    "x1": round(r.x1, 1),
                    "y1": round(r.y1, 1),
                })

            if attempt_len == len(text):
                return results

            # For prefix matches, also grab subsequent lines that are part
            # of the full claim by searching for the tail.
            if len(text) > attempt_len:
                tail = text[-min(40, len(text) - attempt_len) :]
                tail_rects = page.search_for(tail)
                for r in tail_rects:
                    results.append({
                        "page": page_num,
                        "x0": round(r.x0, 1),
                        "y0": round(r.y0, 1),
                        "x1": round(r.x1, 1),
                        "y1": round(r.y1, 1),
                    })
            return results

    return []


def build_pdf_annotations(pdf_path: str, data: dict[str, Any]) -> list[dict]:
    """Build annotation list with PDF coordinates from pipeline results."""
    doc = fitz.open(pdf_path)
    annotations: list[dict] = []

    try:
        for i, vr in enumerate(data.get("verification_results", [])):
            claim = vr.get("claim_text", "")
            if not claim:
                continue
            rects = _search_pdf(doc, claim)
            if not rects:
                logger.debug("No PDF match for claim %d: %.60s...", i, claim)
                continue
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
            if not rects:
                continue
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
