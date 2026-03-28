"""Structured PDF parsing for academic papers using PyMuPDF (fitz)."""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

import fitz

# Reference section headings (standalone line, case-insensitive).
_REFERENCE_HEADING_RE = re.compile(
    r"(?ims)^\s*("
    r"references|bibliography|works cited|literature cited|cited references"
    r")\s*$"
)

# Start of abstract; end before typical next sections.
_ABSTRACT_START_RE = re.compile(r"(?im)^\s*abstract\s*$")

# Lines that likely end the abstract block.
_ABSTRACT_END_RE = re.compile(
    r"(?im)^\s*("
    r"keywords?|key words|index terms|introduction|1\.?\s*introduction|"
    r"background|motivation|highlights|graphical abstract|abbreviations"
    r")\b"
)

_NUMBERED_HEADING_RE = re.compile(
    r"^\s*\d+(?:\.\d+)*\.?\s+(\S.*)$",
)

_ALL_CAPS_HEADING_RE = re.compile(
    r"^[A-Z][A-Z0-9\s\-\.,:;]{2,100}$"
)

# arXiv header / stamp (e.g. "arXiv:2401.12345v2 [cs.CL] 1 Jan 2024")
_ARXIV_ID_IN_LINE_RE = re.compile(r"arXiv:\s*\d+\.\d+", re.IGNORECASE)

# Standalone date-like lines (not titles).
_DATE_ONLY_LINE_RE = re.compile(
    r"^\s*(?:"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|"
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r")\s*$"
)

# DOI-only or DOI-URL-only line.
_DOI_LINE_RE = re.compile(
    r"^\s*(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/\S+\s*$",
    re.IGNORECASE,
)

# Leading section index like ``1.``, ``1.1``, ``2.3.4`` (periods here are not sentence breaks).
_SECTION_INDEX_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")

# Mid-sentence lines mistaken for headings (word-boundary matches).
_VERB_LIKE_IN_HEADING_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|note|omit|make|works|hold)\b",
    re.IGNORECASE,
)

# Inline numeric citations like ``[62]`` (not section indices).
_CITATION_MARKER_IN_LINE_RE = re.compile(r"\[\d+\]")

# Prose lines that start like a sentence, not a titled section.
_SENTENCE_OPENER_LINE_RE = re.compile(
    r"^(?:"
    r"This|We|Our|The|These|They|Very|Here|When|While|Since|Although"
    r")\s+"
    r"|^Note(?:\s|:)\s*"
    r"|^However(?:,\s*|\s+)"
    r"|^A\s+"
    r"|^An\s+"
    r"|^It\s+"
    r"|^As\s+",
    re.IGNORECASE,
)


def _is_excluded_title_span(text: str) -> bool:
    """True if this span should not participate in largest-font title selection."""
    t = text.strip()
    if len(t) < 2:
        return True
    if _ARXIV_ID_IN_LINE_RE.search(t):
        return True
    tl = t.lower()
    if tl.startswith("http://") or tl.startswith("https://"):
        return True
    if _DOI_LINE_RE.match(t):
        return True
    if _DATE_ONLY_LINE_RE.match(t):
        return True
    return False


def _first_non_junk_page_line(lines: list[tuple[str, float]]) -> str:
    for text, _ in lines:
        if not _is_excluded_title_span(text):
            return text[:500]
    return lines[0][0][:500] if lines else ""


def _token_is_number_or_single_char(tok: str) -> bool:
    """Used to drop table fragments and numeric rows mistaken for headings."""
    t = tok.strip(".,;:)\"'[]")
    if not t:
        return True
    if len(t) == 1 and t.isalnum():
        return True
    if re.fullmatch(r"\d+", t):
        return True
    if re.fullmatch(r"\(?\d+\)?", t):
        return True
    if re.fullmatch(r"\d+\.\d+", t):
        return True
    return False


def _heading_numeric_noise_ratio(s: str) -> tuple[list[str], float]:
    """
    Tokens used for the >30% numeric/single-char rule.

    For numbered section lines, only the title part after the numbering is scored
    so that ``1. Introduction`` is not rejected because of the leading ``1.``.
    """
    stripped = s.strip()
    m = _NUMBERED_HEADING_RE.match(stripped)
    if m:
        words = m.group(1).split()
    else:
        words = stripped.split()
    if not words:
        return words, 0.0
    bad = sum(1 for w in words if _token_is_number_or_single_char(w))
    return words, bad / len(words)


def _has_prose_period_outside_section_index(s: str) -> bool:
    """
    True if the line contains a sentence period outside a leading ``1.1``-style index.

    Section titles are not full sentences; internal ``.`` usually means two sentences
    got merged onto one PDF line (e.g. ``... RAM. We make ...``).
    """
    stripped = s.strip()
    body = _SECTION_INDEX_PREFIX_RE.sub("", stripped, count=1)
    return "." in body


def _ends_with_comma_clause_fragment(s: str) -> bool:
    """Lines ending in ``, since`` / ``, which is`` / ``, is`` / ``, are`` (ignoring trailing punct)."""
    core = s.rstrip()
    if not core:
        return False
    tail = core.rstrip('.,;:!?"\'')
    return bool(
        re.search(
            r",\s*(?:since|which\s+is|is|are)\s*$",
            tail,
            re.IGNORECASE,
        )
    )


def _line_opens_with_http_url(s: str) -> bool:
    """True if the line is (or opens with) an http(s) URL after whitespace / wrappers."""
    t = s.replace("\ufeff", "").strip()
    t = t.lstrip(" \t([{<'\"\u00a0\u200b•·")
    return t.lower().startswith(("http://", "https://"))


def _is_plausible_section_heading_line(s: str) -> bool:
    """Filters false positives (URLs, table rows, mid-sentence lines, etc.)."""
    s = s.replace("\ufeff", "").strip()
    if len(s) > 120:
        return False
    tokens = s.split()
    if len(tokens) < 2:
        return False

    if _line_opens_with_http_url(s):
        return False

    if _CITATION_MARKER_IN_LINE_RE.search(s):
        return False

    if _SENTENCE_OPENER_LINE_RE.match(s):
        return False

    core = s.rstrip()
    if not core:
        return False
    if core[-1] in ",.":
        return False

    if _ends_with_comma_clause_fragment(s):
        return False

    if _has_prose_period_outside_section_index(s):
        return False

    if len(_VERB_LIKE_IN_HEADING_RE.findall(s)) > 1:
        return False

    _, ratio = _heading_numeric_noise_ratio(s)
    if ratio > 0.30:
        return False
    return True


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Plain full document text in reading order (same merge logic as ``PaperParser``)."""
    return PaperParser(str(Path(pdf_path))).parse()["full_text"]


class PaperParser:
    """Extract title, abstract, sections, references, and full text from an academic PDF."""

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = str(Path(pdf_path))

    def parse(self) -> dict:
        """
        Return structured content.

        Keys: title, abstract, full_text, sections, references_raw, page_count.
        """
        path = Path(self.pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc = fitz.open(path)
        try:
            page_count = len(doc)
            per_page_lines: list[list[tuple[str, float]]] = []
            flat_lines: list[tuple[str, float]] = []
            for page in doc:
                ol = _ordered_lines_with_fonts(page)
                per_page_lines.append(ol)
                flat_lines.extend(ol)

            full_text = "\n\n".join(
                "\n".join(text for text, _ in ol) for ol in per_page_lines if ol
            ).strip()

            ref_idx = _find_reference_heading_line_index(flat_lines)
            if ref_idx is not None:
                main_lines = flat_lines[:ref_idx]
                ref_lines = flat_lines[ref_idx + 1 :]
                references_raw = "\n".join(text for text, _ in ref_lines)
            else:
                main_lines = list(flat_lines)
                references_raw = ""

            main_text = "\n".join(text for text, _ in main_lines)

            title = _extract_title(doc)
            abstract = _extract_abstract(doc, main_text, title)
            sections = _split_sections(main_lines, main_text, abstract, title)

            return {
                "title": title,
                "abstract": abstract,
                "full_text": full_text,
                "sections": sections,
                "references_raw": references_raw,
                "page_count": page_count,
            }
        finally:
            doc.close()


def _ordered_lines_with_fonts(page: fitz.Page) -> list[tuple[str, float]]:
    """
    Lines in reading order for the page (multi-column: left column, then right).

    Each line is (text, max_font_size_on_line).
    """
    d = page.get_text("dict")
    items: list[tuple[float, float, str, float]] = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        x0_block = float(block.get("bbox", (0, 0, 0, 0))[0])
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join((s.get("text") or "") for s in spans).strip()
            if not text:
                continue
            sizes = [float(s.get("size", 0)) for s in spans]
            mx = max(sizes) if sizes else 0.0
            y0 = float(line.get("bbox", (0, 0, 0, 0))[1])
            items.append((x0_block, y0, text, mx))

    if not items:
        return []

    mid_x = page.rect.width * 0.48
    left = [t for t in items if t[0] < mid_x]
    right = [t for t in items if t[0] >= mid_x]

    if len(left) < 2 or len(right) < 2:
        ordered = sorted(items, key=lambda t: (t[1], t[0]))
    else:
        left.sort(key=lambda t: (t[1], t[0]))
        right.sort(key=lambda t: (t[1], t[0]))
        ordered = left + right

    return [(t[2], t[3]) for t in ordered]


def _find_reference_heading_line_index(flat_lines: list[tuple[str, float]]) -> int | None:
    for i, (text, _) in enumerate(flat_lines):
        if _REFERENCE_HEADING_RE.match(text.strip()):
            return i
    return None


def _extract_title(doc: fitz.Document) -> str:
    """Largest font in the upper portion of page 1, merged into one line."""
    if len(doc) == 0:
        return ""

    page = doc[0]
    d = page.get_text("dict")
    page_h = page.rect.height
    spans: list[tuple[float, str, float]] = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if len(text) < 2:
                    continue
                if _is_excluded_title_span(text):
                    continue
                size = float(span.get("size", 0))
                bbox = span.get("bbox") or (0, 0, 0, 0)
                y0 = float(bbox[1])
                spans.append((size, text, y0))

    if not spans:
        first_page_lines = _ordered_lines_with_fonts(doc[0])
        if not first_page_lines:
            return ""
        return _first_non_junk_page_line(first_page_lines)

    max_size = max(s[0] for s in spans)
    threshold = max_size - 1.2
    top_spans = [
        (y, t)
        for sz, t, y in spans
        if sz >= threshold and y < page_h * 0.42 and not _is_likely_running_header(t)
    ]
    if not top_spans:
        top_spans = [(y, t) for sz, t, y in spans if sz >= threshold]

    top_spans.sort(key=lambda x: x[0])
    parts: list[str] = []
    for _, t in top_spans:
        if _is_excluded_title_span(t):
            continue
        tl = _normalize_whitespace(t).lower()
        if not tl:
            continue
        if tl in {"abstract", "summary"}:
            break
        if re.match(r"^keywords?$", tl):
            break
        if tl == "introduction" and parts:
            break
        parts.append(t)
    title = _normalize_whitespace(" ".join(parts))[:2000]
    if title:
        return title
    first_lines = _ordered_lines_with_fonts(doc[0])
    return _first_non_junk_page_line(first_lines)


def _is_likely_running_header(text: str) -> bool:
    t = text.strip()
    if len(t) > 80:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    return False


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def _extract_abstract(doc: fitz.Document, main_text: str, title: str) -> str:
    """Abstract from heading; else first long paragraph after title on page 1."""
    m = _ABSTRACT_START_RE.search(main_text)
    if m:
        rest = main_text[m.end() :].lstrip("\n")
        end_m = _ABSTRACT_END_RE.search(rest)
        if end_m:
            body = rest[: end_m.start()].strip()
        else:
            body = rest.strip()
        if body:
            return body

    return _abstract_fallback_first_page(doc, title)


def _abstract_fallback_first_page(doc: fitz.Document, title: str) -> str:
    if len(doc) == 0:
        return ""

    page = doc[0]
    d = page.get_text("dict")
    blocks_text: list[tuple[float, str]] = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = block.get("bbox") or (0, 0, 0, 0)
        y0 = float(bbox[1])
        parts: list[str] = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                parts.append(span.get("text") or "")
        text = _normalize_whitespace("".join(parts))
        if text:
            blocks_text.append((y0, text))

    blocks_text.sort(key=lambda x: x[0])
    title_norm = _normalize_whitespace(title).lower() if title else ""

    for _, text in blocks_text:
        if title_norm and _normalize_whitespace(text).lower() == title_norm:
            continue
        if len(text) < 40:
            continue
        if _word_count(text) >= 100:
            return text

    merged_lines = _ordered_lines_with_fonts(doc[0])
    merged = "\n".join(t[0] for t in merged_lines)
    paras = [p.strip() for p in re.split(r"\n\s*\n+", merged) if p.strip()]
    for p in paras:
        if title_norm and _normalize_whitespace(p).lower() == title_norm:
            continue
        if _word_count(p) >= 100:
            return p

    return ""


def _body_font_median(lines: list[tuple[str, float]]) -> float:
    sizes = [sz for _, sz in lines if sz > 0]
    if len(sizes) < 3:
        return statistics.median(sizes) if sizes else 10.0
    trimmed = sorted(sizes)
    lo = int(len(trimmed) * 0.15)
    hi = int(len(trimmed) * 0.85)
    core = trimmed[lo:hi] or trimmed
    return float(statistics.median(core))


def _is_section_heading(
    raw: str,
    font_size: float,
    median_body: float,
    index: int,
    line_texts: list[str],
) -> bool:
    s = raw.strip()
    if not s:
        return False

    if not _is_plausible_section_heading_line(s):
        return False

    if _NUMBERED_HEADING_RE.match(s):
        return True

    if _ALL_CAPS_HEADING_RE.match(s) and any(c.isalpha() for c in s):
        words = s.split()
        if len(words) > 14:
            return False
        return True

    wc = len(s.split())
    if wc <= 14 and font_size >= median_body + 1.25:
        if index > 0:
            prev = ""
            for j in range(index - 1, -1, -1):
                t = line_texts[j].strip()
                if t:
                    prev = t
                    break
            if prev and len(prev) > 180:
                return True
        else:
            return True

    return False


def _split_sections(
    main_lines: list[tuple[str, float]],
    main_text: str,
    abstract: str,
    title: str,
) -> dict[str, str]:
    """Split body into sections using numbered lines, ALL CAPS, or larger font lines."""
    if not main_lines:
        return {}

    line_texts = [t for t, _ in main_lines]
    median_sz = _body_font_median(main_lines)

    headings: list[tuple[int, str]] = []
    for i, (text, sz) in enumerate(main_lines):
        if _is_section_heading(text, sz, median_sz, i, line_texts):
            nm = _NUMBERED_HEADING_RE.match(text.strip())
            label = nm.group(1).strip() if nm else text.strip()
            headings.append((i, label[:300]))

    if not headings:
        return {"Body": main_text.strip()} if main_text.strip() else {}

    sections: dict[str, str] = {}
    for h_idx, (line_no, label) in enumerate(headings):
        start = line_no
        end = headings[h_idx + 1][0] if h_idx + 1 < len(headings) else len(main_lines)
        chunk_lines = [main_lines[j][0] for j in range(start, end)]
        chunk = "\n".join(chunk_lines).strip()
        if not chunk:
            continue
        base = label
        key = base
        n = 1
        while key in sections:
            n += 1
            key = f"{base} ({n})"
        sections[key] = chunk

    preamble_indices = range(0, headings[0][0])
    preamble = "\n".join(main_lines[j][0] for j in preamble_indices).strip()
    if preamble:
        drop_abstract = abstract.strip()
        if drop_abstract and drop_abstract in preamble:
            preamble = preamble.replace(drop_abstract, "").strip()
        title_norm = title.strip()
        if title_norm and title_norm in preamble:
            preamble = preamble.replace(title_norm, "").strip()
        if preamble:
            sections = {"Preamble": preamble, **sections}

    return sections


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils/pdf_parser.py path/to/paper.pdf", file=sys.stderr)
        sys.exit(1)

    result = PaperParser(sys.argv[1]).parse()
    print("Title:", result["title"] or "(empty)")
    ab = result["abstract"] or ""
    preview = ab[:200] + ("…" if len(ab) > 200 else "")
    print("Abstract (first 200 chars):", preview or "(empty)")
    print("Section names:", ", ".join(result["sections"]) if result["sections"] else "(none)")
    ref_newlines = result["references_raw"].count("\n")
    print("Reference newline count (proxy):", ref_newlines)
    print("Pages:", result["page_count"])
