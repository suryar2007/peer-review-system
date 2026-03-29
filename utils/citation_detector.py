"""Deterministic in-text citation detection via regex scanning.

Scans paper section text for bracket citations ([1], [2,3], [1-5]) and
author-year citations ((Smith et al., 2020), Smith et al. (2020)), extracts
the surrounding sentence, and resolves each marker to 0-based bibliography
indices.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CitationMention:
    """A sentence containing one or more in-text citation markers."""

    sentence: str
    section: str
    marker_texts: list[str] = field(default_factory=list)
    citation_indices: list[int] = field(default_factory=list)


# ── Bracket citations: [1]  [1,2]  [1-5]  [1, 3-5, 7] ──────────────────

_BRACKET_RE = re.compile(r"\[\s*(\d+(?:\s*[-–,;]\s*\d+)*)\s*\]")

# Range across separate brackets: [1]–[3] means [1], [2], [3]
_BRACKET_RANGE_RE = re.compile(r"\[(\d+)\]\s*[-–]\s*\[(\d+)\]")


def _parse_bracket_inner(inner: str) -> list[int]:
    """Parse bracket contents into 0-based indices.

    ``'1, 3-5, 7'`` -> ``[0, 2, 3, 4, 6]``
    """
    indices: list[int] = []
    for part in re.split(r"[,;]", inner):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(\d+)\s*[-–]\s*(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            indices.extend(n - 1 for n in range(lo, hi + 1))
        elif re.fullmatch(r"\d+", part):
            indices.append(int(part) - 1)
    return sorted(set(i for i in indices if i >= 0))


# ── Author-year citations ────────────────────────────────────────────────

# Building blocks — _NAME must handle:
#   Simple:      Smith, Peters, Devlin
#   CamelCase:   McCann, McCallum, DeNero
#   Hyphenated:  Al-Rfou, Ben-David, El-Yaniv
#   Multi-word:  Tjong Kim Sang, De Meulder, Van der Wal
_NAME_WORD = r"[A-Z][a-zà-üA-Z]+"
_NAME = rf"{_NAME_WORD}(?:[-']{_NAME_WORD})*(?:\s+(?:[A-Z][a-zà-ü]+|[Dd]e[l]?|[Vv]an|[Dd]er|[Ll]a|[Dd]i))*"
_ET_AL = r"(?:\s+(?:et\s+al\.?))"
_AND_AUTHOR = rf"(?:\s+and\s+{_NAME})"
_AUTHOR = rf"{_NAME}(?:{_ET_AL}|{_AND_AUTHOR})?"
_YEAR = r"\d{4}[a-z]?"

# Parenthetical: (Smith et al., 2020), (Smith, 2020; Jones, 2019),
# and same-author multi-year: (Peters et al., 2017, 2018a)
_PAREN_CITE_RE = re.compile(
    rf"\(\s*"
    rf"(?:{_AUTHOR},?\s*{_YEAR})"
    rf"(?:\s*[;,]\s*(?:{_AUTHOR},?\s*)?{_YEAR})*"
    rf"\s*\)"
)

# Narrative: Smith et al. (2020), Smith and Jones (2020)
_NARRATIVE_CITE_RE = re.compile(
    rf"({_AUTHOR})\s*\(({_YEAR})\)"
)

# Extracts individual (author_str, year_str) pairs from inside a match
_AY_PAIR_RE = re.compile(
    rf"({_AUTHOR}),?\s*({_YEAR})"
)


def _resolve_author_year(
    name: str, year_str: str, bib: list[dict],
) -> int | None:
    """Fuzzy-match author surname + year to a bibliography entry index.

    Tries multiple strategies: first-word match, last-word match, and
    substring containment to handle varied name formats in bibliography
    entries (``"John Smith"`` vs ``"Smith, J."`` vs ``"Tjong Kim Sang"``).
    """
    name_clean = name.replace(" et al", "").replace(" and ", " ").strip().rstrip(".,;:")
    name_lower = name_clean.lower()
    words = name_clean.split()
    candidates: set[str] = set()
    for w in words:
        w_stripped = w.lower().rstrip(".,;:")
        if len(w_stripped) > 1:
            candidates.add(w_stripped)
    candidates.add(name_lower)
    if words:
        candidates.add(words[0].lower().rstrip(".,;:"))

    try:
        year = int(year_str[:4])
    except (ValueError, IndexError):
        return None

    for i, entry in enumerate(bib):
        if entry.get("year") != year:
            continue
        for author in (entry.get("authors") or [])[:1]:
            if not author:
                continue
            a = str(author).lower()
            # Extract bib surname(s) from common formats
            if "," in a:
                bib_surname = a.split(",")[0].strip()
            else:
                parts = a.split()
                bib_surname = parts[-1].strip() if parts else ""
            bib_full = a.strip()
            for cand in candidates:
                if cand == bib_surname or cand in bib_full or bib_surname in candidates:
                    return i
    return None


# ── Sentence boundary helpers ────────────────────────────────────────────

_ABBREVS = frozenset({
    "al", "fig", "figs", "eq", "eqs", "ref", "refs", "tab",
    "vs", "ie", "eg", "etc", "approx", "dept", "dr",
    "prof", "mr", "mrs", "ms", "jr", "sr", "vol", "no",
    "sec", "ch", "pp", "ed", "eds", "rev", "trans", "resp",
    "incl", "est", "max", "min", "avg", "std",
})


def _is_abbrev_period(text: str, dot_pos: int) -> bool:
    """Return True if the ``.`` at *dot_pos* is part of an abbreviation."""
    if dot_pos >= len(text) or text[dot_pos] != ".":
        return False
    i = dot_pos - 1
    while i >= 0 and text[i].isalpha():
        i -= 1
    word = text[i + 1 : dot_pos].lower()
    if word in _ABBREVS:
        return True
    # Single-letter initial  (J.  A.  M.)
    if len(word) <= 1:
        return True
    # "et al." specifically
    if word == "al" and text[max(0, dot_pos - 6) : dot_pos].rstrip().endswith("et"):
        return True
    return False


def _sent_start_before(text: str, pos: int) -> int:
    """Walk backwards from *pos* to find the start of the enclosing sentence."""
    i = pos - 1
    while i >= 0:
        ch = text[i]
        if ch in ".!?":
            if ch == "." and _is_abbrev_period(text, i):
                i -= 1
                continue
            j = i + 1
            while j < pos and text[j] in " \t\n\r":
                j += 1
            if j <= pos:
                return j
        elif ch == "\n" and i > 0 and text[i - 1] == "\n":
            return i + 1
        i -= 1
    return 0


def _sent_end_after(text: str, pos: int) -> int:
    """Walk forwards from *pos* to find the end of the enclosing sentence."""
    i = pos
    while i < len(text):
        ch = text[i]
        if ch in ".!?":
            if ch == "." and _is_abbrev_period(text, i):
                i += 1
                continue
            j = i + 1
            while j < len(text) and text[j] in " \t":
                j += 1
            if j >= len(text) or text[j] in "\n" or text[j].isupper():
                return i + 1
        elif ch == "\n" and i + 1 < len(text) and text[i + 1] == "\n":
            return i
        i += 1
    return len(text)


def _extract_sentence(text: str, span_start: int, span_end: int) -> str:
    """Return the sentence that contains the character span ``[span_start, span_end)``."""
    s = _sent_start_before(text, span_start)
    e = _sent_end_after(text, span_end)
    return text[s:e].strip()


# ── Text normalisation ───────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Join broken PDF lines into paragraphs while preserving paragraph breaks."""
    # Re-join word-wrapping hyphens: "Rad-\nford" → "Radford"
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    # Replace remaining single newlines with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"  +", " ", text)
    return text


# ── Main entry point ─────────────────────────────────────────────────────

_MIN_SENTENCE_LEN = 25  # skip bare references like "see [5]"


def detect_all_citations(
    sections: dict[str, str],
    bib_entries: list[dict],
) -> list[CitationMention]:
    """Deterministically detect every in-text citation across all *sections*.

    Returns one :class:`CitationMention` per unique sentence, with all
    referenced bibliography indices merged.
    """
    num_bib = len(bib_entries)
    all_mentions: list[CitationMention] = []

    for sec_name, sec_text in sections.items():
        text = _normalise(sec_text)
        sent_map: dict[str, CitationMention] = {}

        def _add(marker: str, indices: list[int], start: int, end: int) -> None:
            sentence = _extract_sentence(text, start, end)
            if not sentence or len(sentence) < _MIN_SENTENCE_LEN:
                return
            key = sentence.strip().lower()
            if key in sent_map:
                sent_map[key].citation_indices.extend(indices)
                if marker not in sent_map[key].marker_texts:
                    sent_map[key].marker_texts.append(marker)
            else:
                sent_map[key] = CitationMention(
                    sentence=sentence,
                    section=sec_name,
                    marker_texts=[marker],
                    citation_indices=list(indices),
                )

        # ── Bracket ranges across separate brackets: [1]–[3] ──
        range_spans: set[tuple[int, int]] = set()
        for m in _BRACKET_RANGE_RE.finditer(text):
            lo, hi = int(m.group(1)) - 1, int(m.group(2)) - 1
            indices = [i for i in range(lo, hi + 1) if 0 <= i < num_bib]
            if indices:
                _add(m.group(0), indices, m.start(), m.end())
                range_spans.add((m.start(), m.end()))

        # ── Bracket citations ──
        for m in _BRACKET_RE.finditer(text):
            # Skip if this bracket was already consumed by a range pattern
            if any(rs <= m.start() and m.end() <= re for rs, re in range_spans):
                continue
            indices = _parse_bracket_inner(m.group(1))
            indices = [i for i in indices if i < num_bib]
            if indices:
                _add(m.group(0), indices, m.start(), m.end())

        # ── Parenthetical author-year ──
        for m in _PAREN_CITE_RE.finditer(text):
            matched_text = m.group(0)
            indices: list[int] = []
            last_author: str | None = None
            for pair in _AY_PAIR_RE.finditer(matched_text):
                last_author = pair.group(1)
                idx = _resolve_author_year(pair.group(1), pair.group(2), bib_entries)
                if idx is not None:
                    indices.append(idx)
            if last_author:
                pairs = list(_AY_PAIR_RE.finditer(matched_text))
                if pairs:
                    tail = matched_text[pairs[-1].end():]
                    for ym in re.finditer(r"(\d{4}[a-z]?)", tail):
                        idx = _resolve_author_year(last_author, ym.group(1), bib_entries)
                        if idx is not None:
                            indices.append(idx)
            # Keep even if no indices resolved — the sentence still cites *something*
            _add(matched_text, indices, m.start(), m.end())

        # ── Narrative author-year ──
        for m in _NARRATIVE_CITE_RE.finditer(text):
            idx = _resolve_author_year(m.group(1), m.group(2), bib_entries)
            _add(m.group(0), [idx] if idx is not None else [], m.start(), m.end())

        for mention in sent_map.values():
            mention.citation_indices = sorted(set(mention.citation_indices))
            mention.marker_texts = list(dict.fromkeys(mention.marker_texts))
            all_mentions.append(mention)

    logger.info(
        "Detected %d citation mention(s) across %d section(s)",
        len(all_mentions),
        len(sections),
    )
    return all_mentions
