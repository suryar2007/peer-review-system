"""Academic citation resolution via Semantic Scholar, CrossRef, and OpenAlex APIs.

When a Lava API key is configured, Semantic Scholar requests are routed through
Lava's forward proxy (https://api.lava.so/v1/forward) for automatic usage
tracking and cost metering.  S2 is an explicitly supported Lava provider.

CrossRef and OpenAlex calls always go direct (not in Lava's provider catalog).
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlencode

import httpx

from agents.lava_gateway import LavaEndpointNotSupported, LavaGateway
from config import get_settings
from pipeline.state import Citation

logger = logging.getLogger(__name__)

_S2_API = "https://api.semanticscholar.org/graph/v1"
_CROSSREF_API = "https://api.crossref.org/works"
_OPENALEX_API = "https://api.openalex.org/works"

_TITLE_SEARCH_CONCURRENCY = 6
_TITLE_SEARCH_DELAY = 0.35


def _normalize_title(t: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy comparison."""
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_title(a), _normalize_title(b)).ratio()


def _build_s2_query(cit: Citation) -> str:
    """Build an effective search query from citation metadata."""
    parts: list[str] = []
    if cit.title:
        parts.append(cit.title)
    if cit.authors and len(parts) > 0:
        last_names = []
        for a in cit.authors[:2]:
            tokens = a.replace(",", " ").split()
            if tokens:
                last_names.append(tokens[0])
        if last_names:
            parts.append(" ".join(last_names))
    if cit.year and len(parts) > 0:
        parts.append(str(cit.year))
    return " ".join(parts)


@dataclass
class LavaKnowledgeTools:
    api_key: str
    semantic_scholar_api_key: str | None = None
    lava_gw: LavaGateway | None = field(default=None, repr=False)
    _s2_client: httpx.Client | None = field(default=None, init=False, repr=False)
    _cr_client: httpx.Client | None = field(default=None, init=False, repr=False)
    _lava_s2_ok: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        pass

    def __del__(self) -> None:
        if self._s2_client and not self._s2_client.is_closed:
            self._s2_client.close()
        if self._cr_client and not self._cr_client.is_closed:
            self._cr_client.close()

    @classmethod
    def from_config(cls) -> LavaKnowledgeTools:
        s = get_settings()
        lava_gw: LavaGateway | None = None
        lava_key = s.lava_api_key
        if lava_key and lava_key != "not-set":
            lava_gw = LavaGateway(
                secret_key=lava_key,
                customer_id=s.lava_customer_id,
                meter_slug=s.lava_meter_slug,
            )
            logger.info("Lava gateway enabled for Semantic Scholar API calls")
        return cls(
            api_key=lava_key,
            semantic_scholar_api_key=s.semantic_scholar_api_key,
            lava_gw=lava_gw,
        )

    def _get_s2_client(self) -> httpx.Client:
        if self._s2_client is None or self._s2_client.is_closed:
            headers: dict[str, str] = {}
            if self.semantic_scholar_api_key:
                headers["x-api-key"] = self.semantic_scholar_api_key
            self._s2_client = httpx.Client(
                base_url=_S2_API,
                headers=headers,
                timeout=20.0,
            )
        return self._s2_client

    def _get_cr_client(self) -> httpx.Client:
        if self._cr_client is None or self._cr_client.is_closed:
            self._cr_client = httpx.Client(
                timeout=15.0,
                headers={"User-Agent": "peer-review-system/1.0 (mailto:noreply@example.com)"},
            )
        return self._cr_client

    # ------------------------------------------------------------------
    # S2 request routing (Lava proxy or direct)
    # ------------------------------------------------------------------

    def _s2_extra_headers(self) -> dict[str, str] | None:
        """S2 API key header, forwarded through Lava as a pass-through header."""
        if self.semantic_scholar_api_key:
            return {"x-api-key": self.semantic_scholar_api_key}
        return None

    def _s2_post(
        self, path: str, *, params: dict[str, str] | None = None, json_body: Any = None,
    ) -> httpx.Response:
        """POST to Semantic Scholar, routing through Lava when available."""
        if self.lava_gw and self._lava_s2_ok:
            try:
                url = _S2_API + path
                if params:
                    url += "?" + urlencode(params)
                return self.lava_gw.forward_post(
                    url, json_body=json_body,
                    extra_headers=self._s2_extra_headers(), timeout=20.0,
                )
            except LavaEndpointNotSupported:
                logger.warning("S2 endpoint not supported by Lava, falling back to direct")
                self._lava_s2_ok = False
        client = self._get_s2_client()
        return client.post(path, params=params, json=json_body)

    def _s2_get(self, path: str, *, params: dict[str, str] | None = None) -> httpx.Response:
        """GET from Semantic Scholar, routing through Lava when available."""
        if self.lava_gw and self._lava_s2_ok:
            try:
                url = _S2_API + path
                return self.lava_gw.forward_get(
                    url, params=params,
                    extra_headers=self._s2_extra_headers(), timeout=20.0,
                )
            except LavaEndpointNotSupported:
                logger.warning("S2 endpoint not supported by Lava, falling back to direct")
                self._lava_s2_ok = False
        client = self._get_s2_client()
        return client.get(path, params=params)

    def resolve_citation(self, citation: dict) -> dict:
        """Resolve a single citation dict. Updates in place and returns it."""
        title = citation.get("title") or ""
        doi = citation.get("doi")

        result: dict | None = None
        if doi or title:
            result = self._search_s2_single(title, doi)

        if not result and title:
            result = self._search_crossref(title, citation.get("authors"))

        if result:
            source_text = result.get("abstract") or result.get("tldr") or result.get("summary") or ""
            citation["resolved"] = True
            citation["exists"] = True
            citation["source_text"] = source_text
        else:
            citation["resolved"] = False
            citation["exists"] = None
            citation["source_text"] = None

        return citation

    # ------------------------------------------------------------------
    # Semantic Scholar
    # ------------------------------------------------------------------

    def _s2_batch_by_id(self, citations: list[Citation]) -> dict[int, dict]:
        """Phase 1: batch lookup by DOI / arXiv ID (up to 500 per request)."""
        results: dict[int, dict] = {}
        id_map: list[tuple[int, str]] = []

        for i, cit in enumerate(citations):
            if cit.doi:
                id_map.append((i, f"DOI:{cit.doi}"))
            elif cit.arxiv_id:
                arxiv = cit.arxiv_id.replace("arXiv:", "").strip()
                id_map.append((i, f"ARXIV:{arxiv}"))

        for batch_start in range(0, len(id_map), 500):
            batch = id_map[batch_start : batch_start + 500]
            ids = [pid for _, pid in batch]
            for attempt in range(3):
                try:
                    resp = self._s2_post(
                        "/paper/batch",
                        params={"fields": "title,abstract,year,url,tldr"},
                        json_body={"ids": ids},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for (idx, _), paper in zip(batch, data):
                            if paper and paper.get("title"):
                                abstract = paper.get("abstract") or ""
                                tldr = ""
                                tldr_obj = paper.get("tldr")
                                if isinstance(tldr_obj, dict):
                                    tldr = tldr_obj.get("text") or ""
                                results[idx] = {
                                    "exists": True,
                                    "abstract": abstract or tldr,
                                    "year": paper.get("year"),
                                    "url": paper.get("url") or "",
                                }
                        break
                    if resp.status_code == 429:
                        wait = 3.0 * (attempt + 1)
                        logger.warning("S2 batch rate limited, waiting %.0fs...", wait)
                        time.sleep(wait)
                        continue
                    break
                except Exception as exc:
                    logger.warning("S2 batch lookup failed (attempt %d): %s", attempt + 1, exc)
                    if attempt < 2:
                        time.sleep(2.0)

        return results

    def _s2_title_search(self, cit: Citation) -> dict | None:
        """Search S2 by title + author, with title similarity check."""
        query = _build_s2_query(cit)
        if not query.strip():
            return None

        for attempt in range(2):
            try:
                resp = self._s2_get(
                    "/paper/search",
                    params={"query": query, "limit": "3", "fields": "title,abstract,year,url,tldr"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    papers = data.get("data") or []
                    if not papers:
                        return None
                    best = self._pick_best_match(cit, papers)
                    return best
                if resp.status_code == 429:
                    time.sleep(2.0 + attempt * 3.0)
                    continue
                return None
            except Exception:
                return None
        return None

    def _search_s2_single(self, title: str, doi: str | None = None) -> dict | None:
        """Single-citation S2 lookup for resolve_citation()."""
        if doi:
            try:
                resp = self._s2_get(f"/paper/DOI:{doi}", params={"fields": "title,abstract,year,url,tldr"})
                if resp.status_code == 200:
                    paper = resp.json()
                    if paper and paper.get("title"):
                        abstract = paper.get("abstract") or ""
                        tldr_obj = paper.get("tldr")
                        tldr = tldr_obj.get("text") if isinstance(tldr_obj, dict) else ""
                        return {"exists": True, "abstract": abstract or tldr, "year": paper.get("year"), "url": paper.get("url") or ""}
            except Exception:
                pass

        if not title:
            return None

        for attempt in range(2):
            try:
                resp = self._s2_get("/paper/search", params={"query": title, "limit": "1", "fields": "title,abstract,year,url"})
                if resp.status_code == 200:
                    papers = (resp.json().get("data") or [])
                    if papers:
                        p = papers[0]
                        return {"exists": True, "abstract": p.get("abstract") or "", "year": p.get("year"), "url": p.get("url") or ""}
                    return None
                if resp.status_code == 429:
                    time.sleep(2.0 + attempt * 3.0)
                    continue
                return None
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # CrossRef (free, no key needed, good title matching)
    # ------------------------------------------------------------------

    def _search_crossref(self, title: str, authors: list[str] | None = None) -> dict | None:
        """Query CrossRef for a work by title. Returns abstract if available."""
        if not title or len(title) < 10:
            return None

        client = self._get_cr_client()
        query = title
        if authors:
            first_author = authors[0].split(",")[0].strip() if authors else ""
            if first_author:
                query = f"{title} {first_author}"

        try:
            resp = client.get(
                _CROSSREF_API,
                params={"query.bibliographic": query, "rows": "3", "select": "title,abstract,author,DOI,published-print"},
            )
            if resp.status_code != 200:
                return None
            items = resp.json().get("message", {}).get("items", [])
            if not items:
                return None

            for item in items:
                cr_title = ""
                titles = item.get("title")
                if isinstance(titles, list) and titles:
                    cr_title = titles[0]
                elif isinstance(titles, str):
                    cr_title = titles
                if not cr_title:
                    continue

                sim = _title_similarity(title, cr_title)
                if sim < 0.6:
                    continue

                abstract = item.get("abstract") or ""
                abstract = re.sub(r"<[^>]+>", "", abstract).strip()
                doi = item.get("DOI") or ""
                year = None
                pub = item.get("published-print") or item.get("published-online")
                if isinstance(pub, dict):
                    parts = pub.get("date-parts", [[]])
                    if parts and parts[0]:
                        year = parts[0][0]

                return {"exists": True, "abstract": abstract, "year": year, "url": f"https://doi.org/{doi}" if doi else "", "doi": doi}

            return None
        except Exception as exc:
            logger.debug("CrossRef search failed for '%s': %s", title[:50], exc)
            return None

    # ------------------------------------------------------------------
    # OpenAlex (free, no key, vast coverage)
    # ------------------------------------------------------------------

    def _search_openalex(self, cit: Citation) -> dict | None:
        """Query OpenAlex for a work by title."""
        title = cit.title
        if not title or len(title) < 10:
            return None

        client = self._get_cr_client()
        try:
            search_q = _normalize_title(title)
            resp = client.get(
                _OPENALEX_API,
                params={"search": search_q, "per_page": "3", "select": "title,doi,publication_year,abstract_inverted_index"},
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            if not results:
                return None

            for item in results:
                oa_title = item.get("title") or ""
                if not oa_title:
                    continue
                sim = _title_similarity(title, oa_title)
                if sim < 0.6:
                    continue

                abstract = self._reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
                doi = (item.get("doi") or "").replace("https://doi.org/", "")
                year = item.get("publication_year")

                return {"exists": True, "abstract": abstract, "year": year, "url": f"https://doi.org/{doi}" if doi else "", "doi": doi}
            return None
        except Exception as exc:
            logger.debug("OpenAlex search failed for '%s': %s", (title or "")[:50], exc)
            return None

    @staticmethod
    def _reconstruct_openalex_abstract(inverted_index: dict | None) -> str:
        """OpenAlex stores abstracts as inverted indexes {word: [positions]}."""
        if not inverted_index or not isinstance(inverted_index, dict):
            return ""
        words: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                words.append((pos, word))
        words.sort()
        return " ".join(w for _, w in words)

    # ------------------------------------------------------------------
    # Best-match picker
    # ------------------------------------------------------------------

    def _pick_best_match(self, cit: Citation, papers: list[dict]) -> dict | None:
        """From a list of S2 search results, pick the best match by title similarity."""
        if not cit.title:
            if papers:
                p = papers[0]
                abstract = p.get("abstract") or ""
                tldr_obj = p.get("tldr")
                tldr = tldr_obj.get("text") if isinstance(tldr_obj, dict) else ""
                return {"exists": True, "abstract": abstract or tldr, "year": p.get("year"), "url": p.get("url") or ""}
            return None

        best_paper = None
        best_sim = 0.0

        for p in papers:
            p_title = p.get("title") or ""
            sim = _title_similarity(cit.title, p_title)
            if sim > best_sim:
                best_sim = sim
                best_paper = p

        if best_paper and best_sim >= 0.55:
            abstract = best_paper.get("abstract") or ""
            tldr_obj = best_paper.get("tldr")
            tldr = tldr_obj.get("text") if isinstance(tldr_obj, dict) else ""
            return {"exists": True, "abstract": abstract or tldr, "year": best_paper.get("year"), "url": best_paper.get("url") or ""}

        return None

    # ------------------------------------------------------------------
    # Concurrent multi-source title resolution
    # ------------------------------------------------------------------

    def _resolve_one_by_title(self, idx: int, cit: Citation) -> tuple[int, dict | None]:
        """Try S2 → CrossRef → OpenAlex. Keep going if a match has no abstract."""
        best_no_abstract: dict | None = None

        for searcher in (
            lambda: self._s2_title_search(cit),
            lambda: self._search_crossref(cit.title, cit.authors if cit.authors else None),
            lambda: self._search_openalex(cit),
        ):
            result = searcher()
            if result and result.get("abstract"):
                return idx, result
            if result and best_no_abstract is None:
                best_no_abstract = result

        return idx, best_no_abstract

    def _concurrent_title_search(self, indices: list[int], citations: list[Citation]) -> dict[int, dict]:
        """Run title searches concurrently across multiple APIs."""
        results: dict[int, dict] = {}
        if not indices:
            return results

        logger.info("Running concurrent title search for %d citations across S2 + CrossRef + OpenAlex...", len(indices))

        with ThreadPoolExecutor(max_workers=_TITLE_SEARCH_CONCURRENCY) as executor:
            futures = {}
            for i, idx in enumerate(indices):
                if i > 0 and i % _TITLE_SEARCH_CONCURRENCY == 0:
                    time.sleep(_TITLE_SEARCH_DELAY)
                cit = citations[idx]
                futures[executor.submit(self._resolve_one_by_title, idx, cit)] = idx

            for future in as_completed(futures):
                try:
                    idx, result = future.result(timeout=30.0)
                    if result:
                        results[idx] = result
                except Exception as exc:
                    logger.debug("Title search future failed: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Main batch resolution
    # ------------------------------------------------------------------

    def resolve_citations_batch(self, citations: list[Citation], max_concurrency: int = 5) -> list[Citation]:
        """
        Multi-phase citation resolution:
        1. S2 batch API for DOI/arXiv IDs (fast, 1-2 requests)
        2. Concurrent title search via S2 + CrossRef + OpenAlex for unresolved
        3. Backfill: re-search citations that resolved but got no abstract
        """
        logger.info("Resolving %d citations via multi-source batch API...", len(citations))

        # Phase 1: S2 batch by identifier
        resolved_map = self._s2_batch_by_id(citations)
        logger.info("Phase 1 (S2 batch by ID): resolved %d/%d", len(resolved_map), len(citations))

        # Phase 2: concurrent title search for completely unresolved
        unresolved = [i for i in range(len(citations)) if i not in resolved_map and citations[i].title]
        if unresolved:
            title_results = self._concurrent_title_search(unresolved, citations)
            resolved_map.update(title_results)
            logger.info("Phase 2 (title search): total resolved %d/%d", len(resolved_map), len(citations))

        # Phase 3: backfill — re-search citations that resolved with no abstract
        no_abstract = [
            i for i, r in resolved_map.items()
            if not r.get("abstract") and citations[i].title
        ]
        if no_abstract:
            logger.info("Phase 3 (backfill): %d citations resolved but missing abstracts, retrying...", len(no_abstract))
            backfill = self._concurrent_title_search(no_abstract, citations)
            for idx, result in backfill.items():
                if result.get("abstract"):
                    resolved_map[idx] = result

        with_abstract = sum(1 for r in resolved_map.values() if r.get("abstract"))
        logger.info("Final: %d/%d resolved, %d with abstracts", len(resolved_map), len(citations), with_abstract)

        out: list[Citation] = []
        for i, cit in enumerate(citations):
            if i in resolved_map:
                result = resolved_map[i]
                source_text = result.get("abstract") or ""
                update: dict[str, Any] = {
                    "resolved": True,
                    "exists": True,
                    "source_text": source_text,
                }
                if result.get("doi") and not cit.doi:
                    update["doi"] = result["doi"]
                out.append(cit.model_copy(update=update))
            else:
                out.append(cit.model_copy(update={
                    "resolved": False,
                    "exists": None,
                    "source_text": None,
                }))
        return out


# Keep backward-compatible alias
LavaRetrievalClient = LavaKnowledgeTools
