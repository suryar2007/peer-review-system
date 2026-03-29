"""RAG-based source text retrieval for claim verification.

Downloads full-text PDFs of cited papers, chunks them, embeds via Gemini
through the Lava gateway, and retrieves semantically relevant passages
for each claim — replacing abstract-only evidence.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from agents.lava_gateway import LavaEndpointNotSupported, LavaGateway
from pipeline.state import Citation
from utils.pdf_parser import PaperParser

logger = logging.getLogger(__name__)

_GEMINI_EMBEDDINGS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/embeddings"
)
_GEMINI_MODEL = "gemini-embedding-2-preview"
_EMBED_BATCH_SIZE = 20
_PDF_DOWNLOAD_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str, chunk_size: int = 1000, overlap: int = 200,
) -> list[str]:
    """Split *text* into overlapping chunks, breaking at sentence boundaries."""
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to break at a sentence boundary within the last 20% of the chunk
        if end < length:
            search_start = max(start, end - chunk_size // 5)
            best_break = -1
            for sep in (". ", "? ", "! ", ".\n", "\n\n"):
                pos = text.rfind(sep, search_start, end)
                if pos > best_break:
                    best_break = pos + len(sep)
            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= length:
            break
        # Avoid infinite loop when overlap >= effective chunk
        if start <= (end - chunk_size) and end < length:
            start = end

    return chunks


# ---------------------------------------------------------------------------
# Gemini embedder via Lava
# ---------------------------------------------------------------------------


class GeminiEmbedder:
    """Embed text using Gemini through the Lava forward proxy."""

    def __init__(self, lava_gw: LavaGateway) -> None:
        self._gw = lava_gw
        self._ok = True

    def _post(self, inputs: list[str]) -> list[list[float]]:
        """Single batch POST to Gemini embeddings endpoint via Lava."""
        body: dict[str, Any] = {"model": _GEMINI_MODEL, "input": inputs}
        backoffs = (0.0, 2.0, 5.0)
        last_exc: Exception | None = None

        for attempt, wait in enumerate(backoffs):
            if wait:
                time.sleep(wait)
            try:
                resp = self._gw.forward_post(
                    _GEMINI_EMBEDDINGS_URL, json_body=body, timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                items = sorted(data["data"], key=lambda d: d["index"])
                return [item["embedding"] for item in items]
            except LavaEndpointNotSupported:
                self._ok = False
                raise
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (429, 502, 503, 504) and attempt < len(backoffs) - 1:
                    last_exc = exc
                    logger.warning(
                        "Gemini embed HTTP %d (attempt %d/%d)",
                        status, attempt + 1, len(backoffs),
                    )
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < len(backoffs) - 1:
                    logger.warning(
                        "Gemini embed error (attempt %d/%d): %s",
                        attempt + 1, len(backoffs), exc,
                    )
                    continue
                raise

        raise RuntimeError(f"Gemini embedding failed: {last_exc}") from last_exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            all_embeddings.extend(self._post(batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self._post([text])[0]


# ---------------------------------------------------------------------------
# Chunk store with cosine search
# ---------------------------------------------------------------------------


@dataclass
class PaperChunkStore:
    """In-memory store for a single paper's chunks and their embeddings."""

    chunks: list[str]
    embeddings: np.ndarray  # shape (n_chunks, dim), float32, L2-normalized
    abstract: str

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[str]:
        """Return the top-k most similar chunks by cosine similarity."""
        qvec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec /= norm
        scores = self.embeddings @ qvec
        k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self.chunks[i] for i in top_indices]


# ---------------------------------------------------------------------------
# RAG index: download, chunk, embed, retrieve
# ---------------------------------------------------------------------------


class PaperRAGIndex:
    """Per-pipeline-run index that lazily downloads and indexes cited papers."""

    def __init__(self, embedder: GeminiEmbedder) -> None:
        self._embedder = embedder
        self._stores: dict[int, PaperChunkStore | None] = {}
        self._temp_dir = tempfile.TemporaryDirectory()

    def close(self) -> None:
        self._temp_dir.cleanup()

    def index_paper(self, idx: int, citation: Citation) -> bool:
        """Download, parse, chunk, and embed a cited paper. Returns True on success."""
        if idx in self._stores:
            return self._stores[idx] is not None

        pdf_bytes = self._download_pdf(citation)
        if not pdf_bytes:
            self._stores[idx] = None
            return False

        try:
            pdf_path = Path(self._temp_dir.name) / f"cit_{idx}.pdf"
            pdf_path.write_bytes(pdf_bytes)
            parsed = PaperParser(str(pdf_path)).parse()
            full_text = parsed.get("full_text") or ""
            if len(full_text) < 200:
                logger.debug("Citation %d: paper text too short (%d chars)", idx, len(full_text))
                self._stores[idx] = None
                return False

            chunks = chunk_text(full_text)
            if not chunks:
                self._stores[idx] = None
                return False

            embeddings = self._embedder.embed_texts(chunks)
            emb_array = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb_array /= norms

            abstract = citation.source_text or parsed.get("abstract") or ""
            self._stores[idx] = PaperChunkStore(
                chunks=chunks, embeddings=emb_array, abstract=abstract,
            )
            logger.info("Citation %d indexed: %d chunks", idx, len(chunks))
            return True
        except Exception as exc:
            logger.warning("Citation %d: indexing failed: %s", idx, exc)
            self._stores[idx] = None
            return False

    def retrieve(
        self, idx: int, claim_text: str, abstract_fallback: str, top_k: int = 5,
    ) -> str:
        """Retrieve relevant passages for a claim from an indexed paper.

        Falls back to *abstract_fallback* when the paper was not indexed.
        """
        store = self._stores.get(idx)
        if store is None:
            return abstract_fallback

        try:
            query_emb = self._embedder.embed_query(claim_text)
            top_chunks = store.search(query_emb, top_k=top_k)
        except Exception as exc:
            logger.warning("RAG retrieve failed for citation %d: %s", idx, exc)
            return abstract_fallback

        passages = "\n\n---\n\n".join(top_chunks)
        abstract = store.abstract or abstract_fallback
        return f"ABSTRACT:\n{abstract}\n\nRELEVANT PASSAGES:\n{passages}"

    def has_store(self, idx: int) -> bool:
        return self._stores.get(idx) is not None

    @staticmethod
    def _download_pdf(citation: Citation) -> bytes | None:
        """Try open-access PDF URL, then arXiv fallback."""
        urls: list[str] = []
        if citation.open_access_pdf_url:
            urls.append(citation.open_access_pdf_url)
        if citation.arxiv_id:
            arxiv = citation.arxiv_id.replace("arXiv:", "").strip()
            urls.append(f"https://arxiv.org/pdf/{arxiv}")

        for url in urls:
            try:
                with httpx.Client(
                    timeout=_PDF_DOWNLOAD_TIMEOUT, follow_redirects=True,
                ) as client:
                    resp = client.get(url)
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        logger.info("Downloaded PDF from %s (%d bytes)", url, len(resp.content))
                        return resp.content
            except Exception as exc:
                logger.debug("PDF download failed for %s: %s", url, exc)
        return None
