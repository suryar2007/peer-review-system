"""Tests for the RAG module (agents/rag.py) and its integration with the reasoner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.rag import GeminiEmbedder, PaperChunkStore, PaperRAGIndex, chunk_text
from pipeline.state import Citation


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_input(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text(self):
        text = "Hello world."
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert chunks == ["Hello world."]

    def test_basic_chunking(self):
        text = "A" * 2500
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 3
        # Each chunk should be <= chunk_size
        for c in chunks:
            assert len(c) <= 1000

    def test_overlap_present(self):
        # Build text with clear sentence boundaries
        sentences = [f"Sentence number {i} here. " for i in range(50)]
        text = "".join(sentences)
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        # Adjacent chunks should share some content (overlap)
        for i in range(len(chunks) - 1):
            tail = chunks[i][-40:]
            assert tail in chunks[i + 1] or chunks[i + 1][:60] in chunks[i][-60:]

    def test_sentence_boundary_preference(self):
        text = "First sentence. Second sentence. Third sentence that is a bit longer to fill space."
        chunks = chunk_text(text, chunk_size=40, overlap=10)
        # Should prefer breaking at ". " rather than mid-word
        for c in chunks:
            if len(c) < len(text):
                # Chunk should end at or near a sentence boundary
                assert c.rstrip().endswith(".") or c.rstrip().endswith("?") or len(c) <= 40


# ---------------------------------------------------------------------------
# PaperChunkStore
# ---------------------------------------------------------------------------


class TestPaperChunkStore:
    def test_cosine_search_returns_top_k(self):
        # 5 chunks with known embeddings; query is identical to chunk 2
        dim = 4
        embeddings = np.eye(5, dim, dtype=np.float32)  # one-hot-ish rows
        # Normalize (already unit vectors for the first 4)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms

        chunks = [f"chunk_{i}" for i in range(5)]
        store = PaperChunkStore(chunks=chunks, embeddings=embeddings, abstract="abstract")

        query = [0.0, 0.0, 1.0, 0.0]  # matches chunk_2 perfectly
        results = store.search(query, top_k=2)
        assert results[0] == "chunk_2"
        assert len(results) == 2

    def test_top_k_capped_at_chunk_count(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store = PaperChunkStore(
            chunks=["a", "b"], embeddings=embeddings, abstract="",
        )
        results = store.search([1.0, 0.0], top_k=10)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# GeminiEmbedder
# ---------------------------------------------------------------------------


class TestGeminiEmbedder:
    def test_request_format(self):
        mock_gw = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_gw.forward_post.return_value = mock_resp

        embedder = GeminiEmbedder(mock_gw)
        result = embedder.embed_texts(["hello", "world"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        call_args = mock_gw.forward_post.call_args
        assert "generativelanguage.googleapis.com" in call_args[0][0]
        body = call_args[1]["json_body"]
        assert body["model"] == "gemini-embedding-2-preview"
        assert body["input"] == ["hello", "world"]

    def test_embed_query(self):
        mock_gw = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"embedding": [1.0, 2.0], "index": 0}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_gw.forward_post.return_value = mock_resp

        embedder = GeminiEmbedder(mock_gw)
        result = embedder.embed_query("test")
        assert result == [1.0, 2.0]

    def test_batching(self):
        mock_gw = MagicMock()

        def make_resp(inputs):
            resp = MagicMock()
            resp.json.return_value = {
                "data": [
                    {"embedding": [float(i)], "index": i}
                    for i in range(len(inputs))
                ],
            }
            resp.raise_for_status = MagicMock()
            return resp

        mock_gw.forward_post.side_effect = lambda url, **kw: make_resp(kw["json_body"]["input"])

        embedder = GeminiEmbedder(mock_gw)
        texts = [f"text_{i}" for i in range(25)]
        result = embedder.embed_texts(texts)
        assert len(result) == 25
        # Should have made 2 calls (batch size 20)
        assert mock_gw.forward_post.call_count == 2


# ---------------------------------------------------------------------------
# PaperRAGIndex
# ---------------------------------------------------------------------------


class TestPaperRAGIndex:
    def test_fallback_when_no_pdf(self):
        mock_gw = MagicMock()
        embedder = GeminiEmbedder(mock_gw)
        rag = PaperRAGIndex(embedder)

        cit = Citation(raw_text="Some ref", source_text="the abstract")
        # No open_access_pdf_url or arxiv_id → download fails
        result = rag.index_paper(0, cit)
        assert result is False
        assert not rag.has_store(0)

        # retrieve should return fallback
        text = rag.retrieve(0, "some claim", "fallback abstract")
        assert text == "fallback abstract"
        rag.close()

    def test_retrieve_with_indexed_paper(self):
        """Manually populate a store and verify retrieve returns RAG output."""
        mock_gw = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"embedding": [1.0, 0.0, 0.0], "index": 0}],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_gw.forward_post.return_value = mock_resp

        embedder = GeminiEmbedder(mock_gw)
        rag = PaperRAGIndex(embedder)

        # Manually insert a store
        chunks = ["Methods section text", "Results section text", "Discussion text"]
        emb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        rag._stores[0] = PaperChunkStore(
            chunks=chunks, embeddings=emb, abstract="Paper abstract",
        )

        result = rag.retrieve(0, "query", "fallback", top_k=2)
        assert "ABSTRACT:" in result
        assert "Paper abstract" in result
        assert "RELEVANT PASSAGES:" in result
        rag.close()


# ---------------------------------------------------------------------------
# Integration: _build_sources_for_claim with RAG
# ---------------------------------------------------------------------------


class TestBuildSourcesWithRAG:
    def test_without_rag_unchanged(self):
        """When rag_index is None, behavior is identical to original."""
        from pipeline.nodes.reasoner import _build_sources_for_claim
        from pipeline.state import Claim

        cit = Citation(
            raw_text="ref", title="Paper A", source_text="abstract text",
            resolved=True, exists=True,
        )
        claim = Claim(
            text="They found X",
            paper_section="intro",
            supporting_citation_indices=[0],
            claim_type="empirical",
        )
        sources, has_resolved = _build_sources_for_claim(claim, [cit], rag_index=None)
        assert has_resolved is True
        assert len(sources) == 1
        assert sources[0]["source_text"] == "abstract text"

    def test_with_rag_uses_retrieved_text(self):
        """When rag_index has a store, source_text is RAG-retrieved."""
        from pipeline.nodes.reasoner import _build_sources_for_claim
        from pipeline.state import Claim

        cit = Citation(
            raw_text="ref", title="Paper A", source_text="abstract text",
            resolved=True, exists=True,
        )
        claim = Claim(
            text="They found X",
            paper_section="intro",
            supporting_citation_indices=[0],
            claim_type="empirical",
        )

        mock_rag = MagicMock()
        mock_rag.has_store.return_value = True
        mock_rag.retrieve.return_value = "ABSTRACT:\nabstract\n\nRELEVANT PASSAGES:\nchunk1\n\n---\n\nchunk2"

        sources, has_resolved = _build_sources_for_claim(claim, [cit], rag_index=mock_rag)
        assert has_resolved is True
        assert "RELEVANT PASSAGES:" in sources[0]["source_text"]
        mock_rag.retrieve.assert_called_once()
