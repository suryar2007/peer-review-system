"""Tests for the extraction pipeline — runs offline with mocked API calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from pipeline.state import Citation, Claim, VerificationResult


# ---------------------------------------------------------------------------
# 1. test_pdf_parser_basic
# ---------------------------------------------------------------------------

def test_pdf_parser_basic(tmp_path: Path) -> None:
    """PaperParser returns a dict with all expected keys and non-empty references."""
    from utils.pdf_parser import PaperParser

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "A Great Title", fontsize=18)
    page.insert_text((72, 120), "Abstract")
    page.insert_text((72, 140), "This is the abstract of the paper. " * 10)
    page.insert_text((72, 300), "1 Introduction")
    page.insert_text((72, 320), "Some introduction text here.")
    page.insert_text((72, 400), "References")
    page.insert_text((72, 420), "[1] Smith, J. (2023). A paper title. Journal of Testing, 1(1), 1-10.")
    page.insert_text((72, 440), "[2] Doe, A. (2022). Another paper. arXiv:2201.00001.")
    doc.save(pdf_path)
    doc.close()

    result = PaperParser(str(pdf_path)).parse()

    assert "title" in result
    assert "abstract" in result
    assert "full_text" in result
    assert "sections" in result
    assert "references_raw" in result
    assert "page_count" in result
    assert result["page_count"] == 1
    assert result["references_raw"].strip() != ""


def test_pdf_parser_file_not_found(tmp_path: Path) -> None:
    """PaperParser raises FileNotFoundError for missing PDFs."""
    from utils.pdf_parser import PaperParser

    with pytest.raises(FileNotFoundError):
        PaperParser(str(tmp_path / "nope.pdf")).parse()


# ---------------------------------------------------------------------------
# 2. test_hermes_citation_extraction_schema
# ---------------------------------------------------------------------------

def test_hermes_citation_extraction_schema() -> None:
    """Hermes extract_citations returns dicts that validate as Citation models."""
    mock_response = {
        "citations": [
            {
                "authors": ["Smith, J.", "Doe, A."],
                "title": "Learning with attention",
                "journal": "Journal of ML",
                "year": 2023,
                "doi": "10.1234/test",
                "arxiv_id": None,
                "url": None,
                "raw_text": "[1] Smith, J., & Doe, A. (2023). Learning with attention. Journal of ML.",
            },
            {
                "authors": ["Chen, L."],
                "title": "Efficient transformers",
                "journal": "ICML",
                "year": 2022,
                "doi": None,
                "arxiv_id": "2201.00001",
                "url": None,
                "raw_text": "[2] Chen, L. (2022). Efficient transformers. ICML.",
            },
            {
                "authors": ["Brown, M."],
                "title": "Open problems in optimization",
                "journal": None,
                "year": 2021,
                "doi": None,
                "arxiv_id": "2201.00002",
                "url": None,
                "raw_text": "[3] Brown, M. (2021). Open problems in optimization.",
            },
            {
                "authors": ["Lee, K.", "Park, S."],
                "title": "Neural scaling laws",
                "journal": "NeurIPS",
                "year": 2023,
                "doi": "10.5678/nips",
                "arxiv_id": None,
                "url": "https://example.com/paper4",
                "raw_text": "[4] Lee, K., & Park, S. (2023). Neural scaling laws. NeurIPS.",
            },
            {
                "authors": ["Wang, Z."],
                "title": "Survey of language models",
                "journal": "ACL",
                "year": 2024,
                "doi": None,
                "arxiv_id": None,
                "url": None,
                "raw_text": "[5] Wang, Z. (2024). Survey of language models. ACL.",
            },
        ]
    }

    mock_completion = MagicMock()
    mock_completion.usage = MagicMock(prompt_tokens=100, completion_tokens=200, total_tokens=300)
    mock_completion.choices = [
        MagicMock(message=MagicMock(content=json.dumps(mock_response)))
    ]

    with patch("agents.hermes.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            nous_api_key="test-key",
            nous_base_url="https://test.example.com/v1",
            lava_api_key="not-set",
            lava_customer_id=None,
            lava_meter_slug=None,
        )
        with patch("agents.hermes.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_client

            from agents.hermes import HermesAgent
            agent = HermesAgent()
            result = agent.extract_citations(
                "[1] Smith, J., & Doe, A. (2023). Learning with attention.\n"
                "[2] Chen, L. (2022). Efficient transformers.\n"
                "[3] Brown, M. (2021). Open problems in optimization.\n"
                "[4] Lee, K., & Park, S. (2023). Neural scaling laws.\n"
                "[5] Wang, Z. (2024). Survey of language models."
            )

    assert len(result) == 5
    for item in result:
        assert isinstance(item, dict)
        assert "title" in item
        assert "authors" in item
        assert "raw_text" in item
        # Verify Pydantic Citation model instantiates successfully
        cit = Citation.model_validate(item)
        assert cit.raw_text
        assert isinstance(cit.authors, list)


# ---------------------------------------------------------------------------
# 3. test_citation_resolution_fallback
# ---------------------------------------------------------------------------

def test_citation_resolution_fallback() -> None:
    """When Lava tools fail, fallback to Semantic Scholar direct. If both fail, resolved=False."""
    from agents.lava_tools import LavaKnowledgeTools

    tools = LavaKnowledgeTools(api_key="test", semantic_scholar_api_key=None)

    citation = {
        "title": "A nonexistent paper",
        "authors": ["Nobody"],
        "doi": None,
        "arxiv_id": None,
        "raw_text": "A nonexistent paper reference",
    }

    # Mock all HTTP calls to fail
    with patch("agents.lava_tools.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("Connection failed")
        mock_client.get.side_effect = Exception("Connection failed")
        mock_client_cls.return_value = mock_client

        result = tools.resolve_citation(citation)

    assert result["resolved"] is False
    assert result.get("source_text") is None


# ---------------------------------------------------------------------------
# 4. test_pipeline_state_accumulates_errors
# ---------------------------------------------------------------------------

def test_pipeline_state_accumulates_errors() -> None:
    """The errors list in PipelineState accumulates via operator.add without overwriting."""
    from pipeline.state import initial_state

    state = initial_state("/fake/paper.pdf")
    assert state["errors"] == []

    # Simulate what LangGraph does: merge via operator.add
    import operator
    existing = state["errors"]
    new_errors_1 = ["extraction: failed to parse references"]
    merged = operator.add(existing, new_errors_1)
    assert merged == ["extraction: failed to parse references"]

    new_errors_2 = ["resolver: Lava API timeout"]
    merged = operator.add(merged, new_errors_2)
    assert merged == [
        "extraction: failed to parse references",
        "resolver: Lava API timeout",
    ]
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# 5. test_k2_unavailable_graceful_degradation
# ---------------------------------------------------------------------------

def test_k2_unavailable_graceful_degradation() -> None:
    """When K2 is unreachable, verify_claim returns verdict='unverifiable' without crashing."""
    from agents.k2 import K2ReasoningAgent

    agent = K2ReasoningAgent(
        model_id="test-model",
        base_url="http://unreachable.invalid:9999",
        hf_token=None,
    )

    result = agent.verify_claim(
        claim_text="Transformers outperform RNNs on all NLP tasks",
        cited_sources=[{"title": "Test paper", "source_text": "Some text about models."}],
    )

    assert isinstance(result, VerificationResult)
    assert result.verdict == "unverifiable"
    assert result.confidence == 0.0
    assert "unavailable" in result.explanation.lower() or "error" in result.explanation.lower()


# ---------------------------------------------------------------------------
# Backward compat: extract_text_from_pdf
# ---------------------------------------------------------------------------

def test_extract_text_from_pdf(tmp_path: Path) -> None:
    """extract_text_from_pdf returns full document text."""
    from utils.pdf_parser import extract_text_from_pdf

    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello peer review")
    doc.save(pdf_path)
    doc.close()

    text = extract_text_from_pdf(pdf_path)
    assert "Hello peer review" in text


def test_extract_text_missing_file(tmp_path: Path) -> None:
    from utils.pdf_parser import extract_text_from_pdf

    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(tmp_path / "nope.pdf")


# ---------------------------------------------------------------------------
# 6. test_title_similarity
# ---------------------------------------------------------------------------

def test_title_similarity() -> None:
    """Title similarity helper correctly scores exact, close, and distant pairs."""
    from agents.lava_tools import _title_similarity

    assert _title_similarity("Learning with attention", "Learning with attention") > 0.99
    assert _title_similarity("Learning with Attention", "learning with attention") > 0.99
    assert _title_similarity(
        "Efficient transformers for NLP",
        "Efficient Transformers for Natural Language Processing",
    ) > 0.6
    assert _title_similarity("Quantum computing", "Deep learning for images") < 0.4


# ---------------------------------------------------------------------------
# 7. test_build_sources_for_claim_with_fallback
# ---------------------------------------------------------------------------

def test_build_sources_for_claim_with_fallback() -> None:
    """When no citations are resolved, fallback sources use raw citation text."""
    from pipeline.nodes.reasoner import _build_sources_for_claim

    unresolved_cit = Citation(
        raw_text="[42] Smith, J. (2023). Some important paper. Nature, 1-10.",
        title="Some important paper",
        resolved=False,
        source_text=None,
    )
    claim = Claim(
        text="Models achieve 95% accuracy",
        paper_section="Results",
        supporting_citation_indices=[0],
        claim_type="empirical",
    )

    sources, has_resolved = _build_sources_for_claim(claim, [unresolved_cit])
    assert not has_resolved
    assert len(sources) == 1
    assert sources[0]["resolved"] is False
    assert "[Unresolved citation" in sources[0]["source_text"]


def test_build_sources_prefers_resolved() -> None:
    """When resolved citations exist, they are included and has_resolved is True."""
    from pipeline.nodes.reasoner import _build_sources_for_claim

    resolved_cit = Citation(
        raw_text="[1] Doe, A. (2022). Good paper.",
        title="Good paper",
        resolved=True,
        source_text="This paper studies NLP performance...",
    )
    claim = Claim(
        text="NLP performance improved",
        paper_section="Results",
        supporting_citation_indices=[0],
        claim_type="empirical",
    )

    sources, has_resolved = _build_sources_for_claim(claim, [resolved_cit])
    assert has_resolved
    assert len(sources) == 1
    assert sources[0]["resolved"] is True


# ---------------------------------------------------------------------------
# 8. test_html_report_generation
# ---------------------------------------------------------------------------

def test_html_report_generation(tmp_path: Path) -> None:
    """Reporter generates a valid HTML file with key sections."""
    from pipeline.nodes.reporter import _generate_html_report

    state = {
        "paper_path": str(tmp_path / "test.pdf"),
        "paper_title": "Test Paper Title",
        "citations": [],
        "resolved_citations": [],
        "verification_results": [],
        "statistical_audit_results": [],
        "statistical_assertions": [],
        "errors": ["test error"],
    }
    summary = {
        "total_citations": 10,
        "resolved_count": 7,
        "hallucinated_count": 1,
        "supported_claims": 3,
        "flagged_claims": 2,
    }

    html_content = _generate_html_report(state, summary)
    assert "Test Paper Title" in html_content
    assert "test error" in html_content
    assert "Claim Verification" in html_content
    assert "Statistical Audit" in html_content
    assert "Unresolved Citations" in html_content


# ---------------------------------------------------------------------------
# 9. test_openalex_abstract_reconstruction
# ---------------------------------------------------------------------------

def test_openalex_abstract_reconstruction() -> None:
    """OpenAlex inverted index abstracts are correctly reconstructed."""
    from agents.lava_tools import LavaKnowledgeTools

    inverted = {"Deep": [0], "learning": [1], "is": [2], "powerful": [3]}
    result = LavaKnowledgeTools._reconstruct_openalex_abstract(inverted)
    assert result == "Deep learning is powerful"

    assert LavaKnowledgeTools._reconstruct_openalex_abstract(None) == ""
    assert LavaKnowledgeTools._reconstruct_openalex_abstract({}) == ""


# ---------------------------------------------------------------------------
# 10. test_robust_json_loads
# ---------------------------------------------------------------------------

def test_robust_json_loads() -> None:
    """K2's robust JSON parser handles markdown fences and invalid escapes."""
    from agents.k2 import _robust_json_loads

    assert _robust_json_loads('{"a": 1}') == {"a": 1}
    assert _robust_json_loads('```json\n{"a": 1}\n```') == {"a": 1}
    result = _robust_json_loads('{"verdict": "supported", "confidence": 0.9, "explanation": "test"}')
    assert result["verdict"] == "supported"


# ---------------------------------------------------------------------------
# 11. test_lava_gateway_auth_modes
# ---------------------------------------------------------------------------

def test_lava_gateway_auth_modes() -> None:
    """LavaGateway produces correct auth headers for simple key and forward token modes."""
    import base64
    from agents.lava_gateway import LavaGateway

    gw_simple = LavaGateway(secret_key="aks_live_abc123")
    assert gw_simple._auth_header() == "Bearer aks_live_abc123"
    assert gw_simple.is_configured

    gw_byok = LavaGateway(secret_key="aks_live_abc123", provider_key="sk-nous")
    token_header = gw_byok._auth_header()
    assert token_header.startswith("Bearer ")
    decoded = json.loads(base64.b64decode(token_header.split(" ", 1)[1]))
    assert decoded["secret_key"] == "aks_live_abc123"
    assert decoded["provider_key"] == "sk-nous"
    assert "customer_id" not in decoded

    gw_full = LavaGateway(
        secret_key="aks_live_abc123",
        customer_id="conn_xyz",
        meter_slug="my-meter",
        provider_key="sk-nous",
    )
    full_decoded = json.loads(base64.b64decode(gw_full._auth_header().split(" ", 1)[1]))
    assert full_decoded["customer_id"] == "conn_xyz"
    assert full_decoded["meter_slug"] == "my-meter"

    gw_notset = LavaGateway(secret_key="not-set")
    assert not gw_notset.is_configured


# ---------------------------------------------------------------------------
# 12. test_lava_gateway_url_construction
# ---------------------------------------------------------------------------

def test_lava_gateway_url_construction() -> None:
    """LavaGateway correctly constructs forward URLs with URL-encoded provider URLs."""
    from agents.lava_gateway import LavaGateway

    s2_url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract"
    forward_url = LavaGateway._forward_url(s2_url)
    assert forward_url.startswith("https://api.lava.so/v1/forward?u=")
    assert "semanticscholar" in forward_url
    assert "paper%2Fbatch" in forward_url or "paper/batch" not in forward_url.split("u=")[1].split("&")[0]


# ---------------------------------------------------------------------------
# 13. test_s2_routing_through_lava
# ---------------------------------------------------------------------------

def test_s2_routing_through_lava() -> None:
    """CitationResolver routes S2 calls through Lava when gateway is configured."""
    from agents.lava_gateway import LavaGateway
    from agents.lava_tools import LavaKnowledgeTools

    mock_gw = MagicMock(spec=LavaGateway)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"title": "Test Paper", "abstract": "Test abstract", "year": 2023, "url": "", "tldr": None}
    ]
    mock_gw.forward_post.return_value = mock_resp

    tools = LavaKnowledgeTools(api_key="aks_live_test", lava_gw=mock_gw)

    from pipeline.state import Citation
    citations = [Citation(raw_text="[1] Test", title="Test Paper", doi="10.1234/test")]
    results = tools._s2_batch_by_id(citations)

    mock_gw.forward_post.assert_called_once()
    call_args = mock_gw.forward_post.call_args
    assert "semanticscholar" in call_args[0][0]
    assert "paper/batch" in call_args[0][0]
