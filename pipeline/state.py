"""LangGraph state schema and Pydantic domain models for the peer-review pipeline."""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

import operator
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Citation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_text: str
    authors: list[str] = Field(default_factory=list)
    title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    resolved: bool = False
    source_text: Optional[str] = None
    exists: Optional[bool] = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    paper_section: str
    supporting_citation_indices: list[int] = Field(default_factory=list)
    claim_type: str


class StatisticalAssertion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    confidence_interval: Optional[tuple[float, float]] = None
    section: str

    @field_validator("confidence_interval", mode="before")
    @classmethod
    def _coerce_confidence_interval(
        cls, value: object
    ) -> Optional[tuple[float, float]]:
        if value is None:
            return None
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]), float(value[1])
        if isinstance(value, list) and len(value) == 2:
            return float(value[0]), float(value[1])
        return None


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_text: str
    verdict: str
    confidence: float
    explanation: str
    relevant_passage: Optional[str] = None
    citation_indices: list[int] = Field(default_factory=list)


class StatisticalAuditResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assertion_text: str
    is_internally_consistent: bool
    issues: list[str] = Field(default_factory=list)
    severity: str


class PipelineState(TypedDict):
    # Input
    paper_path: str
    paper_text: str
    paper_title: Optional[str]
    paper_abstract: Optional[str]

    # Phase 1 outputs (Hermes extraction)
    citations: list[Citation]
    claims: list[Claim]
    statistical_assertions: list[StatisticalAssertion]

    # Phase 2 outputs (Lava retrieval)
    resolved_citations: list[Citation]

    # Phase 3 outputs (K2 reasoning)
    verification_results: list[VerificationResult]
    statistical_audit_results: list[StatisticalAuditResult]

    # Phase 4 outputs (reporting)
    hex_run_id: Optional[str]
    dashboard_url: Optional[str]

    # Pipeline control
    current_phase: str
    errors: Annotated[list[str], operator.add]

    # Summary stats (computed in reporter)
    total_citations: int
    resolved_count: int
    hallucinated_count: int
    supported_claims: int
    flagged_claims: int


def initial_state(paper_path: str) -> PipelineState:
    """Return pipeline state for a new run with empty collections and zeroed counters."""
    return {
        "paper_path": paper_path,
        "paper_text": "",
        "paper_title": None,
        "paper_abstract": None,
        "citations": [],
        "claims": [],
        "statistical_assertions": [],
        "resolved_citations": [],
        "verification_results": [],
        "statistical_audit_results": [],
        "hex_run_id": None,
        "dashboard_url": None,
        "current_phase": "ingestion",
        "errors": [],
        "total_citations": 0,
        "resolved_count": 0,
        "hallucinated_count": 0,
        "supported_claims": 0,
        "flagged_claims": 0,
    }
