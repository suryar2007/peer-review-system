"""K2 Think V2 client for claim verification and statistical auditing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from config import get_settings
from pipeline.state import Citation, Claim, StatisticalAssertion, StatisticalAuditResult, VerificationResult


@dataclass
class K2ThinkClient:
    model_id: str
    base_url: str | None
    hf_token: str | None

    @classmethod
    def from_config(cls) -> K2ThinkClient:
        s = get_settings()
        return cls(
            model_id=s.k2_model_id,
            base_url=s.k2_base_url,
            hf_token=s.hf_token,
        )

    def verify_claims(
        self,
        *,
        claims: list[Claim],
        resolved_citations: list[Citation],
    ) -> list[VerificationResult]:
        """Assess whether resolved sources support each claim."""
        if not self.base_url:
            return [
                VerificationResult(
                    claim_text=c.text,
                    verdict="unverifiable",
                    confidence=0.0,
                    explanation="K2_BASE_URL not set; configure inference endpoint.",
                    relevant_passage=None,
                    citation_indices=list(c.supporting_citation_indices),
                )
                for c in claims
            ]

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        prompt = json.dumps(
            {
                "claims": [c.model_dump(mode="json") for c in claims],
                "resolved_citations": [c.model_dump(mode="json") for c in resolved_citations],
            },
            ensure_ascii=False,
        )
        body = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "For each claim, decide if the resolved citations support it. "
                        "Reply with JSON array of objects with keys: "
                        "claim_text, verdict (supported|overstated|contradicted|out_of_scope|unverifiable), "
                        "confidence (0-1), explanation, relevant_passage (string or null), citation_indices (ints)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_verification_results(content, claims)

    def audit_statistical_assertions(
        self,
        assertions: list[StatisticalAssertion],
    ) -> list[StatisticalAuditResult]:
        """Check internal consistency of reported statistics (stub when endpoint unset)."""
        if not self.base_url:
            return [
                StatisticalAuditResult(
                    assertion_text=a.text,
                    is_internally_consistent=True,
                    issues=[],
                    severity="low",
                )
                for a in assertions
            ]

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        prompt = json.dumps(
            [a.model_dump(mode="json") for a in assertions],
            ensure_ascii=False,
        )
        body = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Audit each statistical assertion for internal consistency with standard reporting. "
                        "Reply with JSON array: assertion_text, is_internally_consistent (bool), "
                        "issues (string array), severity (low|medium|high)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_statistical_audit_results(content, assertions)


def _strip_markdown_fence(raw: str) -> str:
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_verification_results(raw: str, claims: list[Claim]) -> list[VerificationResult]:
    raw = _strip_markdown_fence(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_verification(claims, "Failed to parse model output as JSON.")
    if not isinstance(parsed, list):
        return _fallback_verification(claims, "Model output was not a JSON array.")

    results: list[VerificationResult] = []
    for i, claim in enumerate(claims):
        item = parsed[i] if i < len(parsed) else None
        if not isinstance(item, dict):
            results.extend(_fallback_verification([claim], "Missing or invalid model row."))
            continue
        claim_text = str(item.get("claim_text") or claim.text)
        verdict = str(item.get("verdict") or "unverifiable")
        conf = item.get("confidence")
        try:
            confidence = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        explanation = str(item.get("explanation") or "")
        rel = item.get("relevant_passage")
        relevant_passage = str(rel) if rel is not None else None
        ci = item.get("citation_indices")
        citation_indices = (
            [int(x) for x in ci]
            if isinstance(ci, list)
            else list(claim.supporting_citation_indices)
        )
        results.append(
            VerificationResult(
                claim_text=claim_text,
                verdict=verdict,
                confidence=max(0.0, min(1.0, confidence)),
                explanation=explanation,
                relevant_passage=relevant_passage,
                citation_indices=citation_indices,
            )
        )
    return results


def _fallback_verification(claims: list[Claim], reason: str) -> list[VerificationResult]:
    return [
        VerificationResult(
            claim_text=c.text,
            verdict="unverifiable",
            confidence=0.0,
            explanation=reason,
            relevant_passage=None,
            citation_indices=list(c.supporting_citation_indices),
        )
        for c in claims
    ]


def _parse_statistical_audit_results(
    raw: str,
    assertions: list[StatisticalAssertion],
) -> list[StatisticalAuditResult]:
    raw = _strip_markdown_fence(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_audits(assertions, "Failed to parse model output as JSON.")
    if not isinstance(parsed, list):
        return _fallback_audits(assertions, "Model output was not a JSON array.")

    out: list[StatisticalAuditResult] = []
    for i, assertion in enumerate(assertions):
        item = parsed[i] if i < len(parsed) else None
        if not isinstance(item, dict):
            out.extend(_fallback_audits([assertion], "Missing or invalid model row."))
            continue
        text = str(item.get("assertion_text") or assertion.text)
        consistent = bool(item.get("is_internally_consistent", True))
        issues = item.get("issues")
        issue_list = [str(x) for x in issues] if isinstance(issues, list) else []
        severity = str(item.get("severity") or "low")
        out.append(
            StatisticalAuditResult(
                assertion_text=text,
                is_internally_consistent=consistent,
                issues=issue_list,
                severity=severity,
            )
        )
    return out


def _fallback_audits(
    assertions: list[StatisticalAssertion],
    reason: str,
) -> list[StatisticalAuditResult]:
    return [
        StatisticalAuditResult(
            assertion_text=a.text,
            is_internally_consistent=False,
            issues=[reason],
            severity="medium",
        )
        for a in assertions
    ]
