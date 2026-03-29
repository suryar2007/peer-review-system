"""K2 Think V2 client for claim verification and statistical auditing.

Falls back to Hermes (Nous API) when K2_BASE_URL is not configured,
since K2-Think-V2 is currently unavailable on HuggingFace.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import OpenAI

from config import get_settings
from pipeline.state import Citation, Claim, StatisticalAssertion, StatisticalAuditResult, VerificationResult

logger = logging.getLogger(__name__)

_HERMES_MODEL = "nousresearch/hermes-4-70b"


@dataclass
class K2ReasoningAgent:
    model_id: str
    base_url: str | None
    hf_token: str | None
    nous_api_key: str | None = None
    nous_base_url: str | None = None
    _openai_client: OpenAI | None = field(default=None, init=False, repr=False)
    _lava_gw: Any = field(default=None, init=False, repr=False)
    _lava_llm_ok: bool = field(default=True, init=False, repr=False)
    _completions_url: str = field(default="", init=False, repr=False)

    @classmethod
    def from_config(cls) -> K2ReasoningAgent:
        s = get_settings()
        agent = cls(
            model_id=s.k2_model_id,
            base_url=s.k2_base_url,
            hf_token=s.hf_token,
            nous_api_key=s.nous_api_key,
            nous_base_url=s.nous_base_url,
        )
        lava_key = s.lava_api_key
        if lava_key and lava_key != "not-set":
            from agents.lava_gateway import LavaGateway
            agent._lava_gw = LavaGateway(
                secret_key=lava_key,
                customer_id=s.lava_customer_id,
                meter_slug=s.lava_meter_slug,
                provider_key=s.nous_api_key,
            )
        base = (s.nous_base_url or "https://inference-api.nousresearch.com/v1").rstrip("/")
        agent._completions_url = f"{base}/chat/completions"
        return agent

    def _get_openai_client(self) -> OpenAI:
        if self._openai_client is None:
            base = (self.nous_base_url or "https://inference-api.nousresearch.com/v1").rstrip("/")
            self._openai_client = OpenAI(base_url=base, api_key=self.nous_api_key)
        return self._openai_client

    def _chat(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Send a chat completion request and return the content string.

        Uses K2_BASE_URL if configured, otherwise falls back to Hermes via Nous API.
        """
        if self.base_url:
            return self._chat_k2(system, user, max_tokens)
        return self._chat_hermes(system, user, max_tokens)

    def _chat_k2(self, system: str, user: str, max_tokens: int) -> str:
        """Chat via self-hosted K2 endpoint."""
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def _chat_hermes(self, system: str, user: str, max_tokens: int) -> str:
        """Chat via Hermes (Nous API) as fallback for K2, with retry.

        Routes through Lava gateway (BYOK) when configured.
        """
        import time
        from agents.lava_gateway import LavaEndpointNotSupported

        use_lava = self._lava_gw is not None and self._lava_llm_ok
        last_exc: Exception | None = None

        for attempt, wait in enumerate((0.0, 2.0, 6.0)):
            if wait:
                time.sleep(wait)
            try:
                if use_lava:
                    body = {
                        "model": _HERMES_MODEL,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.2,
                        "max_tokens": max_tokens,
                    }
                    resp = self._lava_gw.forward_post(
                        self._completions_url, json_body=body,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                else:
                    client = self._get_openai_client()
                    completion = client.chat.completions.create(
                        model=_HERMES_MODEL,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                        max_tokens=max_tokens,
                    )
                    content = completion.choices[0].message.content

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Empty response from Hermes reasoning fallback")
                return content
            except LavaEndpointNotSupported:
                logger.warning("Nous endpoint not supported by Lava, falling back to direct")
                self._lava_llm_ok = False
                use_lava = False
                continue
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if hasattr(exc, "response"):
                    status = getattr(exc.response, "status_code", status)
                if status in (429, 502, 503, 504) and attempt < 2:
                    logger.warning("Hermes reasoning retry (attempt %d): %s", attempt + 1, exc)
                    continue
                raise
        raise RuntimeError(f"Hermes reasoning failed after retries: {last_exc}") from last_exc

    def verify_claim(
        self,
        claim_text: str,
        cited_sources: list[dict],
        paper_context: str = "",
    ) -> VerificationResult:
        """
        Core verification method for a single claim.

        Determines whether the cited sources actually support the claim.
        Returns a VerificationResult with verdict, confidence, explanation,
        and relevant_passage.
        """
        try:
            sources_json = json.dumps(cited_sources, ensure_ascii=False, indent=2)
            system = (
                "You are a rigorous academic claim verifier. Your job is to determine whether "
                "the cited sources actually support the claim attributed to them.\n\n"
                "IMPORTANT — understand the rhetorical role of each citation before judging:\n"
                "- Contrast citations ('Unlike X...', 'In contrast to X...') reference prior work "
                "to highlight differences. The cited source is NOT expected to support the new claim; "
                "it only needs to exist and describe the prior method.\n"
                "- Background/dataset citations reference a dataset, benchmark, or tool by name. "
                "The source only needs to be the origin of that artifact.\n"
                "- Methodological ancestry citations ('inspired by X', 'based on X') reference the "
                "idea that influenced the work, not a direct replication.\n"
                "Only direct support citations are expected to substantiate the claim's conclusion.\n\n"
                "IMPORTANT — allow reasonable academic interpretation:\n"
                "Citing papers routinely characterize or reframe prior work in their own terms "
                "(e.g., calling a method 'not deeply bidirectional' even if the source never uses "
                "that phrase). This is normal scholarly practice, NOT misrepresentation. As long as "
                "the characterization is a fair reading of what the source describes, mark it "
                "'supported'. Only flag it if the characterization materially distorts the source.\n\n"
                "Verdict rules (apply in order):\n"
                "1. If the source text is missing, only a bibliographic stub, or too short to "
                "evaluate the claim, mark it 'unverifiable'. Do NOT mark it 'out_of_scope' or "
                "'contradicted' when you simply lack the source content.\n"
                "2. If the citation serves a contrast, background, or ancestry role and the source "
                "correctly describes the referenced prior work/dataset/method, mark it 'supported'.\n"
                "3. If the source clearly supports the claim's conclusion, mark it 'supported'.\n"
                "4. If the source supports a weaker version of the claim, mark it 'overstated'.\n"
                "5. If the source directly contradicts the claim, mark it 'contradicted'.\n"
                "6. If the source is genuinely about an unrelated topic AND you have sufficient "
                "source text to be sure, mark it 'out_of_scope'.\n\n"
                "When in doubt between 'out_of_scope' and 'unverifiable', prefer 'unverifiable'.\n\n"
                "Return ONLY a JSON object with keys:\n"
                "- verdict: one of 'supported', 'overstated', 'contradicted', 'out_of_scope', 'unverifiable'\n"
                "- confidence: float 0.0 to 1.0\n"
                "- explanation: 2-3 sentences explaining your verdict\n"
                "- relevant_passage: the most relevant passage from the source (quote it), or null"
            )
            context_part = f"\n\nSurrounding context from the paper:\n{paper_context}" if paper_context else ""
            user = (
                f"Claim: {claim_text}{context_part}\n\n"
                f"Cited sources:\n{sources_json}"
            )
            raw = self._chat(system, user)
            parsed = _robust_json_loads(raw)
            return VerificationResult(
                claim_text=claim_text,
                verdict=str(parsed.get("verdict", "unverifiable")),
                confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.0)))),
                explanation=str(parsed.get("explanation", "")),
                relevant_passage=parsed.get("relevant_passage"),
                citation_indices=[],
            )
        except Exception as exc:
            logger.warning("Reasoning verify_claim failed: %s", exc)
            return VerificationResult(
                claim_text=claim_text,
                verdict="unverifiable",
                confidence=0.0,
                explanation=f"Reasoning model unavailable: {exc}",
                relevant_passage=None,
                citation_indices=[],
            )

    def verify_claims(
        self,
        *,
        claims: list[Claim],
        resolved_citations: list[Citation],
    ) -> list[VerificationResult]:
        """Verify all claims against resolved citations (calls verify_claim per claim)."""
        results: list[VerificationResult] = []
        for claim in claims:
            cited_sources = []
            for idx in claim.supporting_citation_indices:
                if 0 <= idx < len(resolved_citations):
                    cit = resolved_citations[idx]
                    if cit.resolved and cit.source_text:
                        cited_sources.append({
                            "title": cit.title or cit.raw_text[:100],
                            "source_text": cit.source_text,
                        })

            vr = self.verify_claim(claim.text, cited_sources)
            vr = VerificationResult(
                claim_text=vr.claim_text,
                verdict=vr.verdict,
                confidence=vr.confidence,
                explanation=vr.explanation,
                relevant_passage=vr.relevant_passage,
                citation_indices=list(claim.supporting_citation_indices),
            )
            results.append(vr)
        return results

    def audit_statistics(self, assertions: list[dict], batch_size: int = 20) -> list[dict]:
        """
        Check whether reported statistical values are internally consistent.

        Processes all assertions in batches of ``batch_size`` (default 20).
        Returns list of {assertion_text, is_internally_consistent, issues, severity}.
        """
        if not assertions:
            return []

        system = (
            "You are a statistical auditor for academic papers. For each statistical assertion, check:\n"
            "- Is the p-value consistent with the reported sample size and effect size?\n"
            "- Are confidence intervals plausible given n?\n"
            "- Are there impossible values (p < 0, n < 0, CI where low > high)?\n"
            "- Does the precision of the p-value suggest rounding or reporting issues?\n\n"
            "Return ONLY a JSON object with key \"results\" (array). Each item must have:\n"
            "- assertion_text: string\n"
            "- is_internally_consistent: boolean\n"
            "- issues: array of strings describing problems found (empty if consistent)\n"
            "- severity: 'low', 'medium', or 'high'"
        )

        all_results: list[dict] = []
        for start in range(0, len(assertions), batch_size):
            batch = assertions[start : start + batch_size]
            try:
                user = json.dumps(batch, ensure_ascii=False, indent=2)
                raw = self._chat(system, user)
                parsed = _robust_json_loads(raw)
                if isinstance(parsed, list):
                    all_results.extend(parsed)
                elif isinstance(parsed, dict):
                    results = parsed.get("results")
                    if isinstance(results, list):
                        all_results.extend(results)
            except Exception as exc:
                logger.warning("K2 audit_statistics batch %d failed: %s", start // batch_size + 1, exc)
                all_results.extend(
                    {
                        "assertion_text": a.get("text", ""),
                        "is_internally_consistent": False,
                        "issues": [f"Audit failed: {exc}"],
                        "severity": "medium",
                    }
                    for a in batch
                )
        return all_results

    def audit_statistical_assertions(
        self,
        assertions: list[StatisticalAssertion],
    ) -> list[StatisticalAuditResult]:
        """Check internal consistency of reported statistics using Pydantic models."""
        raw_dicts = [a.model_dump(mode="json") for a in assertions]
        raw_results = self.audit_statistics(raw_dicts)
        return _parse_statistical_audit_results_from_dicts(raw_results, assertions)

    def reconstruct_evidence_chain(
        self,
        main_conclusion: str,
        key_citations: list[dict],
    ) -> dict:
        """
        For the paper's primary conclusion, trace the evidence chain.
        Does the evidence cited actually establish this conclusion,
        or does it only establish weaker intermediate results?

        Returns {chain_valid: bool, weakest_link: str, explanation: str}.
        """
        try:
            system = (
                "You are an evidence chain analyst. Given a paper's main conclusion and its key "
                "supporting citations (with abstracts/source text), determine whether the cited "
                "evidence actually establishes the conclusion, or only establishes weaker "
                "intermediate results.\n\n"
                "Return ONLY a JSON object with:\n"
                "- chain_valid: boolean (true if evidence chain fully supports the conclusion)\n"
                "- weakest_link: string describing the weakest point in the evidence chain\n"
                "- explanation: 2-4 sentences explaining your assessment"
            )
            user = json.dumps(
                {"main_conclusion": main_conclusion, "key_citations": key_citations},
                ensure_ascii=False,
                indent=2,
            )
            raw = self._chat(system, user)
            parsed = _robust_json_loads(raw)
            return {
                "chain_valid": bool(parsed.get("chain_valid", False)),
                "weakest_link": str(parsed.get("weakest_link", "")),
                "explanation": str(parsed.get("explanation", "")),
            }
        except Exception as exc:
            logger.warning("K2 reconstruct_evidence_chain failed: %s", exc)
            return {
                "chain_valid": False,
                "weakest_link": "Unable to assess",
                "explanation": f"K2 reasoning model unavailable: {exc}",
            }


# Keep backward-compatible alias
K2ThinkClient = K2ReasoningAgent


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


def _robust_json_loads(raw: str) -> Any:
    """Parse JSON with fallbacks for common LLM output issues."""
    import re
    text = _strip_markdown_fence(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fix invalid escape sequences (\_  \&  etc.)
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Try to find a JSON object in the text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Could not parse JSON from model output", text, 0)


def _parse_statistical_audit_results_from_dicts(
    raw_results: list[dict],
    assertions: list[StatisticalAssertion],
) -> list[StatisticalAuditResult]:
    out: list[StatisticalAuditResult] = []
    for i, assertion in enumerate(assertions):
        item = raw_results[i] if i < len(raw_results) else None
        if not isinstance(item, dict):
            out.append(
                StatisticalAuditResult(
                    assertion_text=assertion.text,
                    is_internally_consistent=False,
                    issues=["Missing or invalid audit result."],
                    severity="medium",
                )
            )
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
