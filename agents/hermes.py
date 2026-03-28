"""Nous Research Hermes agent using the OpenAI-compatible Python client."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "hermes-4-70b"
MAX_PAPER_CHARS = 120_000


class HermesExtractionError(RuntimeError):
    """Raised when Hermes returns an error or output that cannot be parsed."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _env_model() -> str:
    raw = os.getenv("NOUS_MODEL")
    return raw.strip() if raw and raw.strip() else DEFAULT_MODEL


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_json_object(content: str) -> dict[str, Any]:
    text = _strip_json_fence(content)
    try:
        out = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HermesExtractionError(f"Invalid JSON from model: {exc}") from exc
    if not isinstance(out, dict):
        raise HermesExtractionError("Model JSON was not an object")
    return out


class HermesAgent:
    """OpenAI-compatible client for Nous Hermes chat completions."""

    def __init__(self) -> None:
        settings = get_settings()
        base = settings.nous_base_url.rstrip("/")
        self._client = OpenAI(base_url=base, api_key=settings.nous_api_key)
        self._model = _env_model()

    def _log_usage(self, operation: str, usage: Any) -> None:
        if usage is None:
            logger.info("Hermes %s: no usage metadata returned", operation)
            return
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        logger.info(
            "Hermes %s: prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            operation,
            pt,
            ct,
            tt,
        )

    def _chat_json(
        self,
        *,
        operation: str,
        system_prompt: str,
        user_content: str,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt, sleep_s in enumerate((0.0, 2.0)):
            if sleep_s:
                time.sleep(sleep_s)
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                self._log_usage(operation, completion.usage)
                choice = completion.choices[0]
                msg = choice.message
                content = msg.content
                if not isinstance(content, str) or not content.strip():
                    raise HermesExtractionError("Empty model content")
                return _parse_json_object(content)
            except RateLimitError as exc:
                last_exc = exc
                logger.warning("Hermes rate limit (attempt %s): %s", attempt + 1, exc)
                if attempt == 0:
                    continue
                raise HermesExtractionError(
                    f"Hermes rate limited after retry: {exc}",
                    status_code=getattr(exc, "status_code", None) or 429,
                ) from exc
            except APIStatusError as exc:
                if exc.status_code == 429 and attempt == 0:
                    last_exc = exc
                    logger.warning("Hermes HTTP 429 (attempt %s): %s", attempt + 1, exc)
                    continue
                raise HermesExtractionError(
                    f"Hermes API error {exc.status_code}: {exc}",
                    status_code=exc.status_code,
                ) from exc
            except (APIConnectionError, APITimeoutError) as exc:
                raise HermesExtractionError(f"Hermes connection error: {exc}") from exc

        raise HermesExtractionError(f"Hermes request failed: {last_exc}") from last_exc

    def extract_citations(self, references_raw: str) -> list[dict]:
        """
        Parse the raw references / bibliography section into structured citation dicts.

        Each dict should align with the ``Citation`` model: authors, title, journal,
        year, doi, arxiv_id, url, raw_text.
        """
        text = (references_raw or "").strip()
        if not text:
            return []

        system = (
            "You are a citation extraction specialist. Extract every reference from the "
            "provided bibliography section. For each reference, extract all available fields. "
            "Return ONLY a JSON object with a single key \"citations\" whose value is an array. "
            "No markdown, no commentary.\n\n"
            "Each array item must be an object with keys: "
            "authors (array of strings), title (string or null), journal (string or null), "
            "year (integer or null), doi (string or null), arxiv_id (string or null), "
            "url (string or null), raw_text (string, required — verbatim line or entry)."
        )
        user = f"Bibliography section:\n\n{text[:MAX_PAPER_CHARS]}"
        data = self._chat_json(
            operation="extract_citations",
            system_prompt=system,
            user_content=user,
        )
        citations = data.get("citations")
        if not isinstance(citations, list):
            return []
        return [c for c in citations if isinstance(c, dict)]

    def extract_claims(self, paper_text: str, sections: dict) -> list[dict]:
        """
        Extract up to 50 cited empirical / statistical / methodological claims.

        Each item: text, paper_section, supporting_citation_indices, claim_type.
        Indices are 0-based positions in the paper's reference list (bibliography order).
        """
        body = (paper_text or "").strip()
        if not body:
            return []

        sections_json = json.dumps(sections or {}, ensure_ascii=False, indent=2)
        if len(sections_json) > 40_000:
            sections_json = sections_json[:40_000] + "\n…(truncated)"

        system = (
            "You extract empirical claims from academic papers for verification.\n"
            "Return ONLY a JSON object with key \"claims\" (array). No markdown.\n\n"
            "Focus on claims that are clearly supported by explicit citations in the text "
            "(e.g. [12], [3,4], (Smith et al., 2020), superscript numbers). "
            "Skip broad background with no citation. "
            "At most 50 claims; prefer the most important.\n\n"
            "Each claim object:\n"
            "- text (string, required)\n"
            "- paper_section (string): best matching key from the provided sections dict, "
            "or \"unknown\"\n"
            "- supporting_citation_indices (array of integers): 0-based indices into the "
            "numbered bibliography order (first reference = 0)\n"
            "- claim_type: one of \"empirical\", \"statistical\", \"methodological\""
        )
        user = (
            f"Section names and bodies (JSON):\n{sections_json}\n\n"
            f"Full paper text:\n\n{body[:MAX_PAPER_CHARS]}"
        )
        data = self._chat_json(
            operation="extract_claims",
            system_prompt=system,
            user_content=user,
        )
        claims = data.get("claims")
        if not isinstance(claims, list):
            return []
        out = [c for c in claims if isinstance(c, dict)]
        return out[:50]

    def extract_statistical_assertions(self, paper_text: str) -> list[dict]:
        """
        Extract sentences reporting statistical results.

        Each item: text, p_value, effect_size, sample_size, confidence_interval, section.
        """
        body = (paper_text or "").strip()
        if not body:
            return []

        system = (
            "You extract statistical reporting sentences from academic papers.\n"
            "Return ONLY a JSON object with key \"assertions\" (array). No markdown.\n\n"
            "Look for: p-values (p=, p<, p>), effect sizes (d=, r=, β=, Cohen's d, etc.), "
            "confidence intervals (brackets or parentheses with % or CI), "
            "sample sizes (n=, N=).\n\n"
            "Each assertion object:\n"
            "- text (string, required): the sentence or clause\n"
            "- p_value (number or null)\n"
            "- effect_size (number or null)\n"
            "- sample_size (integer or null)\n"
            "- confidence_interval: null or [low, high] numbers\n"
            "- section (string): coarse location or \"unknown\""
        )
        user = f"Paper text:\n\n{body[:MAX_PAPER_CHARS]}"
        data = self._chat_json(
            operation="extract_statistical_assertions",
            system_prompt=system,
            user_content=user,
        )
        assertions = data.get("assertions")
        if not isinstance(assertions, list):
            return []
        normalized: list[dict] = []
        for item in assertions:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            if "section" not in row and isinstance(row.get("paper_section"), str):
                row["section"] = row["paper_section"]
            normalized.append(row)
        return normalized


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    SAMPLE_REFERENCES = """\
[1] Smith, J., & Doe, A. (2023). Learning with attention. Journal of Machine Learning, 15(2), 100–120.
[2] Chen, L. et al. (2022). Efficient transformers. In Proceedings of ICML (pp. 1–12). PMLR.
[3] arXiv:2201.00001 — Brown, M. Open problems in optimization. 2021.
"""
    try:
        agent = HermesAgent()
        result = agent.extract_citations(SAMPLE_REFERENCES)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except HermesExtractionError as e:
        print("HermesExtractionError:", e)
    except Exception as e:
        print("Error:", e)
