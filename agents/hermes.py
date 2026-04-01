"""Nous Research Hermes agent using the OpenAI-compatible Python client.

When a Lava API key is configured, LLM requests are routed through Lava's
forward proxy (BYOK mode) for usage tracking and cost metering. If the
Nous endpoint is not in Lava's supported provider list, falls back to
direct API calls automatically.

This module also provides claim verification and statistical auditing via
the same Hermes model (previously handled by a separate K2 client).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from config import get_settings
from pipeline.state import StatisticalAssertion, StatisticalAuditResult, VerificationResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nousresearch/hermes-4-70b"
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


def _fix_json_escapes(text: str) -> str:
    """Fix invalid escape sequences that LLMs produce (e.g. \\_ or \\&)."""
    import re
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def _try_parse_json(text: str) -> Any:
    """Try parsing JSON, with fallback for invalid escapes."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_fix_json_escapes(text))


def _salvage_truncated_json(text: str) -> dict[str, Any] | None:
    """Attempt to recover a partial JSON object/array from truncated output."""
    last_brace = text.rfind("}")
    if last_brace == -1:
        return None

    # Walk backwards from each } trying to close open brackets
    for pos in range(last_brace + 1, max(last_brace - 20, 0), -1):
        candidate = text[:pos].rstrip().rstrip(",")
        # Count open brackets to figure out what's needed to close
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        suffix = "]" * max(open_brackets, 0) + "}" * max(open_braces, 0)
        if not suffix:
            suffix_options = ["", "}", "]}", "]}"]
        else:
            suffix_options = [suffix]
        for sfx in suffix_options:
            try:
                result = _try_parse_json(candidate + sfx)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _parse_json_object(content: str) -> tuple[dict[str, Any], bool]:
    """Parse JSON from model output.  Returns ``(data, was_salvaged)``."""
    text = _strip_json_fence(content)
    try:
        out = _try_parse_json(text)
        if isinstance(out, dict):
            return out, False
    except json.JSONDecodeError:
        pass

    salvaged = _salvage_truncated_json(text)
    if isinstance(salvaged, dict):
        logger.warning("Salvaged truncated JSON from model output")
        return salvaged, True

    raise HermesExtractionError("Invalid JSON from model — could not parse or salvage")


def _split_references_into_chunks(text: str, max_refs_per_chunk: int = 30) -> list[str]:
    """
    Split a raw references section into chunks of roughly max_refs_per_chunk references.

    Detects reference boundaries by looking for common patterns:
    - [1], [2], ... numbered brackets at line start
    - Blank-line-separated blocks
    - Author-year format (line starting with capitalized name + year pattern)
    """
    import re

    lines = text.split("\n")

    # --- Strategy 1: numbered references [1], [2], etc. ---
    ref_starts: list[int] = []
    for i, line in enumerate(lines):
        if re.match(r"^\s*\[\d+\]", line):
            ref_starts.append(i)

    if len(ref_starts) >= 5:
        return _chunk_by_line_indices(lines, ref_starts, max_refs_per_chunk)

    # --- Strategy 2: blank-line-separated blocks ---
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    if len(blocks) >= 5:
        chunks = []
        for i in range(0, len(blocks), max_refs_per_chunk):
            chunk = "\n\n".join(blocks[i:i + max_refs_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    # --- Strategy 3: author-year format (e.g. "Firstname Lastname, Firstname ...") ---
    # A new reference starts when the previous line ends a reference and the
    # current line starts with an author name.
    ref_starts = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if i > 0:
            prev = ""
            for j in range(i - 1, -1, -1):
                p = lines[j].strip()
                if p:
                    prev = p
                    break
            prev_ends_ref = (
                prev.endswith(".")
                or prev.endswith("}")
                or prev.endswith(")")
                or re.search(r"arXiv:\d+\.\d+", prev) is not None
                or re.search(r"\d{4}", prev) is not None
                or re.search(r"\d+[–-]\d+\.?\s*$", prev) is not None
            )
            if not prev_ends_ref and i > 0:
                continue

        # Author name at start of line:
        #   "Firstname Lastname" | "J. Deng" | "Z. Chen" | "AI@Meta"
        if re.match(r"^[A-Z][a-zà-ü]+\s+[A-Z]", stripped):
            ref_starts.append(i)
        elif re.match(r"^[A-Z]\.\s*[A-Z]", stripped):
            ref_starts.append(i)
        elif re.match(r"^[A-Z][A-Z]@", stripped):
            ref_starts.append(i)

    if len(ref_starts) >= 10:
        return _chunk_by_line_indices(lines, ref_starts, max_refs_per_chunk)

    # --- Strategy 4: split by character count ---
    max_chunk_chars = 6000
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_chars, len(text))
        # Try to break at a newline
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks if chunks else [text]


def _chunk_by_line_indices(lines: list[str], ref_starts: list[int], max_refs: int) -> list[str]:
    """Given line indices where references start, group into chunks of max_refs."""
    chunks: list[str] = []
    for i in range(0, len(ref_starts), max_refs):
        line_start = ref_starts[i]
        next_group = i + max_refs
        line_end = ref_starts[next_group] if next_group < len(ref_starts) else len(lines)
        chunk = "\n".join(lines[line_start:line_end]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else ["\n".join(lines)]


class HermesAgent:
    """OpenAI-compatible client for Nous Hermes chat completions.

    When ``LAVA_API_KEY`` is set, requests are proxied through Lava's forward
    endpoint in BYOK mode (the Nous API key is sent as ``provider_key``).
    If Lava rejects the endpoint, subsequent calls fall back to direct.
    """

    def __init__(self) -> None:
        settings = get_settings()
        base = settings.nous_base_url.rstrip("/")
        self._client = OpenAI(base_url=base, api_key=settings.nous_api_key)
        self._model = _env_model()
        self._last_was_salvaged = False

        # Lava gateway integration (BYOK mode)
        self._lava_gw = None
        self._lava_llm_ok = True
        lava_key = settings.lava_api_key
        if lava_key and lava_key != "not-set":
            from agents.lava_gateway import LavaGateway
            self._lava_gw = LavaGateway(
                secret_key=lava_key,
                customer_id=settings.lava_customer_id,
                meter_slug=settings.lava_meter_slug,
                provider_key=settings.nous_api_key,
            )
            logger.info("Lava gateway enabled for Hermes LLM calls (BYOK mode)")
        self._completions_url = f"{base}/chat/completions"

    def _log_usage(self, operation: str, usage: Any) -> None:
        if usage is None:
            logger.info("Hermes %s: no usage metadata returned", operation)
            return
        if isinstance(usage, dict):
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            tt = usage.get("total_tokens")
        else:
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

    def _complete_via_lava(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Call chat completions through Lava's forward proxy. Returns {content, usage}."""
        from agents.lava_gateway import LavaEndpointNotSupported

        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self._lava_gw.forward_post(self._completions_url, json_body=body)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {"content": content, "usage": data.get("usage")}

    def _chat_json(
        self,
        *,
        operation: str,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        from agents.lava_gateway import LavaEndpointNotSupported

        last_exc: Exception | None = None
        backoffs = (0.0, 2.0, 5.0, 12.0)
        use_lava = self._lava_gw is not None and self._lava_llm_ok

        for attempt, sleep_s in enumerate(backoffs):
            if sleep_s:
                time.sleep(sleep_s)
            try:
                if use_lava:
                    result = self._complete_via_lava(system_prompt, user_content, max_tokens=max_tokens)
                    self._log_usage(operation, result.get("usage"))
                    content = result["content"]
                else:
                    completion = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1,
                        max_tokens=max_tokens,
                    )
                    self._log_usage(operation, completion.usage)
                    content = completion.choices[0].message.content

                if not isinstance(content, str) or not content.strip():
                    raise HermesExtractionError("Empty model content")
                data, salvaged = _parse_json_object(content)
                self._last_was_salvaged = salvaged
                return data
            except LavaEndpointNotSupported:
                logger.warning(
                    "Nous endpoint not supported by Lava gateway, "
                    "falling back to direct API for all subsequent calls"
                )
                self._lava_llm_ok = False
                use_lava = False
                continue
            except (httpx.HTTPStatusError,) as exc:
                status = exc.response.status_code
                if status == 429 and attempt < len(backoffs) - 1:
                    last_exc = exc
                    logger.warning("Hermes HTTP 429 via Lava (attempt %s/%s)", attempt + 1, len(backoffs))
                    continue
                if status in (502, 503, 504) and attempt < len(backoffs) - 1:
                    last_exc = exc
                    logger.warning("Hermes HTTP %d via Lava (attempt %s/%s)", status, attempt + 1, len(backoffs))
                    continue
                raise HermesExtractionError(
                    f"Hermes API error {status}: {exc}", status_code=status,
                ) from exc
            except RateLimitError as exc:
                last_exc = exc
                logger.warning("Hermes rate limit (attempt %s/%s): %s", attempt + 1, len(backoffs), exc)
                if attempt < len(backoffs) - 1:
                    continue
                raise HermesExtractionError(
                    f"Hermes rate limited after {len(backoffs)} attempts: {exc}",
                    status_code=getattr(exc, "status_code", None) or 429,
                ) from exc
            except APIStatusError as exc:
                if exc.status_code == 429 and attempt < len(backoffs) - 1:
                    last_exc = exc
                    logger.warning("Hermes HTTP 429 (attempt %s/%s): %s", attempt + 1, len(backoffs), exc)
                    continue
                if exc.status_code in (502, 503, 504) and attempt < len(backoffs) - 1:
                    last_exc = exc
                    logger.warning("Hermes HTTP %d (attempt %s/%s): %s", exc.status_code, attempt + 1, len(backoffs), exc)
                    continue
                raise HermesExtractionError(
                    f"Hermes API error {exc.status_code}: {exc}",
                    status_code=exc.status_code,
                ) from exc
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < len(backoffs) - 1:
                    logger.warning("Hermes connection error (attempt %s/%s): %s", attempt + 1, len(backoffs), exc)
                    continue
                raise HermesExtractionError(f"Hermes connection error: {exc}") from exc

        raise HermesExtractionError(f"Hermes request failed: {last_exc}") from last_exc

    def extract_citations(self, references_raw: str) -> list[dict]:
        """
        Parse the raw references / bibliography section into structured citation dicts.

        Splits large reference lists into chunks of ~30 references each to avoid
        output truncation, then merges the results.
        """
        text = (references_raw or "").strip()
        if not text:
            return []

        chunks = _split_references_into_chunks(text, max_refs_per_chunk=12)
        logger.info("Split references into %d chunk(s)", len(chunks))

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

        all_citations: list[dict] = []
        for i, chunk in enumerate(chunks):
            logger.info("Extracting citations from chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            user = f"Bibliography section (part {i + 1} of {len(chunks)}):\n\n{chunk[:MAX_PAPER_CHARS]}"
            data = self._chat_json(
                operation=f"extract_citations_chunk_{i + 1}",
                system_prompt=system,
                user_content=user,
                max_tokens=8192,
            )
            citations = data.get("citations")
            if isinstance(citations, list):
                all_citations.extend(c for c in citations if isinstance(c, dict))

        logger.info("Total citations extracted: %d", len(all_citations))
        return all_citations

    _CLASSIFY_SYSTEM_PROMPT = (
        "You are an expert claim classifier for academic peer review.\n"
        "You are given sentences from a paper that contain in-text citations.\n"
        "For EACH sentence, decide its claim type.\n\n"
        "Return ONLY a JSON object: {\"results\": [...]}. No markdown.\n\n"

        "## Claim types\n"
        "- \"empirical\": attributes a finding or result to the cited work\n"
        "  e.g. \"Smith et al. [2] showed that attention improves MT\"\n"
        "- \"statistical\": cites a specific number, metric, or quantitative result\n"
        "  e.g. \"ELMo improved F1 by 4.7% (Peters et al., 2018)\"\n"
        "- \"methodological\": attributes a method, technique, or approach\n"
        "  e.g. \"The Transformer [47] uses self-attention\"\n"
        "- \"not_a_claim\": bare reference with no factual assertion about the cited work\n"
        "  e.g. \"see [5] for details\", table headers, dataset attributions\n\n"

        "## Output schema\n"
        "The \"results\" array must have exactly one entry per input sentence, "
        "in the same order:\n"
        "- id (int): the sentence number from the input\n"
        "- claim_type (string): one of the four types above\n"
    )

    _CLASSIFY_BATCH_SIZE = 20

    def classify_citation_mentions(
        self,
        mentions: list[dict],
    ) -> list[dict]:
        """Classify pre-detected citation mentions into claim types.

        *mentions* is a list of dicts with ``sentence``, ``section``, and
        ``citation_indices``.  Returns the same dicts with an added
        ``claim_type`` field.  Mentions classified as ``not_a_claim`` are
        excluded from the result.
        """
        if not mentions:
            return []

        # Default everything to "empirical" (fallback if LLM fails)
        for m in mentions:
            m.setdefault("claim_type", "empirical")

        # Process in batches
        for batch_start in range(0, len(mentions), self._CLASSIFY_BATCH_SIZE):
            batch = mentions[batch_start : batch_start + self._CLASSIFY_BATCH_SIZE]
            numbered = "\n".join(
                f"{i + 1}. [{m['section']}] {m['sentence']}"
                for i, m in enumerate(batch)
            )
            user = (
                f"Classify each of the following {len(batch)} sentences.\n\n"
                f"{numbered}"
            )
            try:
                data = self._chat_json(
                    operation=f"classify_mentions_batch_{batch_start // self._CLASSIFY_BATCH_SIZE + 1}",
                    system_prompt=self._CLASSIFY_SYSTEM_PROMPT,
                    user_content=user,
                    max_tokens=2000,
                )
                results = data.get("results")
                if isinstance(results, list):
                    by_id = {int(r.get("id", 0)): r.get("claim_type", "empirical") for r in results if isinstance(r, dict)}
                    for i, m in enumerate(batch):
                        ct = by_id.get(i + 1, "empirical")
                        if ct in ("empirical", "statistical", "methodological", "not_a_claim"):
                            m["claim_type"] = ct
            except Exception as exc:
                logger.warning("Claim classification batch failed: %s", exc)

        out = [m for m in mentions if m.get("claim_type") != "not_a_claim"]
        logger.info(
            "Classification: %d/%d mentions kept (filtered %d not_a_claim)",
            len(out), len(mentions), len(mentions) - len(out),
        )
        return out

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


    _VERIFY_SYSTEM_PROMPT = (
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

    _AUDIT_SYSTEM_PROMPT = (
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

    _AUDIT_BATCH_SIZE = 20

    def verify_claim(
        self,
        claim_text: str,
        cited_sources: list[dict],
        paper_context: str = "",
    ) -> VerificationResult:
        """Verify whether cited sources support the claim. Returns a VerificationResult."""
        try:
            import json as _json
            sources_json = _json.dumps(cited_sources, ensure_ascii=False, indent=2)
            context_part = f"\n\nSurrounding context from the paper:\n{paper_context}" if paper_context else ""
            user = (
                f"Claim: {claim_text}{context_part}\n\n"
                f"Cited sources:\n{sources_json}"
            )
            data = self._chat_json(
                operation="verify_claim",
                system_prompt=self._VERIFY_SYSTEM_PROMPT,
                user_content=user,
                max_tokens=4096,
            )
            return VerificationResult(
                claim_text=claim_text,
                verdict=str(data.get("verdict", "unverifiable")),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
                explanation=str(data.get("explanation", "")),
                relevant_passage=data.get("relevant_passage"),
                citation_indices=[],
            )
        except Exception as exc:
            logger.warning("verify_claim failed: %s", exc)
            return VerificationResult(
                claim_text=claim_text,
                verdict="unverifiable",
                confidence=0.0,
                explanation=f"Reasoning model unavailable: {exc}",
                relevant_passage=None,
                citation_indices=[],
            )

    def audit_statistical_assertions(
        self,
        assertions: list[StatisticalAssertion],
    ) -> list[StatisticalAuditResult]:
        """Check internal consistency of reported statistics."""
        if not assertions:
            return []
        import json as _json
        raw_dicts = [a.model_dump(mode="json") for a in assertions]
        all_results: list[dict] = []
        for start in range(0, len(raw_dicts), self._AUDIT_BATCH_SIZE):
            batch = raw_dicts[start: start + self._AUDIT_BATCH_SIZE]
            try:
                user = _json.dumps(batch, ensure_ascii=False, indent=2)
                data = self._chat_json(
                    operation=f"audit_stats_batch_{start // self._AUDIT_BATCH_SIZE + 1}",
                    system_prompt=self._AUDIT_SYSTEM_PROMPT,
                    user_content=user,
                    max_tokens=4096,
                )
                results = data.get("results")
                if isinstance(results, list):
                    all_results.extend(results)
            except Exception as exc:
                logger.warning("audit_statistical_assertions batch %d failed: %s", start // self._AUDIT_BATCH_SIZE + 1, exc)
                all_results.extend(
                    {
                        "assertion_text": a.get("text", ""),
                        "is_internally_consistent": False,
                        "issues": [f"Audit failed: {exc}"],
                        "severity": "medium",
                    }
                    for a in batch
                )
        return _parse_statistical_audit_results_from_dicts(all_results, assertions)


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
