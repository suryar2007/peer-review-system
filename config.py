"""Application configuration loaded from the environment."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)


class ConfigurationError(ValueError):
    """Raised when required configuration is missing or invalid."""


def _truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _strip_or_none(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = raw.strip()
    return s if s else None


def _require(name: str, raw: str | None) -> str:
    value = _strip_or_none(raw)
    if value is None:
        raise ConfigurationError(
            f"Required environment variable {name} is missing or empty. "
            f"Copy .env.example to .env, set {name}, and try again."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Typed configuration for the peer-review pipeline."""

    nous_api_key: str
    nous_base_url: str
    hf_token: str | None
    k2_model_id: str
    k2_base_url: str | None
    lava_api_key: str
    langchain_tracing_v2: bool
    langchain_api_key: str | None
    langchain_project: str
    hex_api_key: str | None
    hex_project_id: str | None
    e2b_api_key: str | None
    semantic_scholar_api_key: str | None


def _load_settings() -> Settings:
    """
    Load and validate settings from the environment.

    Required:
        NOUS_API_KEY — Hermes / Nous inference API.
        LAVA_API_KEY — Lava MCP retrieval.

    Conditionally required:
        LANGCHAIN_API_KEY — required when LANGCHAIN_TRACING_V2 is enabled.
    """
    nous_api_key = _require("NOUS_API_KEY", os.getenv("NOUS_API_KEY"))
    lava_api_key = _require("LAVA_API_KEY", os.getenv("LAVA_API_KEY"))

    tracing = _truthy(os.getenv("LANGCHAIN_TRACING_V2"))
    langchain_api_key = _strip_or_none(os.getenv("LANGCHAIN_API_KEY"))
    if tracing and not langchain_api_key:
        raise ConfigurationError(
            "LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is missing or empty. "
            "Set LANGCHAIN_API_KEY in .env, or set LANGCHAIN_TRACING_V2=false."
        )

    nous_base_url = _strip_or_none(os.getenv("NOUS_BASE_URL"))
    if not nous_base_url:
        nous_base_url = "https://inference-api.nousresearch.com/v1"

    k2_model_id = _strip_or_none(os.getenv("K2_MODEL_ID"))
    if not k2_model_id:
        k2_model_id = "LLM360/K2-Think-V2"

    langchain_project = _strip_or_none(os.getenv("LANGCHAIN_PROJECT"))
    if not langchain_project:
        langchain_project = "peer-review-pipeline"

    return Settings(
        nous_api_key=nous_api_key,
        nous_base_url=nous_base_url.rstrip("/"),
        hf_token=_strip_or_none(os.getenv("HF_TOKEN")),
        k2_model_id=k2_model_id,
        k2_base_url=_strip_or_none(os.getenv("K2_BASE_URL")),
        lava_api_key=lava_api_key,
        langchain_tracing_v2=tracing,
        langchain_api_key=langchain_api_key,
        langchain_project=langchain_project,
        hex_api_key=_strip_or_none(os.getenv("HEX_API_KEY")),
        hex_project_id=_strip_or_none(os.getenv("HEX_PROJECT_ID")),
        e2b_api_key=_strip_or_none(os.getenv("E2B_API_KEY")),
        semantic_scholar_api_key=_strip_or_none(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return validated settings (cached). Call ``get_settings.cache_clear()`` in tests to reload."""
    return _load_settings()


load_settings = get_settings
