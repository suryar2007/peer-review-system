"""Lava MCP / API wrappers for academic retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from config import get_settings
from pipeline.state import Citation


@dataclass
class LavaRetrievalClient:
    api_key: str

    @classmethod
    def from_config(cls) -> LavaRetrievalClient:
        s = get_settings()
        return cls(api_key=s.lava_api_key)

    def resolve_citations(self, citations: list[Citation]) -> list[Citation]:
        """
        Resolve citations against academic databases; returns citations with retrieval fields set.

        Replace the HTTP call with your Lava MCP bridge or REST surface when available.
        """
        payload = {"citations": [c.model_dump(mode="json") for c in citations]}
        base = "https://api.lava.example"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(f"{base}/resolve", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            resolved = data.get("resolved")
            if isinstance(resolved, list):
                return _citations_from_payload(resolved, citations)
        except Exception:
            pass
        return [
            c.model_copy(
                update={
                    "resolved": False,
                    "source_text": None,
                    "exists": None,
                }
            )
            for c in citations
        ]


def _citations_from_payload(
    resolved: list[Any],
    originals: list[Citation],
) -> list[Citation]:
    out: list[Citation] = []
    for i, item in enumerate(resolved):
        base = originals[i] if i < len(originals) else None
        if isinstance(item, Citation):
            out.append(item)
            continue
        if isinstance(item, dict) and base is not None:
            merged = base.model_dump() | {k: v for k, v in item.items() if v is not None}
            try:
                out.append(Citation.model_validate(merged))
            except Exception:
                out.append(base.model_copy(update={"resolved": False}))
        elif base is not None:
            out.append(base.model_copy(update={"resolved": False}))
    if len(out) < len(originals):
        for j in range(len(out), len(originals)):
            out.append(originals[j].model_copy(update={"resolved": False}))
    return out
