"""Hex API client for triggering dashboard / notebook runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from config import get_settings


@dataclass
class HexClient:
    api_key: str | None
    project_id: str | None

    @classmethod
    def from_config(cls) -> HexClient:
        s = get_settings()
        return cls(api_key=s.hex_api_key, project_id=s.hex_project_id)

    def trigger_dashboard_update(self, payload: dict[str, Any]) -> str | None:
        """Trigger a Hex run; returns run id when configured, else None."""
        if not self.api_key or not self.project_id:
            return None

        url = f"https://app.hex.tech/api/v1/projects/{self.project_id}/runs"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {"payload": payload}
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
        return data.get("runId") or data.get("id")
