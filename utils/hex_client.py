"""Hex API client for triggering dashboard / notebook runs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from config import get_settings

logger = logging.getLogger(__name__)

_HEX_BASE_URL = "https://app.hex.tech"


class HexError(RuntimeError):
    """Raised when a Hex API request fails."""


@dataclass
class HexClient:
    api_key: str | None
    project_id: str | None

    @classmethod
    def from_config(cls) -> HexClient:
        s = get_settings()
        return cls(api_key=s.hex_api_key, project_id=s.hex_project_id)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    def _is_configured(self) -> bool:
        return bool(self.api_key and self.project_id)

    def trigger_run(self, input_params: dict[str, Any]) -> str | None:
        """
        Trigger a Hex project run with the given input parameters.

        Returns the run_id from the response.
        Returns None if Hex is not configured (mock mode).
        Raises HexError if the request fails.
        """
        if not self._is_configured():
            logger.info("Hex not configured (HEX_API_KEY or HEX_PROJECT_ID missing); returning mock run_id")
            return None

        url = f"{_HEX_BASE_URL}/api/v1/project/{self.project_id}/run"
        body = {
            "inputParams": {
                "paper_data": json.dumps(input_params),
            },
            "dryRun": False,
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=self._headers(), json=body)
            if response.status_code >= 400:
                raise HexError(f"Hex trigger failed: {response.status_code} {response.text}")
            data = response.json()
        run_id = data.get("runId") or data.get("id")
        if not run_id:
            raise HexError(f"Hex trigger returned no run ID: {data}")
        return run_id

    def poll_run(self, run_id: str, timeout_seconds: int = 120) -> dict:
        """
        Poll a run until it completes or times out.

        GET /api/v1/project/{project_id}/run/{run_id} every 5 seconds.
        Returns the final run status dict when status is COMPLETE or ERRORED.
        Raises TimeoutError if timeout_seconds is exceeded.
        """
        if not self._is_configured():
            return {"status": "COMPLETE"}

        url = f"{_HEX_BASE_URL}/api/v1/project/{self.project_id}/run/{run_id}"
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout_seconds:
                raise TimeoutError(f"Hex run {run_id} did not complete within {timeout_seconds}s")

            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, headers=self._headers())
                response.raise_for_status()
                data = response.json()

            status = data.get("status", "").upper()
            if status in ("COMPLETE", "COMPLETED"):
                return data
            if status == "ERRORED":
                raise HexError(f"Hex run {run_id} errored: {data}")

            time.sleep(5)

    def get_dashboard_url(self, run_id: str) -> str:
        """
        Return the public URL for viewing the completed run's output.
        """
        return f"{_HEX_BASE_URL}/app/{self.project_id}?run={run_id}"

    def trigger_and_wait(self, input_params: dict[str, Any], timeout_seconds: int = 120) -> str | None:
        """
        Convenience: trigger a run, wait for completion, return dashboard URL.

        Returns None if Hex is not configured (mock mode).
        """
        if not self._is_configured():
            logger.info("Hex not configured; skipping dashboard trigger")
            return None

        run_id = self.trigger_run(input_params)
        if not run_id:
            return None

        self.poll_run(run_id, timeout_seconds=timeout_seconds)
        return self.get_dashboard_url(run_id)

    def trigger_dashboard_update(self, payload: dict[str, Any]) -> str | None:
        """Backward-compatible alias for trigger_run."""
        return self.trigger_run(payload)
