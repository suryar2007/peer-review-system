"""Lava gateway client for routing API calls through the Lava proxy.

Lava (lava.so) is a usage-based billing and AI proxy service. Its forward
endpoint proxies requests to 100+ supported providers (AI models and data APIs)
with automatic usage tracking and cost metering.

Supported providers include Semantic Scholar, arXiv, OpenAI, Anthropic,
Together.ai, DeepInfra, Nebius, and many more.

Docs: https://lava.so/docs/gateway/forward-proxy
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlencode

import httpx

logger = logging.getLogger(__name__)

_LAVA_API = "https://api.lava.so/v1"


class LavaEndpointNotSupported(Exception):
    """Raised when Lava's forward endpoint rejects the target URL."""


@dataclass
class LavaGateway:
    """Thin wrapper around Lava's ``/v1/forward`` endpoint.

    Authentication modes (determined by which fields are set):
    - secret_key only → merchant pays, no customer billing
    - + customer_id + meter_slug → customer billing
    - + provider_key → BYOK (bring your own key to the upstream provider)
    """

    secret_key: str
    customer_id: str | None = None
    meter_slug: str | None = None
    provider_key: str | None = None
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    @property
    def is_configured(self) -> bool:
        return bool(self.secret_key and self.secret_key != "not-set")

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=120.0)
        return self._client

    def close(self) -> None:
        if self._client and not self._client.is_closed:
            self._client.close()

    def __del__(self) -> None:
        self.close()

    def _auth_header(self) -> str:
        """Build the Authorization header value.

        Simple secret key when no customer/meter/provider fields are set,
        otherwise a base64-encoded forward token.
        """
        needs_token = self.customer_id or self.meter_slug or self.provider_key
        if not needs_token:
            return f"Bearer {self.secret_key}"

        token_data: dict[str, Any] = {"secret_key": self.secret_key}
        if self.customer_id:
            token_data["customer_id"] = self.customer_id
        if self.meter_slug:
            token_data["meter_slug"] = self.meter_slug
        if self.provider_key:
            token_data["provider_key"] = self.provider_key
        encoded = base64.b64encode(json.dumps(token_data).encode()).decode()
        return f"Bearer {encoded}"

    @staticmethod
    def _forward_url(provider_url: str) -> str:
        return f"{_LAVA_API}/forward?u={quote(provider_url, safe='')}"

    @staticmethod
    def _check_endpoint_support(resp: httpx.Response) -> None:
        """Raise LavaEndpointNotSupported if the provider/endpoint is rejected."""
        if resp.status_code != 400:
            return
        try:
            data = resp.json()
        except Exception:
            return
        err = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(err, str):
            if "not supported" in err.lower():
                raise LavaEndpointNotSupported(err)
            return
        if not isinstance(err, dict):
            return
        code = err.get("code", "")
        if code in (
            "forward_endpoint_not_supported",
            "forward_model_not_supported",
            "forward_model_invalid",
        ):
            raise LavaEndpointNotSupported(err.get("message", code))

    def forward_post(
        self,
        provider_url: str,
        *,
        json_body: dict | list | None = None,
        extra_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Forward a POST request through Lava's gateway."""
        client = self._get_client()
        headers: dict[str, str] = {
            "Authorization": self._auth_header(),
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        if metadata:
            headers["x-lava-metadata"] = json.dumps(metadata)

        url = self._forward_url(provider_url)
        logger.debug("Lava forward POST → %s", provider_url)
        resp = client.post(url, json=json_body, headers=headers, timeout=timeout)
        self._check_endpoint_support(resp)
        return resp

    def forward_get(
        self,
        provider_url: str,
        *,
        params: dict[str, str] | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Forward a GET request through Lava's gateway."""
        target = provider_url
        if params:
            target = f"{provider_url}?{urlencode(params)}"

        client = self._get_client()
        headers: dict[str, str] = {"Authorization": self._auth_header()}
        if extra_headers:
            headers.update(extra_headers)

        url = self._forward_url(target)
        logger.debug("Lava forward GET → %s", target)
        resp = client.get(url, headers=headers, timeout=timeout)
        self._check_endpoint_support(resp)
        return resp
