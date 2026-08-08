"""Adapter base classes, provenance, cache, and safe probing."""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..contract import Capability, ContractError, SourceDescriptor


class MarketAdapter(ABC):
    market: str
    credential: str | None = None
    redistribution: str = "metadata_and_derived_metrics_only"

    @abstractmethod
    def descriptors(self) -> list[SourceDescriptor]:
        raise NotImplementedError

    @abstractmethod
    def probe_url(self) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def credential_available(self) -> bool:
        return self.credential is None or bool(os.environ.get(self.credential))

    def capability(self) -> Capability:
        if not self.credential_available():
            return Capability(
                market=self.market,
                status="BLOCKED",
                capability="historical_primary_tuple",
                reason=f"required credential {self.credential} is not set",
                credential=self.credential,
                details={"fail_closed": True},
            )
        return Capability(
            market=self.market,
            status="READY_TO_PROBE",
            capability="historical_primary_tuple",
            reason="required authentication is available",
            credential=self.credential,
        )

    def probe(self, timeout: int = 30) -> Capability:
        blocked = self.capability()
        if blocked.status == "BLOCKED":
            return blocked
        url, kwargs = self.probe_url()
        headers = {
            "User-Agent": "dc-energy-joint-optimization/energy-model-v3",
            **kwargs.pop("headers", {}),
        }
        started = time.monotonic()
        try:
            response = requests.get(
                url, headers=headers, timeout=timeout, stream=True, **kwargs
            )
            status = response.status_code
            content_type = response.headers.get("Content-Type")
            content_length = response.headers.get("Content-Length")
            response.close()
        except requests.RequestException as exc:
            return Capability(
                market=self.market,
                status="BLOCKED",
                capability="network_probe",
                reason=f"{type(exc).__name__}: {exc}",
                credential=self.credential,
                details={"url": url, "fail_closed": True},
            )
        return Capability(
            market=self.market,
            status="AVAILABLE" if status < 400 else "BLOCKED",
            capability="network_probe",
            reason=f"official endpoint returned HTTP {status}",
            credential=self.credential,
            details={
                "url": url,
                "http_status": status,
                "content_type": content_type,
                "content_length": content_length,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "probe_only": True,
            },
        )

    def write_descriptors(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = [descriptor.to_dict() for descriptor in self.descriptors()]
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def download_raw(
    url: str,
    destination: Path,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    retries: int = 6,
) -> dict[str, Any]:
    """Download immutable raw evidence with bounded 429 backoff."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    delay = 2.0
    for attempt in range(retries):
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        if response.status_code == 429:
            response.close()
            if attempt + 1 < retries:
                time.sleep(delay)
                delay = min(delay * 2.0, 60.0)
                continue
            raise ContractError(f"rate limit persisted for {url}")
        response.raise_for_status()
        content = response.content
        if not content:
            raise ContractError(f"empty response from {response.url}")
        destination.write_bytes(content)
        return {
            "url": response.url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "http_headers": {
                key: response.headers.get(key)
                for key in ("Content-Type", "ETag", "Last-Modified")
            },
        }
    raise ContractError(f"download retries exhausted for {url}")
