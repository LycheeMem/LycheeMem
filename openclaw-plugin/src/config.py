"""Configuration helpers for the thin OpenClaw adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class PluginConfig:
    base_url: str = "http://127.0.0.1:8000"
    transport: str = "mcp"
    timeout: float = 15.0
    api_token: str = ""

    @property
    def mcp_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/mcp"

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/health"


def load_config() -> PluginConfig:
    return PluginConfig(
        base_url=os.getenv("LYCHEEMEM_BASE_URL", "http://127.0.0.1:8000"),
        transport=os.getenv("LYCHEEMEM_TRANSPORT", "mcp").strip().lower() or "mcp",
        timeout=float(os.getenv("LYCHEEMEM_TIMEOUT", "15.0")),
        api_token=os.getenv("LYCHEEMEM_API_TOKEN", ""),
    )
