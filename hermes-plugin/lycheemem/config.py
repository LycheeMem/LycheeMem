"""Configuration for the LycheeMem Hermes plugin."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _normalize_transport(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"http", "mcp"}:
        return value
    return "http"


def _normalize_mode(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"raw", "compact", "full"}:
        return value
    return "compact"


def _normalize_response_level(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"minimal", "compact", "full"}:
        return value
    return "minimal"


@dataclass(frozen=True)
class PluginConfig:
    base_url: str = "http://127.0.0.1:8000"
    transport: str = "http"
    api_token: str = ""
    mcp_url_override: str = ""
    timeout: float = 120.0
    session_prefix: str = "hermes"
    enable_pre_llm_context: bool = True
    enable_auto_append: bool = True
    enable_auto_consolidate: bool = True
    enable_finalize_consolidation: bool = True
    auto_consolidate_background: bool = True
    auto_consolidate_cooldown_seconds: int = 60
    smart_search_top_k: int = 5
    smart_search_mode: str = "compact"
    response_level: str = "minimal"
    include_graph: bool = True
    include_skills: bool = True
    synthesize: bool = True
    command_consolidate_background: bool = False

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/health"

    @property
    def mcp_url(self) -> str:
        if self.mcp_url_override:
            return self.mcp_url_override
        return f"{self.base_url.rstrip('/')}/mcp"


def load_config() -> PluginConfig:
    return PluginConfig(
        base_url=_env_str("LYCHEEMEM_BASE_URL", "http://127.0.0.1:8000"),
        transport=_normalize_transport(_env_str("LYCHEEMEM_TRANSPORT", "http")),
        api_token=os.getenv("LYCHEEMEM_API_TOKEN", "").strip()
        or os.getenv("LYCHEEMEM_API_KEY", "").strip(),
        mcp_url_override=os.getenv("LYCHEEMEM_MCP_URL", "").strip(),
        timeout=_env_float("LYCHEEMEM_TIMEOUT", 120.0),
        session_prefix=os.getenv("LYCHEEMEM_SESSION_PREFIX", "hermes").strip(),
        enable_pre_llm_context=_env_bool("LYCHEEMEM_ENABLE_PRE_LLM_CONTEXT", True),
        enable_auto_append=_env_bool("LYCHEEMEM_ENABLE_AUTO_APPEND", True),
        enable_auto_consolidate=_env_bool("LYCHEEMEM_ENABLE_AUTO_CONSOLIDATE", True),
        enable_finalize_consolidation=_env_bool(
            "LYCHEEMEM_ENABLE_FINALIZE_CONSOLIDATION",
            True,
        ),
        auto_consolidate_background=_env_bool(
            "LYCHEEMEM_AUTO_CONSOLIDATE_BACKGROUND",
            True,
        ),
        auto_consolidate_cooldown_seconds=max(
            0,
            _env_int("LYCHEEMEM_AUTO_CONSOLIDATE_COOLDOWN_SECONDS", 60),
        ),
        smart_search_top_k=max(1, min(50, _env_int("LYCHEEMEM_SMART_SEARCH_TOP_K", 5))),
        smart_search_mode=_normalize_mode(
            _env_str("LYCHEEMEM_SMART_SEARCH_MODE", "compact")
        ),
        response_level=_normalize_response_level(
            _env_str("LYCHEEMEM_RESPONSE_LEVEL", "minimal")
        ),
        include_graph=_env_bool("LYCHEEMEM_INCLUDE_GRAPH", True),
        include_skills=_env_bool("LYCHEEMEM_INCLUDE_SKILLS", True),
        synthesize=_env_bool("LYCHEEMEM_SYNTHESIZE", True),
        command_consolidate_background=_env_bool(
            "LYCHEEMEM_COMMAND_CONSOLIDATE_BACKGROUND",
            False,
        ),
    )
