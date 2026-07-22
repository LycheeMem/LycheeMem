"""Provider-specific helpers for LiteLLM-backed model configuration."""

from __future__ import annotations

ATLASCLOUD_API_BASE = "https://api.atlascloud.ai/v1"
ATLASCLOUD_MODEL_PREFIX = "atlascloud/"


def resolve_litellm_model_provider(
    model: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    atlascloud_api_key: str | None = None,
    atlascloud_api_base: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Resolve project-level provider shortcuts into LiteLLM arguments."""

    normalized_model = (model or "").strip()
    resolved_api_key = api_key or None
    resolved_api_base = api_base or None

    if not normalized_model.lower().startswith(ATLASCLOUD_MODEL_PREFIX):
        return normalized_model, resolved_api_key, resolved_api_base

    atlas_model = normalized_model[len(ATLASCLOUD_MODEL_PREFIX) :].strip()
    if not atlas_model:
        raise ValueError("atlascloud/ model names must include a model id")

    return (
        f"openai/{atlas_model}",
        resolved_api_key or atlascloud_api_key or None,
        resolved_api_base or atlascloud_api_base or ATLASCLOUD_API_BASE,
    )
