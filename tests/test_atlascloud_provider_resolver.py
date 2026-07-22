from __future__ import annotations

from types import SimpleNamespace

import pytest

from main import _create_llm
from src.core.provider_resolver import (
    ATLASCLOUD_API_BASE,
    resolve_litellm_model_provider,
)


def _settings(**overrides):
    values = {
        "llm_model": "atlascloud/qwen/qwen3.5-flash",
        "llm_api_key": "",
        "llm_api_base": "",
        "llm_temperature": 0.7,
        "llm_max_tokens": 0,
        "llm_top_p": 0.8,
        "atlascloud_api_key": "atlas-key",
        "atlascloud_api_base": ATLASCLOUD_API_BASE,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_atlascloud_shortcut_maps_to_openai_compatible_litellm_args() -> None:
    llm = _create_llm(_settings())

    assert llm.model == "openai/qwen/qwen3.5-flash"
    assert llm._api_key == "atlas-key"
    assert llm._api_base == ATLASCLOUD_API_BASE


def test_explicit_llm_config_overrides_atlascloud_defaults() -> None:
    llm = _create_llm(
        _settings(
            llm_model="atlascloud/deepseek-ai/deepseek-v4-pro",
            llm_api_key="proxy-key",
            llm_api_base="https://proxy.example/v1",
        )
    )

    assert llm.model == "openai/deepseek-ai/deepseek-v4-pro"
    assert llm._api_key == "proxy-key"
    assert llm._api_base == "https://proxy.example/v1"


def test_non_atlascloud_model_is_left_unchanged() -> None:
    model, api_key, api_base = resolve_litellm_model_provider(
        "openai/gpt-4o-mini",
        api_key="openai-key",
        api_base="https://openai-compatible.example/v1",
        atlascloud_api_key="atlas-key",
    )

    assert model == "openai/gpt-4o-mini"
    assert api_key == "openai-key"
    assert api_base == "https://openai-compatible.example/v1"


def test_atlascloud_shortcut_requires_a_model_id() -> None:
    with pytest.raises(ValueError, match="model id"):
        resolve_litellm_model_provider("atlascloud/")
