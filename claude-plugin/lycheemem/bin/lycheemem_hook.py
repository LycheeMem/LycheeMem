#!/usr/bin/env python3
"""Claude Code hook adapter for LycheeMem.

The hook is intentionally stdlib-only so the plugin can run without installing
extra Python packages into the user's Claude Code environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


def _option(key: str, default: str = "") -> str:
    """Read Claude plugin option envs plus LycheeMem-compatible envs."""

    plugin_keys = [
        f"CLAUDE_PLUGIN_OPTION_{key}",
        f"CLAUDE_PLUGIN_OPTION_{key.upper()}",
    ]
    for env_key in plugin_keys:
        raw = os.getenv(env_key)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


def _option_bool(key: str, env_name: str, default: bool) -> bool:
    raw = _option(key)
    if raw:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return _env_bool(env_name, default)


def _option_int(key: str, env_name: str, default: int) -> int:
    raw = _option(key)
    if raw:
        try:
            return int(float(raw.strip()))
        except ValueError:
            return _env_int(env_name, default)
    return _env_int(env_name, default)


@dataclass(frozen=True)
class Config:
    base_url: str
    api_token: str
    timeout: float
    session_prefix: str
    enable_auto_recall: bool
    enable_auto_append: bool
    enable_auto_consolidate: bool
    auto_consolidate_background: bool
    auto_consolidate_cooldown_seconds: int
    smart_search_top_k: int
    smart_search_mode: str
    response_level: str
    include_graph: bool
    include_skills: bool
    synthesize: bool
    max_context_chars: int
    debug: bool


def load_config() -> Config:
    base_url = (
        os.getenv("LYCHEEMEM_BASE_URL", "").strip()
        or _option("base_url", "http://127.0.0.1:8000")
    )
    return Config(
        base_url=base_url.rstrip("/"),
        api_token=(
            os.getenv("LYCHEEMEM_API_TOKEN", "").strip()
            or os.getenv("LYCHEEMEM_API_KEY", "").strip()
        ),
        timeout=_env_float("LYCHEEMEM_TIMEOUT", 120.0),
        session_prefix=os.getenv("LYCHEEMEM_SESSION_PREFIX", "claude").strip()
        or "claude",
        enable_auto_recall=_option_bool(
            "enable_auto_recall",
            "LYCHEEMEM_ENABLE_PRE_LLM_CONTEXT",
            True,
        ),
        enable_auto_append=_option_bool(
            "enable_auto_append",
            "LYCHEEMEM_ENABLE_AUTO_APPEND",
            True,
        ),
        enable_auto_consolidate=_option_bool(
            "enable_auto_consolidate",
            "LYCHEEMEM_ENABLE_AUTO_CONSOLIDATE",
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
        smart_search_top_k=max(
            1,
            min(50, _option_int("smart_search_top_k", "LYCHEEMEM_SMART_SEARCH_TOP_K", 5)),
        ),
        smart_search_mode=_normalize_choice(
            os.getenv("LYCHEEMEM_SMART_SEARCH_MODE", "").strip()
            or _option("smart_search_mode", "compact"),
            {"raw", "compact", "full"},
            "compact",
        ),
        response_level=_normalize_choice(
            os.getenv("LYCHEEMEM_RESPONSE_LEVEL", "").strip()
            or _option("response_level", "minimal"),
            {"minimal", "compact", "full"},
            "minimal",
        ),
        include_graph=_env_bool("LYCHEEMEM_INCLUDE_GRAPH", True),
        include_skills=_env_bool("LYCHEEMEM_INCLUDE_SKILLS", True),
        synthesize=_env_bool("LYCHEEMEM_SYNTHESIZE", True),
        max_context_chars=max(500, _env_int("LYCHEEMEM_MAX_CONTEXT_CHARS", 6000)),
        debug=_env_bool("LYCHEEMEM_CLAUDE_DEBUG", False),
    )


def _normalize_choice(raw: str, allowed: set[str], default: str) -> str:
    value = raw.strip().lower()
    return value if value in allowed else default


class LycheeMemClient:
    def __init__(self, config: Config):
        self.config = config

    def smart_search(self, query: str, session_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "session_id": session_id,
            "top_k": self.config.smart_search_top_k,
            "include_graph": self.config.include_graph,
            "include_skills": self.config.include_skills,
            "synthesize": self.config.synthesize,
            "mode": self.config.smart_search_mode,
            "response_level": self.config.response_level,
        }
        return self._post("/memory/smart-search", payload)

    def append_turn(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        return self._post(
            "/memory/append-turn",
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "token_count": 0,
            },
        )

    def consolidate(self, session_id: str) -> dict[str, Any]:
        return self._post(
            "/memory/consolidate",
            {
                "session_id": session_id,
                "background": self.config.auto_consolidate_background,
            },
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout) as response:
            body = response.read().decode("utf-8")
        if not body:
            return {}
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else {"result": parsed}


def main(argv: list[str]) -> int:
    action = argv[1] if len(argv) > 1 else ""
    if action not in {"user-prompt-submit", "stop", "session-end"}:
        _debug(True, f"unknown hook action: {action}")
        return 0

    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        _debug(True, f"invalid hook JSON: {exc}")
        return 0

    config = load_config()
    client = LycheeMemClient(config)
    session_id = _lycheemem_session_id(config, event)

    try:
        if action == "user-prompt-submit":
            return handle_user_prompt_submit(config, client, session_id, event)
        if action == "stop":
            return handle_stop(config, client, session_id, event)
        return handle_session_end(config, client, session_id)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        _debug(config.debug, f"LycheeMem hook skipped after error: {exc}")
        return 0


def handle_user_prompt_submit(
    config: Config,
    client: LycheeMemClient,
    session_id: str,
    event: dict[str, Any],
) -> int:
    prompt = str(event.get("prompt") or "").strip()
    additional_context = ""

    if config.enable_auto_recall and prompt:
        result = client.smart_search(prompt, session_id)
        additional_context = _format_recalled_context(result, config.max_context_chars)

    if config.enable_auto_append and prompt:
        client.append_turn(session_id, "user", prompt)

    if additional_context:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": additional_context,
                    },
                    "suppressOutput": True,
                },
                ensure_ascii=False,
            )
        )
    return 0


def handle_stop(
    config: Config,
    client: LycheeMemClient,
    session_id: str,
    event: dict[str, Any],
) -> int:
    assistant_text = str(event.get("last_assistant_message") or "").strip()
    if config.enable_auto_append and assistant_text:
        client.append_turn(session_id, "assistant", assistant_text)

    if config.enable_auto_consolidate and _should_consolidate(config, session_id):
        client.consolidate(session_id)
        _mark_consolidated(session_id)
    return 0


def handle_session_end(config: Config, client: LycheeMemClient, session_id: str) -> int:
    if config.enable_auto_consolidate:
        client.consolidate(session_id)
        _mark_consolidated(session_id)
    return 0


def _format_recalled_context(result: dict[str, Any], max_chars: int) -> str:
    context = str(result.get("background_context") or "").strip()
    if not context:
        context = _fallback_constructed_context(result)
    if not context:
        return ""

    if len(context) > max_chars:
        context = context[: max_chars - 80].rstrip() + "\n...[LycheeMem context truncated]"

    return (
        "[LycheeMem recalled context]\n"
        "The following long-term memory was retrieved for this prompt. "
        "Use it as background evidence when relevant, and do not mention "
        "LycheeMem unless it helps the user.\n\n"
        f"{context}"
    )


def _fallback_constructed_context(result: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("graph_results", "semantic_results", "skill_results"):
        values = result.get(key)
        if not isinstance(values, list):
            continue
        for item in values[:3]:
            if not isinstance(item, dict):
                continue
            constructed = item.get("constructed_context")
            if isinstance(constructed, str) and constructed.strip():
                chunks.append(constructed.strip())
                continue
            semantic_text = item.get("semantic_text")
            if isinstance(semantic_text, str) and semantic_text.strip():
                chunks.append(semantic_text.strip())
    return "\n\n".join(chunks)


def _lycheemem_session_id(config: Config, event: dict[str, Any]) -> str:
    raw = str(event.get("session_id") or event.get("transcript_path") or "default")
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw).strip("_") or "default"
    return f"{config.session_prefix}:{safe}"


def _state_dir() -> Path:
    root = os.getenv("CLAUDE_PLUGIN_DATA", "").strip()
    if root:
        path = Path(root)
    else:
        path = Path("/tmp") / "lycheemem-claude-plugin"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_file(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", session_id).strip("_") or "default"
    return _state_dir() / f"{safe}.json"


def _should_consolidate(config: Config, session_id: str) -> bool:
    cooldown = config.auto_consolidate_cooldown_seconds
    if cooldown <= 0:
        return True
    path = _state_file(session_id)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    last = float(state.get("last_consolidated_at") or 0)
    return (time.time() - last) >= cooldown


def _mark_consolidated(session_id: str) -> None:
    path = _state_file(session_id)
    path.write_text(
        json.dumps({"last_consolidated_at": time.time()}, ensure_ascii=False),
        encoding="utf-8",
    )


def _debug(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[lycheemem] {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
