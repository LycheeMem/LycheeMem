#!/usr/bin/env python3
"""Verify the LycheeMem OpenClaw plugin configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PLUGIN_ID = "lycheemem-tools"
SKILL_ID = "lycheemem"
OPENCLAW_CONFIG_ENV = "OPENCLAW_CONFIG_PATH"
DEFAULT_CONFIG_PATHS = (
    Path.home() / ".openclaw" / "openclaw.json",
    Path.home() / ".config" / "openclaw" / "openclaw.json",
)
DEFAULT_AGENT_ID = "main"
REQUIRED_AGENT_TOOLS = (
    "lychee_memory_smart_search",
    "lychee_memory_append_turn",
    "lychee_memory_consolidate",
)
REQUIRED_CONFIG_KEYS = {
    "baseUrl": str,
    "transport": str,
    "timeout": (int, float),
    "apiToken": str,
    "enableHostLifecycle": bool,
    "enablePromptPresence": bool,
    "enableAutoAppendTurns": bool,
    "enableBoundaryConsolidation": bool,
    "enableProactiveConsolidation": bool,
    "proactiveConsolidationCooldownSeconds": (int, float),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check OpenClaw config and LycheeMem backend reachability."
    )
    parser.add_argument(
        "--config-path",
        help=(
            "Path to openclaw.json. Defaults to OPENCLAW_CONFIG_PATH, then "
            "~/.openclaw/openclaw.json."
        ),
    )
    parser.add_argument(
        "--agent-id",
        default=DEFAULT_AGENT_ID,
        help="Agent id to verify under agents.list. Defaults to main.",
    )
    parser.add_argument(
        "--skip-reachability",
        action="store_true",
        help="Only validate config; do not call the LycheeMem backend.",
    )
    parser.add_argument(
        "--health-path",
        default="/health",
        help="Backend health endpoint path. Defaults to /health.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5,
        help="Reachability timeout in seconds. Defaults to 5.",
    )
    return parser.parse_args()


def resolve_config_path(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser()

    env_path = os.environ.get(OPENCLAW_CONFIG_ENV)
    if env_path:
        return Path(env_path).expanduser()

    for candidate in DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return candidate

    return DEFAULT_CONFIG_PATHS[0]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"OpenClaw config not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"OpenClaw config is empty: {path}")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("OpenClaw config root must be a JSON object")
    return data


def get_plugin_entry(root: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    plugins = root.get("plugins")
    if not isinstance(plugins, dict):
        return None, "plugins object is missing"

    entries = plugins.get("entries")
    if not isinstance(entries, dict):
        return None, "plugins.entries object is missing"

    entry = entries.get(PLUGIN_ID)
    if not isinstance(entry, dict):
        return None, f"plugins.entries.{PLUGIN_ID} object is missing"

    return entry, None


def get_skill_entry(root: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    skills = root.get("skills")
    if not isinstance(skills, dict):
        return None, "skills object is missing"

    entries = skills.get("entries")
    if not isinstance(entries, dict):
        return None, "skills.entries object is missing"

    entry = entries.get(SKILL_ID)
    if not isinstance(entry, dict):
        return None, f"skills.entries.{SKILL_ID} object is missing"

    return entry, None


def get_agent_entry(
    root: dict[str, Any],
    agent_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    agents = root.get("agents")
    if not isinstance(agents, dict):
        return None, "agents object is missing"

    entries = agents.get("list")
    if not isinstance(entries, list):
        return None, "agents.list array is missing"

    for item in entries:
        if isinstance(item, dict) and item.get("id") == agent_id:
            return item, None

    return None, f'agents.list does not contain agent "{agent_id}"'


def status_line(ok: bool, message: str) -> None:
    prefix = "PASS" if ok else "FAIL"
    print(f"[{prefix}] {message}")


def validate_config(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    all_ok = True

    enabled = entry.get("enabled")
    if enabled is True:
        status_line(True, f"{PLUGIN_ID} plugin entry is enabled")
    else:
        status_line(False, f"{PLUGIN_ID} plugin entry is not enabled")
        all_ok = False

    config = entry.get("config")
    if not isinstance(config, dict):
        status_line(False, f"{PLUGIN_ID}.config object is missing")
        return None, False

    for key, expected_type in REQUIRED_CONFIG_KEYS.items():
        value = config.get(key)
        if isinstance(value, expected_type):
            status_line(True, f"config.{key} is present")
        else:
            status_line(False, f"config.{key} is missing or has the wrong type")
            all_ok = False

    base_url = config.get("baseUrl")
    if isinstance(base_url, str):
        parsed = urlparse(base_url)
        url_ok = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        status_line(url_ok, f"config.baseUrl is a valid HTTP URL: {base_url}")
        all_ok = all_ok and url_ok

    transport = config.get("transport")
    transport_ok = transport in {"mcp", "http"}
    status_line(transport_ok, "config.transport is mcp or http")
    all_ok = all_ok and transport_ok

    timeout = config.get("timeout")
    timeout_ok = isinstance(timeout, (int, float)) and timeout > 0
    status_line(timeout_ok, "config.timeout is positive")
    all_ok = all_ok and timeout_ok

    cooldown = config.get("proactiveConsolidationCooldownSeconds")
    cooldown_ok = isinstance(cooldown, (int, float)) and cooldown >= 15
    status_line(cooldown_ok, "config.proactiveConsolidationCooldownSeconds >= 15")
    all_ok = all_ok and cooldown_ok

    return config, all_ok


def validate_skill_entry(entry: dict[str, Any]) -> bool:
    enabled = entry.get("enabled")
    ok = enabled is True
    status_line(ok, f"skills.entries.{SKILL_ID}.enabled is true")
    return ok


def validate_agent_entry(entry: dict[str, Any], agent_id: str) -> bool:
    all_ok = True

    skills = entry.get("skills")
    if isinstance(skills, list) and SKILL_ID in skills:
        status_line(True, f'agents.list["{agent_id}"].skills contains {SKILL_ID}')
    else:
        status_line(False, f'agents.list["{agent_id}"].skills is missing {SKILL_ID}')
        all_ok = False

    tools = entry.get("tools")
    if not isinstance(tools, dict):
        status_line(False, f'agents.list["{agent_id}"].tools object is missing')
        return False

    allow = tools.get("alsoAllow")
    if not isinstance(allow, list):
        status_line(False, f'agents.list["{agent_id}"].tools.alsoAllow array is missing')
        return False

    for tool_name in REQUIRED_AGENT_TOOLS:
        ok = tool_name in allow
        status_line(
            ok,
            f'agents.list["{agent_id}"].tools.alsoAllow contains {tool_name}',
        )
        all_ok = all_ok and ok

    return all_ok


def join_url(base_url: str, path: str) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{base_url.rstrip('/')}{normalized_path}"


def check_reachability(base_url: str, health_path: str, timeout: float) -> bool:
    url = join_url(base_url, health_path)
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
    except HTTPError as exc:
        status_line(False, f"LycheeMem health check failed: HTTP {exc.code} at {url}")
        return False
    except URLError as exc:
        status_line(False, f"LycheeMem health check failed: {exc.reason} at {url}")
        return False
    except TimeoutError:
        status_line(False, f"LycheeMem health check timed out at {url}")
        return False

    ok = 200 <= status < 400
    status_line(ok, f"LycheeMem backend reachable at {url}")
    return ok


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config_path)
    print(f"OpenClaw config: {config_path}")

    try:
        root = load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status_line(False, str(exc))
        return 1

    entry, error = get_plugin_entry(root)
    if error:
        status_line(False, error)
        return 1
    skill_entry, skill_error = get_skill_entry(root)
    if skill_error:
        status_line(False, skill_error)
        return 1
    agent_entry, agent_error = get_agent_entry(root, args.agent_id)
    if agent_error:
        status_line(False, agent_error)
        return 1

    assert entry is not None
    assert skill_entry is not None
    assert agent_entry is not None
    config, config_ok = validate_config(entry)
    if config is None:
        return 1
    skill_ok = validate_skill_entry(skill_entry)
    agent_ok = validate_agent_entry(agent_entry, args.agent_id)

    reachability_ok = True
    if args.skip_reachability:
        print("[SKIP] LycheeMem backend reachability check")
    else:
        base_url = config.get("baseUrl")
        if isinstance(base_url, str):
            reachability_ok = check_reachability(
                base_url,
                args.health_path,
                args.timeout,
            )
        else:
            reachability_ok = False

    if config_ok and skill_ok and agent_ok and reachability_ok:
        print("\nLycheeMem OpenClaw plugin config looks ready.")
        return 0

    print("\nLycheeMem OpenClaw plugin config needs attention.")
    print("Hint: run setup_openclaw_plugin.py, restart gateway, and ensure the backend is up.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
