#!/usr/bin/env python3
"""Configure LycheeMem-related OpenClaw config entries.

This script intentionally only manages the OpenClaw config file. It does not
install OpenClaw, install the plugin package, or start the LycheeMem backend.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLUGIN_ID = "lycheemem-tools"
SKILL_ID = "lycheemem"
OPENCLAW_CONFIG_ENV = "OPENCLAW_CONFIG_PATH"
DEFAULT_CONFIG_PATHS = (
    Path.home() / ".openclaw" / "openclaw.json",
    Path.home() / ".config" / "openclaw" / "openclaw.json",
)
DEFAULT_AGENT_ID = "main"
DEFAULT_AGENT_TOOLS = (
    "lychee_memory_smart_search",
    "lychee_memory_append_turn",
    "lychee_memory_consolidate",
)

DEFAULT_PLUGIN_CONFIG: dict[str, Any] = {
    "baseUrl": "http://127.0.0.1:8000",
    "transport": "mcp",
    "timeout": 300,
    "apiToken": "",
    "enableHostLifecycle": True,
    "enablePromptPresence": True,
    "enableAutoAppendTurns": True,
    "enableBoundaryConsolidation": True,
    "enableProactiveConsolidation": False,
    "proactiveConsolidationCooldownSeconds": 180,
}


class ConfigError(RuntimeError):
    """Raised when the OpenClaw config cannot be safely updated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a conservative LycheeMem plugin config into OpenClaw's "
            "openclaw.json."
        )
    )
    parser.add_argument(
        "--config-path",
        help=(
            "Path to openclaw.json. Defaults to OPENCLAW_CONFIG_PATH, then "
            "~/.openclaw/openclaw.json."
        ),
    )
    parser.add_argument("--base-url", help="LycheeMem base URL.")
    parser.add_argument("--transport", choices=("mcp", "http"), help="Plugin transport.")
    parser.add_argument("--timeout", type=float, help="Request timeout in seconds.")
    parser.add_argument(
        "--api-token",
        help="Optional Bearer token for older authenticated LycheeMem deployments.",
    )
    parser.add_argument(
        "--plugin-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the lycheemem-tools plugin entry.",
    )
    parser.add_argument(
        "--skill-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable skills.entries.lycheemem.",
    )
    parser.add_argument(
        "--agent-id",
        default=DEFAULT_AGENT_ID,
        help="Agent id to update under agents.list. Defaults to main.",
    )
    parser.add_argument(
        "--tool",
        action="append",
        dest="agent_tools",
        help=(
            "Additional agent tools to allow. Repeat for multiple tools. "
            "Defaults to the core LycheeMem tool set."
        ),
    )
    parser.add_argument(
        "--host-lifecycle",
        dest="enableHostLifecycle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable host lifecycle hooks.",
    )
    parser.add_argument(
        "--prompt-presence",
        dest="enablePromptPresence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable LycheeMem prompt presence injection.",
    )
    parser.add_argument(
        "--auto-append-turns",
        dest="enableAutoAppendTurns",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable automatic user/assistant turn mirroring.",
    )
    parser.add_argument(
        "--boundary-consolidation",
        dest="enableBoundaryConsolidation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable consolidate on session boundaries.",
    )
    parser.add_argument(
        "--proactive-consolidation",
        dest="enableProactiveConsolidation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable signal-triggered proactive consolidation.",
    )
    parser.add_argument(
        "--proactive-cooldown",
        dest="proactiveConsolidationCooldownSeconds",
        type=float,
        help="Minimum seconds between proactive consolidations in one session.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing plugin config fields with defaults and CLI values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the merged plugin entry without writing openclaw.json.",
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
        return {}
    if path.is_dir():
        raise ConfigError(f"Config path is a directory: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"OpenClaw config root must be a JSON object: {path}")
    return data


def require_mapping(parent: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        value = {}
        parent[key] = value
    if not isinstance(value, dict):
        raise ConfigError(f"{path}.{key} must be a JSON object")
    return value


def require_list(parent: dict[str, Any], key: str, path: str) -> list[Any]:
    value = parent.get(key)
    if value is None:
        value = []
        parent[key] = value
    if not isinstance(value, list):
        raise ConfigError(f"{path}.{key} must be a JSON array")
    return value


def cli_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    candidates = {
        "baseUrl": args.base_url,
        "transport": args.transport,
        "timeout": args.timeout,
        "apiToken": args.api_token,
        "enableHostLifecycle": args.enableHostLifecycle,
        "enablePromptPresence": args.enablePromptPresence,
        "enableAutoAppendTurns": args.enableAutoAppendTurns,
        "enableBoundaryConsolidation": args.enableBoundaryConsolidation,
        "enableProactiveConsolidation": args.enableProactiveConsolidation,
        "proactiveConsolidationCooldownSeconds": (
            args.proactiveConsolidationCooldownSeconds
        ),
    }
    return {key: value for key, value in candidates.items() if value is not None}


def merge_plugin_entry(
    root: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], list[str]]:
    plugins = require_mapping(root, "plugins", "root")
    entries = require_mapping(plugins, "entries", "root.plugins")

    entry = entries.get(PLUGIN_ID)
    if entry is None:
        entry = {}
        entries[PLUGIN_ID] = entry
    if not isinstance(entry, dict):
        raise ConfigError(f"root.plugins.entries.{PLUGIN_ID} must be a JSON object")

    written: list[str] = []
    preserved: list[str] = []

    if args.plugin_enabled is not None:
        entry["enabled"] = args.plugin_enabled
        written.append("enabled")
    elif args.force or "enabled" not in entry:
        entry["enabled"] = True
        written.append("enabled")
    else:
        preserved.append("enabled")

    config = entry.get("config")
    if config is None:
        config = {}
        entry["config"] = config
    if not isinstance(config, dict):
        raise ConfigError(f"root.plugins.entries.{PLUGIN_ID}.config must be a JSON object")

    overrides = cli_config_overrides(args)
    desired = dict(DEFAULT_PLUGIN_CONFIG)
    desired.update(overrides)

    for key, value in desired.items():
        has_override = key in overrides
        if has_override or args.force or key not in config:
            config[key] = value
            written.append(f"config.{key}")
        else:
            preserved.append(f"config.{key}")

    return entry, written, preserved


def merge_skill_entry(
    root: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], list[str]]:
    skills = require_mapping(root, "skills", "root")
    entries = require_mapping(skills, "entries", "root.skills")

    entry = entries.get(SKILL_ID)
    if entry is None:
        entry = {}
        entries[SKILL_ID] = entry
    if not isinstance(entry, dict):
        raise ConfigError(f"root.skills.entries.{SKILL_ID} must be a JSON object")

    written: list[str] = []
    preserved: list[str] = []

    desired_enabled = True if args.skill_enabled is None else args.skill_enabled
    if args.force or "enabled" not in entry or args.skill_enabled is not None:
        entry["enabled"] = desired_enabled
        written.append(f"skills.entries.{SKILL_ID}.enabled")
    else:
        preserved.append(f"skills.entries.{SKILL_ID}.enabled")

    return entry, written, preserved


def unique_extend(items: list[Any], new_items: list[Any]) -> list[Any]:
    seen = set()
    merged: list[Any] = []
    for item in items + new_items:
        marker = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(
            item, (dict, list)
        ) else item
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(item)
    return merged


def merge_agent_entry(
    root: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], list[str]]:
    agents = require_mapping(root, "agents", "root")
    agent_list = agents.get("list")
    if agent_list is None:
        agent_list = []
        agents["list"] = agent_list
    if not isinstance(agent_list, list):
        raise ConfigError("root.agents.list must be a JSON array")

    agent_id = args.agent_id.strip() or DEFAULT_AGENT_ID
    target_agent: dict[str, Any] | None = None
    for item in agent_list:
        if isinstance(item, dict) and item.get("id") == agent_id:
            target_agent = item
            break

    if target_agent is None:
        target_agent = {"id": agent_id}
        agent_list.append(target_agent)

    written: list[str] = []
    preserved: list[str] = []

    skills = target_agent.get("skills")
    if skills is None:
        skills = []
        target_agent["skills"] = skills
    if not isinstance(skills, list):
        raise ConfigError(f"root.agents.list[{agent_id}].skills must be a JSON array")

    merged_skills = unique_extend(skills, [SKILL_ID])
    if merged_skills != skills:
        target_agent["skills"] = merged_skills
        written.append(f'agents.list[{agent_id}].skills += ["{SKILL_ID}"]')
    else:
        preserved.append(f'agents.list[{agent_id}].skills contains "{SKILL_ID}"')

    tools = require_mapping(target_agent, "tools", f"root.agents.list[{agent_id}]")
    allow = require_list(tools, "alsoAllow", f"root.agents.list[{agent_id}].tools")
    desired_tools = list(DEFAULT_AGENT_TOOLS)
    if args.agent_tools:
        desired_tools = unique_extend(desired_tools, args.agent_tools)
    merged_allow = unique_extend(allow, desired_tools)
    if merged_allow != allow:
        tools["alsoAllow"] = merged_allow
        written.append(f"agents.list[{agent_id}].tools.alsoAllow updated")
    else:
        preserved.append(f"agents.list[{agent_id}].tools.alsoAllow already contains LycheeMem tools")

    return target_agent, written, preserved


def write_config(path: Path, data: dict[str, Any]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)

    backup_path: Path | None = None
    if path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_name(f"{path.name}.lycheemem.{timestamp}.bak")
        shutil.copy2(path, backup_path)

    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)
    return backup_path


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config_path)

    try:
        root = load_config(config_path)
        plugin_entry, plugin_written, plugin_preserved = merge_plugin_entry(root, args)
        skill_entry, skill_written, skill_preserved = merge_skill_entry(root, args)
        agent_entry, agent_written, agent_preserved = merge_agent_entry(root, args)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"OpenClaw config: {config_path}")
    print(f"Plugin entry: plugins.entries.{PLUGIN_ID}")
    print(f"Skill entry: skills.entries.{SKILL_ID}")
    print(f"Agent entry: agents.list[{args.agent_id}]")

    written = plugin_written + skill_written + agent_written
    preserved = plugin_preserved + skill_preserved + agent_preserved

    if args.dry_run:
        print("\nDry run only. Merged plugin entry:")
        print(json.dumps(plugin_entry, indent=2, ensure_ascii=False))
        print("\nDry run only. Merged skill entry:")
        print(json.dumps(skill_entry, indent=2, ensure_ascii=False))
        print("\nDry run only. Merged agent entry:")
        print(json.dumps(agent_entry, indent=2, ensure_ascii=False))
        return 0

    backup_path = write_config(config_path, root)
    if backup_path:
        print(f"Backup written: {backup_path}")
    else:
        print("Backup skipped: config file did not exist yet")

    if written:
        print("\nWritten fields:")
        for item in written:
            print(f"  - {item}")
    if preserved:
        print("\nPreserved existing fields:")
        for item in preserved:
            print(f"  - {item}")

    print("\nNext steps:")
    print("  1. Restart OpenClaw gateway after changing plugin config.")
    print("  2. Start the LycheeMem backend, then run verify_openclaw_plugin.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
