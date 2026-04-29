"""Runtime state and hook/tool handlers for the LycheeMem Hermes plugin."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
import logging
import re
import shlex
import threading
import time
from typing import Any

from tools.registry import tool_error, tool_result

from .client import LycheeMemClient, LycheeMemPluginError
from .config import PluginConfig, load_config
from .tools import COMMAND_USAGE

logger = logging.getLogger(__name__)

_SESSION_MAX_LEN = 128
_SAFE_SESSION_CHARS = re.compile(r"[^A-Za-z0-9:_-]+")


@dataclass
class SessionState:
    dirty_turns: int = 0
    next_consolidate_at: float = 0.0
    last_pair_fingerprint: str = ""
    last_background_context: str = ""
    last_retrieved_context: str = ""


class LycheeMemHermesRuntime:
    """Shared runtime for tools and hooks registered by the plugin."""

    def __init__(
        self,
        config: PluginConfig | None = None,
        client: LycheeMemClient | None = None,
    ):
        self.config = config or load_config()
        self.client = client or LycheeMemClient(self.config)
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}
        self._current_session_id: str | None = None

    def on_session_start(self, **kwargs: Any) -> None:
        session_id = self._resolve_session_id(kwargs.get("session_id"), allow_default=False)
        if not session_id:
            return
        with self._lock:
            self._current_session_id = session_id
            self._sessions.setdefault(session_id, SessionState())

    def pre_llm_call(self, **kwargs: Any) -> dict[str, str] | None:
        if not self.config.enable_pre_llm_context:
            return None

        session_id = self._resolve_session_id(kwargs.get("session_id"))
        user_message = str(kwargs.get("user_message") or "").strip()
        if not session_id or not user_message:
            return None

        try:
            result = self.client.smart_search(
                user_message,
                top_k=self.config.smart_search_top_k,
                session_id=session_id,
                include_graph=self.config.include_graph,
                include_skills=self.config.include_skills,
                synthesize=self.config.synthesize,
                mode=self.config.smart_search_mode,
                response_level=self.config.response_level,
            )
        except Exception as exc:
            logger.warning("LycheeMem pre_llm_call recall failed: %s", exc)
            return None

        background_context = str(result.get("background_context") or "").strip()
        if not background_context:
            return None

        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            state.last_background_context = background_context
            raw_context = result.get("novelty_retrieved_context")
            if isinstance(raw_context, str):
                state.last_retrieved_context = raw_context
            self._current_session_id = session_id

        return {"context": f"Relevant memory from LycheeMem:\n{background_context}"}

    def post_llm_call(self, **kwargs: Any) -> None:
        if not self.config.enable_auto_append:
            return None

        session_id = self._resolve_session_id(kwargs.get("session_id"))
        user_message = str(kwargs.get("user_message") or "").strip()
        assistant_response = str(kwargs.get("assistant_response") or "").strip()
        if not session_id or not user_message or not assistant_response:
            return None

        fingerprint = sha1(
            f"{session_id}\0{user_message}\0{assistant_response}".encode("utf-8")
        ).hexdigest()
        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            if state.last_pair_fingerprint == fingerprint:
                return None

        try:
            self.client.append_turn(
                session_id=session_id,
                role="user",
                content=user_message,
                token_count=0,
            )
            self.client.append_turn(
                session_id=session_id,
                role="assistant",
                content=assistant_response,
                token_count=0,
            )
        except Exception as exc:
            logger.warning("LycheeMem post_llm_call append failed: %s", exc)
            return None

        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            state.dirty_turns += 2
            state.last_pair_fingerprint = fingerprint
            self._current_session_id = session_id
        return None

    def on_session_end(self, **kwargs: Any) -> None:
        if not self.config.enable_auto_consolidate:
            return None
        if kwargs.get("interrupted"):
            return None
        session_id = self._resolve_session_id(kwargs.get("session_id"))
        if not session_id:
            return None
        self._maybe_consolidate(
            session_id,
            background=self.config.auto_consolidate_background,
            reason="turn_end",
        )
        return None

    def on_session_finalize(self, **kwargs: Any) -> None:
        if not self.config.enable_finalize_consolidation:
            return None
        session_id = self._resolve_session_id(kwargs.get("session_id"))
        if not session_id:
            return None
        self._maybe_consolidate(
            session_id,
            background=self.config.auto_consolidate_background,
            force=True,
            reason="session_finalize",
        )
        return None

    def handle_health(self, args: dict[str, Any], **_: Any) -> str:
        try:
            result = self.client.health_check()
        except Exception as exc:
            return tool_error(str(exc), success=False)
        return tool_result(
            success=True,
            transport=self.config.transport,
            base_url=self.config.base_url,
            health=result,
        )

    def handle_smart_search(self, args: dict[str, Any], **_: Any) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required", success=False)

        session_id = self._resolve_session_id(args.get("session_id"))
        top_k = self._coerce_int(args.get("top_k"), self.config.smart_search_top_k, 1, 50)
        include_graph = self._coerce_bool(args.get("include_graph"), self.config.include_graph)
        include_skills = self._coerce_bool(args.get("include_skills"), self.config.include_skills)
        synthesize = self._coerce_bool(args.get("synthesize"), self.config.synthesize)
        mode = self._coerce_choice(
            args.get("mode"),
            self.config.smart_search_mode,
            {"raw", "compact", "full"},
        )
        response_level = self._coerce_choice(
            args.get("response_level"),
            self.config.response_level,
            {"minimal", "compact", "full"},
        )

        try:
            result = self.client.smart_search(
                query,
                top_k=top_k,
                session_id=session_id,
                include_graph=include_graph,
                include_skills=include_skills,
                synthesize=synthesize,
                mode=mode,
                response_level=response_level,
            )
        except Exception as exc:
            return tool_error(str(exc), success=False)

        with self._lock:
            if session_id:
                state = self._sessions.setdefault(session_id, SessionState())
                raw_context = result.get("novelty_retrieved_context")
                if isinstance(raw_context, str):
                    state.last_retrieved_context = raw_context
                background_context = result.get("background_context")
                if isinstance(background_context, str):
                    state.last_background_context = background_context
                self._current_session_id = session_id

        response = {
            "success": True,
            "session_id": session_id,
            "transport": self.config.transport,
        }
        response.update(result)
        return tool_result(response)

    def handle_append_turn(self, args: dict[str, Any], **_: Any) -> str:
        role = str(args.get("role") or "").strip()
        content = str(args.get("content") or "").strip()
        if not role:
            return tool_error("role is required", success=False)
        if not content:
            return tool_error("content is required", success=False)

        session_id = self._resolve_session_id(args.get("session_id"))
        if not session_id:
            return tool_error("session_id is required", success=False)
        token_count = self._coerce_int(args.get("token_count"), 0, 0, 1_000_000)

        try:
            result = self.client.append_turn(
                session_id=session_id,
                role=role,
                content=content,
                token_count=token_count,
            )
        except Exception as exc:
            return tool_error(str(exc), success=False)

        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            state.dirty_turns += 1
            self._current_session_id = session_id

        response = {"success": True, "session_id": session_id}
        response.update(result)
        return tool_result(response)

    def handle_consolidate(self, args: dict[str, Any], **_: Any) -> str:
        session_id = self._resolve_session_id(args.get("session_id"))
        if not session_id:
            return tool_error("session_id is required", success=False)

        retrieved_context = str(args.get("retrieved_context") or "")
        background = self._coerce_bool(
            args.get("background"),
            self.config.command_consolidate_background,
        )
        try:
            result = self.client.consolidate(
                session_id,
                retrieved_context=retrieved_context,
                background=background,
            )
        except Exception as exc:
            return tool_error(str(exc), success=False)

        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            state.dirty_turns = 0
            state.next_consolidate_at = time.time() + self.config.auto_consolidate_cooldown_seconds
            self._current_session_id = session_id

        response = {"success": True, "session_id": session_id}
        response.update(result)
        return tool_result(response)

    def handle_command(self, raw_args: str) -> str:
        raw_args = (raw_args or "").strip()
        if not raw_args:
            return COMMAND_USAGE

        try:
            parts = shlex.split(raw_args)
        except ValueError as exc:
            return f"Invalid command arguments: {exc}"

        if not parts:
            return COMMAND_USAGE

        command = parts[0].lower()
        if command in {"help", "--help", "-h"}:
            return COMMAND_USAGE

        if command == "status":
            session_id = self._resolve_session_id(None)
            with self._lock:
                state = self._sessions.get(session_id or "", SessionState())
            status = {
                "transport": self.config.transport,
                "base_url": self.config.base_url,
                "current_session_id": session_id,
                "dirty_turns": state.dirty_turns,
                "cooldown_remaining_seconds": max(
                    0,
                    int(state.next_consolidate_at - time.time()),
                ),
            }
            return json.dumps(status, ensure_ascii=False, indent=2)

        if command == "health":
            return self.handle_health({})

        if command == "recall":
            if len(parts) < 2:
                return "Usage: /lycheemem recall <query>"
            return self.handle_smart_search({"query": " ".join(parts[1:])})

        if command == "consolidate":
            return self.handle_consolidate({"background": False})

        return COMMAND_USAGE

    def _maybe_consolidate(
        self,
        session_id: str,
        *,
        background: bool,
        force: bool = False,
        reason: str = "",
    ) -> None:
        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            now = time.time()
            if state.dirty_turns <= 0:
                return
            if not force and now < state.next_consolidate_at:
                return
            retrieved_context = state.last_retrieved_context

        try:
            result = self.client.consolidate(
                session_id,
                retrieved_context=retrieved_context,
                background=background,
            )
        except Exception as exc:
            logger.warning(
                "LycheeMem auto consolidate failed for %s (%s): %s",
                session_id,
                reason,
                exc,
            )
            return

        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState())
            state.dirty_turns = 0
            state.next_consolidate_at = time.time() + self.config.auto_consolidate_cooldown_seconds
            self._current_session_id = session_id

        logger.info(
            "LycheeMem auto consolidate ok session=%s reason=%s status=%s",
            session_id,
            reason,
            result.get("status"),
        )

    def _resolve_session_id(
        self,
        raw_session_id: Any,
        *,
        allow_default: bool = True,
    ) -> str | None:
        raw = str(raw_session_id or "").strip()
        if not raw and allow_default:
            with self._lock:
                raw = self._current_session_id or ""
        if not raw:
            return None

        prefix = self.config.session_prefix.strip()
        if prefix and not raw.startswith(f"{prefix}:"):
            raw = f"{prefix}:{raw}"

        normalized = _SAFE_SESSION_CHARS.sub("_", raw)
        if len(normalized) <= _SESSION_MAX_LEN:
            return normalized

        digest = sha1(normalized.encode("utf-8")).hexdigest()[:12]
        keep = _SESSION_MAX_LEN - len(digest) - 1
        return f"{normalized[:keep]}:{digest}"

    @staticmethod
    def _coerce_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _coerce_choice(value: Any, default: str, allowed: set[str]) -> str:
        if not isinstance(value, str):
            return default
        candidate = value.strip().lower()
        if candidate in allowed:
            return candidate
        return default
