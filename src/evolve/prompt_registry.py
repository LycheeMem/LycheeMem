"""Prompt 注册表：版本管理 + 热切换 + 全局单例接口。

核心设计：
- 每个 prompt 以 name 标识，对应一组版本链
- `get_prompt(name)` 返回当前 active 版本文本
- 首次调用时自动 fallback 到 hardcoded 默认值
- 版本提升/回滚通过 PromptStore 持久化

全局接口：
- `init_registry(db_path)`: 在 factory.py 中调用一次
- `get_prompt(name, fallback)`: 在所有 prompt 消费方中调用
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from src.evolve.prompt_store import PromptStore, PromptVersion

logger = logging.getLogger("src.evolve.registry")

# 不可被自动优化的 prompt（安全护栏）
IMMUTABLE_PROMPTS: frozenset[str] = frozenset({
    "feedback_classification",
    "novelty_check",
})


class PromptRegistry:
    """Prompt 版本注册表。

    - 在内存中缓存 active 版本文本，避免每次 LLM 调用都查库
    - 支持热切换：promote/rollback 后自动刷新缓存
    - 线程安全
    """

    def __init__(self, store: PromptStore):
        self._store = store
        self._cache: dict[str, str] = {}  # name → prompt_text
        self._version_cache: dict[str, int] = {}  # name → version number
        self._defaults: dict[str, str] = {}  # name → hardcoded default
        self._lock = threading.Lock()

    @property
    def store(self) -> PromptStore:
        return self._store

    def register_default(self, name: str, prompt_text: str) -> None:
        """注册一个 hardcoded 默认 prompt。

        如果 SQLite 中不存在该 prompt 的任何版本，则创建 version=0 作为基线。
        """
        self._defaults[name] = prompt_text

        existing = self._store.get_active_version(name)
        if existing is None:
            pv = PromptVersion(
                prompt_name=name,
                version=0,
                prompt_text=prompt_text,
                status="active",
                reason="hardcoded default",
            )
            self._store.save_version(pv)
            with self._lock:
                self._cache[name] = prompt_text
                self._version_cache[name] = 0
        else:
            with self._lock:
                self._cache[name] = existing.prompt_text
                self._version_cache[name] = existing.version

    def get(self, name: str, fallback: str | None = None) -> str:
        """获取 prompt 当前 active 文本。

        优先级：缓存 → SQLite active → fallback → 默认值 → 空串
        """
        with self._lock:
            if name in self._cache:
                return self._cache[name]

        active = self._store.get_active_version(name)
        if active is not None:
            with self._lock:
                self._cache[name] = active.prompt_text
                self._version_cache[name] = active.version
            return active.prompt_text

        text = fallback or self._defaults.get(name, "")
        return text

    def get_active_version_number(self, name: str) -> int:
        """获取当前 active 版本号。"""
        with self._lock:
            if name in self._version_cache:
                return self._version_cache[name]
        active = self._store.get_active_version(name)
        if active:
            with self._lock:
                self._version_cache[name] = active.version
            return active.version
        return 0

    def create_candidate(
        self,
        name: str,
        prompt_text: str,
        reason: str = "",
    ) -> PromptVersion:
        """创建一个候选版本（status=candidate），不影响当前 active。"""
        if name in IMMUTABLE_PROMPTS:
            raise ValueError(f"Prompt '{name}' is immutable and cannot be optimized")

        latest = self._store.get_latest_version_number(name)
        new_version = latest + 1
        pv = PromptVersion(
            prompt_name=name,
            version=new_version,
            prompt_text=prompt_text,
            status="candidate",
            reason=reason,
            parent_version=latest if latest >= 0 else 0,
        )
        self._store.save_version(pv)
        logger.info("Created candidate v%d for prompt '%s': %s", new_version, name, reason)
        return pv

    def promote_candidate(self, name: str, version: int, eval_score: float | None = None) -> None:
        """将候选版本提升为 active，旧版本归档。"""
        if name in IMMUTABLE_PROMPTS:
            raise ValueError(f"Prompt '{name}' is immutable")

        pv = self._store.get_version(name, version)
        if pv is None:
            raise ValueError(f"Version {version} not found for prompt '{name}'")

        if eval_score is not None:
            pv.eval_score = eval_score
            self._store.save_version(pv)

        self._store.promote(name, version)

        with self._lock:
            self._cache[name] = pv.prompt_text
            self._version_cache[name] = version
        logger.info("Promoted prompt '%s' to v%d (score=%.3f)", name, version, eval_score or 0)

    def rollback(self, name: str, version: int) -> None:
        """回滚某版本，恢复其 parent 为 active。"""
        pv = self._store.get_version(name, version)
        if pv is None:
            return

        self._store.rollback(name, version)

        parent = self._store.get_version(name, pv.parent_version)
        if parent:
            self._store.promote(name, parent.version)
            with self._lock:
                self._cache[name] = parent.prompt_text
                self._version_cache[name] = parent.version
            logger.info("Rolled back prompt '%s' v%d → v%d", name, version, parent.version)
        else:
            fallback = self._defaults.get(name, "")
            with self._lock:
                self._cache[name] = fallback
                self._version_cache[name] = 0
            logger.warning("Rolled back prompt '%s' v%d → default (no parent)", name, version)

    def refresh(self, name: str | None = None) -> None:
        """从 SQLite 刷新缓存。name=None 刷新全部。"""
        if name:
            active = self._store.get_active_version(name)
            if active:
                with self._lock:
                    self._cache[name] = active.prompt_text
                    self._version_cache[name] = active.version
        else:
            with self._lock:
                self._cache.clear()
                self._version_cache.clear()
            for pname in self._store.list_all_prompt_names():
                active = self._store.get_active_version(pname)
                if active:
                    with self._lock:
                        self._cache[pname] = active.prompt_text
                        self._version_cache[pname] = active.version

    def list_registered(self) -> list[str]:
        """列出所有已注册的 prompt 名称。"""
        return list(self._defaults.keys())

    def is_immutable(self, name: str) -> bool:
        return name in IMMUTABLE_PROMPTS


# ═══════════════════════════════════════════════════════════════
# 全局单例接口
# ═══════════════════════════════════════════════════════════════

_registry: PromptRegistry | None = None
_registry_lock = threading.Lock()


def init_registry(db_path: str = "data/prompt_evolve.db") -> PromptRegistry:
    """初始化全局 PromptRegistry（在 factory.py 中调用一次）。"""
    global _registry
    with _registry_lock:
        store = PromptStore(db_path=db_path)
        _registry = PromptRegistry(store=store)
        _register_all_defaults(_registry)
        return _registry


def get_registry() -> PromptRegistry | None:
    """获取全局 PromptRegistry（可能为 None）。"""
    return _registry


def get_prompt(name: str, fallback: str = "") -> str:
    """全局快捷接口：获取 prompt active 文本。

    如果 registry 未初始化，返回 fallback。
    这保证即使 evolve 子系统未启用，系统也能正常运行。
    """
    reg = _registry
    if reg is not None:
        return reg.get(name, fallback=fallback)
    return fallback


def get_active_versions_snapshot() -> dict[str, int]:
    """获取所有已注册 prompt 当前 active 版本号的快照。

    返回一个独立的 dict 拷贝，调用方可安全保留；
    后续 promote/rollback 不会影响已返回的快照。
    """
    reg = _registry
    if reg is None:
        return {}
    result: dict[str, int] = {}
    for name in reg.list_registered():
        result[name] = reg.get_active_version_number(name)
    return result


def select_prompt_versions(
    prompt_names: list[str] | tuple[str, ...] | set[str] | frozenset[str],
    *,
    snapshot: dict[str, int] | None = None,
) -> dict[str, int]:
    """从快照中筛选指定 prompt 的版本号。

    若未提供 snapshot，则读取当前 active 版本快照。
    返回独立 dict，便于调用方长期保留用于归因或计数。
    """
    source = snapshot if isinstance(snapshot, dict) else get_active_versions_snapshot()
    if not source:
        return {}

    selected: dict[str, int] = {}
    for raw_name in prompt_names:
        name = str(raw_name or "").strip()
        if not name or name not in source:
            continue
        try:
            selected[name] = int(source.get(name, 0) or 0)
        except Exception:
            selected[name] = 0
    return selected


# ── 注册所有默认 prompt ──

def _register_all_defaults(registry: PromptRegistry) -> None:
    """将所有 hardcoded prompt 注册到 registry。"""
    from src.memory.semantic.prompts import (
        COMPACT_ENCODING_SYSTEM,
        NOVELTY_CHECK_SYSTEM,
        SYNTHESIS_EXECUTE_SYSTEM,
        FEEDBACK_CLASSIFICATION_SYSTEM,
        RETRIEVAL_PLANNING_SYSTEM,
        RETRIEVAL_ADEQUACY_CHECK_SYSTEM,
        RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM,
        COMPOSITE_FILTER_SYSTEM,
    )
    from src.agents.prompts import (
        SEARCH_COORDINATOR_SYSTEM_PROMPT,
        SYNTHESIS_SYSTEM_PROMPT,
        REASONING_SYSTEM_PROMPT,
        CONSOLIDATION_SYSTEM_PROMPT,
    )
    from src.memory.working.compressor import WorkingMemoryCompressor

    defaults = {
        "compact_encoding": COMPACT_ENCODING_SYSTEM,
        "novelty_check": NOVELTY_CHECK_SYSTEM,
        "synthesis_execute": SYNTHESIS_EXECUTE_SYSTEM,
        "feedback_classification": FEEDBACK_CLASSIFICATION_SYSTEM,
        "retrieval_planning": RETRIEVAL_PLANNING_SYSTEM,
        "retrieval_adequacy_check": RETRIEVAL_ADEQUACY_CHECK_SYSTEM,
        "retrieval_additional_queries": RETRIEVAL_ADDITIONAL_QUERIES_SYSTEM,
        "composite_filter": COMPOSITE_FILTER_SYSTEM,
        "search_coordinator": SEARCH_COORDINATOR_SYSTEM_PROMPT,
        "synthesis": SYNTHESIS_SYSTEM_PROMPT,
        "reasoning": REASONING_SYSTEM_PROMPT,
        "consolidation": CONSOLIDATION_SYSTEM_PROMPT,
        "wm_compression": WorkingMemoryCompressor._default_compression_prompt(),
    }

    for name, text in defaults.items():
        registry.register_default(name, text)

    logger.info("Registered %d default prompts", len(defaults))
