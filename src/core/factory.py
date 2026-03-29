"""
Pipeline 工厂：一键组装所有组件。

提供 `create_pipeline()` 入口，注入所有依赖。
语义记忆后端固定使用 Compact（SQLite+LanceDB）。
"""

from __future__ import annotations

from src.agents.consolidator_agent import ConsolidatorAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.search_coordinator import SearchCoordinator
from src.agents.synthesizer_agent import SynthesizerAgent
from src.agents.wm_manager import WMManager
from src.core.graph import LycheePipeline
from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.procedural.sqlite_skill_store import SQLiteSkillStore
from src.memory.working.compressor import WorkingMemoryCompressor
from src.memory.working.session_store import InMemorySessionStore


def _create_session_store(settings=None):
    """根据配置创建会话存储。"""
    backend = getattr(settings, "session_backend", "memory") if settings else "memory"
    if backend == "sqlite":
        from src.memory.working.sqlite_session_store import SQLiteSessionStore

        return SQLiteSessionStore(db_path=settings.sqlite_db_path)
    return InMemorySessionStore()


def create_pipeline(
    llm: BaseLLM,
    embedder: BaseEmbedder,
    *,
    settings,
) -> LycheePipeline:
    """一键组装 LycheeMem Pipeline。

    Args:
        llm: LLM 适配器实例。
        embedder: Embedding 适配器实例。
        settings: 配置对象，控制存储后端选择。
    Returns:
        组装好的 LycheePipeline 实例。
    """
    wm_max_tokens = settings.wm_max_tokens
    warn_threshold = settings.wm_warn_threshold
    block_threshold = settings.wm_block_threshold
    min_recent_turns = settings.min_recent_turns
    graph_top_k = settings.graph_top_k
    skill_top_k = settings.skill_top_k
    session_store = _create_session_store(settings)
    skill_store = SQLiteSkillStore(
        db_path=getattr(settings, "skill_db_path", "data/skill_store.db"),
        vector_db_path=getattr(settings, "skill_vector_db_path", "data/skill_vector"),
        embedder=embedder,
        embedding_dim=getattr(settings, "embedding_dim", 1536),
    )

    compressor = WorkingMemoryCompressor(
        llm=llm,
        max_tokens=wm_max_tokens,
        warn_threshold=warn_threshold,
        block_threshold=block_threshold,
        min_recent_turns=min_recent_turns,
    )

    from src.memory.semantic.engine import CompactSemanticEngine

    semantic_engine = CompactSemanticEngine(
        llm=llm,
        embedder=embedder,
        sqlite_db_path=getattr(settings, "compact_memory_db_path", "data/compact_memory.db"),
        vector_db_path=getattr(settings, "compact_vector_db_path", "data/compact_vector"),
        dedup_threshold=getattr(settings, "compact_dedup_threshold", 0.85),
        synthesis_min_records=getattr(settings, "compact_synthesis_min_records", 2),
        synthesis_similarity=getattr(settings, "compact_synthesis_similarity", 0.75),
        embedding_dim=getattr(settings, "embedding_dim", 1536),
    )

    wm_manager = WMManager(session_store=session_store, compressor=compressor)
    search_coordinator = SearchCoordinator(
        llm=llm,
        embedder=embedder,
        skill_store=skill_store,
        semantic_engine=semantic_engine,
        graph_top_k=graph_top_k,
        skill_top_k=skill_top_k,
    )
    synthesizer = SynthesizerAgent(llm=llm)
    reasoner = ReasoningAgent(llm=llm)
    consolidator = ConsolidatorAgent(
        llm=llm,
        embedder=embedder,
        skill_store=skill_store,
        semantic_engine=semantic_engine,
    )

    return LycheePipeline(
        wm_manager=wm_manager,
        search_coordinator=search_coordinator,
        synthesizer=synthesizer,
        reasoner=reasoner,
        consolidator=consolidator,
    )
