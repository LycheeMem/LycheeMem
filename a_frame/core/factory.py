"""
Pipeline 工厂：一键组装所有组件。

提供 `create_pipeline()` 入口，注入所有依赖。
"""

from __future__ import annotations

from a_frame.agents.consolidator_agent import ConsolidatorAgent
from a_frame.agents.reasoning_agent import ReasoningAgent
from a_frame.agents.search_coordinator import SearchCoordinator
from a_frame.agents.synthesizer_agent import SynthesizerAgent
from a_frame.agents.wm_manager import WMManager
from a_frame.core.graph import AFramePipeline
from a_frame.embedder.base import BaseEmbedder
from a_frame.llm.base import BaseLLM
from a_frame.memory.graph.entity_extractor import EntityExtractor
from a_frame.memory.graph.graph_store import NetworkXGraphStore
from a_frame.memory.procedural.skill_store import InMemorySkillStore
from a_frame.memory.working.compressor import WorkingMemoryCompressor
from a_frame.memory.working.session_store import InMemorySessionStore


def _create_session_store(settings=None):
    """根据配置创建会话存储。"""
    backend = getattr(settings, "session_backend", "memory") if settings else "memory"
    if backend == "sqlite":
        from a_frame.memory.working.sqlite_session_store import SQLiteSessionStore
        return SQLiteSessionStore(db_path=settings.sqlite_db_path)
    return InMemorySessionStore()


def _create_graph_store(settings=None, embedder: BaseEmbedder | None = None):
    """根据配置创建图谱存储。"""
    backend = getattr(settings, "graph_backend", "memory") if settings else "memory"
    enable_semantic_search = getattr(settings, "graph_semantic_search", True) if settings else True
    enable_semantic_merge = getattr(settings, "graph_semantic_merge", False) if settings else False
    merge_threshold = getattr(settings, "graph_semantic_merge_threshold", 0.88) if settings else 0.88
    search_threshold = getattr(settings, "graph_semantic_search_threshold", 0.55) if settings else 0.55
    scan_limit = getattr(settings, "graph_semantic_scan_limit", 5000) if settings else 5000
    if backend == "neo4j":
        from a_frame.memory.graph.neo4j_graph_store import Neo4jGraphStore
        return Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            embedder=embedder,
            enable_semantic_search=enable_semantic_search,
            enable_semantic_merge=enable_semantic_merge,
            semantic_merge_threshold=merge_threshold,
            semantic_search_threshold=search_threshold,
            semantic_scan_limit=scan_limit,
        )
    return NetworkXGraphStore(
        embedder=embedder,
        enable_semantic_search=enable_semantic_search,
        enable_semantic_merge=enable_semantic_merge,
        semantic_merge_threshold=merge_threshold,
        semantic_search_threshold=search_threshold,
    )


def _create_skill_store(settings=None, embedding_dim: int = 768):
    """根据配置创建技能库。"""
    backend = getattr(settings, "skill_backend", "memory") if settings else "memory"
    if backend == "file":
        from a_frame.memory.procedural.file_skill_store import FileSkillStore
        return FileSkillStore(file_path=settings.skill_file_path)
    if backend == "lancedb":
        from a_frame.memory.procedural.lancedb_skill_store import LanceDBSkillStore
        return LanceDBSkillStore(
            db_path=settings.lancedb_path,
            embedding_dim=embedding_dim,
        )
    return InMemorySkillStore()


def create_pipeline(
    llm: BaseLLM,
    embedder: BaseEmbedder,
    *,
    settings=None,
    max_tokens: int = 128_000,
    warn_threshold: float = 0.7,
    block_threshold: float = 0.9,
    min_recent_turns: int = 4,
    graph_search_depth: int = 1,
    skill_top_k: int = 3,
) -> AFramePipeline:
    """一键组装 A-Frame Pipeline。

    传入 settings 时使用配置指定的存储后端（SQLite/Neo4j/LanceDB）。
    不传 settings 时使用内存存储（开发/测试模式）。

    Args:
        llm: LLM 适配器实例。
        embedder: Embedding 适配器实例。
        settings: 可选配置对象，控制存储后端选择。
        max_tokens: 工作记忆 token 上限。
        warn_threshold: 预警压缩阈值。
        block_threshold: 阻塞压缩阈值。
        min_recent_turns: 压缩时保留的最近轮数。

    Returns:
        组装好的 AFramePipeline 实例。
    """
    # 根据 settings 选择存储后端
    embedding_dim = getattr(settings, "embedding_dim", 768) if settings else 768
    session_store = _create_session_store(settings)
    graph_store = _create_graph_store(settings, embedder=embedder)
    skill_store = _create_skill_store(settings, embedding_dim=embedding_dim)

    # 压缩器
    compressor = WorkingMemoryCompressor(
        llm=llm,
        max_tokens=max_tokens,
        warn_threshold=warn_threshold,
        block_threshold=block_threshold,
        min_recent_turns=min_recent_turns,
    )

    # 实体抽取器
    entity_extractor = EntityExtractor(llm=llm)

    # 5 个认知组件
    wm_manager = WMManager(session_store=session_store, compressor=compressor)
    search_coordinator = SearchCoordinator(
        llm=llm,
        embedder=embedder,
        graph_store=graph_store,
        skill_store=skill_store,
        graph_search_depth=graph_search_depth,
        skill_top_k=skill_top_k,
    )
    synthesizer = SynthesizerAgent(llm=llm)
    reasoner = ReasoningAgent(llm=llm)
    consolidator = ConsolidatorAgent(
        llm=llm,
        embedder=embedder,
        graph_store=graph_store,
        skill_store=skill_store,
        entity_extractor=entity_extractor,
    )

    return AFramePipeline(
        wm_manager=wm_manager,
        search_coordinator=search_coordinator,
        synthesizer=synthesizer,
        reasoner=reasoner,
        consolidator=consolidator,
    )
