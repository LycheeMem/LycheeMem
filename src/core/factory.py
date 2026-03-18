"""
Pipeline 工厂：一键组装所有组件。

提供 `create_pipeline()` 入口，注入所有依赖。
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
from src.memory.graph.entity_extractor import EntityExtractor
from src.memory.graph.graph_store import NetworkXGraphStore
from src.memory.procedural.skill_store import InMemorySkillStore
from src.memory.working.compressor import WorkingMemoryCompressor
from src.memory.working.session_store import InMemorySessionStore


def _create_session_store(settings=None):
    """根据配置创建会话存储。"""
    backend = getattr(settings, "session_backend", "memory") if settings else "memory"
    if backend == "sqlite":
        from src.memory.working.sqlite_session_store import SQLiteSessionStore

        return SQLiteSessionStore(db_path=settings.sqlite_db_path)
    return InMemorySessionStore()


def _create_graph_store(settings=None, embedder: BaseEmbedder | None = None):
    """根据配置创建图谱存储。"""
    backend = getattr(settings, "graph_backend", "memory") if settings else "memory"
    enable_semantic_search = getattr(settings, "graph_semantic_search", True) if settings else True
    enable_semantic_merge = getattr(settings, "graph_semantic_merge", False) if settings else False
    merge_threshold = (
        getattr(settings, "graph_semantic_merge_threshold", 0.88) if settings else 0.88
    )
    search_threshold = (
        getattr(settings, "graph_semantic_search_threshold", 0.55) if settings else 0.55
    )
    scan_limit = getattr(settings, "graph_semantic_scan_limit", 5000) if settings else 5000
    if backend == "neo4j":
        from src.memory.graph.neo4j_graph_store import Neo4jGraphStore

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
        from src.memory.procedural.file_skill_store import FileSkillStore

        return FileSkillStore(file_path=settings.skill_file_path)
    if backend == "lancedb":
        from src.memory.procedural.lancedb_skill_store import LanceDBSkillStore

        return LanceDBSkillStore(
            db_path=settings.lancedb_path,
            embedding_dim=embedding_dim,
        )
    return InMemorySkillStore()


def create_pipeline(
    llm: BaseLLM,
    embedder: BaseEmbedder,
    *,
    settings,
) -> LycheePipeline:
    """一键组装 LycheeMemOS Pipeline。

    传入 settings 时使用配置指定的存储后端（SQLite/Neo4j/LanceDB）。
    不传 settings 时使用内存存储（开发/测试模式）。

    Args:
        llm: LLM 适配器实例。
        embedder: Embedding 适配器实例。
        settings: 可选配置对象，控制存储后端选择。
    Returns:
        组装好的 LycheePipeline 实例。
    """
    # 根据 settings 选择存储后端
    embedding_dim = settings.embedding_dim
    wm_max_tokens = settings.wm_max_tokens
    warn_threshold = settings.wm_warn_threshold
    block_threshold = settings.wm_block_threshold
    min_recent_turns = settings.min_recent_turns
    graph_search_depth = settings.graph_search_depth
    graph_top_k = settings.graph_top_k
    skill_top_k = settings.skill_top_k
    session_store = _create_session_store(settings)
    graph_store = _create_graph_store(settings, embedder=embedder)
    skill_store = _create_skill_store(settings, embedding_dim=embedding_dim)

    # 压缩器
    compressor = WorkingMemoryCompressor(
        llm=llm,
        max_tokens=wm_max_tokens,
        warn_threshold=warn_threshold,
        block_threshold=block_threshold,
        min_recent_turns=min_recent_turns,
    )

    # 实体抽取器
    entity_extractor = EntityExtractor(llm=llm)

    graphiti_engine = None
    if settings and getattr(settings, "graphiti_enabled", False):
        if getattr(settings, "graph_backend", "memory") != "neo4j":
            raise RuntimeError("Graphiti requires graph_backend=neo4j")

        from src.memory.graph.graphiti_engine import GraphitiEngine
        from src.memory.graph.graphiti_neo4j_store import GraphitiNeo4jStore

        strict = bool(getattr(settings, "graphiti_strict", False))

        # Infer vector dim for Neo4j vector indexes.
        vector_dim = int(getattr(settings, "graphiti_vector_dim", 0) or 0)
        if vector_dim <= 0:
            # Prefer Gemini explicit dim when using Gemini embedder; else fall back to embedding_dim.
            if getattr(settings, "embedding_backend", "").lower() == "gemini":
                gem_dim = getattr(settings, "gemini_embedding_dim", None)
                if gem_dim is not None:
                    vector_dim = int(gem_dim)
            if vector_dim <= 0:
                vector_dim = int(embedding_dim)

        vector_similarity = str(
            getattr(settings, "graphiti_vector_similarity_function", "cosine") or "cosine"
        )

        graphiti_store = GraphitiNeo4jStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=getattr(settings, "graphiti_database", "neo4j"),
            init_schema=True,
            vector_dim=vector_dim,
            vector_similarity_function=vector_similarity,
        )

        if strict:
            graphiti_store.preflight(
                require_gds=bool(getattr(settings, "graphiti_require_gds", True)),
                require_vector_index=bool(getattr(settings, "graphiti_require_vector_index", True)),
                vector_dim=vector_dim,
            )

        # Optional: cross-encoder rerank (paper-parity, via main LLM adapter)
        cross_encoder = None
        cross_encoder_enabled = bool(getattr(settings, "graphiti_cross_encoder_enabled", False))
        if cross_encoder_enabled:
            if llm is None:
                if strict:
                    raise RuntimeError("Graphiti strict cross-encoder enabled but llm is missing")
            else:
                from src.memory.graph.cross_encoder import (  # noqa: PLC0415
                    LLMCrossEncoderReranker,
                )

                cross_encoder = LLMCrossEncoderReranker(llm=llm)

        graphiti_engine = GraphitiEngine(
            store=graphiti_store,
            strict=strict,
            community_llm=llm,
            embedder=embedder,
            gds_distance_max_depth=int(getattr(settings, "graphiti_gds_distance_max_depth", 4)),
            cross_encoder=cross_encoder,
            cross_encoder_top_n=int(getattr(settings, "graphiti_cross_encoder_top_n", 20)),
            cross_encoder_weight=float(getattr(settings, "graphiti_cross_encoder_weight", 1.0)),
            mmr_lambda=float(getattr(settings, "graphiti_mmr_lambda", 0.5)),
            bfs_recent_episode_limit=int(getattr(settings, "graphiti_bfs_recent_episode_limit", 4)),
        )

    # 5 个认知组件
    wm_manager = WMManager(session_store=session_store, compressor=compressor)
    search_coordinator = SearchCoordinator(
        llm=llm,
        embedder=embedder,
        graph_store=graph_store,
        skill_store=skill_store,
        graphiti_engine=graphiti_engine,
        graph_search_depth=graph_search_depth,
        graph_top_k=graph_top_k,
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
        graphiti_engine=graphiti_engine,
        community_refresh_every=settings.graphiti_community_refresh_every,
    )

    return LycheePipeline(
        wm_manager=wm_manager,
        search_coordinator=search_coordinator,
        synthesizer=synthesizer,
        reasoner=reasoner,
        consolidator=consolidator,
    )
