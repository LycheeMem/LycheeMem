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
    merge_threshold = (
        getattr(settings, "graph_semantic_merge_threshold", 0.88) if settings else 0.88
    )
    search_threshold = (
        getattr(settings, "graph_semantic_search_threshold", 0.55) if settings else 0.55
    )
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

    graphiti_engine = None
    if settings and getattr(settings, "graphiti_enabled", False):
        if getattr(settings, "graph_backend", "memory") != "neo4j":
            raise RuntimeError("Graphiti requires graph_backend=neo4j")

        from a_frame.memory.graph.graphiti_engine import GraphitiEngine
        from a_frame.memory.graph.graphiti_neo4j_store import GraphitiNeo4jStore

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

        # Optional: Gemini cross-encoder rerank (paper-parity)
        cross_encoder = None
        cross_encoder_enabled = bool(getattr(settings, "graphiti_cross_encoder_enabled", False))
        if cross_encoder_enabled:
            gemini_key = str(getattr(settings, "gemini_api_key", "") or "").strip()
            if not gemini_key:
                if strict:
                    raise RuntimeError(
                        "Graphiti strict cross-encoder enabled but gemini_api_key is empty"
                    )
            else:
                from a_frame.memory.graph.gemini_cross_encoder import (  # noqa: PLC0415
                    GeminiCrossEncoderReranker,
                )

                cross_encoder = GeminiCrossEncoderReranker(
                    api_key=gemini_key,
                    model=str(
                        getattr(settings, "graphiti_cross_encoder_model", "")
                        or "gemini-3.1-flash-lite-preview"
                    ),
                )

        graphiti_engine = GraphitiEngine(
            store=graphiti_store,
            strict=strict,
            gds_distance_max_depth=int(getattr(settings, "graphiti_gds_distance_max_depth", 4)),
            cross_encoder=cross_encoder,
            cross_encoder_top_n=int(getattr(settings, "graphiti_cross_encoder_top_n", 20)),
            cross_encoder_weight=float(getattr(settings, "graphiti_cross_encoder_weight", 1.0)),
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
    )

    return AFramePipeline(
        wm_manager=wm_manager,
        search_coordinator=search_coordinator,
        synthesizer=synthesizer,
        reasoner=reasoner,
        consolidator=consolidator,
    )
