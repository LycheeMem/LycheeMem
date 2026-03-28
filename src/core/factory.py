"""
Pipeline 工厂：一键组装所有组件。

提供 `create_pipeline()` 入口，注入所有依赖。
支持两种语义记忆后端：graphiti（Neo4j 图谱）/ compact（SQLite+LanceDB）。
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
from src.memory.procedural.file_skill_store import FileSkillStore
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

    传入 settings 时使用配置指定的存储后端（SQLite/Neo4j/LanceDB）。
    不传 settings 时使用内存存储（开发/测试模式）。

    Args:
        llm: LLM 适配器实例。
        embedder: Embedding 适配器实例。
        settings: 可选配置对象，控制存储后端选择。
    Returns:
        组装好的 LycheePipeline 实例。
    """
    # 存储后端选择
    embedding_dim = settings.embedding_dim
    wm_max_tokens = settings.wm_max_tokens
    warn_threshold = settings.wm_warn_threshold
    block_threshold = settings.wm_block_threshold
    min_recent_turns = settings.min_recent_turns
    graph_top_k = settings.graph_top_k
    skill_top_k = settings.skill_top_k
    session_store = _create_session_store(settings)
    skill_store = FileSkillStore(file_path=settings.skill_file_path)

    # 压缩器
    compressor = WorkingMemoryCompressor(
        llm=llm,
        max_tokens=wm_max_tokens,
        warn_threshold=warn_threshold,
        block_threshold=block_threshold,
        min_recent_turns=min_recent_turns,
    )

    # ── 语义记忆后端选择 ──
    semantic_backend = getattr(settings, "semantic_memory_backend", "compact")

    if semantic_backend == "compact":
        semantic_engine, graphiti_engine = _create_compact_backend(
            llm=llm, embedder=embedder, settings=settings,
        )
    else:
        semantic_engine, graphiti_engine = _create_graphiti_backend(
            llm=llm, embedder=embedder, settings=settings, embedding_dim=embedding_dim,
        )

    # 5 个认知组件
    wm_manager = WMManager(session_store=session_store, compressor=compressor)
    search_coordinator = SearchCoordinator(
        llm=llm,
        embedder=embedder,
        skill_store=skill_store,
        semantic_engine=semantic_engine,
        graphiti_engine=graphiti_engine,
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
        graphiti_engine=graphiti_engine,
        community_refresh_every=settings.graphiti_community_refresh_every if graphiti_engine else 0,
    )

    return LycheePipeline(
        wm_manager=wm_manager,
        search_coordinator=search_coordinator,
        synthesizer=synthesizer,
        reasoner=reasoner,
        consolidator=consolidator,
    )


def _create_compact_backend(
    *,
    llm: BaseLLM,
    embedder: BaseEmbedder,
    settings,
) -> tuple:
    """创建 Compact Semantic Memory 后端。返回 (semantic_engine, None)。"""
    from src.memory.semantic.engine import CompactSemanticEngine
    from src.memory.semantic.scorer import ScoringWeights

    engine = CompactSemanticEngine(
        llm=llm,
        embedder=embedder,
        sqlite_db_path=getattr(settings, "compact_memory_db_path", "data/compact_memory.db"),
        vector_db_path=getattr(settings, "compact_vector_db_path", "data/compact_vector"),
        dedup_threshold=getattr(settings, "compact_dedup_threshold", 0.85),
        synthesis_min_units=getattr(settings, "compact_synthesis_min_units", 2),
        synthesis_similarity=getattr(settings, "compact_synthesis_similarity", 0.75),
    )
    return engine, None


def _create_graphiti_backend(
    *,
    llm: BaseLLM,
    embedder: BaseEmbedder,
    settings,
    embedding_dim: int,
) -> tuple:
    """创建 Graphiti (Neo4j) 后端。返回 (None, graphiti_engine)。"""
    from src.memory.graph.graphiti_engine import GraphitiEngine
    from src.memory.graph.graphiti_neo4j_store import GraphitiNeo4jStore

    strict = bool(getattr(settings, "graphiti_strict", True))

    vector_dim = int(getattr(settings, "graphiti_vector_dim", 0) or 0)
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

    cross_encoder = None
    if bool(getattr(settings, "graphiti_cross_encoder_enabled", False)):
        if llm is None:
            raise RuntimeError("Graphiti cross-encoder requires a valid llm adapter")
        from src.memory.graph.cross_encoder import LLMCrossEncoderReranker

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
    return None, graphiti_engine
