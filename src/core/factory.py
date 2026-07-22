"""
Pipeline 工厂：一键组装所有组件。

提供 `create_pipeline()` 入口，注入所有依赖。
语义记忆后端固定使用 Compact（SQLite+LanceDB）。
支持可选的视觉记忆模块。
"""

from __future__ import annotations

import logging

from src.agents.consolidator_agent import ConsolidatorAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.search_coordinator import SearchCoordinator
from src.agents.wm_manager import WMManager
from src.core.graph import LycheePipeline
from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.procedural.sqlite_skill_store import SQLiteSkillStore
from src.memory.working.compressor import WorkingMemoryCompressor
from src.memory.working.session_store import InMemorySessionStore

logger = logging.getLogger(__name__)


def _resolve_embedding_dim(embedder: BaseEmbedder, settings) -> int:
    """Resolve the dimension returned by the active embedder.
    
    The configured value is also forwarded to providers that support output
    truncation, but it cannot be trusted as the storage dimension: compatible
    APIs may ignore ``dimensions`` and return their model default. Probe once at
    startup so LanceDB is built from the real vector shape.
    """
    cached_dim = int(getattr(embedder, "_lychee_resolved_dimension", 0) or 0)
    if cached_dim > 0:
        return cached_dim

    def remember(dimension: int) -> int:
        try:
            setattr(embedder, "_lychee_resolved_dimension", dimension)
        except Exception:
            pass
        return dimension

    configured_dim = int(getattr(settings, "embedding_dim", 0) or 0)
    # 本地模型：直接读属性（会触发懒加载，但无网络调用）
    if hasattr(embedder, "dimension"):
        try:
            actual_dim = int(embedder.dimension)
            if configured_dim > 0 and actual_dim != configured_dim:
                logger.warning(
                    "配置 embedding_dim=%d，但本地模型实际返回 %d；使用实际维度",
                    configured_dim,
                    actual_dim,
                )
            else:
                logger.info("本地模型 embedding_dim = %d", actual_dim)
            return remember(actual_dim)
        except Exception:
            pass
    logger.info(
        "探测实际 embedding_dim（配置值=%d）…",
        configured_dim,
    )
    try:
        vec = embedder.embed_query("probe")
        actual_dim = len(vec)
        if actual_dim <= 0:
            raise ValueError("embedding probe returned an empty vector")
        if configured_dim > 0 and actual_dim != configured_dim:
            logger.warning(
                "配置 embedding_dim=%d，但 embedding API 实际返回 %d；使用实际维度",
                configured_dim,
                actual_dim,
            )
        else:
            logger.info("实际 embedding_dim = %d", actual_dim)
        return remember(actual_dim)
    except Exception as exc:
        if configured_dim > 0:
            logger.warning(
                "实际 embedding_dim 探测失败，暂用配置值 %d: %s",
                configured_dim,
                exc,
            )
            return remember(configured_dim)
        logger.warning("实际 embedding_dim 探测失败，回退到变长模式: %s", exc)
        return 0


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
    skill_top_k = settings.skill_top_k
    session_store = _create_session_store(settings)
    embedding_dim = _resolve_embedding_dim(embedder, settings)
    skill_store = SQLiteSkillStore(
        db_path=getattr(settings, "skill_db_path", "data/skill_store.db"),
        vector_db_path=getattr(settings, "skill_vector_db_path", "data/skill_vector"),
        embedder=embedder,
        embedding_dim=embedding_dim,
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
        session_store=session_store,
        sqlite_db_path=getattr(settings, "compact_memory_db_path", "data/compact_memory.db"),
        vector_db_path=getattr(settings, "compact_vector_db_path", "data/compact_vector"),
        embedding_dim=embedding_dim,
        reranker_enabled=getattr(settings, "reranker_enabled", False),
        reranker_backend=getattr(settings, "reranker_backend", "local"),
        reranker_model=getattr(settings, "reranker_model", "BAAI/bge-reranker-v2-m3"),
        reranker_api_base=getattr(settings, "reranker_api_base", ""),
        reranker_api_key=getattr(settings, "reranker_api_key", ""),
        reranker_device=getattr(settings, "reranker_device", "auto"),
        reranker_batch_size=getattr(settings, "reranker_batch_size", 16),
        reranker_max_length=getattr(settings, "reranker_max_length", 512),
        reranker_candidate_limit=getattr(settings, "reranker_candidate_limit", 100),
    )
    # 启动时补全尚未向量化的原始对话 turns（增量，已索引的跳过）
    semantic_engine.index_unvectorized_turns()

    wm_manager = WMManager(session_store=session_store, compressor=compressor)
    search_coordinator = SearchCoordinator(
        llm=llm,
        embedder=embedder,
        skill_store=skill_store,
        semantic_engine=semantic_engine,
        skill_top_k=skill_top_k,
    )
    reasoner = ReasoningAgent(llm=llm)
    consolidator = ConsolidatorAgent(
        llm=llm,
        embedder=embedder,
        skill_store=skill_store,
        semantic_engine=semantic_engine,
    )

    from src.core.config import settings as global_settings
    from src.memory.visual.visual_extractor import VisualExtractor
    from src.memory.visual.visual_forgetter import VisualForgetter
    from src.memory.visual.visual_retriever import VisualRetriever
    from src.memory.visual.visual_store import VisualStore

    multimodal_embedder = None
    use_multimodal = getattr(global_settings, "use_multimodal_embedding", False)
    if use_multimodal:
        try:
            logger.info("Initializing multimodal embedder...")
            from src.memory.visual.multimodal_embedder import MultimodalEmbedder
            
            dashscope_key = getattr(global_settings, "llm_api_key", "")
            use_dashscope = bool(dashscope_key)
            
            if use_dashscope:
                logger.info("Using DashScope API for multimodal embedding")
                multimodal_model = "dashscope/image-embedding-v1"
            else:
                logger.info("Using local CLIP model for multimodal embedding")
                multimodal_model = getattr(
                    global_settings,
                    "multimodal_embedding_model",
                    "openai/clip-vit-base-patch32"
                )
            
            multimodal_embedder = MultimodalEmbedder(
                model_name=multimodal_model,
                device="cpu",
                lazy_load=True,  # 延迟加载，避免启动时阻塞
                cache_size=1000,
                use_dashscope=use_dashscope,
                dashscope_api_key=dashscope_key if use_dashscope else None,
            )
            logger.info("Multimodal embedding enabled: model=%s", multimodal_model)
        except ImportError as e:
            logger.warning("Multimodal embedder not available (ImportError): %s", e)
            logger.warning("Falling back to text-only embedding")
        except Exception as e:
            logger.warning("Failed to initialize multimodal embedder: %s", e)
            logger.warning("Falling back to text-only embedding")

    visual_store = VisualStore(
        db_path=getattr(global_settings, "visual_memory_db_path", "data/visual_memory.db"),
        vector_db_path=getattr(global_settings, "visual_vector_db_path", "data/visual_vector"),
        image_storage_path=getattr(global_settings, "visual_image_path", "data/visual_memory"),
        embedding_dim=_resolve_embedding_dim(embedder, global_settings),
        embedder=embedder,  # 文本 embedder（向后兼容）
        multimodal_embedder=multimodal_embedder,
    )

    vlm_model = getattr(global_settings, "vlm_model", None)
    logger.info("VLM 配置检查：vlm_model=%s, vlm_api_base=%s", 
                vlm_model, 
                getattr(global_settings, "vlm_api_base", "NOT SET"))
    if vlm_model:
        from src.core.provider_resolver import resolve_litellm_model_provider
        from src.llm.litellm_llm import LiteLLMLLM

        resolved_vlm_model, resolved_vlm_api_key, resolved_vlm_api_base = (
            resolve_litellm_model_provider(
                vlm_model,
                api_key=(
                    getattr(global_settings, "vlm_api_key", "")
                    or getattr(global_settings, "llm_api_key", "")
                ),
                api_base=(
                    getattr(global_settings, "vlm_api_base", "")
                    or getattr(global_settings, "llm_api_base", "")
                ),
                atlascloud_api_key=getattr(global_settings, "atlascloud_api_key", ""),
                atlascloud_api_base=getattr(global_settings, "atlascloud_api_base", ""),
            )
        )

        vlm_llm = LiteLLMLLM(
            model=resolved_vlm_model,
            api_key=resolved_vlm_api_key,
            api_base=resolved_vlm_api_base,
            default_temperature=getattr(global_settings, "vlm_temperature", 0.7),
            default_max_tokens=(
                getattr(global_settings, "vlm_max_tokens", 0)
                if getattr(global_settings, "vlm_max_tokens", 0) > 0
                else None
            ),
            default_top_p=getattr(global_settings, "vlm_top_p", 1.0),
        )
        logger.info("VLM 初始化成功：model=%s", vlm_model)
    else:
        vlm_llm = llm  # 降级：复用主 LLM
        logger.warning("未配置 VLM 模型，将复用主 LLM（可能不支持图片识别）")

    use_fast_mode = getattr(global_settings, "visual_fast_mode", True)
    max_image_size = getattr(global_settings, "visual_max_image_size", 1024)

    visual_extractor = VisualExtractor(
        llm=vlm_llm,
        fast_mode=use_fast_mode,
        max_image_size=max_image_size,
        cache_size=128,
    )
    visual_retriever = VisualRetriever(visual_store=visual_store)
    visual_forgetter = VisualForgetter(visual_store=visual_store)

    pipeline = LycheePipeline(
        wm_manager=wm_manager,
        search_coordinator=search_coordinator,
        reasoner=reasoner,
        consolidator=consolidator,
    )

    # 注入视觉组件到 pipeline（供后续扩展使用）
    pipeline.visual_store = visual_store
    pipeline.visual_extractor = visual_extractor
    pipeline.visual_retriever = visual_retriever
    pipeline.visual_forgetter = visual_forgetter

    return pipeline
