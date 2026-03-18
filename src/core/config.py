"""
Pydantic Settings 配置。

从 .env 文件或环境变量读取全部配置项。
所有字段扁平定义在 Settings 中，直接映射环境变量。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """统一配置，所有字段直接从环境变量读取。"""

    # ─── LLM ───
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite-preview"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5"

    # ─── Embedder ───
    embedding_backend: str = "gemini"  # "openai" or "gemini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    gemini_embedding_model: str = "gemini-embedding-2-preview"
    gemini_embedding_dim: int | None = None  # None = 默认 3072, 可选 768/1536

    # ─── 工作记忆预算 ───
    wm_max_tokens: int = 128_000
    wm_warn_threshold: float = 0.7
    wm_block_threshold: float = 0.9

    # ─── 存储后端选择 ───
    session_backend: str = "sqlite"  # "memory" | "sqlite"
    graph_backend: str = "neo4j"  # "memory" | "neo4j"
    skill_backend: str = "file"  # "memory" | "file" | "lancedb"

    # ─── SQLite (会话持久化) ───
    sqlite_db_path: str = "lychee_memos_sessions.db"

    # ─── Neo4j (图谱持久化) ───
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ─── Graphiti(论文) 图谱引擎 ───
    graphiti_enabled: bool = True
    graphiti_database: str = "neo4j"

    # ─── Graphiti Strict Mode (Graphiti-only / Fail-fast) ───
    # 当 graphiti_enabled=true 且 graphiti_strict=true 时：
    # - 不允许任何 Graphiti→legacy 回退
    # - 关键依赖缺失（Neo4j/GDS/vector index）应在启动时直接失败
    graphiti_strict: bool = True
    graphiti_require_gds: bool = True
    graphiti_require_vector_index: bool = True

    # Graphiti vector index 配置（默认与 embedding_dim 对齐；若为 0 则自动推断）
    graphiti_vector_dim: int = 0
    graphiti_vector_similarity_function: str = "cosine"

    # ─── Graphiti GDS-based rerank/community ───
    graphiti_gds_distance_max_depth: int = 4
    # Paper §3.1: number of recent episodes used as BFS seeds (φ_bfs).
    # "particularly valuable when using recent episodes as seeds for the
    # breadth-first search" — default 4 mirrors the n=4 context window used
    # for entity extraction (§2.2.1).
    graphiti_bfs_recent_episode_limit: int = 4
    # Paper §2.3: periodic full-graph community refresh.
    # refresh_all_communities() is called automatically every N episodes
    # ingested globally (across all sessions) to correct drift from
    # incremental dynamic extension.  Set to 0 to disable.
    graphiti_community_refresh_every: int = 50

    # ─── Graphiti MMR (Maximal Marginal Relevance) ───
    graphiti_mmr_lambda: float = 0.5  # 1.0 = pure relevance, 0.0 = pure diversity

    # ─── Graphiti Cross-Encoder Rerank (Gemini) ───
    graphiti_cross_encoder_enabled: bool = True
    graphiti_cross_encoder_model: str = "gemini-3.1-flash-lite-preview"
    graphiti_cross_encoder_top_n: int = 20
    graphiti_cross_encoder_weight: float = 1.0

    # ─── Graph Semantic (向量检索 / 同义合并) ───
    graph_semantic_search: bool = True
    graph_semantic_merge: bool = True
    graph_semantic_merge_threshold: float = 0.88
    graph_semantic_search_threshold: float = 0.55
    graph_semantic_scan_limit: int = 5000

    # ─── LanceDB (向量持久化) ───
    lancedb_path: str = "lychee_memos_lancedb"

    # ─── File Skill Store (轻量级向量持久化) ───
    skill_file_path: str = "lychee_memos_skills.json"

    # ─── API ───
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")


settings = Settings()
