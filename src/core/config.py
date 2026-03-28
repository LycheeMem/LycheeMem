"""
Pydantic Settings 配置。

从 .env 文件或环境变量读取全部配置项。
所有字段扁平定义在 Settings 中，直接映射环境变量。
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 在 Settings 初始化前，显式加载 .env 文件
# override=True 确保 .env 中的值优先于系统环境变量
# （开发环境约定：.env 提高优先级）
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)


class Settings(BaseSettings):
    """统一配置，所有字段直接从环境变量读取。"""

    # ─── LLM（litellm 统一入口）───
    # model 使用完整 litellm 模型字符串，provider 前缀决定供应商：
    #   OpenAI / 兼容代理：openai/<model>，例如 openai/gpt-4o-mini
    #   Gemini：           gemini/<model>，例如 gemini/gemini-2.0-flash
    #   Ollama：           ollama_chat/<model>，例如 ollama_chat/qwen2.5
    llm_model: str = "openai/gpt-4o-mini"
    llm_api_key: str = ""
    llm_api_base: str = ""

    # ─── Embedder（litellm 统一入口）───
    # model 同样使用完整 litellm 模型字符串，例如：
    #   openai/text-embedding-3-small
    #   openai/<custom-model>（配合 embedding_api_base 使用）
    #   gemini/gemini-embedding-001
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_api_key: str = ""  # 可选
    embedding_api_base: str = ""  # 可选

    # ─── 工作记忆预算 ───
    wm_max_tokens: int = 128000
    wm_warn_threshold: float = 0.7
    wm_block_threshold: float = 0.9
    min_recent_turns: int = 4

    # ─── 存储后端选择 ───
    session_backend: str = "sqlite"  # "memory" | "sqlite"
    semantic_memory_backend: str = "compact"  # "graphiti" | "compact"

    # ─── SQLite (会话持久化) ───
    sqlite_db_path: str = "lychee_memos_sessions.db"

    # ─── Compact Semantic Memory ───
    compact_memory_db_path: str = "data/compact_memory.db"
    compact_vector_db_path: str = "data/compact_vector"
    compact_dedup_threshold: float = 0.85
    compact_synthesis_min_units: int = 2
    compact_synthesis_similarity: float = 0.75

    # ─── Neo4j (图谱持久化) ───
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ─── Graphiti(论文) 图谱引擎（唯一图谱实现）───
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

    # ─── Graphiti Cross-Encoder Rerank (复用主 LLM 适配器) ───
    graphiti_cross_encoder_enabled: bool = True
    graphiti_cross_encoder_top_n: int = 20
    graphiti_cross_encoder_weight: float = 1.0

    # ─── File Skill Store (轻量级向量持久化) ───
    skill_file_path: str = "lychee_memos_skills.json"
    skill_top_k: int = 3

    # ─── 图谱检索 ───
    graph_top_k: int = 5  # 图谱记忆检索返回条数（Graphiti 与 legacy 共用）

    # ─── API ───
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ─── 用户认证 ───
    jwt_secret_key: str = "lychee-dev-secret-change-me"
    jwt_expire_hours: int = 168  # 7 天
    user_db_path: str = "lychee_memos_users.db"

    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")


settings = Settings()