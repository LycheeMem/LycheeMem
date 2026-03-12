"""
Pydantic Settings 配置。

从 .env 文件或环境变量读取全部配置项。
所有字段扁平定义在 Settings 中，直接映射环境变量。
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """统一配置，所有字段直接从环境变量读取。"""

    # ─── LLM ───
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5"

    # ─── Embedder ───
    embedding_backend: str = "openai"  # "openai" or "gemini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    gemini_embedding_model: str = "gemini-embedding-2-preview"
    gemini_embedding_dim: int | None = None  # None = 默认 3072, 可选 768/1536

    # ─── 工作记忆预算 ───
    wm_max_tokens: int = 128_000
    wm_warn_threshold: float = 0.7
    wm_block_threshold: float = 0.9

    # ─── 存储后端选择 ───
    session_backend: str = "sqlite"   # "memory" | "sqlite"
    graph_backend: str = "neo4j"     # "memory" | "neo4j"
    skill_backend: str = "lancedb"    # "memory" | "lancedb"

    # ─── SQLite (会话持久化) ───
    sqlite_db_path: str = "a_frame_sessions.db"

    # ─── Neo4j (图谱持久化) ───
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # ─── LanceDB (向量持久化) ───
    lancedb_path: str = "a_frame_lancedb"

    # ─── API ───
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
