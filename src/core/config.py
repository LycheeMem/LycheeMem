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

    # ─── SQLite (会话持久化) ───
    sqlite_db_path: str = "sessions.db"

    # ─── Compact Semantic Memory ───
    compact_memory_db_path: str = "data/compact_memory.db"
    compact_vector_db_path: str = "data/compact_vector"
    compact_dedup_threshold: float = 0.85
    compact_synthesis_min_records: int = 2
    compact_synthesis_similarity: float = 0.75

    # ─── Skill Store (SQLite + LanceDB) ───
    skill_db_path: str = "data/skill_store.db"
    skill_vector_db_path: str = "data/skill_vector"
    skill_top_k: int = 3

    # ─── 图谱检索 ───
    graph_top_k: int = 5  # 语义记忆检索返回条数

    # ─── API ───
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ─── 用户认证 ───
    jwt_secret_key: str = "lychee-dev-secret-change-me"
    jwt_expire_hours: int = 168  # 7 天
    user_db_path: str = "users.db"

    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")


settings = Settings()