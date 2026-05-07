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

    llm_model: str = "openai/gpt-4o-mini"
    llm_api_key: str = ""
    llm_api_base: str = ""


    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 0  # 0 = 自动探测（启动时调用一次 API 获取实际维度）
    embedding_api_key: str = ""  # 可选
    embedding_api_base: str = ""  # 可选

    # ─── 本地 Embedding（sentence-transformers）───
    # 设为 true 时使用本地模型，EMBEDDING_MODEL 填 HuggingFace 模型路径
    # 例如：Qwen/Qwen3-Embedding-0.6B、BAAI/bge-m3、sentence-transformers/all-MiniLM-L6-v2
    embedding_local: bool = False
    embedding_device: str = "auto"          # auto / cpu / cuda / mps

    wm_max_tokens: int = 128000
    wm_warn_threshold: float = 0.7
    wm_block_threshold: float = 0.9
    min_recent_turns: int = 4

    session_backend: str = "sqlite"

    sqlite_db_path: str = "data/sessions.db"

    compact_memory_db_path: str = "data/compact_memory.db"
    compact_vector_db_path: str = "data/compact_vector"
    compact_dedup_threshold: float = 0.85
    compact_synthesis_min_records: int = 2
    compact_synthesis_similarity: float = 0.75

    skill_db_path: str = "data/skill_store.db"
    skill_vector_db_path: str = "data/skill_vector"
    skill_top_k: int = 3

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    vlm_model: str = ""  # 例如 openai/qwen-vl-max，留空则复用主 LLM
    vlm_api_key: str = ""
    vlm_api_base: str = ""
    visual_memory_db_path: str = "data/visual_memory.db"
    visual_vector_db_path: str = "data/visual_vector"
    visual_image_path: str = "data/visual_memory"

    visual_fast_mode: bool = True  # 快速模式：更短 Prompt、更低 token 限制
    visual_max_image_size: int = 1024  # 图片最大边长（像素），超过则缩放
    visual_skip_embedding: bool = True  # 是否跳过嵌入生成（更快，但无法向量检索）

    use_multimodal_embedding: bool = False  # 是否启用多模态嵌入 (需要 HuggingFace 网络)
    multimodal_embedding_model: str = "openai/clip-vit-base-patch32"  # CLIP 模型

    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")


settings = Settings()