"""
使用开发模式 LLM/Embedder 启动 FastAPI 服务。
生产环境应通过 uvicorn 直接运行。
"""

import argparse
from pathlib import Path

curr_dir = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="LycheeMem: Compact, efficient, and extensible long-term memory for LLM agents")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server")
    args = parser.parse_args()
    
    # 创建data/
    data_dir = curr_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # 延迟导入，仅在实际运行时加载
    import uvicorn

    from src.api.server import create_app
    from src.core.config import settings
    from src.core.factory import create_pipeline

    llm = _create_llm(settings)
    embedder = _create_embedder(settings)

    pipeline = create_pipeline(llm=llm, embedder=embedder, settings=settings)
    app = create_app(pipeline)

    host = settings.api_host
    port = args.port or settings.api_port

    print(f"   LycheeMem server starting on http://{host}:{port}")
    print(f"   LLM:  {settings.llm_model}")
    print(f"   Embed:{settings.embedding_model}")
    print(f"   Docs: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)


def _create_llm(settings):
    from src.llm.litellm_llm import LiteLLMLLM

    return LiteLLMLLM(
        model=settings.llm_model,
        api_key=settings.llm_api_key or None,
        api_base=settings.llm_api_base or None,
        default_temperature=settings.llm_temperature,
        default_max_tokens=(settings.llm_max_tokens if settings.llm_max_tokens > 0 else None),
        default_top_p=settings.llm_top_p,
    )


def _create_embedder(settings):
    backend = str(getattr(settings, "embedding_backend", "litellm") or "litellm").lower()
    embedding_model = str(getattr(settings, "embedding_model", "") or "").strip()
    embedding_api_base = str(getattr(settings, "embedding_api_base", "") or "").strip()
    looks_like_shared_embedding_server = (
        embedding_model == "local-embed"
        or embedding_api_base.rstrip("/").endswith(":8765")
    )

    if backend == "http" or looks_like_shared_embedding_server:
        from src.embedder.http_embedder import HTTPEmbeddingServerEmbedder

        if not embedding_api_base:
            raise RuntimeError(
                "EMBEDDING_BACKEND=http or EMBEDDING_MODEL=local-embed requires "
                "EMBEDDING_API_BASE, for example http://localhost:8765"
            )
        return HTTPEmbeddingServerEmbedder(
            api_base=embedding_api_base,
            api_key=settings.embedding_api_key or None,
            model=embedding_model or "local-embed",
        )

    # 本地模式：使用 sentence-transformers，不调用远程 API
    if backend == "local" or getattr(settings, "embedding_local", False):
        from src.embedder.st_embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            dimensions=settings.embedding_dim
        )

    from src.embedder.litellm_embedder import LiteLLMEmbedder

    return LiteLLMEmbedder(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key or None,
        api_base=settings.embedding_api_base or None,
        # task_type 仅对 gemini/ 和 vertex_ai/ 生效，其他 provider 自动忽略
        task_type="RETRIEVAL_DOCUMENT",
        query_task_type="RETRIEVAL_QUERY",
    )


if __name__ == "__main__":
    main()
