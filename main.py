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
    port = settings.api_port

    print(f"🚀 LycheeMem server starting on http://{host}:{port}")
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
    )


def _create_embedder(settings):
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
