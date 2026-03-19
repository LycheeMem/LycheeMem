"""
启动入口: python -m lychee_memos

使用开发模式 LLM/Embedder 启动 FastAPI 服务。
生产环境应通过 uvicorn 直接运行。
"""

import argparse
from pathlib import Path

curr_dir = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="LycheeMemOS Cognitive Memory Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    # 延迟导入，仅在实际运行时加载
    import uvicorn

    from src.api.server import create_app
    from src.auth.auth import configure_jwt
    from src.auth.user_store import UserStore
    from src.core.config import settings
    from src.core.factory import create_pipeline

    llm = _create_llm(settings)
    embedder = _create_embedder(settings)

    # 用户存储 + JWT 配置
    user_store = UserStore(db_path=settings.user_db_path)
    configure_jwt(settings.jwt_secret_key, settings.jwt_expire_hours)

    pipeline = create_pipeline(llm=llm, embedder=embedder, settings=settings)
    app = create_app(pipeline, user_store=user_store)

    print(f"🚀 LycheeMemOS server starting on http://{args.host}:{args.port}")
    print(f"   LLM:  {settings.llm_model}")
    print(f"   Embed:{settings.embedding_model}")
    print(
        f"   Storage: session={settings.session_backend}, graph={settings.graph_backend}, skill={settings.skill_backend}"
    )
    print(f"   Auth:  user_db={settings.user_db_path}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port)


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
