"""
启动入口: python -m a_frame

使用开发模式 LLM/Embedder 启动 FastAPI 服务。
生产环境应通过 uvicorn 直接运行。
"""

import argparse
from pathlib import Path

curr_dir = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="A-Frame Cognitive Memory Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--llm",
        default="gemini",
        choices=["openai", "gemini", "ollama"],
        help="LLM backend (default: openai)",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    # 延迟导入，仅在实际运行时加载
    import uvicorn

    from a_frame.api.server import create_app
    from a_frame.core.config import settings
    from a_frame.core.factory import create_pipeline

    # 根据选择创建 LLM
    llm = _create_llm(args.llm, settings)
    embedder = _create_embedder(settings)

    pipeline = create_pipeline(llm=llm, embedder=embedder, settings=settings)
    app = create_app(pipeline)

    print(f"🚀 A-Frame server starting on http://{args.host}:{args.port}")
    print(f"   LLM backend: {args.llm}")
    print(
        f"   Storage: session={settings.session_backend}, graph={settings.graph_backend}, skill={settings.skill_backend}"
    )
    print(f"   Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port)


def _create_llm(backend: str, settings):
    if backend == "openai":
        from a_frame.llm.openai_llm import OpenAILLM

        return OpenAILLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_api_base,
        )
    elif backend == "gemini":
        from a_frame.llm.gemini_llm import GeminiLLM

        return GeminiLLM(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
    elif backend == "ollama":
        from a_frame.llm.ollama_llm import OllamaLLM

        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


def _create_embedder(settings):
    backend = settings.embedding_backend
    if backend == "gemini":
        from a_frame.embedder.gemini_embedder import GeminiEmbedder

        return GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
            output_dimensionality=settings.gemini_embedding_dim,
        )
    else:
        from a_frame.embedder.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            base_url=settings.openai_api_base,
        )


if __name__ == "__main__":
    main()
