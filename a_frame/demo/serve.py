"""
启动 A-Frame Demo 服务器。

用法:
    python -m a_frame.demo.serve
    # 或
    python a_frame/demo/serve.py

打开浏览器访问: http://localhost:8000/demo
"""

from __future__ import annotations
import webbrowser
from threading import Timer


def _make_llm(settings):
    """根据配置创建 LLM 实例。"""
    if settings.gemini_api_key:
        from a_frame.llm.gemini_llm import GeminiLLM
        return GeminiLLM(api_key=settings.gemini_api_key, model=settings.gemini_model)
    if settings.openai_api_key:
        from a_frame.llm.openai_llm import OpenAILLM
        return OpenAILLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_api_base,
        )
    from a_frame.llm.ollama_llm import OllamaLLM
    return OllamaLLM(base_url=settings.ollama_base_url, model=settings.ollama_model)


def _make_embedder(settings):
    """根据配置创建 Embedder 实例。"""
    if settings.embedding_backend == "gemini" and settings.gemini_api_key:
        from a_frame.embedder.gemini_embedder import GeminiEmbedder
        return GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
        )
    from a_frame.embedder.openai_embedder import OpenAIEmbedder
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        base_url=settings.openai_api_base,
    )


def main():
    import uvicorn

    from a_frame.api.server import create_app
    from a_frame.core.config import Settings
    from a_frame.core.factory import create_pipeline

    settings = Settings()
    llm = _make_llm(settings)
    embedder = _make_embedder(settings)
    pipeline = create_pipeline(llm=llm, embedder=embedder, settings=settings)

    app = create_app(pipeline)

    # Auto-open browser after a short delay
    Timer(1.5, lambda: webbrowser.open("http://localhost:8000/demo")).start()

    print("\n" + "=" * 52)
    print("  A-Frame Console Demo")
    print("  http://localhost:8000/demo")
    print("=" * 52 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
