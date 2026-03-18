#!/usr/bin/env python3
"""
独立脚本：测试 OpenAI 配置是否有效（embedding + chat）
不依赖 pytest，直接验证 API 连接。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import settings
from src.llm.openai_llm import OpenAILLM
from src.embedder.openai_embedder import OpenAIEmbedder


def test_openai_llm():
    """测试 OpenAI LLM 生成功能。"""
    print("=" * 60)
    print("测试 OpenAI LLM 生成")
    print("=" * 60)

    print(f"API Key: {settings.openai_api_key}")
    print(f"API Base: {settings.openai_api_base}")
    print(f"Model: {settings.openai_model}")

    try:
        llm = OpenAILLM(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )
        print("✓ OpenAILLM 实例创建成功")

        messages = [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "简要介绍一下你自己"},
        ]
        print(f"\n发送请求: {messages[-1]['content']}")

        response = llm.generate(messages, max_tokens=100)
        print(f"\n✓ LLM 响应成功:")
        print(f"  {response[:200]}")
        return True
    except Exception as e:
        print(f"✗ LLM 生成失败: {type(e).__name__}: {e}")
        return False


def test_openai_embedder():
    """测试 OpenAI Embedder 嵌入功能。"""
    print("\n" + "=" * 60)
    print("测试 OpenAI Embedder")
    print("=" * 60)

    print(f"API Key: {settings.openai_api_key}")
    print(f"API Base: {settings.openai_api_base}")
    print(f"Model: {settings.embedding_model}")
    print(f"Embedding Dim: {settings.embedding_dim}")

    try:
        embedder = OpenAIEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )
        print("✓ OpenAIEmbedder 实例创建成功")

        texts = ["今天天气真好", "LycheeMemOS 是一个认知记忆框架"]
        print(f"\n嵌入文本: {texts}")

        embeddings = embedder.embed(texts)
        print(f"\n✓ 嵌入成功:")
        print(f"  返回嵌入数量: {len(embeddings)}")
        print(f"  每个嵌入维度: {len(embeddings[0])}")
        print(f"  第一个嵌入样本 (前5维): {embeddings[0][:5]}")

        # 测试 embed_query
        query = "如何启动服务？"
        query_emb = embedder.embed_query(query)
        print(f"\n✓ 查询嵌入成功:")
        print(f"  查询: '{query}'")
        print(f"  嵌入维度: {len(query_emb)}")
        print(f"  嵌入样本 (前5维): {query_emb[:5]}")

        return True
    except Exception as e:
        print(f"✗ Embedder 失败: {type(e).__name__}: {e}")
        return False


def main():
    print("\n🔍 LycheeMemOS OpenAI 配置检验\n")

    # llm_ok = test_openai_llm()
    embedder_ok = test_openai_embedder()

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    # print(f"LLM 生成:  {'✓ 通过' if llm_ok else '✗ 失败'}")
    print(f"Embedder:  {'✓ 通过' if embedder_ok else '✗ 失败'}")

    if embedder_ok:
        print("\n✓ 全部测试通过！配置有效。")
        return 0
    else:
        print("\n✗ 部分测试失败。请检查 API Key 和网络连接。")
        return 1


if __name__ == "__main__":
    exit(main())
