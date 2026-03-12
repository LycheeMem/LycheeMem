# A-Frame: Training-free Agentic Cognitive Memory Framework

无需训练的智能体认知记忆框架。基于认知科学四类记忆分层 + 六个认知角色 Agent + LangGraph Pipeline 编排。

## 架构

```
API Layer (FastAPI)
        │
Pipeline Layer (LangGraph StateGraph)
        │
  ┌─────┼──────────────────────────────────────┐
  │  WM Manager → Router → Search → Synthesize → Reason
  │                                                │
  │                              Consolidation Agent (后台)
  └────────────────────────────────────────────────┘
        │
Memory Substrate Layer
  ├── Working Memory     (Session + Compressor)
  ├── Graph Memory       (NetworkX / Neo4j)
  ├── Procedural Memory  (Vector Skill Store)
  └── Sensory Memory     (FIFO Buffer)
```

## 快速开始

### 安装

```bash
cd a_frame
pip install -e ".[dev]"
```

### 启动服务

```bash
# 使用 OpenAI 后端
python -m a_frame --llm openai

# 使用 Ollama 本地模型
python -m a_frame --llm ollama --port 8080

# 开发模式
python -m a_frame --reload
```

服务启动后访问 `http://localhost:8000/docs` 查看 API 文档。

### 环境变量

复制 `.env.example` 为 `.env` 并填写：

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/chat/complete` | POST | 非流式完整对话 |
| `/chat` | POST | SSE 流式对话 |
| `/memory/graph` | GET | 查看知识图谱 |
| `/memory/skills` | GET | 查看技能库 |
| `/memory/session/{id}` | GET | 查看会话历史 |
| `/memory/session/{id}` | DELETE | 删除会话 |
| `/health` | GET | 健康检查 |

### 对话示例

```bash
curl -X POST http://localhost:8000/chat/complete \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "你好，我叫Alice"}'
```

### SSE 流式

```bash
curl -N http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "你还记得我的名字吗？"}'
```

## 代码中使用

```python
from a_frame.core.factory import create_pipeline

pipeline = create_pipeline(llm=my_llm, embedder=my_embedder)
result = pipeline.run(user_query="你好", session_id="s1")
print(result["final_response"])
```

## 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
a_frame/
├── agents/          # 6 个认知 Agent
├── api/             # FastAPI 服务
├── core/            # Pipeline 编排 + 配置 + 状态
├── embedder/        # Embedding 适配器
├── llm/             # LLM 适配器 (OpenAI/Gemini/Ollama)
├── memory/          # 4 类记忆基质
│   ├── working/     #   工作记忆 (会话 + 压缩器)
│   ├── graph/       #   语义情景图谱
│   ├── procedural/  #   程序记忆 (技能库)
│   └── sensory/     #   感觉记忆 (缓冲区)
└── prompts/         # Agent 提示词模板
```

## 设计决策

- **压缩而非丢弃**: 双阈值预算驱动压缩，超长对话不会丢失关键状态
- **图谱双写**: 实体三元组同时写入知识图谱和向量库，支持多跳推理
- **HyDE 检索增强**: 先生成假设文档再做向量检索，提升召回率
- **LLM-as-Judge**: 整合器用 LLM 二元打分过滤不相关记忆
- **异步固化**: 对话后台提取实体和技能，不阻塞主流程

## License

Apache-2.0
