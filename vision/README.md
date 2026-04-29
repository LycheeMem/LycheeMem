# Vision Module - 视觉记忆模块

视觉记忆和多模态记忆相关代码的集合。

## 目录结构

```
vision/
├── __init__.py                 # 模块入口
├── models.py                   # 数据模型
├── visual_store.py             # 存储引擎
├── visual_extractor.py         # VLM 图片提取器
├── visual_extractor_fast.py    # 快速版提取器
├── visual_retriever.py         # 检索器
├── visual_forgetter.py         # 遗忘机制
├── multimodal_embedder.py      # 多模态嵌入器
├── multimodal_embedder_fast.py
└── api/
    └── routers/
        └── visual.py           # API 路由
```

## 使用方式

### 导入

```python
# 从 vision 目录导入
from vision import VisualExtractor, VisualStore
from vision.models import VisualMemoryRecord
```

### 配置

在 `.env` 文件中配置：

```bash
VLM_MODEL=openai/qwen-vl-max
VLM_API_KEY=your-api-key
VLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 启动

```bash
# 使用 main.py 启动（保持不变）
python main.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `models.py` | VisualMemoryRecord 数据模型 |
| `visual_store.py` | SQLite + LanceDB 存储引擎 |
| `visual_extractor.py` | VLM 图片理解和结构化提取 |
| `visual_retriever.py` | 跨模态检索（文本↔图片） |
| `visual_forgetter.py` | 基于 TTL 和重要性的遗忘机制 |
| `multimodal_embedder.py` | CLIP 多模态嵌入 |
| `api/routers/visual.py` | FastAPI 路由端点 |

## API 端点

- `GET /visual/memories` - 获取视觉记忆列表
- `GET /visual/memories/{record_id}` - 获取详情
- `GET /visual/memories/{record_id}/image` - 获取图片
- `POST /visual/search` - 搜索视觉记忆
- `GET /visual/stats` - 获取统计信息
- `DELETE /visual/memories/{record_id}` - 删除记忆

## 依赖

- Pillow
- litellm
- lancedb
- sentence-transformers (可选)
- torch (可选)
