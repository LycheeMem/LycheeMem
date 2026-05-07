"""本地 SentenceTransformer Embedder。

通过 sentence-transformers 库在本地运行 HuggingFace 模型，无需调用远程 API。
适合离线场景或使用 Qwen3-Embedding、BGE、all-MiniLM 等开源模型。

使用方式（.env）：
  EMBEDDING_LOCAL=true
  EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
  # 可选：
  EMBEDDING_DEVICE=auto   # auto / cpu / cuda / mps
  EMBEDDING_DIM=1024      # 若模型支持 Matryoshka 则可截断维度；0 = 自动

查询侧 prompt（如 Qwen3 的 "query" prompt）会在模型加载后自动检测并启用，无需手动配置。
注意：首次运行会从 HuggingFace Hub 下载模型，可预先设置 HF_HOME 或 HF_ENDPOINT。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.embedder.base import BaseEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """使用 sentence-transformers 在本地运行 Embedding 模型。

    参数
    ----
    model_name : HuggingFace 模型路径，如 "Qwen/Qwen3-Embedding-0.6B"
    device     : 推理设备，"auto" / "cpu" / "cuda" / "mps"，默认 auto
    normalize  : 是否对输出向量做 L2 归一化，默认 True
    dimensions : 截断到指定维度（Matryoshka 模型支持）；0 表示不截断
    trust_remote_code : 传给 SentenceTransformer 的 trust_remote_code 参数
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        normalize: bool = True,
        dimensions: int = 0,
        trust_remote_code: bool = True,
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._query_prompt_name: str | None = None  # 加载后自动检测
        self._normalize = normalize
        self._dimensions = dimensions
        self._trust_remote_code = trust_remote_code
        self._model_kwargs = model_kwargs
        self._model = None  # 懒加载

    # ── 懒加载 ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers 未安装。请执行：pip install sentence-transformers"
            ) from exc

        logger.info("加载本地 Embedding 模型: %s (device=%s)", self.model_name, self._device)
        t0 = time.perf_counter()

        # 解析 device
        device = self._device if self._device != "auto" else None  # None 让 ST 自动选

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
        }
        if device:
            load_kwargs["device"] = device
        if self._model_kwargs:
            load_kwargs["model_kwargs"] = self._model_kwargs

        # 尝试优化加载（flash_attention_2），失败则降级
        if self._device in ("auto", "cuda"):
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                    processor_kwargs={"padding_side": "left"},
                    trust_remote_code=self._trust_remote_code,
                )
                logger.info("已启用 flash_attention_2 优化")
            except Exception as e:
                logger.debug("flash_attention_2 不可用（%s），使用标准加载", e)
                self._model = SentenceTransformer(self.model_name, **load_kwargs)
        else:
            self._model = SentenceTransformer(self.model_name, **load_kwargs)

        elapsed = time.perf_counter() - t0
        dim = self._model.get_embedding_dimension()
        logger.info("模型加载完成，维度=%s，耗时=%.1fs", dim, elapsed)

        # 自动检测模型是否内置 query prompt（Qwen3 等非对称模型）
        prompts = getattr(self._model, "prompts", {}) or {}
        if "query" in prompts:
            self._query_prompt_name = "query"
            logger.info("检测到 query prompt，查询侧将自动使用")

    # ── 核心接口 ─────────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文档 embedding（文档侧）。"""
        if not texts:
            return []
        self._load()
        t0 = time.perf_counter()
        vecs = self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=self._normalize,
            truncate_dim=self._dimensions if self._dimensions > 0 else None,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        self._accumulate_usage(len(texts), 0, latency_ms)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        """单条查询 embedding（查询侧，使用 query prompt）。"""
        self._load()
        t0 = time.perf_counter()
        encode_kwargs: dict[str, Any] = {
            "show_progress_bar": False,
            "normalize_embeddings": self._normalize,
        }
        if self._dimensions > 0:
            encode_kwargs["truncate_dim"] = self._dimensions
        if self._query_prompt_name:
            encode_kwargs["prompt_name"] = self._query_prompt_name

        vecs = self._model.encode([text], **encode_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        self._accumulate_usage(1, 0, latency_ms)
        return vecs[0].tolist()

    @property
    def dimension(self) -> int:
        """实际输出维度（懒加载后可用）。"""
        self._load()
        dim = self._model.get_embedding_dimension()
        if self._dimensions > 0:
            return min(self._dimensions, dim)
        return dim
