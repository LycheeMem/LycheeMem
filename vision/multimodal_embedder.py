"""多模态嵌入器 - 高性能版。

使用 CLIP 风格的模型实现统一的文本 - 图像嵌入空间。
支持：
- 文本嵌入
- 图像嵌入
- 跨模态相似度计算

优化:
- LRU 缓存避免重复计算
- 批量处理提高效率
- 图片预加载和缓存

注意：
- 如果无法访问 HuggingFace，可使用 DashScope API 作为替代
- 配置 USE_MULTIMODAL_EMBEDDING=False 时自动降级为文本嵌入
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from PIL import Image

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.debug("sentence-transformers not installed")

# 尝试导入 DashScope（阿里云）
try:
    import dashscope
    from dashscope import ImageEmbedding
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger.debug("dashscope not installed")


class MultimodalEmbedder:
    """多模态嵌入器（高性能版）。

    支持两种后端：
    1. sentence-transformers (CLIP 模型，本地运行)
    2. DashScope API (阿里云，需要网络)

    使用 CLIP 风格模型将文本和图像映射到同一向量空间。
    支持跨模态相似度计算（文本查询→图像检索，图像查询→文本检索）。

    Attributes:
        model: SentenceTransformer 模型实例（延迟加载）
        model_name: 模型名称
        device: 运行设备 (cuda/cpu)
        embedding_dim: 嵌入维度
        use_dashscope: 是否使用 DashScope API
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        lazy_load: bool = True,  # 延迟加载模型
        cache_size: int = 1000,  # LRU 缓存大小
        use_dashscope: bool = False,
        dashscope_api_key: Optional[str] = None,
    ) -> None:
        """初始化多模态嵌入器。

        Args:
            model_name: HuggingFace 模型名称。选项：
                - openai/clip-vit-base-patch32: 512 维，快速，良好质量
                - openai/clip-vit-large-patch14: 768 维，较慢，更好质量
                - BAAI/bge-visual: 中文优化的多模态模型
            device: 运行设备 ('cuda', 'cpu', 或 None 自动选择)
            lazy_load: 是否延迟加载模型（默认 True，首次使用时加载）
            cache_size: LRU 缓存大小（默认 1000）
            use_dashscope: 是否使用 DashScope API（默认 False）
            dashscope_api_key: DashScope API 密钥（可选）
        """
        self.use_dashscope = use_dashscope and DASHSCOPE_AVAILABLE
        self.dashscope_api_key = dashscope_api_key
        
        if self.use_dashscope:
            logger.info("Using DashScope API for multimodal embedding")
            if dashscope_api_key:
                dashscope.api_key = dashscope_api_key
            self._embedding_dim = 1024  # DashScope 默认维度
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers is required for local multimodal embedding. "
                "Install with: pip install sentence-transformers"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self._model = None  # 延迟加载
        self._embedding_dim = None
        self.lazy_load = lazy_load
        self.cache_size = cache_size

        # LRU 缓存：hash -> embedding
        self._text_cache: dict[str, List[float]] = {}
        self._image_cache: dict[str, List[float]] = {}
        # 图片预加载缓存：path -> PIL Image
        self._image_cache_pil: dict[str, Image.Image] = {}

        if not lazy_load:
            self._load_model()
        else:
            logger.info(
                "MultimodalEmbedder initialized (lazy load): model=%s, device=%s, cache_size=%d",
                model_name, device, cache_size,
            )

    @property
    def embedding_dim(self) -> int:
        """获取嵌入维度（延迟加载）。"""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim

    def _load_model(self) -> None:
        """延迟加载模型。"""
        if self._model is not None:
            return

        logger.info("Loading multimodal model: %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Multimodal model loaded: dim=%d",
            self._embedding_dim,
        )
    
    @property
    def model(self):
        """模型属性（延迟加载）。"""
        if self._model is None:
            self._load_model()
        return self._model

    def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        """将文本嵌入为向量（带缓存）。

        Args:
            text: 文本字符串
            normalize: 是否 L2 归一化

        Returns:
            嵌入向量（列表形式）
        """
        # 使用 DashScope API
        if self.use_dashscope:
            cache_key = f"dashscope_text:{text}:{normalize}"
            text_hash = hash(cache_key)
            if text_hash in self._text_cache:
                return self._text_cache[text_hash].copy()

            try:
                # 使用 text-embedding-v3 模型
                from src.core.config import settings
                response = dashscope.TextEmbedding.call(
                    model="text-embedding-v3",
                    input=text,
                    api_key=self.dashscope_api_key or settings.embedding_api_key,
                )
                embedding = response.output["embeddings"][0]["embedding"]
                self._text_cache[text_hash] = embedding
                return embedding
            except Exception as e:
                logger.warning("DashScope text embedding failed: %s", e)
                # 返回零向量作为降级
                return [0.0] * self._embedding_dim

        # 使用本地模型
        cache_key = f"{text}:{normalize}"
        text_hash = hash(cache_key)
        if text_hash in self._text_cache:
            logger.debug("Text embedding cache hit")
            return self._text_cache[text_hash].copy()

        # 生成嵌入
        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )
        result = embedding.cpu().tolist()

        # 更新缓存（带大小限制）
        if len(self._text_cache) >= self.cache_size:
            remove_count = self.cache_size // 4
            for key in list(self._text_cache.keys())[:remove_count]:
                del self._text_cache[key]
        self._text_cache[text_hash] = result

        return result

    def embed_image(
        self,
        image: Union[str, Path, Image.Image],
        normalize: bool = True
    ) -> List[float]:
        """将图像嵌入为向量（与文本同一空间，带缓存）。

        Args:
            image: 图像路径（str/Path）或 PIL Image 对象
            normalize: 是否 L2 归一化

        Returns:
            嵌入向量（列表形式）
        """
        # 使用 DashScope API
        if self.use_dashscope:
            cache_key = f"dashscope_image:{str(image)}:{normalize}"
            image_hash = hash(cache_key)
            if image_hash in self._image_cache:
                logger.debug("DashScope image embedding cache hit")
                return self._image_cache[image_hash].copy()

            try:
                # 获取图片路径
                if isinstance(image, (str, Path)):
                    image_path = str(image)
                elif isinstance(image, Image.Image):
                    # 保存到临时文件
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        image.save(tmp, format='JPEG')
                        image_path = tmp.name
                else:
                    raise ValueError("Image must be a path or PIL Image object")

                # 使用阿里云视觉嵌入 API
                from src.core.config import settings
                response = dashscope.ImageEmbedding.call(
                    model="image-embedding-v1",
                    image=image_path,
                    api_key=self.dashscope_api_key or settings.llm_api_key,
                )
                embedding = response.output["embeddings"][0]["embedding"]
                self._image_cache[image_hash] = embedding
                
                if isinstance(image, Image.Image):
                    Path(image_path).unlink(missing_ok=True)
                    
                return embedding
            except Exception as e:
                logger.warning("DashScope image embedding failed: %s", e)
                # 返回零向量作为降级
                return [0.0] * self._embedding_dim

        # 使用本地 CLIP 模型
        # 转换为 PIL Image
        if isinstance(image, (str, Path)):
            image_path = str(image)
            # 检查图片缓存
            if image_path in self._image_cache_pil:
                logger.debug("Image PIL cache hit: %s", image_path)
                pil_image = self._image_cache_pil[image_path]
            else:
                logger.debug("Loading image: %s", image_path)
                pil_image = Image.open(image_path).convert("RGB")
                # 更新图片缓存
                if len(self._image_cache_pil) >= self.cache_size:
                    remove_count = self.cache_size // 4
                    for key in list(self._image_cache_pil.keys())[:remove_count]:
                        del self._image_cache_pil[key]
                self._image_cache_pil[image_path] = pil_image
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a path or PIL Image object")

        # 计算缓存 key（使用图片路径或 id）
        if isinstance(image, (str, Path)):
            cache_key = f"{image_path}:{normalize}"
        else:
            cache_key = f"id({id(image)}):{normalize}"
        image_hash = hash(cache_key)

        # 检查嵌入缓存
        if image_hash in self._image_cache:
            logger.debug("Image embedding cache hit")
            return self._image_cache[image_hash].copy()

        # 生成嵌入
        embedding = self.model.encode(
            pil_image,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )
        result = embedding.cpu().tolist()

        # 更新缓存
        if len(self._image_cache) >= self.cache_size:
            remove_count = self.cache_size // 4
            for key in list(self._image_cache.keys())[:remove_count]:
                del self._image_cache[key]
        self._image_cache[image_hash] = result

        return result
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """计算两个嵌入向量的余弦相似度。
        
        由于嵌入已归一化，这等同于点积。
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
            
        Returns:
            余弦相似度分数（归一化嵌入为 0-1）
        """
        import numpy as np
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # 对于归一化嵌入，点积 = 余弦相似度
        similarity = np.dot(vec1, vec2)
        
        # 截断处理数值误差
        return float(np.clip(similarity, -1.0, 1.0))
    
    def batch_embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """批量嵌入多个文本。
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            show_progress: 是否显示进度条
            
        Returns:
            嵌入向量列表
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        return embeddings.cpu().tolist()
    
    def batch_embed_images(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """批量嵌入多个图像。
        
        Args:
            images: 图像列表（路径或 PIL Image）
            batch_size: 批次大小
            show_progress: 是否显示进度条
            
        Returns:
            嵌入向量列表
        """
        # 加载图像（如果是路径）
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError(f"Invalid image type: {type(img)}")
        
        embeddings = self.model.encode(
            pil_images,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        return embeddings.cpu().tolist()
