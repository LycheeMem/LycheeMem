"""快速多模态嵌入器。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from PIL import Image

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.debug("sentence-transformers not installed")

try:
    import dashscope
    from dashscope import TextEmbedding
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger.debug("dashscope not installed")


class MultimodalEmbedderFast:
    """支持本地 CLIP 模型和 DashScope API 的多模态嵌入器。"""

    RECOMMENDED_MODELS = {
        "fastest": "openai/clip-vit-base-patch32",      # 512 维，最快
        "fast": "openai/clip-vit-base-patch16",         # 512 维，较快
        "balanced": "openai/clip-vit-large-patch14",    # 768 维，平衡
        "quality": "BAAI/bge-visual",                   # 中文优化
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        lazy_load: bool = True,
        cache_size: int = 2000,
        use_dashscope: bool = False,
        dashscope_api_key: Optional[str] = None,
        use_fp16: bool = True,
        batch_size: int = 64,
    ) -> None:
        """初始化多模态嵌入器。

        Args:
            model_name: HuggingFace 模型名称。选项：
                - openai/clip-vit-base-patch32: 512 维，快速，良好质量（推荐）
                - openai/clip-vit-base-patch16: 512 维，较快
                - openai/clip-vit-large-patch14: 768 维，较慢，更好质量
                - BAAI/bge-visual: 中文优化的多模态模型
            device: 运行设备 ('cuda', 'cpu', 或 None 自动选择)
            lazy_load: 是否延迟加载模型（默认 True）
            cache_size: LRU 缓存大小（默认 2000）
            use_dashscope: 是否使用 DashScope API（默认 False）
            dashscope_api_key: DashScope API 密钥
            use_fp16: 是否使用 FP16 半精度（默认 True，GPU 加速）
            batch_size: 批量处理大小（默认 64）
        """
        if model_name is None:
            model_name = self.RECOMMENDED_MODELS["fastest"]
        
        self.use_dashscope = use_dashscope and DASHSCOPE_AVAILABLE
        self.dashscope_api_key = dashscope_api_key
        self.use_fp16 = use_fp16
        self.batch_size = batch_size

        if self.use_dashscope:
            logger.info("Using DashScope API for multimodal embedding")
            if dashscope_api_key:
                dashscope.api_key = dashscope_api_key
            self._embedding_dim = 1024
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
        self._model = None
        self._embedding_dim = None
        self.lazy_load = lazy_load
        self.cache_size = cache_size

        self._text_cache: dict[str, List[float]] = {}
        self._image_cache: dict[str, List[float]] = {}
        self._image_cache_pil: dict[str, Image.Image] = {}

        if not lazy_load:
            self._load_model()
        else:
            logger.info(
                "MultimodalEmbedderFast initialized: model=%s, device=%s, fp16=%s, batch=%d, cache=%d",
                model_name, device, use_fp16, batch_size, cache_size,
            )

    @property
    def embedding_dim(self) -> int:
        """获取嵌入维度（延迟加载）。"""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim

    def _load_model(self) -> None:
        """延迟加载模型（支持 FP16 加速）。"""
        if self._model is not None:
            return

        logger.info("Loading multimodal model: %s on %s", self.model_name, self.device)
        
        if self.use_fp16 and self.device == "cuda":
            logger.info("Using FP16 half-precision for faster inference")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16}
            )
        else:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info("Multimodal model loaded: dim=%d, device=%s", self._embedding_dim, self.device)

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
        if self.use_dashscope:
            cache_key = f"dashscope_text:{text}:{normalize}"
            text_hash = hash(cache_key)
            if text_hash in self._text_cache:
                return self._text_cache[text_hash].copy()

            try:
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
                return [0.0] * self._embedding_dim

        cache_key = f"{text}:{normalize}"
        text_hash = hash(cache_key)
        if text_hash in self._text_cache:
            logger.debug("Text embedding cache hit")
            return self._text_cache[text_hash].copy()

        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )
        result = embedding.cpu().tolist()

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
        if self.use_dashscope:
            cache_key = f"dashscope_image:{str(image)}:{normalize}"
            image_hash = hash(cache_key)
            if image_hash in self._image_cache:
                logger.debug("DashScope image embedding cache hit")
                return self._image_cache[image_hash].copy()

            try:
                from src.core.config import settings
                import tempfile
                
                if isinstance(image, (str, Path)):
                    image_path = str(image)
                elif isinstance(image, Image.Image):
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        image.save(tmp, format='JPEG')
                        image_path = tmp.name
                else:
                    raise ValueError("Image must be a path or PIL Image object")

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
                return [0.0] * self._embedding_dim

        if isinstance(image, (str, Path)):
            image_path = str(image)
            if image_path in self._image_cache_pil:
                logger.debug("Image PIL cache hit: %s", image_path)
                pil_image = self._image_cache_pil[image_path]
            else:
                logger.debug("Loading image: %s", image_path)
                pil_image = Image.open(image_path).convert("RGB")
                if len(self._image_cache_pil) >= self.cache_size:
                    remove_count = self.cache_size // 4
                    for key in list(self._image_cache_pil.keys())[:remove_count]:
                        del self._image_cache_pil[key]
                self._image_cache_pil[image_path] = pil_image
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a path or PIL Image object")

        if isinstance(image, (str, Path)):
            cache_key = f"{image_path}:{normalize}"
        else:
            cache_key = f"id({id(image)}):{normalize}"
        image_hash = hash(cache_key)

        if image_hash in self._image_cache:
            logger.debug("Image embedding cache hit")
            return self._image_cache[image_hash].copy()

        embedding = self.model.encode(
            pil_image,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )
        result = embedding.cpu().tolist()

        if len(self._image_cache) >= self.cache_size:
            remove_count = self.cache_size // 4
            for key in list(self._image_cache.keys())[:remove_count]:
                del self._image_cache[key]
        self._image_cache[image_hash] = result

        return result

    def batch_embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """批量嵌入多个文本。

        Args:
            texts: 文本列表
            batch_size: 批次大小（默认使用实例的 batch_size）
            show_progress: 是否显示进度条

        Returns:
            嵌入向量列表
        """
        if batch_size is None:
            batch_size = self.batch_size
            
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
        batch_size: Optional[int] = None,
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
        if batch_size is None:
            batch_size = self.batch_size
            
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

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """计算两个嵌入向量的余弦相似度。"""
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        similarity = np.dot(vec1, vec2)
        return float(np.clip(similarity, -1.0, 1.0))
