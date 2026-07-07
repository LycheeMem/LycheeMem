"""视觉记忆提取器。"""

from __future__ import annotations

import base64
import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

SCENE_TYPES = {"screenshot", "chart", "photo", "document", "ui", "code", "other"}

VISUAL_EXTRACTION_PROMPT_FAST = """\
分析图片并返回 JSON:
{"caption":"1-2 句描述","scene_type":"screenshot|chart|photo|document|ui|code|other","entities":[{"type":"类型","name":"名称","confidence":0.9}],"structured_data":{},"importance_score":0.5,"importance_reason":"理由"}
要求：caption 简洁准确，scene_type 必选，只返回 JSON。"""

VISUAL_EXTRACTION_PROMPT_DETAILED = """\
你是一个专业的视觉内容分析器。请分析图片并返回 JSON:

```json
{
  "caption": "1-3 句简洁描述",
  "scene_type": "screenshot|chart|photo|document|ui|code|other",
  "entities": [{"type": "类型", "name": "名称", "confidence": 0.9}],
  "structured_data": {},
  "importance_score": 0.5,
  "importance_reason": "简短理由"
}
```

要求：
- caption: 简洁准确，50-100 字
- scene_type: 必须从选项中选择
- entities: 识别关键实体 (3-5 个)
- importance_score: 0.0-1.0
- 只返回 JSON，不要其他文字
"""


class VisualExtractor:
    """使用 VLM 进行图片理解和结构化提取。

    Args:
        llm: LLM 适配器（需支持多模态输入，如 qwen-vl-max）。
        fast_mode: 是否启用快速模式（更短的 Prompt 和更低的 token 限制）。
        max_image_size: 图片最大边长（像素），超过则缩放，默认 1024。
        image_quality: JPEG 压缩质量 (1-100)，默认 85。
        cache_size: LRU 缓存大小，默认 128。
    """

    def __init__(
        self,
        llm: BaseLLM,
        fast_mode: bool = True,
        max_image_size: int = 1024,
        image_quality: int = 85,
        cache_size: int = 128,
    ) -> None:
        self.llm = llm
        self.fast_mode = fast_mode
        self.max_image_size = max_image_size
        self.image_quality = image_quality
        self._prompt = VISUAL_EXTRACTION_PROMPT_FAST if fast_mode else VISUAL_EXTRACTION_PROMPT_DETAILED
        self._max_tokens = 800 if fast_mode else 1500
        self._temperature = 0.01 if fast_mode else 0.1
        self._cache_size = cache_size
        self._result_cache: dict[str, dict[str, Any]] = {}
        self._compressed_cache: dict[str, str] = {}
        logger.info(
            "VisualExtractor initialized: fast_mode=%s, max_image_size=%d, max_tokens=%d, cache_size=%d",
            fast_mode, max_image_size, self._max_tokens, cache_size,
        )

    async def extract_from_base64(
        self,
        image_b64: str,
        mime_type: str = "image/jpeg",
        session_id: str = "",
    ) -> dict[str, Any]:
        """从 Base64 编码的图片提取结构化信息。

        Args:
            image_b64: Base64 编码的图片数据。
            mime_type: 图片 MIME 类型。
            session_id: 来源会话 ID（用于日志追踪）。

        Returns:
            提取结果的字典，包含 caption, entities, scene_type, structured_data,
            importance_score, image_hash 等字段。
        """
        image_bytes = base64.b64decode(image_b64)
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

        logger.info(
            "Extracting visual memory: hash=%s, size=%d bytes, session=%s",
            image_hash, len(image_bytes), session_id,
        )

        if image_hash in self._result_cache:
            logger.debug("Cache hit for image hash: %s", image_hash)
            cached_result = self._result_cache[image_hash].copy()
            cached_result["from_cache"] = True
            return cached_result

        compressed_b64 = self._compress_image(image_bytes, mime_type, image_hash)
        if compressed_b64:
            logger.debug(
                "Image compressed: %d -> %d bytes (%.1f%%)",
                len(image_bytes), len(compressed_b64) * 3 // 4,
                (1 - len(compressed_b64) * 3 / 4 / len(image_bytes)) * 100,
            )
        else:
            compressed_b64 = image_b64

        result = await self._call_vlm(compressed_b64, mime_type)

        result["image_hash"] = image_hash
        result["image_mime_type"] = mime_type
        result["image_size"] = len(image_bytes)
        result["from_cache"] = False

        result = self._validate_and_fix(result)

        self._update_cache(image_hash, result, compressed_b64)

        logger.info(
            "Visual extraction completed: scene_type=%s, importance=%.2f, caption_len=%d",
            result["scene_type"], result["importance_score"], len(result.get("caption", "")),
        )

        return result

    def _update_cache(self, image_hash: str, result: dict[str, Any], compressed_b64: str) -> None:
        """更新缓存，保持缓存大小在限制内。

        Args:
            image_hash: 图片哈希
            result: 提取结果
            compressed_b64: 压缩后的图片
        """
        if len(self._result_cache) >= self._cache_size:
            remove_count = self._cache_size // 4
            for key in list(self._result_cache.keys())[:remove_count]:
                del self._result_cache[key]
                if key in self._compressed_cache:
                    del self._compressed_cache[key]

        self._result_cache[image_hash] = result.copy()
        if compressed_b64:
            self._compressed_cache[image_hash] = compressed_b64

    def _compress_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        image_hash: str = "",
    ) -> Optional[str]:
        """压缩/缩放图片，减少 VLM 处理时间（带缓存）。

        Args:
            image_bytes: 原始图片字节数据。
            mime_type: 图片 MIME 类型。
            image_hash: 图片哈希（用于缓存查找）。

        Returns:
            Base64 编码的压缩图片，如果压缩失败返回 None。
        """
        if image_hash and image_hash in self._compressed_cache:
            logger.debug("Compression cache hit for hash: %s", image_hash)
            return self._compressed_cache[image_hash]

        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_bytes))

            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            width, height = img.size
            max_dim = max(width, height)

            if max_dim > self.max_image_size:
                scale = self.max_image_size / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug("Image resized: %dx%d -> %dx%d", width, height, new_width, new_height)

            output = io.BytesIO()
            img.save(output, format="JPEG", quality=self.image_quality, optimize=True)
            output.seek(0)

            return base64.b64encode(output.getvalue()).decode("utf-8")

        except ImportError:
            logger.warning("PIL not available, skipping image compression")
            return None
        except Exception as e:
            logger.warning("Image compression failed: %s", e)
            return None

    async def _call_vlm(self, image_b64: str, mime_type: str) -> dict[str, Any]:
        """调用 VLM 进行图片理解。

        Args:
            image_b64: Base64 编码的图片。
            mime_type: MIME 类型。

        Returns:
            VLM 返回的解析结果字典。
        """
        import asyncio
        import json
        import re

        if mime_type not in SUPPORTED_MIME_TYPES:
            logger.warning("Unsupported MIME type: %s, defaulting to image/jpeg", mime_type)
            mime_type = "image/jpeg"

        messages = [
            {
                "role": "system",
                "content": self._prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": "分析图片，返回 JSON。"},
                ],
            },
        ]

        try:
            logger.info("Calling VLM with model: %s", getattr(self.llm, 'model', 'unknown'))
            response = await asyncio.wait_for(
                self.llm.agenerate(
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                ),
                timeout=30,
            )
            logger.info("VLM response received, length: %d chars", len(response))
        except asyncio.TimeoutError:
            logger.error("VLM call timeout after 30s")
            return self._get_fallback_result("VLM 响应超时")
        except Exception as e:
            logger.error("VLM call failed: %s", e, exc_info=True)
            raise RuntimeError(f"VLM API call failed: {e}") from e

        text = response.strip()
        logger.debug("VLM raw response: %s", text[:500])

        # 提取 JSON 块（支持多种格式）
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        else:
            text = text.strip()

        try:
            result = json.loads(text)
            return result
        except json.JSONDecodeError as e:
            logger.warning("VLM returned invalid JSON: %s. Response: %s", str(e)[:100], text[:200])
            
            # 尝试提取 JSON 片段（处理 VLM 返回额外文本的情况）
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                try:
                    result = json.loads(json_str)
                    logger.info("Successfully extracted JSON from partial text")
                    return result
                except json.JSONDecodeError as e2:
                    logger.warning("Partial JSON extraction also failed: %s", str(e2)[:100])
            
            # 尝试修复常见的 JSON 格式问题
            fixed_text = text.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
            try:
                result = json.loads(fixed_text)
                logger.info("Successfully parsed JSON after fixing common issues")
                return result
            except json.JSONDecodeError:
                pass
            
            return self._get_fallback_result(f"JSON 解析失败")

        return result

    def _get_fallback_result(self, error_msg: str) -> dict[str, Any]:
        """返回降级的默认结果。

        Args:
            error_msg: 错误信息

        Returns:
            降级结果字典
        """
        return {
            "caption": f"[{error_msg}]",
            "scene_type": "other",
            "entities": [],
            "structured_data": {},
            "importance_score": 0.3,
            "importance_reason": "解析失败",
        }

    def _validate_and_fix(self, result: dict[str, Any]) -> dict[str, Any]:
        """验证和修正提取结果，确保字段完整性。

        Args:
            result: VLM 返回的原始结果。

        Returns:
            修正后的结果字典。
        """
        required_fields = {
            "caption": "",
            "scene_type": "other",
            "entities": [],
            "structured_data": {},
            "importance_score": 0.5,
            "importance_reason": "",
        }

        for field, default in required_fields.items():
            result.setdefault(field, default)

        if result["scene_type"] not in SCENE_TYPES:
            result["scene_type"] = "other"

        score = result.get("importance_score", 0.5)
        try:
            score = float(score)
            result["importance_score"] = max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            result["importance_score"] = 0.5

        if not isinstance(result.get("entities"), list):
            result["entities"] = []

        if not isinstance(result.get("structured_data"), dict):
            result["structured_data"] = {}

        return result
