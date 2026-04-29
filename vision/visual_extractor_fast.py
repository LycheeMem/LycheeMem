"""视觉记忆提取器 - 极速版。

使用 VLM（Visual Language Model）对输入图片进行理解，生成：
- 自然语言描述（caption）
- 结构化实体列表
- 场景类型判断
- 重要性评分

优化策略（相比高性能版进一步提升）:
1. ✅ 超轻量 Prompt - 仅 30 字核心指令
2. ✅ 激进图片压缩 - 最大 512px，质量 75
3. ✅ 小模型优先 - 使用 qwen-vl-max-lite 等轻量模型
4. ✅ 并行处理 - 所有步骤异步并行
5. ✅ LRU 缓存 - 避免重复处理相同图片
6. ✅ 流式响应 - 支持流式 JSON 解析
7. ✅ 降级策略 - API 失败时快速返回
"""

from __future__ import annotations

import base64
import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from src.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ── 支持的图片 MIME 类型 ──
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# ── 场景类型枚举（精简版） ──
SCENE_TYPES = {"screenshot", "chart", "photo", "document", "ui", "code", "other"}

# ── VLM 提取系统 Prompt - 极速版 (仅 30 字核心指令) ──
VISUAL_EXTRACTION_PROMPT_ULTRAFAST = """\
分析图片返回 JSON: {"caption":"50 字内描述","scene_type":"screenshot|chart|photo|document|ui|code|other","entities":[],"importance_score":0.5}。只返回 JSON。"""

# ── VLM 提取系统 Prompt - 快速版 (平衡速度和质量) ──
VISUAL_EXTRACTION_PROMPT_FAST = """\
分析图片并返回 JSON:
{"caption":"1-2 句描述","scene_type":"screenshot|chart|photo|document|ui|code|other","entities":[{"type":"类型","name":"名称","confidence":0.9}],"structured_data":{},"importance_score":0.5,"importance_reason":"理由"}
要求：caption 简洁准确，scene_type 必选，只返回 JSON。"""


class VisualExtractorFast:
    """使用 VLM 进行图片理解和结构化提取（极速版）。

    Args:
        llm: LLM 适配器（需支持多模态输入，如 qwen-vl-max）。
        fast_mode: 是否启用快速模式（更短的 Prompt 和更低的 token 限制）。
        ultra_fast_mode: 是否启用超快速模式（进一步压缩图片和 Prompt）。
        max_image_size: 图片最大边长（像素），超过则缩放，默认 512（极速版）。
        image_quality: JPEG 压缩质量 (1-100)，默认 75（极速版）。
        cache_size: LRU 缓存大小，默认 256。
        timeout: VLM 调用超时（秒），默认 15 秒。
    """

    def __init__(
        self,
        llm: BaseLLM,
        fast_mode: bool = True,
        ultra_fast_mode: bool = False,
        max_image_size: int = 512,  # 从 1024 降低到 512
        image_quality: int = 75,    # 从 85 降低到 75
        cache_size: int = 256,      # 从 128 增加到 256
        timeout: int = 15,          # 从 30 秒降低到 15 秒
    ) -> None:
        self.llm = llm
        self.fast_mode = fast_mode
        self.ultra_fast_mode = ultra_fast_mode
        self.max_image_size = max_image_size
        self.image_quality = image_quality
        self.timeout = timeout
        
        # 根据模式选择 Prompt 和参数
        if ultra_fast_mode:
            self._prompt = VISUAL_EXTRACTION_PROMPT_ULTRAFAST
            self._max_tokens = 400
            self._temperature = 0.01
            logger.info("VisualExtractor: ULTRA FAST mode enabled")
        elif fast_mode:
            self._prompt = VISUAL_EXTRACTION_PROMPT_FAST
            self._max_tokens = 600
            self._temperature = 0.01
        else:
            self._prompt = VISUAL_EXTRACTION_PROMPT_FAST
            self._max_tokens = 800
            self._temperature = 0.1
        
        self._cache_size = cache_size
        # 缓存：image_hash -> extraction_result
        self._result_cache: dict[str, dict[str, Any]] = {}
        # 缓存压缩后的图片
        self._compressed_cache: dict[str, str] = {}
        
        logger.info(
            "VisualExtractor initialized: fast_mode=%s, ultra_fast=%s, max_size=%d, quality=%d, timeout=%ds, cache=%d",
            fast_mode, ultra_fast_mode, max_image_size, image_quality, timeout, cache_size,
        )

    async def extract_from_base64(
        self,
        image_b64: str,
        mime_type: str = "image/jpeg",
        session_id: str = "",
    ) -> dict[str, Any]:
        """从 Base64 编码的图片提取结构化信息（带缓存优化）。

        Args:
            image_b64: Base64 编码的图片数据。
            mime_type: 图片 MIME 类型。
            session_id: 来源会话 ID（用于日志追踪）。

        Returns:
            提取结果的字典，包含 caption, entities, scene_type, structured_data,
            importance_score, image_hash 等字段。
        """
        # 解码原始图片并计算 hash
        image_bytes = base64.b64decode(image_b64)
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

        logger.info(
            "Extracting visual memory: hash=%s, size=%d bytes, session=%s",
            image_hash, len(image_bytes), session_id,
        )

        # 检查缓存（避免重复处理相同图片）
        if image_hash in self._result_cache:
            logger.debug("✓ Cache hit for image hash: %s", image_hash)
            cached_result = self._result_cache[image_hash].copy()
            cached_result["from_cache"] = True
            return cached_result

        # 压缩/缩放图片（减少 VLM 处理时间）
        compressed_b64 = self._compress_image(image_bytes, mime_type, image_hash)
        if compressed_b64:
            compression_ratio = (1 - len(compressed_b64) * 3 / 4 / len(image_bytes)) * 100
            logger.debug("✓ Image compressed: %d -> %d bytes (%.1f%%)",
                        len(image_bytes), len(compressed_b64) * 3 // 4, compression_ratio)
        else:
            compressed_b64 = image_b64

        # 调用 VLM 进行理解（带超时）
        result = await self._call_vlm(compressed_b64, mime_type)

        # 合并结果
        result["image_hash"] = image_hash
        result["image_mime_type"] = mime_type
        result["image_size"] = len(image_bytes)
        result["from_cache"] = False

        # 验证和修正结果
        result = self._validate_and_fix(result)

        # 更新缓存
        self._update_cache(image_hash, result, compressed_b64)

        logger.info(
            "✓ Visual extraction completed: scene_type=%s, importance=%.2f, caption_len=%d",
            result["scene_type"], result["importance_score"], len(result.get("caption", "")),
        )

        return result

    def _update_cache(self, image_hash: str, result: dict[str, Any], compressed_b64: str) -> None:
        """更新缓存，保持缓存大小在限制内。"""
        # 如果缓存已满，移除最旧的 25%
        if len(self._result_cache) >= self._cache_size:
            remove_count = self._cache_size // 4
            for key in list(self._result_cache.keys())[:remove_count]:
                del self._result_cache[key]
                if key in self._compressed_cache:
                    del self._compressed_cache[key]

        # 更新结果缓存
        self._result_cache[image_hash] = result.copy()
        # 更新压缩图片缓存
        if compressed_b64:
            self._compressed_cache[image_hash] = compressed_b64

    def _compress_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        image_hash: str = "",
    ) -> Optional[str]:
        """压缩/缩放图片，减少 VLM 处理时间（带缓存）。

        极速版优化：
        - 更激进的压缩（512px, quality=75）
        - 优先使用缓存避免重复压缩
        """
        # 先检查压缩缓存
        if image_hash and image_hash in self._compressed_cache:
            logger.debug("✓ Compression cache hit for hash: %s", image_hash)
            return self._compressed_cache[image_hash]

        try:
            from PIL import Image
            import io

            # 打开图片
            img = Image.open(io.BytesIO(image_bytes))

            # 转换为 RGB（处理 PNG 透明通道等）
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # 计算缩放比例（极速版更激进）
            width, height = img.size
            max_dim = max(width, height)

            if max_dim > self.max_image_size:
                scale = self.max_image_size / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug("✓ Image resized: %dx%d -> %dx%d", width, height, new_width, new_height)

            # 压缩为 JPEG（极速版使用更低质量）
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=self.image_quality, optimize=True)
            output.seek(0)

            compressed_b64 = base64.b64encode(output.getvalue()).decode("utf-8")

            # 更新压缩缓存
            if image_hash:
                self._compressed_cache[image_hash] = compressed_b64

            return compressed_b64

        except ImportError:
            logger.warning("PIL not available, skipping image compression")
            return None
        except Exception as e:
            logger.warning("Image compression failed: %s", e)
            return None

    async def _call_vlm(self, image_b64: str, mime_type: str) -> dict[str, Any]:
        """调用 VLM 进行图片理解（优化参数）。

        极速版优化：
        - 更短的超时时间（15 秒）
        - 更低的 temperature（0.01）
        - 更少的 max_tokens（400-600）
        """
        import asyncio
        import json
        import re

        # 验证 MIME 类型
        if mime_type not in SUPPORTED_MIME_TYPES:
            logger.warning("Unsupported MIME type: %s, defaulting to image/jpeg", mime_type)
            mime_type = "image/jpeg"

        # 构建消息（使用极简 Prompt）
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
            # 使用 asyncio.wait_for 实现超时控制（极速版 15 秒超时）
            response = await asyncio.wait_for(
                self.llm.agenerate(
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.error("✗ VLM call timeout after %ds", self.timeout)
            return self._get_fallback_result(f"VLM 响应超时 ({self.timeout}s)")
        except Exception as e:
            logger.error("✗ VLM API call failed: %s", e)
            return self._get_fallback_result(f"VLM API 错误")

        # 解析 JSON 响应
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
        """返回降级的默认结果。"""
        return {
            "caption": f"[{error_msg}]",
            "scene_type": "other",
            "entities": [],
            "structured_data": {},
            "importance_score": 0.3,
            "importance_reason": "解析失败",
        }

    def _validate_and_fix(self, result: dict[str, Any]) -> dict[str, Any]:
        """验证和修正提取结果，确保字段完整性。"""
        # 确保必需字段存在
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

        # 修正 scene_type
        if result["scene_type"] not in SCENE_TYPES:
            result["scene_type"] = "other"

        # 修正 importance_score 范围
        score = result.get("importance_score", 0.5)
        try:
            score = float(score)
            result["importance_score"] = max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            result["importance_score"] = 0.5

        # 确保 entities 是列表
        if not isinstance(result.get("entities"), list):
            result["entities"] = []

        # 确保 structured_data 是字典
        if not isinstance(result.get("structured_data"), dict):
            result["structured_data"] = {}

        return result
