"""对话端点。"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_pipeline
from src.api.models import ChatRequest, ChatResponse
from src.api.trace_builders import (
    _build_chat_response,
    _build_reasoner_trace,
    _build_search_trace,
    _build_synthesizer_trace,
    _build_trace,
    _build_wm_trace,
)

logger = logging.getLogger("src.api.chat")

router = APIRouter()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/complete", response_model=ChatResponse)
async def chat_complete(req: ChatRequest, pipeline=Depends(get_pipeline)):
    """非流式对话端点（带错误处理）。"""
    try:
        # 处理图片输入
        visual_memory_ids = []
        if req.images:
            logger.info("Processing %d images for session %s", len(req.images), req.session_id)
            try:
                # 构建带 MIME 类型的图片数据
                images_with_mime = []
                for i, img_b64 in enumerate(req.images):
                    mime_type = req.image_mime_types[i] if i < len(req.image_mime_types) else "image/jpeg"
                    images_with_mime.append({"base64": img_b64, "mime_type": mime_type})
                visual_memory_ids = await _process_images_with_mime(images_with_mime, req.session_id, pipeline)
                logger.info("Processed images, stored %d visual memories", len(visual_memory_ids))
            except Exception as e:
                logger.error("Failed to process images: %s", e, exc_info=True)
                # 继续执行，不阻断对话

        # 运行 pipeline
        result = pipeline.run(
            user_query=req.message,
            session_id=req.session_id,
            input_images=req.images,
        )

        response = _build_chat_response(req.session_id, result)

        # 附加视觉记忆检索信息
        if visual_memory_ids:
            response.memories_retrieved += len(visual_memory_ids)

        return response

    except Exception as e:
        logger.error("Chat complete failed: %s", e, exc_info=True)
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_stream(req: ChatRequest, pipeline=Depends(get_pipeline)):
    """SSE 流式对话（带错误处理）。

    流式 chunk 格式 (Server-Sent Events):
      data: {"type": "step", "step": "wm_manager", "status": "done", "trace_fragment": {...}}
      data: {"type": "step", "step": "search",     "status": "done", "trace_fragment": {...}}
      data: {"type": "step", "step": "synthesize",  "status": "done", "trace_fragment": {...}}
      data: {"type": "step", "step": "reason",      "status": "done", "trace_fragment": {...}}
      data: {"type": "answer", "content": "最终回答文本"}
      data: {"type": "done", "session_id": "...", "memories_retrieved": N, "trace": {...}}
    """
    # 处理图片输入 - 在开始对话前先完成图片识别
    visual_memory_ids = []
    visual_captions = []  # 存储图片识别结果
    
    if req.images:
        logger.info("Processing %d images for session %s BEFORE conversation", len(req.images), req.session_id)
        try:
            # 构建带 MIME 类型的图片数据
            images_with_mime = []
            for i, img_b64 in enumerate(req.images):
                mime_type = req.image_mime_types[i] if i < len(req.image_mime_types) else "image/jpeg"
                images_with_mime.append({"base64": img_b64, "mime_type": mime_type})
            
            # 处理图片并获取识别结果 - 等待完成
            logger.info("Waiting for image recognition to complete...")
            stored_ids = await _process_images_with_mime(images_with_mime, req.session_id, pipeline)
            visual_memory_ids = stored_ids
            logger.info("✅ Image recognition completed, stored %d visual memories", len(visual_memory_ids))
            
            # 获取图片识别的 caption 用于在回答中引用
            for record_id in stored_ids:
                try:
                    record = pipeline.visual_store.get_by_id(record_id)
                    if record and record.caption:
                        visual_captions.append(record.caption)
                        logger.info("  - Image caption: %s", record.caption[:60])
                except Exception as e:
                    logger.warning("Failed to get caption for %s: %s", record_id, e)
                    
        except Exception as e:
            logger.error("Failed to process images: %s", e, exc_info=True)
            # 继续执行，不阻断对话
    
    # 如果有图片识别结果，将图片信息添加到用户消息中
    enhanced_message = req.message
    if visual_captions:
        # 将图片描述添加到消息中，让 LLM 可以基于图片内容回答
        image_context = "\n\n[图片内容]\n" + "\n".join(f"- {cap}" for cap in visual_captions)
        enhanced_message = req.message + image_context
        logger.info("Enhanced message with image context: %d chars", len(enhanced_message))

    async def event_stream():
        try:
            accumulated: dict[str, Any] = {}
            final_result: dict = {}
            step_name = ""

            # 注意：不再传递 input_images，因为图片已经在上面处理完成了
            async for evt in pipeline.astream_steps(
                user_query=enhanced_message,  # 使用增强后的消息
                session_id=req.session_id,
                input_images=[],  # 传空列表，因为图片已处理
            ):
                if evt["type"] == "step":
                    step_name = evt["step"]
                    patch = evt.get("patch", {})
                    accumulated.update(patch)

                    # Build a display-ready trace fragment for this step only.
                    fragment: dict[str, Any] = {}
                    if step_name == "wm_manager":
                        fragment["wm_manager"] = _build_wm_trace(accumulated).model_dump()
                    elif step_name == "search":
                        fragment["search_coordinator"] = _build_search_trace(accumulated).model_dump()
                    elif step_name == "synthesize":
                        fragment["synthesizer"] = _build_synthesizer_trace(accumulated).model_dump()
                    elif step_name == "reason":
                        fragment["reasoner"] = _build_reasoner_trace(accumulated).model_dump()

                    yield _sse({"type": "step", "step": step_name, "status": "done", "trace_fragment": fragment})

                elif evt["type"] == "token":
                    yield _sse({"type": "token", "content": evt["content"]})

                elif evt["type"] == "done":
                    final_result = evt.get("result", {})
                    break

            # 循环结束后发送最终响应
            memories = len(final_result.get("retrieved_graph_memories", [])) + len(
                final_result.get("retrieved_skills", [])
            ) + len(final_result.get("retrieved_visual_memories", []))

            yield _sse({"type": "answer", "content": final_result.get("final_response", "")})
            trace = _build_trace(final_result)
            yield _sse(
                {
                    "type": "done",
                    "session_id": req.session_id,
                    "memories_retrieved": memories,
                    "wm_token_usage": final_result.get("wm_token_usage", 0),
                    "turn_input_tokens": final_result.get("turn_input_tokens", 0),
                    "turn_output_tokens": final_result.get("turn_output_tokens", 0),
                    "trace": trace.model_dump(),
                }
            )

        except Exception as e:
            logger.error("Event stream error: %s", e, exc_info=True)
            yield _sse({"type": "error", "content": f"Server error: {str(e)}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _process_images_with_mime(
    images_with_mime: list[dict[str, str]],
    session_id: str,
    pipeline
) -> list[str]:
    """处理输入图片（带 MIME 类型），提取并存储视觉记忆（双嵌入：文本 + 视觉）。

    Args:
        images_with_mime: 包含 base64 和 mime_type 的字典列表。
        session_id: 会话 ID。
        pipeline: Pipeline 实例。

    Returns:
        存储的视觉记忆 record_id 列表。
    """
    import base64
    import hashlib
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    from src.memory.visual.models import VisualMemoryRecord

    stored_ids = []

    # 检查 pipeline 是否有视觉组件
    if not hasattr(pipeline, 'visual_extractor') or not hasattr(pipeline, 'visual_store'):
        logger.warning("Pipeline missing visual components, skipping image processing")
        return []

    for i, img_data in enumerate(images_with_mime):
        try:
            img_b64 = img_data["base64"]
            mime_type = img_data.get("mime_type", "image/jpeg")
            logger.info("Processing image %d for session %s: mime_type=%s", i, session_id, mime_type)

            # 提取视觉信息（使用 VLM 进行图片理解）
            try:
                extraction = await pipeline.visual_extractor.extract_from_base64(
                    image_b64=img_b64,
                    mime_type=mime_type,
                    session_id=session_id,
                )
            except Exception as e:
                logger.error("Visual extraction failed for image %d: %s", i, e, exc_info=True)
                # 使用降级结果继续处理
                extraction = {
                    "caption": f"[Image {i+1}]",
                    "scene_type": "other",
                    "image_hash": hashlib.sha256(img_b64.encode()).hexdigest()[:16],
                    "image_mime_type": mime_type,
                    "image_size": len(img_b64) * 3 // 4,
                    "entities": [],
                    "structured_data": {},
                    "importance_score": 0.5,
                }

            logger.info(
                "Visual extraction completed: caption=%s, scene_type=%s",
                extraction.get("caption", "")[:60],
                extraction.get("scene_type", "other"),
            )

            # 保存图片到本地文件系统
            try:
                image_path = pipeline.visual_store.save_image_file(
                    image_b64=img_b64,
                    image_hash=extraction["image_hash"],
                    mime_type=extraction.get("image_mime_type", mime_type),
                    session_id=session_id,
                )
            except Exception as e:
                logger.error("Failed to save image file: %s", e)
                continue  # 跳过此图片

            # 生成 record_id
            record_id = hashlib.sha256(
                f"{extraction['image_hash']}:{session_id}:{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:32]

            # 创建视觉记忆记录
            record = VisualMemoryRecord(
                record_id=record_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                image_path=image_path,
                image_hash=extraction["image_hash"],
                image_size=extraction.get("image_size", 0),
                image_mime_type=extraction.get("image_mime_type", mime_type),
                caption=extraction.get("caption", ""),
                entities=extraction.get("entities", []),
                scene_type=extraction.get("scene_type", "other"),
                structured_data=extraction.get("structured_data", {}),
                importance_score=extraction.get("importance_score", 0.5),
                source_role="user",
            )

            # 生成双嵌入：caption embedding + visual embedding
            # 1. Caption embedding（文本嵌入，向后兼容）
            if pipeline.visual_store.embedder:
                try:
                    embedding = pipeline.visual_store.embedder.embed_query(record.caption)
                    if embedding:
                        record.caption_embedding = embedding
                        logger.info(
                            "Caption embedding generated: dim=%d", len(embedding)
                        )
                except Exception as e:
                    logger.warning("Failed to generate caption embedding: %s", e)

            # 2. Visual embedding（CLIP 视觉嵌入，支持跨模态检索）
            if hasattr(pipeline.visual_store, 'multimodal_embedder') and pipeline.visual_store.multimodal_embedder:
                try:
                    # 解码图像数据
                    image_bytes = base64.b64decode(img_b64)

                    # 使用临时文件生成视觉嵌入
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        tmp.write(image_bytes)
                        tmp_path = tmp.name

                    try:
                        # 生成 CLIP 视觉嵌入（与文本同一空间，支持跨模态检索）
                        visual_embedding = pipeline.visual_store.multimodal_embedder.embed_image(tmp_path)
                        if visual_embedding:
                            record.visual_embedding = visual_embedding
                            logger.info(
                                "Visual embedding generated: dim=%d",
                                len(visual_embedding),
                            )
                    finally:
                        # 清理临时文件
                        Path(tmp_path).unlink(missing_ok=True)

                except Exception as e:
                    logger.warning("Failed to generate visual embedding: %s", e)

            # 存储记录到 SQLite + LanceDB
            stored_id = pipeline.visual_store.store(record)
            if stored_id:
                stored_ids.append(stored_id)
                logger.info(
                    "✓ Visual memory stored: id=%s, caption=%s",
                    stored_id, record.caption[:60],
                )
            else:
                logger.warning("Visual memory store returned empty id (possible duplicate)")

        except Exception as e:
            logger.error("Failed to process image %d: %s", i, e, exc_info=True)

    logger.info("Processed %d images, stored %d visual memories", len(images_with_mime), len(stored_ids))
    return stored_ids


async def _process_images(images: list[str], session_id: str, pipeline) -> list[str]:
    """处理输入图片，提取并存储视觉记忆（双嵌入：文本 + 视觉）。

    Args:
        images: Base64 编码的图片列表。
        session_id: 会话 ID。
        pipeline: Pipeline 实例。

    Returns:
        存储的视觉记忆 record_id 列表。
    """
    import base64
    import hashlib
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path

    from src.memory.visual.models import VisualMemoryRecord

    stored_ids = []

    # 检查 pipeline 是否有视觉组件
    if not hasattr(pipeline, 'visual_extractor') or not hasattr(pipeline, 'visual_store'):
        logger.warning("Pipeline missing visual components, skipping image processing")
        return []

    for i, img_b64 in enumerate(images):
        try:
            logger.info("Processing image %d for session %s", i, session_id)

            # 提取视觉信息（使用 VLM 进行图片理解）
            try:
                extraction = await pipeline.visual_extractor.extract_from_base64(
                    image_b64=img_b64,
                    mime_type="image/jpeg",  # 默认，前端可以传更准确的类型
                    session_id=session_id,
                )
            except Exception as e:
                logger.error("Visual extraction failed for image %d: %s", i, e, exc_info=True)
                # 使用降级结果继续处理
                extraction = {
                    "caption": f"[Image {i+1}]",
                    "scene_type": "other",
                    "image_hash": hashlib.sha256(img_b64.encode()).hexdigest()[:16],
                    "image_mime_type": "image/jpeg",
                    "image_size": len(img_b64) * 3 // 4,
                    "entities": [],
                    "structured_data": {},
                    "importance_score": 0.5,
                }

            logger.info(
                "Visual extraction completed: caption=%s, scene_type=%s",
                extraction.get("caption", "")[:60],
                extraction.get("scene_type", "other"),
            )

            # 保存图片到本地文件系统
            try:
                image_path = pipeline.visual_store.save_image_file(
                    image_b64=img_b64,
                    image_hash=extraction["image_hash"],
                    mime_type=extraction.get("image_mime_type", "image/jpeg"),
                    session_id=session_id,
                )
            except Exception as e:
                logger.error("Failed to save image file: %s", e)
                continue  # 跳过此图片

            # 生成 record_id
            record_id = hashlib.sha256(
                f"{extraction['image_hash']}:{session_id}:{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:32]

            # 创建视觉记忆记录
            record = VisualMemoryRecord(
                record_id=record_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                image_path=image_path,
                image_hash=extraction["image_hash"],
                image_size=extraction.get("image_size", 0),
                image_mime_type=extraction.get("image_mime_type", "image/jpeg"),
                caption=extraction.get("caption", ""),
                entities=extraction.get("entities", []),
                scene_type=extraction.get("scene_type", "other"),
                structured_data=extraction.get("structured_data", {}),
                importance_score=extraction.get("importance_score", 0.5),
                source_role="user",
            )

            # 生成双嵌入：caption embedding + visual embedding
            # 1. Caption embedding（文本嵌入，向后兼容）
            if pipeline.visual_store.embedder:
                try:
                    embedding = pipeline.visual_store.embedder.embed_query(record.caption)
                    if embedding:
                        record.caption_embedding = embedding
                        logger.info(
                            "Caption embedding generated: dim=%d", len(embedding)
                        )
                except Exception as e:
                    logger.warning("Failed to generate caption embedding: %s", e)

            # 2. Visual embedding（CLIP 视觉嵌入，支持跨模态检索）
            if hasattr(pipeline.visual_store, 'multimodal_embedder') and pipeline.visual_store.multimodal_embedder:
                try:
                    # 解码图像数据
                    image_bytes = base64.b64decode(img_b64)

                    # 使用临时文件生成视觉嵌入
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        tmp.write(image_bytes)
                        tmp_path = tmp.name

                    try:
                        # 生成 CLIP 视觉嵌入（与文本同一空间，支持跨模态检索）
                        visual_embedding = pipeline.visual_store.multimodal_embedder.embed_image(tmp_path)
                        if visual_embedding:
                            record.visual_embedding = visual_embedding
                            logger.info(
                                "Visual embedding generated: dim=%d",
                                len(visual_embedding),
                            )
                    finally:
                        # 清理临时文件
                        Path(tmp_path).unlink(missing_ok=True)

                except Exception as e:
                    logger.warning("Failed to generate visual embedding: %s", e)

            # 存储记录到 SQLite + LanceDB
            stored_id = pipeline.visual_store.store(record)
            if stored_id:
                stored_ids.append(stored_id)
                logger.info(
                    "✓ Visual memory stored: id=%s, caption=%s",
                    stored_id, record.caption[:60],
                )
            else:
                logger.warning("Visual memory store returned empty id (possible duplicate)")

        except Exception as e:
            logger.error("Failed to process image %d: %s", i, e, exc_info=True)

    logger.info("Processed %d images, stored %d visual memories", len(images), len(stored_ids))
    return stored_ids
