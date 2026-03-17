"""对话端点。"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from a_frame.api.dependencies import get_pipeline
from a_frame.api.models import ChatRequest, ChatResponse
from a_frame.api.trace_builders import (
    _build_chat_response,
    _build_reasoner_trace,
    _build_search_trace,
    _build_synthesizer_trace,
    _build_trace,
    _build_wm_trace,
)

router = APIRouter()


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/complete", response_model=ChatResponse)
async def chat_complete(req: ChatRequest, pipeline=Depends(get_pipeline)):
    result = pipeline.run(user_query=req.message, session_id=req.session_id)
    return _build_chat_response(req.session_id, result)


@router.post("/chat")
async def chat_stream(req: ChatRequest, pipeline=Depends(get_pipeline)):
    """SSE 流式对话。

    流式 chunk 格式 (Server-Sent Events):
      data: {"type": "step", "step": "wm_manager", "status": "done", "trace_fragment": {...}}
      data: {"type": "step", "step": "search",     "status": "done", "trace_fragment": {...}}
      data: {"type": "step", "step": "synthesize",  "status": "done", "trace_fragment": {...}}
      data: {"type": "step", "step": "reason",      "status": "done", "trace_fragment": {...}}
      data: {"type": "answer", "content": "最终回答文本"}
      data: {"type": "done", "session_id": "...", "memories_retrieved": N, "trace": {...}}
    """

    async def event_stream():
        accumulated: dict[str, Any] = {}
        final_result: dict = {}
        async for evt in pipeline.astream_steps(
            user_query=req.message, session_id=req.session_id
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
            elif evt["type"] == "done":
                final_result = evt.get("result", {})

        memories = len(final_result.get("retrieved_graph_memories", [])) + len(
            final_result.get("retrieved_skills", [])
        )
        yield _sse({"type": "answer", "content": final_result.get("final_response", "")})
        trace = _build_trace(final_result)
        yield _sse(
            {
                "type": "done",
                "session_id": req.session_id,
                "memories_retrieved": memories,
                "wm_token_usage": final_result.get("wm_token_usage", 0),
                "trace": trace.model_dump(),
            }
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
