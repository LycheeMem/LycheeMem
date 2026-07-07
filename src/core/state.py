"""
全局 Pipeline 状态定义。

所有 LangGraph 节点通过读写这个 TypedDict 来传递数据。
"""

from __future__ import annotations

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    """LangGraph 全局状态。

    每个节点可以读取任意字段、返回部分字段的更新 patch。
    """

    user_query: str
    session_id: str
    reference_time: str

    compressed_history: list[dict[str, str]]  # 压缩后的对话上下文
    raw_recent_turns: list[dict[str, str]]  # 最近 N 轮原始对话
    wm_token_usage: int  # 当前 token 占用

    retrieved_graph_memories: list[dict[str, Any]]
    retrieved_skills: list[dict[str, Any]]
    retrieval_plan: dict[str, Any]
    action_state: dict[str, Any]
    search_mode: str
    semantic_usage_log_id: str
    feedback_update: dict[str, Any]

    background_context: str  # 融合后的上下文注入字符串
    skill_reuse_plan: list[dict[str, Any]]  # 可复用技能执行计划
    provenance: list[dict[str, Any]]  # 记忆溯源信息

    final_response: str
    tool_calls: list[dict[str, Any]]

    auto_consolidate: bool
    consolidation_pending: bool

    turn_input_tokens: int   # 输入 token 总量
    turn_output_tokens: int  # 输出 token 总量

    input_images: list[str] = []  # Base64 编码的输入图片列表（当前轮次）
    retrieved_visual_memories: list[dict[str, Any]] = []  # 检索到的视觉记忆
    visual_context: str = ""  # 融合后的视觉记忆上下文（供 Reasoner 使用）
