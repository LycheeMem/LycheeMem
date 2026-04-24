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

    # ─── 输入 ───
    user_query: str
    session_id: str

    # ─── 工作记忆管理器输出 ───
    compressed_history: list[dict[str, str]]  # 压缩后的对话上下文
    raw_recent_turns: list[dict[str, str]]  # 最近 N 轮原始对话
    wm_token_usage: int  # 当前 token 占用

    # ─── 检索协调器输出 ───
    retrieved_graph_memories: list[dict[str, Any]]
    retrieved_skills: list[dict[str, Any]]
    novelty_retrieved_context: str  # search 阶段原始语义记忆片段，用于 novelty check
    retrieval_plan: dict[str, Any]
    action_state: dict[str, Any]
    search_mode: str
    semantic_usage_log_id: str
    feedback_update: dict[str, Any]

    # ─── 整合器输出 ───
    background_context: str  # 融合后的上下文注入字符串
    skill_reuse_plan: list[dict[str, Any]]  # 可复用技能执行计划
    provenance: list[dict[str, Any]]  # 记忆溯源信息

    # ─── 推理器输出 ───
    final_response: str
    tool_calls: list[dict[str, Any]]

    # ─── 固化器 (异步后台，不阻塞主流程) ───
    consolidation_pending: bool

    # ─── 本轮 token 统计（主流程所有 LLM 调用之和，不含后台固化） ───
    turn_input_tokens: int   # 输入 token 总量
    turn_output_tokens: int  # 输出 token 总量

    # ─── 多模态 / 视觉记忆 ───
    input_images: list[str] = []  # Base64 编码的输入图片列表（当前轮次）
    retrieved_visual_memories: list[dict[str, Any]] = []  # 检索到的视觉记忆
    visual_context: str = ""  # 融合后的视觉记忆上下文（供 Reasoner 使用）
