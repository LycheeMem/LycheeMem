"""
检索协调器 (Memory Search Coordinator)。

语义记忆检索直接调用 CompactSemanticEngine.search()，并在进入检索前构造
ActionState，让 planner 看到“当前想做什么、最近发生了什么、受什么约束”。
技能库检索则按 answer / action / mixed 三种模式自适应启用。
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.base_agent import BaseAgent
from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.procedural.sqlite_skill_store import SQLiteSkillStore
from src.memory.semantic.base import BaseSemanticMemoryEngine
from src.memory.semantic.models import ActionState

HYDE_SYSTEM_PROMPT = """\
你是 HyDE 假设性回答生成器。

你的任务：
- 给定用户查询，为"程序/技能类"意图生成一段 **假设性的理想回答文本（Draft Answer）**。
- 这段草稿回答不会直接返回给用户，而是作为向量检索的"锚点文本"，用来提高召回率。

要求：
1. 假装你已经成功完成了用户想要的任务，用 2-3 句话描述一个合理的解决方案草稿。
2. 文本中应自然包含：可能会调用的工具名称、关键参数名、重要中间产物等关键信息。
3. 保持简洁，聚焦关键实体、步骤和概念，不要展开长篇解释。
4. 不要使用列表或 JSON，只输出连续自然语言段落。

## 示例（仅供参考，不要原样抄写）

- 用户查询："帮我写一个脚本，每天凌晨 3 点备份 PostgreSQL 数据库到 S3。"
    参考输出：
    "我为你编写了一个使用 `pg_dump` 的备份脚本，并通过 crontab 配置在每天凌晨 3 点运行。脚本会将生成的备份文件上传到你指定的 S3 bucket，并使用时间戳作为文件名，方便后续检索和清理。"

- 用户查询："搭一个最简单的 FastAPI 服务，并用 Docker 部署。"
    参考输出：
    "我创建了一个包含单个 `/health` 路由的 FastAPI 应用，并编写了一个使用 `python:3.10-slim` 基础镜像的 Dockerfile。通过 `docker build` 构建镜像后，在服务器上使用 `docker run -p 8000:8000` 运行该服务。"
"""


class SearchCoordinator(BaseAgent):
    """检索协调器：每次请求均同时检索语义记忆和技能库。"""

    _ACTION_HINT_PATTERNS = (
        "部署", "发布", "上线", "回滚", "排查", "修复", "配置", "执行", "运行",
        "创建", "实现", "安装", "迁移", "备份", "重启", "导入", "导出", "调试",
        "怎么做", "如何", "步骤", "流程", "命令",
    )
    _CONSTRAINT_MARKERS = (
        "必须", "只能", "不要", "不能", "禁止", "限定", "预算", "不超过", "至少", "先不要",
    )
    _FAILURE_MARKERS = (
        "报错", "失败", "不行", "异常", "超时", "错误", "冲突", "崩溃", "没生效", "卡住",
    )
    _KNOWN_TOOLS = (
        "python", "docker", "kubernetes", "k8s", "helm", "fastapi", "postgresql", "redis",
        "neo4j", "sqlite", "lancedb", "git", "node", "npm", "vite", "react", "typescript",
        "jwt", "openai", "gemini", "ollama", "uvicorn",
    )

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        skill_store: SQLiteSkillStore,
        semantic_engine: BaseSemanticMemoryEngine,
        skill_top_k: int = 3,
        skill_reuse_threshold: float = 0.85,
    ):
        super().__init__(llm=llm, prompt_template=HYDE_SYSTEM_PROMPT)
        self.embedder = embedder
        self.skill_store = skill_store
        self.semantic_engine = semantic_engine
        self.skill_top_k = skill_top_k
        self.skill_reuse_threshold = skill_reuse_threshold

    def run(
        self,
        user_query: str,
        **kwargs,
    ) -> dict[str, Any]:
        """同时检索语义记忆和技能库。"""
        session_id = kwargs.get("session_id")
        if session_id is not None:
            session_id = str(session_id)
        user_id = kwargs.get("user_id", "")
        top_k = kwargs.get("top_k")
        include_skills = bool(kwargs.get("include_skills", True))

        recent_context = self._build_recent_context(
            raw_recent_turns=kwargs.get("raw_recent_turns") or [],
            compressed_history=kwargs.get("compressed_history") or [],
        )
        action_state = self._build_action_state(
            user_query=user_query,
            recent_context=recent_context,
            wm_token_usage=int(kwargs.get("wm_token_usage", 0) or 0),
            tool_calls=kwargs.get("tool_calls") or [],
        )

        feedback_update: dict[str, Any] = {}
        if session_id:
            try:
                feedback_update = self.semantic_engine.apply_feedback_from_user_turn(
                    session_id=session_id,
                    user_turn=user_query,
                    user_id=user_id,
                )
            except Exception:
                feedback_update = {}

        result = self._run_compact(
            user_query,
            session_id=session_id,
            user_id=user_id,
            recent_context=recent_context,
            action_state=action_state,
            top_k=int(top_k) if top_k is not None else None,
            include_skills=include_skills,
        )
        result["feedback_update"] = feedback_update
        return result

    def _run_compact(
        self,
        user_query: str,
        *,
        session_id: str | None = None,
        user_id: str = "",
        recent_context: str = "",
        action_state: ActionState | None = None,
        top_k: int | None = None,
        include_skills: bool = True,
    ) -> dict[str, Any]:
        """Compact 后端路径：semantic_engine.search() + mode-aware 技能检索。"""
        result = self.semantic_engine.search(
            query=user_query,
            session_id=session_id,
            top_k=int(top_k or 0),
            user_id=user_id,
            recent_context=recent_context,
            action_state=self._action_state_to_dict(action_state),
        )

        graph_memories = []
        if result.context.strip():
            graph_memories = [
                {
                    "anchor": {
                        "node_id": "compact_context",
                        "name": "CompactSemanticMemory",
                        "label": "SemanticContext",
                        "score": 1.0,
                    },
                    "subgraph": {"nodes": [], "edges": []},
                    "constructed_context": result.context,
                    "provenance": result.provenance,
                }
            ]

        skill_results: list[dict[str, Any]] = []
        if include_skills:
            skill_results = self._search_skills(
                user_query,
                plan=result.retrieval_plan,
                action_state=result.action_state,
                user_id=user_id,
                top_k=top_k,
            )

        return {
            "retrieved_graph_memories": graph_memories,
            "retrieved_skills": skill_results,
            "retrieval_plan": result.retrieval_plan,
            "action_state": result.action_state,
            "search_mode": result.mode,
            "semantic_usage_log_id": result.usage_log_id,
        }

    def _search_skills(
        self,
        query: str,
        *,
        plan: dict[str, Any] | None = None,
        action_state: dict[str, Any] | None = None,
        user_id: str = "",
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """按 mode / decision state 自适应检索技能库。"""
        plan = plan or {}
        action_state = action_state or {}
        mode = str(plan.get("mode") or "answer").strip().lower() or "answer"

        if mode == "answer" and not self._looks_procedural_query(query):
            return []

        skill_query = self._build_skill_query(query, plan=plan, action_state=action_state)
        hyde_doc = self._call_llm(
            skill_query,
            system_content=self.prompt_template,
            add_time_basis=True,
        )
        hyde_embedding = self.embedder.embed_query(hyde_doc)

        default_top_k = self.skill_top_k
        if mode == "mixed":
            default_top_k = max(1, self.skill_top_k - 1)
        elif mode == "answer":
            default_top_k = 1
        skill_top_k = top_k if top_k is not None else default_top_k

        results = self.skill_store.search(
            query=skill_query,
            top_k=skill_top_k,
            query_embedding=hyde_embedding,
            user_id=user_id,
        )

        reuse_threshold = self.skill_reuse_threshold
        if mode == "action":
            reuse_threshold = max(0.0, self.skill_reuse_threshold - 0.05)

        for skill in results:
            skill["reusable"] = skill.get("score", 0) >= reuse_threshold
            skill["retrieval_mode"] = mode

        return results

    def _build_recent_context(
        self,
        *,
        raw_recent_turns: list[dict[str, Any]],
        compressed_history: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []

        summaries = [m for m in (compressed_history or []) if m.get("role") == "system"]
        for msg in summaries[-2:]:
            content = str(msg.get("content") or "").strip()
            if content:
                parts.append(f"Summary: {content}")

        recent_turns = raw_recent_turns or [m for m in (compressed_history or []) if m.get("role") != "system"]
        for turn in recent_turns[-6:]:
            role = str(turn.get("role") or "").strip() or "unknown"
            content = str(turn.get("content") or "").strip()
            if content:
                parts.append(f"{role}: {content}")

        context = "\n".join(parts).strip()
        if len(context) > 2000:
            context = context[-2000:]
        return context

    def _build_action_state(
        self,
        *,
        user_query: str,
        recent_context: str,
        wm_token_usage: int,
        tool_calls: list[dict[str, Any]],
    ) -> ActionState:
        last_tool_name = ""
        last_tool_result = ""
        if tool_calls:
            last_tool = tool_calls[-1] or {}
            last_tool_name = str(
                last_tool.get("name") or last_tool.get("tool_name") or last_tool.get("id") or ""
            ).strip()
            last_tool_result = str(
                last_tool.get("result") or last_tool.get("output") or last_tool.get("content") or ""
            ).strip()

        return ActionState(
            current_subgoal=user_query.strip(),
            tentative_action=self._infer_tentative_action(user_query, recent_context),
            last_tool_name=last_tool_name,
            last_tool_result=last_tool_result[:500],
            missing_slots=[],
            known_constraints=self._extract_known_constraints(user_query, recent_context),
            available_tools=self._infer_available_tools(user_query, recent_context, tool_calls),
            failure_signal=self._infer_failure_signal(user_query, recent_context),
            token_budget=max(0, wm_token_usage),
            recent_context_excerpt=recent_context[:1000],
        )

    def _infer_tentative_action(self, user_query: str, recent_context: str) -> str:
        haystack = f"{user_query}\n{recent_context}".lower()
        if any(keyword in haystack for keyword in self._ACTION_HINT_PATTERNS):
            return str(user_query or "").strip()
        return ""

    def _extract_known_constraints(self, user_query: str, recent_context: str) -> list[str]:
        text = "\n".join([str(recent_context or ""), str(user_query or "")])
        segments = re.split(r"[\n。！？!?；;]", text)
        constraints: list[str] = []
        seen: set[str] = set()
        for seg in segments:
            piece = seg.strip()
            if not piece:
                continue
            if any(marker in piece for marker in self._CONSTRAINT_MARKERS):
                key = piece.casefold()
                if key in seen:
                    continue
                seen.add(key)
                constraints.append(piece[:120])
        return constraints[:6]

    def _infer_available_tools(
        self,
        user_query: str,
        recent_context: str,
        tool_calls: list[dict[str, Any]],
    ) -> list[str]:
        combined = f"{user_query}\n{recent_context}".lower()
        tools: list[str] = []
        seen: set[str] = set()

        for tool_name in self._KNOWN_TOOLS:
            if tool_name in combined and tool_name not in seen:
                seen.add(tool_name)
                tools.append(tool_name)

        for call in tool_calls or []:
            tool_name = str(call.get("name") or call.get("tool_name") or "").strip()
            if not tool_name:
                continue
            key = tool_name.casefold()
            if key in seen:
                continue
            seen.add(key)
            tools.append(tool_name)

        return tools[:8]

    def _infer_failure_signal(self, user_query: str, recent_context: str) -> str:
        segments = [str(user_query or "").strip()] + [s.strip() for s in recent_context.splitlines() if s.strip()]
        for segment in segments:
            lowered = segment.lower()
            if any(marker in lowered for marker in self._FAILURE_MARKERS):
                return segment[:160]
        return ""

    def _build_skill_query(
        self,
        query: str,
        *,
        plan: dict[str, Any],
        action_state: dict[str, Any],
    ) -> str:
        mode = str(plan.get("mode") or "answer").strip().lower() or "answer"
        parts: list[str] = [str(query or "").strip()]

        tentative_action = str(action_state.get("tentative_action") or "").strip()
        if tentative_action and tentative_action not in parts:
            parts.append(tentative_action)

        if mode in {"action", "mixed"}:
            for key in (
                "pragmatic_queries",
                "tool_hints",
                "required_constraints",
                "required_affordances",
                "missing_slots",
            ):
                for value in plan.get(key, []) or []:
                    item = str(value or "").strip()
                    if item and item not in parts:
                        parts.append(item)

        return "；".join(parts)

    def _looks_procedural_query(self, query: str) -> bool:
        text = str(query or "").lower()
        return any(keyword in text for keyword in self._ACTION_HINT_PATTERNS)

    @staticmethod
    def _action_state_to_dict(action_state: ActionState | None) -> dict[str, Any] | None:
        if action_state is None:
            return None
        return {
            "current_subgoal": action_state.current_subgoal,
            "tentative_action": action_state.tentative_action,
            "last_tool_name": action_state.last_tool_name,
            "last_tool_result": action_state.last_tool_result,
            "missing_slots": list(action_state.missing_slots),
            "known_constraints": list(action_state.known_constraints),
            "available_tools": list(action_state.available_tools),
            "failure_signal": action_state.failure_signal,
            "token_budget": action_state.token_budget,
            "recent_context_excerpt": action_state.recent_context_excerpt,
        }
