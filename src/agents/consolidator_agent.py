"""
记忆固化 Agent (Memory Consolidation Agent)。

异步后台进程，在每次交互结束后：
1. 调用语义记忆引擎执行新颖性检查与语义固化
2. 仅在检测到新信息时，再提取成功的工具调用链并写入技能库
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.prompts import CONSOLIDATION_SYSTEM_PROMPT
from src.evolve.prompt_registry import get_prompt
from src.embedder.base import BaseEmbedder, set_embed_call_source
from src.llm.base import BaseLLM, set_llm_call_source
from src.memory.procedural.sqlite_skill_store import SQLiteSkillStore
from src.memory.semantic.base import BaseSemanticMemoryEngine


class ConsolidatorAgent(BaseAgent):
    """记忆固化 Agent：异步分析对话并更新长期记忆（CompactSemanticEngine）。"""

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        skill_store: SQLiteSkillStore,
        semantic_engine: BaseSemanticMemoryEngine,
    ):
        super().__init__(llm=llm, prompt_template=CONSOLIDATION_SYSTEM_PROMPT)
        self.embedder = embedder
        self.skill_store = skill_store
        self.semantic_engine = semantic_engine

    def run(
        self,
        turns: list[dict[str, Any]],
        session_id: str | None = None,
        retrieved_context: str = "",
        turn_index_offset: int = 0,
        skip_skills: bool = False,
        session_date: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """分析对话并固化到长期记忆。

        Args:
            turns: 完整的对话轮次列表。
            session_id: 会话 ID。
            retrieved_context: search 阶段召回的原始已有语义记忆片段，
                用于与本轮对话比对，判断是否有新信息需要固化；
                应优先传 pre-synthesis raw context，而不是回答期的 background_context。

        Returns:
            dict 包含：entities_added (int), skills_added (int), facts_added (int)
        """
        if not turns:
            return {"entities_added": 0, "skills_added": 0, "steps": []}

        if not session_id:
            raise RuntimeError("session_id is required for consolidation")

        return self._run_compact(
            turns=turns,
            session_id=session_id,
            retrieved_context=retrieved_context,
            turn_index_offset=turn_index_offset,
            skip_skills=skip_skills,
            session_date=session_date,
        )

    def _run_compact(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        retrieved_context: str = "",
        turn_index_offset: int = 0,
        skip_skills: bool = False,
        session_date: str | None = None,
    ) -> dict[str, Any]:
        """Compact 后端路径：语义固化 → 按新颖性门控技能抽取。

        流程：
        1. 先执行 semantic_engine.ingest_conversation()；其中已包含新颖性检查。
        2. 仅当语义引擎确认存在新信息时，再执行技能提取（skip_skills=True 时跳过）。
        """
        ingest_result = self.semantic_engine.ingest_conversation(
            turns=turns,
            session_id=session_id,
            retrieved_context=retrieved_context,
            turn_index_offset=turn_index_offset,
            reference_timestamp=session_date,
        )

        steps: list[dict[str, Any]] = list(ingest_result.steps)
        has_novelty = bool(ingest_result.has_novelty)

        if not has_novelty or skip_skills:
            steps.append({
                "name": "skill_extraction",
                "status": "skipped",
                "detail": "无新信息，跳过技能提取" if not has_novelty else "skip_skills=True，跳过技能提取",
            })
            return {
                "entities_added": ingest_result.records_added,
                "skills_added": 0,
                "facts_added": ingest_result.records_merged,
                "records_expired": ingest_result.records_expired,
                "has_novelty": False,
                "steps": steps,
            }

        conversation_text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)

        # 已有技能列表注入（LLM 层去重防线）
        existing_skills = self.skill_store.get_all()
        if existing_skills:
            existing_block_lines = ["## Existing Skill List (for reference only, to avoid duplicate extraction)"]
            for idx, s in enumerate(existing_skills[:30], 1):
                existing_block_lines.append(f"{idx}. {s['intent']}")
            existing_block_lines.append("")
            conversation_text = "\n".join(existing_block_lines) + "\n## Current Conversation Log\n" + conversation_text

        set_llm_call_source("skill_extraction")
        response = self._call_llm(
            conversation_text,
            system_content=get_prompt("consolidation", self.prompt_template),
            add_time_basis=True,
        )
        analysis = self._safe_parse(response)

        def _write_skills() -> int:
            """写入新技能，含双层去重：LLM 层（已提前注入已有技能列表）+ 向量相似度兜底。

            去重阈值 0.85：intent 向量余弦相似度达到此值即认为语义相同。
            - 相似技能已存在 → upsert（复用已有 skill_id，更新 doc_markdown），不新建
            - 无相似技能 → 新建
            这样既防止重复，也允许技能被更新升级。
            """
            DEDUP_THRESHOLD = 0.85
            count = 0
            for skill in analysis.get("new_skills", []):
                intent = skill.get("intent", "")
                doc_markdown = skill.get("doc_markdown", "")
                if not intent or not doc_markdown:
                    continue
                set_embed_call_source("skill_search")
                embedding = self.embedder.embed_query(intent)
                # 代码层兜底：向量相似度去重 / 合并
                top_existing = self.skill_store.search(
                    query=intent,
                    top_k=1,
                    query_embedding=embedding,
                )
                if top_existing and top_existing[0].get("score", 0.0) >= DEDUP_THRESHOLD:
                    # 相似技能已存在：upsert 到已有条目（更新内容而非新建）
                    existing_id = top_existing[0]["id"]
                    self.skill_store.add(
                        [{"id": existing_id, "intent": intent, "embedding": embedding, "doc_markdown": doc_markdown}],
                    )
                else:
                    self.skill_store.add(
                        [{"intent": intent, "embedding": embedding, "doc_markdown": doc_markdown}],
                    )
                count += 1
            return count

        skills_added = _write_skills()
        steps.append({
            "name": "skill_extraction",
            "status": "done",
            "detail": f"{skills_added} 个技能" if skills_added else "无新技能",
        })

        return {
            "entities_added": ingest_result.records_added,
            "skills_added": skills_added,
            "facts_added": ingest_result.records_merged,
            "records_expired": ingest_result.records_expired,
            "has_novelty": has_novelty,
            "steps": steps,
        }

    def _safe_parse(self, response: str) -> dict[str, Any]:
        """安全解析 LLM 输出，失败时返回安全默认值。"""
        try:
            return self._parse_json(response)
        except (ValueError, KeyError):
            return {"new_skills": []}
