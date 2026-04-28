"""
记忆固化 Agent (Memory Consolidation Agent)。

异步后台进程，在每次交互结束后：
1. 调用语义记忆引擎执行新颖性检查与语义固化
2. 仅在检测到新信息时，再提取成功的工具调用链并写入技能库
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.procedural.sqlite_skill_store import SQLiteSkillStore
from src.memory.semantic.base import BaseSemanticMemoryEngine

CONSOLIDATION_SYSTEM_PROMPT = """\
You are a Memory Consolidator.
The semantic novelty check has already been done before you are called.
Your only task is to decide whether the conversation contains a **new reusable procedural skill** worth storing.

Reply in JSON using exactly the following structure and field names:
{
    "new_skills": [
        {
            "intent": "One-sentence description of the task intent",
            "doc_markdown": "# Skill Title\\n\\nWrite a reusable Markdown operation guide that may include steps, commands, notes, and input/output details"
        }
    ]
}

Rules:
- If the conversation does not contain a complex operational pattern worth saving, `new_skills` must be an empty array.
- If the main content of this turn is **using an already existing skill** to complete a task, rather than **defining or teaching a new skill**, then `new_skills` must be empty because the skill already exists.
  The message block "Existing Skill List" will provide all current skill intents. Use it when making this judgment.
- Ignore repeated phrasing, obviously failed attempts, and similar content that is not worth saving long term.
- **The output must be strict JSON** with no code fences. JSON strings must not contain raw line breaks; use `\\n` instead.

Requirements for `doc_markdown`:
- It must be plain Markdown text, not JSON or YAML.
- It should preferably include: applicable scenario, prerequisites, numbered steps, key commands or code blocks, common errors, and troubleshooting.

Below are several examples for format and extraction criteria only. Do not copy them verbatim.

## Example 1: Contains a new skill
<session_log>
user: Help me design a blue-green deployment flow for user-service this time. I want to canary it on half of the prod-a nodes first.
assistant: We can do it this way:
    1) Update the Helm values and tag the new user-service image as v2.
    2) Apply the new Deployment with kubectl.
    3) Observe Prometheus alerts and logs. If there is no anomaly, shift all replicas to the new version.
</session_log>

Expected JSON output:
{
    "new_skills": [
        {
            "intent": "Perform a blue-green deployment of user-service to the prod-a cluster",
            "doc_markdown": "# user-service Blue-Green Deployment (prod-a)\\n\\n## Applicable Scenario\\n- Deploy user-service to prod-a and canary it on half of the nodes first\\n\\n## Steps\\n1. Update Helm values and mark the image as v2\\n2. Use `kubectl apply` to deploy to part of the nodes\\n3. Observe Prometheus alerts and logs. If there is no anomaly, shift all replicas to v2\\n"
        }
    ]
}

## Example 2: No new skill
<session_log>
user: From now on, all documents in this project must be written in Chinese. Do not give me English templates again.
assistant: Understood. Documentation for this project will use Chinese consistently.
</session_log>

Expected JSON output:
{
    "new_skills": []
}
"""


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
        )

    def _run_compact(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        retrieved_context: str = "",
        turn_index_offset: int = 0,
    ) -> dict[str, Any]:
        """Compact 后端路径：语义固化 → 按新颖性门控技能抽取。

        流程：
        1. 先执行 semantic_engine.ingest_conversation()；其中已包含新颖性检查。
        2. 仅当语义引擎确认存在新信息时，再执行技能提取。
        """
        ingest_result = self.semantic_engine.ingest_conversation(
            turns=turns,
            session_id=session_id,
            retrieved_context=retrieved_context,
            turn_index_offset=turn_index_offset,
        )

        steps: list[dict[str, Any]] = list(ingest_result.steps)
        has_novelty = bool(ingest_result.has_novelty)

        if not has_novelty:
            steps.append({
                "name": "skill_extraction",
                "status": "skipped",
                "detail": "无新信息，跳过技能提取",
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

        response = self._call_llm(
            conversation_text,
            system_content=self.prompt_template,
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
