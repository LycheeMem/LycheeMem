"""
记忆固化 Agent (Memory Consolidation Agent)。

异步后台进程，在每次交互结束后：
1. 分析完整对话记录
2. 提取并更新语义记忆（CompactSemanticEngine）
3. 提取成功的工具调用链 → 存入技能库
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

from src.agents.base_agent import BaseAgent
from src.embedder.base import BaseEmbedder
from src.llm.base import BaseLLM
from src.memory.procedural.sqlite_skill_store import SQLiteSkillStore
from src.memory.semantic.base import BaseSemanticMemoryEngine

CONSOLIDATION_SYSTEM_PROMPT = """\
You are a Memory Consolidator.
You need to review the complete conversation log that just ended and determine whether it contains content worth storing as long-term memory.

Focus on two kinds of information:
1. Graph facts:
     - User preferences, project attributes, stable objective facts, and similar information that can be represented as subject-relation-object triples.
     - A dedicated entity-recognition / triple-generation component will produce the actual triples later,
         so you only need to decide whether such facts are worth extracting. Do not output triples directly.
2. Procedural skills:
     - If the conversation contains a **successful multi-step tool-use chain or operating procedure**,
         distill it into a reusable workflow template.

Reply in JSON using exactly the following structure and field names:
{
    "new_skills": [
        {
            "intent": "One-sentence description of the task intent",
            "doc_markdown": "# Skill Title\\n\\nWrite a reusable Markdown operation guide that may include steps, commands, notes, and input/output details"
        }
    ],
    "should_extract_entities": true/false
}

Rules:
- If the conversation does not contain a complex operational pattern worth saving, `new_skills` must be an empty array.
- If the main content of this turn is **using an already existing skill** to complete a task, rather than **defining or teaching a new skill**, then `new_skills` must be empty because the skill already exists.
  The message block "Existing Skill List" will provide all current skill intents. Use it when making this judgment.
- Set `should_extract_entities` to `true` if any of the following is present:
    · User preferences, technology choices, coding habits, or similar personal / project preferences
    · Concrete plans, schedules, deadlines, or milestones
    · Team assignments, members, or role responsibilities
    · Objective facts such as locations, project names, tools, organizational relationships, contracts, or agreements
    · Updates or corrections to existing information
    · Operational procedures, agreed steps, or conventions
    · Any content the user explicitly asked the system to remember
- Set it to `false` only when the conversation is purely casual chat, simple querying of already known information, repetition of known facts, or contains no substantive information at all.
- **Bias toward `true`. Return `false` only when you are highly confident that the conversation contains no new fact worth long-term storage.**
- Ignore repeated phrasing, obviously failed attempts, and similar content that is not worth saving long term.
- **The output must be strict JSON** with no code fences. JSON strings must not contain raw line breaks; use `\\n` instead.

Requirements for `doc_markdown`:
- It must be plain Markdown text, not JSON or YAML.
- It should preferably include: applicable scenario, prerequisites, numbered steps, key commands or code blocks, common errors, and troubleshooting.

Note: "Graph" here means relationship graph / long-term fact storage. You do not need to output triples directly in this step.

Below are several examples for format and extraction criteria only. Do not copy them verbatim.

## Example 1: Contains both graph facts and a new skill
<session_log>
user: I want this project to standardize on Python 3.10, and all new services should be deployed to the prod-a Kubernetes cluster.
assistant: Understood. I will remember that Python 3.10 is the standard language and prod-a is the deployment target.
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
    ],
    "should_extract_entities": true
}

## Example 2: Contains graph facts only, with no reusable skill
<session_log>
user: From now on, all documents in this project must be written in Chinese. Do not give me English templates again.
assistant: Understood. Documentation for this project will use Chinese consistently.
</session_log>

Expected JSON output:
{
    "new_skills": [],
    "should_extract_entities": true
}

## Example 3: Pure small talk, no consolidation needed
<session_log>
user: I'm in a good mood today. Let's just chat casually.
assistant: Sure, we can talk about something light.
</session_log>

Expected JSON output:
{
    "new_skills": [],
    "should_extract_entities": false
}

## Example 4: Team assignment + project plan + deadline, should be consolidated
<session_log>
user: In the next iteration of our recommendation-system project, due on 3.31, we need to expand the feature dimensions for user profiles. Wang Ming, Zhao Lin, and I are responsible for this module. Wang Ming will handle new behavioral-feature event tracking design and data reporting, Zhao Lin will handle the ETL processing logic, and I will complete feature adaptation and offline training validation for the recommendation model. The design review is expected to be completed within the first 2 days.
assistant: Noted. The next recommendation-system iteration involves you, Wang Ming, and Zhao Lin. The feature-expansion project is due on 3.31, the assignments are confirmed, and the design review should finish within 2 days.
</session_log>

Expected JSON output:
{
    "new_skills": [],
    "should_extract_entities": true
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
        """Compact 后端路径：LLM 分析 → 条件语义固化 + 技能抽取。

        流程（串行+并行结合）：
        1. LLM 分析对话，输出 should_extract_entities + new_skills（串行，因为 ingest 依赖其结果）
        2. 若 should_extract_entities=True，则执行 semantic_engine.ingest_conversation()
           与 _write_skills() 并行（两者互不依赖）
        3. should_extract_entities=False 时跳过语义固化，只写技能（如有）
        """
        steps: list[dict[str, Any]] = []
        conversation_text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)

        # Step 0: 已有技能列表注入（LLM 层去重防线）
        # 拉取当前用户所有技能 intent，以 numbered list 形式前置于对话文本，
        # 让 LLM 感知"哪些技能已经存在"，从而不再重复抽取"使用已有技能"的对话。
        existing_skills = self.skill_store.get_all()
        if existing_skills:
            existing_block_lines = ["## Existing Skill List (for reference only, to avoid duplicate extraction)"]
            for idx, s in enumerate(existing_skills[:30], 1):
                existing_block_lines.append(f"{idx}. {s['intent']}")
            existing_block_lines.append("")
            conversation_text = "\n".join(existing_block_lines) + "\n## Current Conversation Log\n" + conversation_text

        # Step 1: LLM 整体分析（串行，结果决定后续步骤）
        response = self._call_llm(
            conversation_text,
            system_content=self.prompt_template,
            add_time_basis=True,
        )
        analysis = self._safe_parse(response)
        should_extract: bool = bool(analysis.get("should_extract_entities", True))

        steps.append({
            "name": "novelty_check",
            "status": "done",
            "detail": "提取语义" if should_extract else "跳过语义固化（LLM 判定无值得固化的事实）",
        })

        # Step 2: 条件语义固化与技能抽取（并行）
        def _do_ingest():
            return self.semantic_engine.ingest_conversation(
                turns=turns,
                session_id=session_id,
                retrieved_context=retrieved_context,
                turn_index_offset=turn_index_offset,
            )

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

        from src.memory.semantic.base import ConsolidationResult as _CR

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ingest_future = executor.submit(_do_ingest) if should_extract else None
            skill_future = executor.submit(_write_skills)

            ingest_result: _CR = (
                ingest_future.result()
                if ingest_future is not None
                else _CR(records_added=0, records_merged=0, records_expired=0, steps=[{
                    "name": "semantic_ingest", "status": "skipped",
                    "detail": "should_extract_entities=false，跳过语义固化",
                }])
            )
            skills_added: int = skill_future.result()

        steps.extend(ingest_result.steps)
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
            "has_novelty": ingest_result.records_added > 0 or ingest_result.records_merged > 0 or ingest_result.records_expired > 0,
            "steps": steps,
        }

    def _safe_parse(self, response: str) -> dict[str, Any]:
        """安全解析 LLM 输出，失败时返回安全默认值。"""
        try:
            return self._parse_json(response)
        except (ValueError, KeyError):
            return {"new_skills": [], "should_extract_entities": False}
