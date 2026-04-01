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
你是记忆固化专家（Memory Consolidator）。
你需要审查刚刚结束的完整对话日志，从中判断是否有值得沉淀为长期记忆的内容。

需要关注两类信息：
1. 图谱事实 (Graph Facts)：
     - 用户偏好、项目属性、稳定的客观事实等，可以表示为 [主体, 关系, 客体] 的三元组；
     - 本系统会在后续步骤中调用专门的实体识别/三元组生成组件来产出具体三元组，
         因此你只需判断是否存在值得沉淀的事实，不必直接输出三元组。
2. 程序技能 (Procedural Skills)：
     - 如果在本次对话中出现了 **成功的多步工具调用/操作流程**，
         请将其提炼为可复用的“工作流模板”。

请以 JSON 格式回复，结构如下（字段名必须保持一致）：
{
    "new_skills": [
        {
            "intent": "任务意图的一句话描述",
            "doc_markdown": "# 技能标题\\n\\n用 Markdown 写一份可复用的操作说明文档（可包含步骤、命令、注意事项、输入输出等）"
        }
    ],
    "should_extract_entities": true/false
}

说明：
- 如果对话没有值得保存的复杂操作模式，`new_skills` 应为一个空数组；
- 如果本轮对话的主要内容是**调用/使用了已存在的技能**来完成任务（例如：使用"以指定格式总结学习周报"技能来写某周的周报），
  而不是在**定义/教授新技能**，则 `new_skills` 应为空数组，因为对应技能已经存在。
  消息块 "已存在技能列表" 中会列出当前技能库的所有 intent，请在判断时参考。
- 满足以下任意一条，即应将 `should_extract_entities` 设为 true：
    · 用户偏好、技术选型、编程习惯等个人或项目偏好
    · 具体计划、日程安排、截止日期、里程碑
    · 人员分工、小组成员、角色职责
    · 客观事实（地点、项目名称、工具、组织关系、合同/协议等）
    · 对已有信息的更新或纠正
    · 操作流程、步骤约定、规范
    · 用户明确要求记住的任何内容（如"帮我记住"、"你记一下"）
- 只有对话内容是纯粹的寒暄、单纯查询已有信息、重复已知事实，或完全没有实质信息时，才设为 false。
- **倾向于 true。只有非常确信对话中没有任何值得长期保存的新事实时，才返回 false。**
- 忽略重复说法、明显错误尝试等不值得长期保存的内容。
- **输出必须是严格 JSON**（不要代码块）。注意：JSON 字符串里不能出现裸换行，换行必须写成 `\\n`。

技能文档（doc_markdown）要求：
- 必须是 Markdown 纯文本，不要输出 JSON/YAML。
- 建议包含：适用场景、前置条件、步骤（编号列表）、关键命令/代码块、常见错误与排查。

注意：这里的图谱指“关系图谱/长期事实存储”，你不需要在本步骤直接输出三元组。

下面是几个示例（只用于帮助你理解格式与抽取标准，不要原样抄写示例中的中文内容）：

## 示例 1：既有图谱事实，也有新技能
<session_log>
user: 我想在这个项目里统一用 Python 3.10，并且所有新服务都部署到 k8s 集群 prod-a 上。
assistant: 好的，我会记住：语言用 Python 3.10，部署目标是 prod-a 集群。
user: 这次帮我把 user-service 做一个蓝绿发布的流程，我想先在 prod-a 的一半节点上灰度。
assistant: 我们可以这么做：
    1）更新 Helm values，把 user-service 的新版本镜像打上 v2 标签；
    2）使用 kubectl apply 应用新的 Deployment；
    3）观察 prometheus 的告警和日志，如果无异常，再将所有副本切到新版本。
</session_log>

期望 JSON 输出示例：
{
    "new_skills": [
        {
            "intent": "对 user-service 执行蓝绿发布到 prod-a 集群",
            "doc_markdown": "# user-service 蓝绿发布（prod-a）\\n\\n## 适用场景\\n- 需要将 user-service 发布到 prod-a，并先灰度一半节点\\n\\n## 步骤\\n1. 更新 Helm values，将镜像标记为 v2\\n2. `kubectl apply` 部署到部分节点\\n3. 观察 Prometheus 告警与日志，无异常后将全部副本切到 v2\\n"
        }
    ],
    "should_extract_entities": true
}

## 示例 2：只有图谱事实，没有可复用技能
<session_log>
user: 以后在这个项目里，所有文档一律用中文撰写，不要再给我英文模版了。
assistant: 明白了，这个项目的文档统一使用中文。
</session_log>

期望 JSON 输出示例：
{
    "new_skills": [],
    "should_extract_entities": true
}

## 示例 3：纯闲聊，不需要固化
<session_log>
user: 哈哈，今天心情不错，随便聊聊八卦吧。
assistant: 好的，我们可以聊点轻松的话题～
</session_log>

期望 JSON 输出示例：
{
    "new_skills": [],
    "should_extract_entities": false
}

## 示例 4：团队分工 + 项目计划 + 截止日期，应固化
<session_log>
user: 我们的推荐系统项目下个迭代（3.31）要做用户画像特征维度扩展。我、王明、赵琳负责这个模块。分工安排：王明负责新增行为特征埋点设计和数据上报，赵琳来处理特征的ETL加工逻辑，我需要完成推荐模型的特征适配和离线训练验证。前期预计2天内要完成设计评审。
assistant: 好的，已记录。推荐系统下个迭代成员是你、王明和赵琳。特征维度扩展项目，3.31截止，分工确认。设计评审计划2天完成，我会提醒。
</session_log>

期望 JSON 输出示例：
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
        user_id: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """分析对话并固化到长期记忆。

        Args:
            turns: 完整的对话轮次列表。
            session_id: 会话 ID。
            retrieved_context: Pipeline 检索阶段合成的已有记忆上下文，
                用于与本轮对话比对，判断是否有新信息需要固化。

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
            user_id=user_id,
        )

    def _run_compact(
        self,
        *,
        turns: list[dict[str, Any]],
        session_id: str,
        retrieved_context: str = "",
        user_id: str = "",
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
        existing_skills = self.skill_store.get_all(user_id=user_id)
        if existing_skills:
            existing_block_lines = ["## 已存在技能列表（仅供参考，避免重复抽取）"]
            for idx, s in enumerate(existing_skills[:30], 1):
                existing_block_lines.append(f"{idx}. {s['intent']}")
            existing_block_lines.append("")
            conversation_text = "\n".join(existing_block_lines) + "\n## 本轮对话日志\n" + conversation_text

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
                user_id=user_id,
                retrieved_context=retrieved_context,
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
                    user_id=user_id,
                )
                if top_existing and top_existing[0].get("score", 0.0) >= DEDUP_THRESHOLD:
                    # 相似技能已存在：upsert 到已有条目（更新内容而非新建）
                    existing_id = top_existing[0]["id"]
                    self.skill_store.add(
                        [{"id": existing_id, "intent": intent, "embedding": embedding, "doc_markdown": doc_markdown}],
                        user_id=user_id,
                    )
                else:
                    self.skill_store.add(
                        [{"intent": intent, "embedding": embedding, "doc_markdown": doc_markdown}],
                        user_id=user_id,
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
