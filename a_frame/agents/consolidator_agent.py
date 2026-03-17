"""
记忆固化 Agent (Memory Consolidation Agent)。

异步后台进程，在每次交互结束后：
1. 分析完整对话记录
2. 提取新的事实/偏好变化 → 更新图谱
3. 提取成功的工具调用链 → 存入技能库
"""

from __future__ import annotations

import datetime
import hashlib
from typing import Any

from typing import TYPE_CHECKING

from a_frame.agents.base_agent import BaseAgent
from a_frame.embedder.base import BaseEmbedder
from a_frame.llm.base import BaseLLM
from a_frame.memory.graph.entity_extractor import EntityExtractor
from a_frame.memory.graph.graph_store import NetworkXGraphStore
from a_frame.memory.procedural.skill_store import InMemorySkillStore

if TYPE_CHECKING:
    from a_frame.memory.graph.graphiti_engine import GraphitiEngine

CONSOLIDATION_SYSTEM_PROMPT = """\
你是一个「记忆固化专家（Memory Consolidator）」。
你需要审查刚刚结束的完整对话日志，从中判断是否有值得沉淀为长期记忆的内容。

需要关注两类信息：
1. 图谱事实 (Graph Facts)：
     - 用户偏好、项目属性、稳定的客观事实等，可以表示为 [主体, 关系, 客体] 的三元组；
     - 本系统会在后续步骤中调用专门的实体识别/三元组生成组件来产出具体三元组，
         因此你只需判断「是否存在值得沉淀的事实」，不必直接输出三元组。
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
- 当你认为对话中包含稳定的用户偏好/事实/关系，适合写入图谱存储时，
    请将 `should_extract_entities` 设为 true；否则为 false；
- 忽略闲聊、重复说法、明显错误尝试等不值得长期保存的内容。
- **输出必须是严格 JSON**（不要代码块）。注意：JSON 字符串里不能出现裸换行，换行必须写成 `\\n`。

技能文档（doc_markdown）要求：
- 必须是 Markdown 纯文本，不要输出 JSON/YAML。
- 建议包含：适用场景、前置条件、步骤（编号列表）、关键命令/代码块、常见错误与排查。

注意：这里的图谱指“关系图谱/长期事实存储”，你不需要在本步骤直接输出三元组。

下面是几个示例（只用于帮助你理解格式与抽取标准，不要原样抄写示例中的中文内容）：

【示例 1：既有图谱事实，也有新技能】
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

【示例 2：只有图谱事实，没有可复用技能】
<session_log>
user: 以后在这个项目里，所有文档一律用中文撰写，不要再给我英文模版了。
assistant: 明白了，这个项目的文档统一使用中文。
</session_log>

期望 JSON 输出示例：
{
    "new_skills": [],
    "should_extract_entities": true
}

【示例 3：纯闲聊，不需要固化】
<session_log>
user: 哈哈，今天心情不错，随便聊聊八卦吧。
assistant: 好的，我们可以聊点轻松的话题～
</session_log>

期望 JSON 输出示例：
{
    "new_skills": [],
    "should_extract_entities": false
}
"""


class ConsolidatorAgent(BaseAgent):
    """记忆固化 Agent：异步分析对话并更新长期记忆。"""

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        graph_store: NetworkXGraphStore,
        skill_store: InMemorySkillStore,
        entity_extractor: EntityExtractor,
        graphiti_engine: "GraphitiEngine | None" = None,
    ):
        super().__init__(llm=llm, prompt_template=CONSOLIDATION_SYSTEM_PROMPT)
        self.embedder = embedder
        self.graph_store = graph_store
        self.skill_store = skill_store
        self.entity_extractor = entity_extractor
        self.graphiti_engine = graphiti_engine

        self._graphiti_last_episode_ingested: dict[str, int] = {}
        self._graphiti_last_user_semantic: dict[str, int] = {}

        self._graphiti_semantic_builder = None
        if self.graphiti_engine is not None and hasattr(self.graphiti_engine, "store"):
            from a_frame.memory.graph.graphiti_semantic import GraphitiSemanticBuilder

            self._graphiti_semantic_builder = GraphitiSemanticBuilder(
                llm=self.llm,
                embedder=self.embedder,
                store=self.graphiti_engine.store,
            )

    @staticmethod
    def _episode_id(*, session_id: str, turn_index: int, role: str, content: str) -> str:
        raw = f"{session_id}|{turn_index}|{role}|{content}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def run(
        self,
        turns: list[dict[str, Any]],
        session_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """分析对话并固化到长期记忆。

        Args:
            turns: 完整的对话轮次列表。

        Returns:
            dict 包含：entities_added (int), skills_added (int)
        """
        if not turns:
            return {"entities_added": 0, "skills_added": 0}

        # 0. Graphiti: Episode raw ingestion + Semantic build（可选；不影响现有检索/回答）
        graphiti_entities_added = 0
        graphiti_facts_added = 0
        if self.graphiti_engine is not None and session_id:
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # 0.1 仅写入新增 turns 的 Episode（幂等）
            last_ingested = self._graphiti_last_episode_ingested.get(session_id, -1)
            for idx in range(last_ingested + 1, len(turns)):
                t = turns[idx] or {}
                role = str(t.get("role") or "")
                content = str(t.get("content") or "")
                if not role and not content:
                    continue
                t_ref = str(t.get("created_at") or now_iso)
                self.graphiti_engine.ingest_episode(
                    session_id=session_id,
                    turn_index=idx,
                    role=role,
                    content=content,
                    t_ref=t_ref,
                    episode_id=self._episode_id(
                        session_id=session_id,
                        turn_index=idx,
                        role=role,
                        content=content,
                    ),
                )
            self._graphiti_last_episode_ingested[session_id] = len(turns) - 1

            # 0.2 只对“最新 user turn”做语义抽取（结合最近 n=4 上下文）
            if self._graphiti_semantic_builder is not None:
                latest_user_idx = None
                for i in range(len(turns) - 1, -1, -1):
                    if str((turns[i] or {}).get("role") or "") == "user":
                        latest_user_idx = i
                        break

                if latest_user_idx is not None:
                    last_semantic = self._graphiti_last_user_semantic.get(session_id, -1)
                    if latest_user_idx > last_semantic:
                        current_turn = turns[latest_user_idx] or {}
                        previous_turns = turns[max(0, latest_user_idx - 4) : latest_user_idx]
                        episode_id = self._episode_id(
                            session_id=session_id,
                            turn_index=latest_user_idx,
                            role=str(current_turn.get("role") or ""),
                            content=str(current_turn.get("content") or ""),
                        )
                        reference_timestamp = str(current_turn.get("created_at") or now_iso)
                        built = self._graphiti_semantic_builder.ingest_user_turn(
                            session_id=session_id,
                            episode_id=episode_id,
                            previous_turns=previous_turns,
                            current_turn=current_turn,
                            reference_timestamp=reference_timestamp,
                        )
                        graphiti_entities_added = int(built.get("entities_added", 0))
                        graphiti_facts_added = int(built.get("facts_added", 0))
                        self._graphiti_last_user_semantic[session_id] = latest_user_idx

                        # PR5: best-effort 社区 refresh（不阻塞主流程；失败不影响固化）
                        try:
                            if hasattr(self.graphiti_engine, "refresh_communities_for_session"):
                                self.graphiti_engine.refresh_communities_for_session(
                                    session_id=session_id,
                                    limit=50,
                                )
                        except Exception:
                            pass

        # 格式化对话用于分析
        conversation_text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)

        # 1. 用 LLM 分析对话，判断是否有新技能和是否需要实体抽取
        response = self._call_llm(
            conversation_text,
            system_content=self.prompt_template,
            add_time_basis=True,
        )
        analysis = self._safe_parse(response)

        entities_added = 0
        skills_added = 0

        # 2. 实体抽取 → 更新图谱
        # Graphiti 模式下使用 Fact-node 语义子图；暂不写入 legacy triples（避免双写与语义冲突）。
        if self.graphiti_engine is not None and session_id:
            entities_added = graphiti_entities_added
        elif analysis.get("should_extract_entities", False):
            triples = self.entity_extractor.extract_from_turns(
                turns, source_session=session_id or ""
            )
            if triples:
                self.graph_store.add(triples)
                entities_added = len(triples)

        # 3. 新技能 → 存入技能库
        new_skills = analysis.get("new_skills", [])
        for skill in new_skills:
            intent = skill.get("intent", "")
            doc_markdown = skill.get("doc_markdown", "")
            if intent and doc_markdown:
                embedding = self.embedder.embed_query(intent)
                self.skill_store.add(
                    [
                        {
                            "intent": intent,
                            "embedding": embedding,
                            "doc_markdown": doc_markdown,
                        }
                    ]
                )
                skills_added += 1

        result: dict[str, Any] = {
            "entities_added": entities_added,
            "skills_added": skills_added,
        }
        if self.graphiti_engine is not None and session_id:
            result["facts_added"] = graphiti_facts_added
        return result

    def _safe_parse(self, response: str) -> dict[str, Any]:
        """安全解析 LLM 输出，失败时返回安全默认值。"""
        try:
            return self._parse_json(response)
        except (ValueError, KeyError):
            return {"new_skills": [], "should_extract_entities": False}
