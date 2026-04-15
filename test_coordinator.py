# -*- coding: utf-8 -*-
from src.core.config import settings
from src.llm.litellm_llm import LiteLLMLLM
from src.embedder.litellm_embedder import LiteLLMEmbedder
from src.memory.procedural.sqlite_skill_store import SQLiteSkillStore
from src.memory.semantic.engine import CompactSemanticEngine
from src.memory.working.session_store import InMemorySessionStore
from src.agents.search_coordinator import SearchCoordinator
import json

llm = LiteLLMLLM(model=settings.llm_model)
embedder = LiteLLMEmbedder(model=settings.embedding_model)
session_store = InMemorySessionStore()

skill_store = SQLiteSkillStore(db_path=":memory:", vector_db_path="data/test_skill_vector", embedder=embedder, embedding_dim=1536)
semantic_engine = CompactSemanticEngine(llm=llm, embedder=embedder, session_store=session_store, sqlite_db_path=":memory:", vector_db_path="data/test_compact_vector")

coordinator = SearchCoordinator(
    llm=llm,
    embedder=embedder,
    skill_store=skill_store,
    semantic_engine=semantic_engine
)

user_query = "我的PostgreSQL连接池配置对吗？max_client_conn=1000会不会太大？"
recent_context = "Summary: 用户正在配置生产环境数据库。之前遇到过 'FATAL: sorry, too many clients already' 的报错。"

print(f"Testing query: {user_query}")
print(f"Recent context: {recent_context}")
print("-" * 50)

analysis = coordinator._analyze_query_and_context(user_query, recent_context)
print("\nLLM Analysis Result:")
print(json.dumps(analysis, indent=2, ensure_ascii=False))

print("\nActionState Output:")
action_state = coordinator._build_action_state(
    user_query=user_query,
    recent_context=recent_context,
    wm_token_usage=100,
    tool_calls=[],
    analysis=analysis
)
print(f"tentative_action: '{action_state.tentative_action}'")
print(f"known_constraints: {action_state.known_constraints}")
print(f"available_tools: {action_state.available_tools}")
print(f"failure_signal: '{action_state.failure_signal}'")
