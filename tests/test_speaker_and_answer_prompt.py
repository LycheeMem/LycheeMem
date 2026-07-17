from __future__ import annotations

from src.agents.reasoning_agent import ReasoningAgent
from src.memory.semantic.encoder import CompactSemanticEncoder
from src.memory.working.sqlite_session_store import SQLiteSessionStore


class _NoopLLM:
    def generate(self, messages: list[dict[str, str]]) -> str:
        return "unused"


def test_encoder_renders_explicit_turn_index_and_speaker() -> None:
    rendered = CompactSemanticEncoder._format_section([
        {
            "role": "user",
            "speaker": "Caroline",
            "content": "Caroline: I moved from Sweden.",
        },
        {
            "role": "user",
            "speaker": "Melanie",
            "content": "Melanie: I painted a sunrise.\n[Image caption]: a lake",
        },
    ])

    assert '<TURN index=0 role="user" speaker="Caroline">' in rendered
    assert '<TURN index=1 role="user" speaker="Melanie">' in rendered
    assert "Caroline: I moved" not in rendered
    assert "I moved from Sweden." in rendered
    assert "[Image caption]: a lake" in rendered


def test_evidence_turns_are_limited_to_current_turns() -> None:
    normalized = CompactSemanticEncoder._normalize_evidence_turns(
        [0, 1, 2, -1, "bad"],
        turn_index_offset=10,
        turn_count=2,
    )

    assert normalized == [10, 11]


def test_sqlite_session_store_round_trips_speaker(tmp_path) -> None:
    store = SQLiteSessionStore(str(tmp_path / "sessions.db"))
    store.append_turn(
        "locomo-session",
        "user",
        "I painted a sunrise.",
        speaker="Melanie",
        created_at="2023-05-08",
    )

    turns = store.get_turns("locomo-session")
    assert turns[0]["speaker"] == "Melanie"
    assert turns[0]["content"] == "I painted a sunrise."
    store.close()


def test_retrieval_answer_rules_only_apply_with_background_context() -> None:
    agent = ReasoningAgent(_NoopLLM())

    with_facts = agent._build_messages(
        "When did Melanie paint a sunrise?",
        background_context="Melanie painted a sunrise in 2022.",
        reference_time="2023-12-31",
    )
    without_facts = agent._build_messages(
        "Write a detailed project plan.",
        background_context="",
    )

    assert "Retrieved-fact answer mode" in with_facts[0]["content"]
    assert "normally no more than 10 words" in with_facts[0]["content"]
    assert "Retrieved-fact answer mode" not in without_facts[0]["content"]
