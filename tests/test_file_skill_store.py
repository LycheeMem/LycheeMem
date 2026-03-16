"""测试本地文件技能库（FileSkillStore）。"""

import json
from pathlib import Path

from a_frame.memory.procedural.file_skill_store import FileSkillStore


class TestFileSkillStore:
    def test_add_and_persist_and_reload(self, tmp_path: Path):
        fp = tmp_path / "skills.json"
        store = FileSkillStore(file_path=str(fp))

        store.add([
            {
                "id": "s1",
                "intent": "test",
                "embedding": [0.1] * 8,
                "doc_markdown": "# test\n\nstep1\n",
                "metadata": {"a": 1},
                "conditions": "always",
            }
        ])

        assert fp.exists()
        # 重新加载
        store2 = FileSkillStore(file_path=str(fp))
        all_skills = store2.get_all()
        assert len(all_skills) == 1
        assert all_skills[0]["id"] == "s1"
        assert all_skills[0]["conditions"] == "always"

    def test_search_cosine_similarity(self, tmp_path: Path):
        fp = tmp_path / "skills.json"
        store = FileSkillStore(file_path=str(fp))
        store.add([
            {
                "id": "s1",
                "intent": "A",
                "embedding": [1.0, 0.0, 0.0],
                "doc_markdown": "# A\n",
            },
            {
                "id": "s2",
                "intent": "B",
                "embedding": [0.0, 1.0, 0.0],
                "doc_markdown": "# B\n",
            },
        ])

        results = store.search("q", top_k=1, query_embedding=[1.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0]["id"] == "s1"

    def test_delete_persists(self, tmp_path: Path):
        fp = tmp_path / "skills.json"
        store = FileSkillStore(file_path=str(fp))
        store.add([
            {"id": "s1", "intent": "A", "embedding": [0.1], "doc_markdown": "# A"},
            {"id": "s2", "intent": "B", "embedding": [0.2], "doc_markdown": "# B"},
        ])
        store.delete(["s1"])

        store2 = FileSkillStore(file_path=str(fp))
        ids = {s["id"] for s in store2.get_all()}
        assert "s1" not in ids
        assert "s2" in ids

    def test_record_usage_updates_fields(self, tmp_path: Path):
        fp = tmp_path / "skills.json"
        store = FileSkillStore(file_path=str(fp))
        store.add([
            {"id": "s1", "intent": "A", "embedding": [0.1], "doc_markdown": "# A"},
        ])
        store.record_usage("s1")

        all_skills = store.get_all()
        assert all_skills[0]["success_count"] == 1
        assert all_skills[0]["last_used"] is not None

    def test_load_invalid_file_graceful(self, tmp_path: Path):
        fp = tmp_path / "skills.json"
        fp.write_text("not json", encoding="utf-8")
        store = FileSkillStore(file_path=str(fp))
        assert store.get_all() == []

    def test_load_non_list_json_graceful(self, tmp_path: Path):
        fp = tmp_path / "skills.json"
        fp.write_text(json.dumps({"a": 1}), encoding="utf-8")
        store = FileSkillStore(file_path=str(fp))
        assert store.get_all() == []
