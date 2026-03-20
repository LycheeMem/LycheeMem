import { DeleteOutlined, SearchOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { useCallback, useEffect, useState } from "react";
import { clearSkillMemory, deleteSkill, fetchSkills } from "../../api";
import { useStore } from "../../state";
import type { SkillItem } from "../../types";
import { escapeHtml, formatContent } from "../../utils";

export default function SkillsTab() {
  const skills = useStore((s) => s.skills);
  const setSkills = useStore((s) => s.setSkills);
  const [filterText, setFilterText] = useState("");

  const reload = useCallback(async () => {
    try { setSkills(await fetchSkills()); } catch { /* */ }
  }, [setSkills]);

  useEffect(() => { reload(); }, [reload]);

  const handleDelete = async (skill: SkillItem) => {
    const id = skill.id || skill.skill_id || "";
    if (!id) return;
    const name = skill.intent || skill.name || id;
    if (!window.confirm(`确定删除技能「${name}」？此操作不可撤销。`)) return;
    try {
      await deleteSkill(id);
      await reload();
    } catch { /* ignore */ }
  };

  const handleClearAll = async () => {
    if (!window.confirm(
      `确定清空所有技能记忆？\n当前共 ${skills.length} 条技能将被永久删除，此操作不可撤销。`
    )) return;
    try {
      await clearSkillMemory();
      await reload();
    } catch { /* ignore */ }
  };

  const filtered = filterText.trim()
    ? skills.filter((s) => {
        const q = filterText.toLowerCase();
        const title = (s.intent || s.name || s.skill_id || s.id || "").toLowerCase();
        const doc = (s.doc_markdown || s.doc || s.markdown || "").toString().toLowerCase();
        return title.includes(q) || doc.includes(q);
      })
    : skills;

  return (
    <>
      <div className="crud-toolbar">
        <div className="crud-search">
          <input
            className="crud-search-input"
            type="text"
            placeholder="过滤技能…"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />
          <SearchOutlined style={{ color: "var(--text-muted)", fontSize: 12 }} />
        </div>
        <button
          className="crud-btn crud-btn-sm"
          style={{ color: "var(--red)", borderColor: "var(--red)" }}
          onClick={handleClearAll}
          title="清空当前用户的所有技能记忆"
          disabled={skills.length === 0}
        >
          <DeleteOutlined /> 清空
        </button>
      </div>

      {filtered.length === 0 ? (
        <div className="empty-hint">{filterText ? "未找到匹配技能" : "暂无技能记忆"}</div>
      ) : (
        <div className="memory-list">
          {filtered.map((s, i) => {
            const skillId = s.id || s.skill_id || "";
            const title = s.intent || s.name || s.skill_id || s.id || "skill";
            const doc = (s.doc_markdown || s.doc || s.markdown || "").toString();
            const desc = doc || s.conditions || "";

            return (
              <div key={i} className="memory-item">
                <div className="mem-label skill mem-label-row">
                  <span><ThunderboltOutlined /> {escapeHtml(title)}</span>
                  {skillId && (
                    <button
                      className="crud-btn-icon crud-btn-danger"
                      title="删除技能"
                      onClick={() => handleDelete(s)}
                    >
                      <DeleteOutlined />
                    </button>
                  )}
                </div>
                <div
                  className="mem-content"
                  dangerouslySetInnerHTML={{
                    __html: formatContent(String(desc).slice(0, 800)),
                  }}
                />
                {(s.success_count !== undefined ||
                  s.score !== undefined ||
                  s.last_used) && (
                  <div className="mem-meta">
                    {s.success_count !== undefined &&
                      `成功次数: ${s.success_count}`}
                    {s.score !== undefined &&
                      ` | 评分: ${(s.score || 0).toFixed(2)}`}
                    {s.last_used &&
                      ` | 最近使用: ${escapeHtml(String(s.last_used))}`}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
