import { ThunderboltOutlined } from "@ant-design/icons";
import { useEffect } from "react";
import { fetchSkills } from "../../api";
import { useStore } from "../../state";
import { escapeHtml, formatContent } from "../../utils";

export default function SkillsTab() {
  const skills = useStore((s) => s.skills);
  const setSkills = useStore((s) => s.setSkills);

  useEffect(() => {
    fetchSkills()
      .then(setSkills)
      .catch(() => {});
  }, [setSkills]);

  if (!skills.length) {
    return <div className="empty-hint">暂无技能记忆</div>;
  }

  return (
    <div className="memory-list">
      {skills.map((s, i) => {
        const title = s.intent || s.name || s.skill_id || s.id || "skill";
        const doc = (s.doc_markdown || s.doc || s.markdown || "").toString();
        const desc = doc || s.conditions || "";

        return (
          <div key={i} className="memory-item">
            <div className="mem-label skill"><ThunderboltOutlined /> {escapeHtml(title)}</div>
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
  );
}
