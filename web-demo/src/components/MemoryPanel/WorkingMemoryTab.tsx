import { RobotOutlined, UserOutlined } from "@ant-design/icons";
import { useStore } from "../../state";
import { escapeHtml } from "../../utils";

export default function WorkingMemoryTab() {
  const wmTokenUsage = useStore((s) => s.wmTokenUsage);
  const wmMaxTokens = useStore((s) => s.wmMaxTokens);
  const wmTurns = useStore((s) => s.wmTurns);

  const pct = Math.min(100, (wmTokenUsage / wmMaxTokens) * 100);

  return (
    <>
      <div className="wm-stats">
        <div className="stat-bar">
          <div className="stat-label">Token 使用量</div>
          <div className="progress-bar">
            <div
              className={`progress-fill${pct > 70 ? " warn" : ""}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="stat-value">
            {wmTokenUsage.toLocaleString()} / {wmMaxTokens.toLocaleString()}
          </div>
        </div>
      </div>
      <div className="memory-list">
        {wmTurns.slice(-20).map((t, i) => (
          <div key={i} className="memory-item">
            <div className="mem-label turn">
              {t.role === "user" ? <><UserOutlined /> USER</> : <><RobotOutlined /> ASSISTANT</>}
            </div>
            <div className="mem-content">
              {escapeHtml((t.content || "").slice(0, 200))}
              {t.content && t.content.length > 200 ? "…" : ""}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}
