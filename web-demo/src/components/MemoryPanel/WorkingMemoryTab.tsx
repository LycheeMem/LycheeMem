import { CompressOutlined, DownOutlined, RobotOutlined, UserOutlined } from "@ant-design/icons";
import { useState } from "react";
import { useStore } from "../../state";
import { escapeHtml } from "../../utils";

export default function WorkingMemoryTab() {
  const wmTokenUsage = useStore((s) => s.wmTokenUsage);
  const wmMaxTokens = useStore((s) => s.wmMaxTokens);
  const wmTurns = useStore((s) => s.wmTurns);
  const wmBoundaryIndex = useStore((s) => s.wmBoundaryIndex);
  const wmSummaries = useStore((s) => s.wmSummaries);
  const [isCompressedExpanded, setIsCompressedExpanded] = useState(false);

  const pct = Math.min(100, (wmTokenUsage / wmMaxTokens) * 100);

  // 分离被压缩的对话和近期对话
  const compressedTurns = wmBoundaryIndex > 0 ? wmTurns.slice(0, wmBoundaryIndex) : [];
  const recentTurns = wmBoundaryIndex > 0 ? wmTurns.slice(wmBoundaryIndex) : wmTurns;

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
        {/* 被压缩的对话（可展开/收起） */}
        {compressedTurns.length > 0 && (
          <>
            <div
              className="compressed-header"
              onClick={() => setIsCompressedExpanded(!isCompressedExpanded)}
              style={{ cursor: "pointer", padding: "8px" }}
            >
              <span>
                <DownOutlined
                  style={{
                    transform: isCompressedExpanded ? "" : "rotate(-90deg)",
                    transition: "transform 0.2s",
                    marginRight: "8px",
                  }}
                />
                被压缩的对话 ({compressedTurns.length} 条)
              </span>
            </div>
            {isCompressedExpanded && (
              <div style={{ opacity: 0.5, borderLeft: "2px solid #ccc", paddingLeft: "8px" }}>
                {compressedTurns.map((t, i) => (
                  <div key={i} className="memory-item">
                    <div className="mem-label turn">
                      {t.role === "user" ? (
                        <><UserOutlined /> USER</>
                      ) : (
                        <><RobotOutlined /> ASSISTANT</>
                      )}
                    </div>
                    <div className="mem-content">
                      {escapeHtml((t.content || "").slice(0, 150))}
                      {t.content && t.content.length > 150 ? "…" : ""}
                    </div>
                    {!!t.token_count && (
                      <div className="mem-token-count">{t.token_count} tokens</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* 压缩摘要卡片 */}
        {wmSummaries.length > 0 && (
          <div className="memory-item">
            <div className="mem-label">
              <CompressOutlined /> 压缩摘要
            </div>
            <div className="mem-content" style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
              {wmSummaries[wmSummaries.length - 1].content}
            </div>
            {!!wmSummaries[wmSummaries.length - 1].token_count && (
              <div className="mem-token-count">{wmSummaries[wmSummaries.length - 1].token_count} tokens</div>
            )}
          </div>
        )}

        {/* 近期对话（最多 20 条） */}
        {recentTurns.slice().map((t, i) => (
          <div key={i} className="memory-item">
            <div className="mem-label turn">
              {t.role === "user" ? (
                <><UserOutlined /> USER</>
              ) : t.role === "assistant" ? (
                <><RobotOutlined /> ASSISTANT</>
              ) : (
                <><CompressOutlined /> 摘要</>
              )}
            </div>
            <div className="mem-content">
              {escapeHtml((t.content || "").slice(0, 200))}
              {t.content && t.content.length > 200 ? "…" : ""}
            </div>
            {!!t.token_count && (
              <div className="mem-token-count">{t.token_count} tokens</div>
            )}
          </div>
        ))}
      </div>
    </>
  );
}
