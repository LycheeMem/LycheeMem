import { EditOutlined, PlusOutlined } from "@ant-design/icons";
import { useCallback, useEffect, useRef, useState } from "react";
import {
    fetchGraphData,
    fetchGraphEdges,
    fetchSessions,
    fetchSessionTurns,
    fetchSkills,
    updateSessionMeta,
} from "../api";
import { useStore } from "../state";
import type { SessionInfo } from "../types";

function getTitle(s: SessionInfo): string {
  if (s.topic) return s.topic;
  if (s.title) return s.title;
  return "新对话";
}

function formatTime(updatedAt?: string | null): string {
  if (!updatedAt) return "";
  try {
    const d = new Date(updatedAt);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffDays = Math.floor(diffMs / 86400000);
    if (diffDays === 0) return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
    if (diffDays === 1) return "昨天";
    if (diffDays < 7) return `${diffDays}天前`;
    return d.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
  } catch {
    return "";
  }
}

export default function SessionSidebar() {
  const sessions = useStore((s) => s.sessions);
  const sessionId = useStore((s) => s.sessionId);
  const setSessions = useStore((s) => s.setSessions);
  const setSessionId = useStore((s) => s.setSessionId);
  const setMessages = useStore((s) => s.setMessages);
  const resetAgents = useStore((s) => s.resetAgents);
  const setGraphData = useStore((s) => s.setGraphData);
  const setGraphEdges = useStore((s) => s.setGraphEdges);
  const setSkills = useStore((s) => s.setSkills);
  const setWmTurns = useStore((s) => s.setWmTurns);
  const setWmTokenUsage = useStore((s) => s.setWmTokenUsage);
  const setWmMaxTokens = useStore((s) => s.setWmMaxTokens);
  const setWmSummaries = useStore((s) => s.setWmSummaries);
  const setWmBoundaryIndex = useStore((s) => s.setWmBoundaryIndex);
  const newSession = useStore((s) => s.newSession);

  // Inline rename state
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);

  const loadSessions = useCallback(async () => {
    try {
      setSessions(await fetchSessions());
    } catch { /* ignore */ }
  }, [setSessions]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Focus input when editing starts
  useEffect(() => {
    if (editingId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingId]);

  const handleSelect = useCallback(async (sid: string) => {
    if (sid === sessionId) return;
    setSessionId(sid);
    setMessages([]);
    resetAgents();
    try {
      const { turns, summaries, boundary_index, wm_current_tokens, wm_max_tokens } =
        await fetchSessionTurns(sid);
      setMessages(
        turns
          .filter((t) => t.role === "user" || t.role === "assistant")
          .map((t) => ({
            role: t.role as "user" | "assistant",
            content: t.content,
            meta: null,
          }))
      );
      setWmTurns(turns);
      setWmSummaries(summaries);
      setWmBoundaryIndex(boundary_index);
      setWmTokenUsage(wm_current_tokens);
      setWmMaxTokens(wm_max_tokens);
    } catch { /* ignore */ }
    try { setGraphData(await fetchGraphData()); } catch { /* */ }
    try { setGraphEdges(await fetchGraphEdges()); } catch { /* */ }
    try { setSkills(await fetchSkills()); } catch { /* */ }
  }, [
    sessionId, setSessionId, setMessages, resetAgents,
    setWmTurns, setWmSummaries, setWmBoundaryIndex, setWmTokenUsage, setWmMaxTokens,
    setGraphData, setGraphEdges, setSkills,
  ]);

  const startEdit = (e: React.MouseEvent, s: SessionInfo) => {
    e.stopPropagation();
    setEditingId(s.session_id);
    setEditValue(s.topic || s.title || "");
  };

  const commitEdit = async (sid: string) => {
    const name = editValue.trim();
    setEditingId(null);
    if (!name) return;
    try {
      await updateSessionMeta(sid, name);
      await loadSessions();
    } catch { /* ignore */ }
  };

  const handleEditKeyDown = (e: React.KeyboardEvent, sid: string) => {
    if (e.key === "Enter") { e.preventDefault(); commitEdit(sid); }
    if (e.key === "Escape") setEditingId(null);
  };

  return (
    <aside id="session-sidebar">
      <div className="sidebar-header">
        <span className="sidebar-title">历史会话</span>
        <button
          className="sidebar-new-btn"
          onClick={newSession}
          title="新建会话"
        >
          <PlusOutlined />
        </button>
      </div>

      <div className="sidebar-list">
        {sessions.length === 0 && (
          <p className="sidebar-empty">暂无会话记录</p>
        )}
        {sessions.map((s) => {
          const isActive = s.session_id === sessionId;
          const isEditing = editingId === s.session_id;
          return (
            <div
              key={s.session_id}
              className={`sidebar-item${isActive ? " active" : ""}`}
              onClick={() => !isEditing && handleSelect(s.session_id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && !isEditing && handleSelect(s.session_id)}
            >
              {isEditing ? (
                <input
                  ref={editInputRef}
                  className="sidebar-rename-input"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={(e) => handleEditKeyDown(e, s.session_id)}
                  onBlur={() => commitEdit(s.session_id)}
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <>
                  <div className="sidebar-item-row">
                    <span className="sidebar-item-title">{getTitle(s)}</span>
                    <button
                      className="sidebar-edit-btn"
                      title="重命名"
                      onClick={(e) => startEdit(e, s)}
                    >
                      <EditOutlined />
                    </button>
                  </div>
                  <div className="sidebar-item-meta">
                    {(s.turn_count ?? 0) > 0 && <span>{s.turn_count} 轮</span>}
                    {s.updated_at && <span>{formatTime(s.updated_at)}</span>}
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
