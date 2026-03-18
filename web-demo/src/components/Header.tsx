import { FileTextOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { useCallback, useEffect } from "react";
import {
    fetchGraphData,
    fetchGraphEdges,
    fetchPipelineStatus,
    fetchSessions,
    fetchSessionTurns,
    fetchSkills,
} from "../api";
import { useStore } from "../state";

export default function Header() {
  const sessionId = useStore((s) => s.sessionId);
  const sessions = useStore((s) => s.sessions);
  const pipelineStatus = useStore((s) => s.pipelineStatus);
  const setSessions = useStore((s) => s.setSessions);
  const setSessionId = useStore((s) => s.setSessionId);
  const setMessages = useStore((s) => s.setMessages);
  const resetAgents = useStore((s) => s.resetAgents);
  const setGraphData = useStore((s) => s.setGraphData);
  const setGraphEdges = useStore((s) => s.setGraphEdges);
  const setSkills = useStore((s) => s.setSkills);
  const setPipelineStatus = useStore((s) => s.setPipelineStatus);
  const setWmTurns = useStore((s) => s.setWmTurns);
  const newSession = useStore((s) => s.newSession);

  const loadAll = useCallback(async () => {
    try {
      const s = await fetchSessions();
      setSessions(s);
    } catch {
      /* ignore */
    }
    try {
      setPipelineStatus(await fetchPipelineStatus());
    } catch {
      /* ignore */
    }
  }, [setSessions, setPipelineStatus]);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  const handleSessionChange = async (
    e: React.ChangeEvent<HTMLSelectElement>
  ) => {
    const sid = e.target.value;
    if (!sid) return;
    setSessionId(sid);
    setMessages([]);
    resetAgents();

    try {
      const turns = await fetchSessionTurns(sid);
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
    } catch {
      /* ignore */
    }

    try {
      setGraphData(await fetchGraphData());
    } catch {
      /* ignore */
    }
    try {
      setGraphEdges(await fetchGraphEdges());
    } catch {
      /* ignore */
    }
    try {
      setSkills(await fetchSkills());
    } catch {
      /* ignore */
    }
  };

  const handleNewSession = () => {
    newSession();
  };

  return (
    <header id="app-header">
      <div className="header-left">
        <div className="logo">
          <img className="logo-img" src="/logo.png" alt="Logo" />
          <span className="logo-text">
            立知大模型<span className="logo-text">记忆系统</span>
          </span>
        </div>
      </div>
      <div className="header-center">
        <div className="session-selector">
          <select
            id="session-select"
            value={sessionId}
            onChange={handleSessionChange}
          >
            <option value="">新建会话</option>
            {sessions.map((s) => (
              <option key={s.session_id} value={s.session_id}>
                {(s.topic || s.session_id).slice(0, 40)}
              </option>
            ))}
          </select>
          <button
            className="icon-btn"
            title="新建会话"
            onClick={handleNewSession}
          >
            ＋
          </button>
        </div>
      </div>
      <div className="header-right">
        <div className="status-chips" id="status-chips">
          <span className="chip" title="会话数">
            <FileTextOutlined /> {pipelineStatus.session_count}
          </span>
          <span className="chip" title="图谱节点">
            ● {pipelineStatus.graph_node_count}
          </span>
          <span className="chip" title="技能数">
            <ThunderboltOutlined /> {pipelineStatus.skill_count}
          </span>
        </div>
      </div>
    </header>
  );
}
