import { FileTextOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { useCallback, useEffect } from "react";
import { fetchPipelineStatus } from "../api";
import { useStore } from "../state";

export default function Header() {
  const pipelineStatus = useStore((s) => s.pipelineStatus);
  const setPipelineStatus = useStore((s) => s.setPipelineStatus);

  const loadAll = useCallback(async () => {
    try {
      setPipelineStatus(await fetchPipelineStatus());
    } catch {
      /* ignore */
    }
  }, [setPipelineStatus]);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  return (
    <header id="app-header">
      <div className="header-left">
        <div className="logo">
          <img className="logo-img" src="/logo.png" alt="Logo" />
          <span className="logo-text">
            LycheeMem 立知大模型记忆系统
          </span>
        </div>
      </div>
      <div className="header-right">
        <div className="status-chips" id="status-chips">
          <span className="chip" title="会话数">
            <FileTextOutlined /> {pipelineStatus.session_count}
          </span>
          <span className="chip" title="记忆树节点">
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
