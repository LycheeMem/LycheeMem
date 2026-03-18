import { LinkOutlined, PushpinOutlined } from "@ant-design/icons";
import { useEffect } from "react";
import { fetchGraphEdges } from "../../api";
import { useStore } from "../../state";
import type { GraphEdge } from "../../types";
import { escapeHtml } from "../../utils";

function edgeKey(e: GraphEdge): string {
  const s = typeof e.source === "object" ? (e.source as { id: string }).id : e.source;
  const t = typeof e.target === "object" ? (e.target as { id: string }).id : e.target;
  return `${s}::${e.relation}::${t}::${e.timestamp}`;
}

function getEdgeFact(e: GraphEdge, nodeById: Map<string, { label: string }>): string {
  const fact = (e.fact || "").toString().trim();
  if (fact) return fact;
  const s = typeof e.source === "object" ? (e.source as { id: string }).id : e.source;
  const t = typeof e.target === "object" ? (e.target as { id: string }).id : e.target;
  const srcLabel = nodeById.get(s as string)?.label || s || "?";
  const tgtLabel = nodeById.get(t as string)?.label || t || "?";
  return `${srcLabel} ${e.relation || "?"} ${tgtLabel}`;
}

export default function GraphMemoryTab() {
  const graphEdges = useStore((s) => s.graphEdges);
  const graphData = useStore((s) => s.graphData);
  const hoveredEdge = useStore((s) => s.hoveredEdge);
  const selectedEdge = useStore((s) => s.selectedEdge);
  const setGraphEdges = useStore((s) => s.setGraphEdges);
  const setHoveredEdge = useStore((s) => s.setHoveredEdge);
  const setSelectedEdge = useStore((s) => s.setSelectedEdge);

  useEffect(() => {
    fetchGraphEdges()
      .then(setGraphEdges)
      .catch(() => {});
  }, [setGraphEdges]);

  const nodeById = new Map(
    (graphData.nodes || []).map((n) => [n.id, n])
  );

  const activeEdge = selectedEdge || hoveredEdge;
  const activeKey = activeEdge ? edgeKey(activeEdge) : null;

  return (
    <>
      {/* Edge detail card */}
      {activeEdge && (
        <div className="memory-item edge-detail">
          <div className="mem-label graph">
            {selectedEdge ? <><PushpinOutlined /> 选中边</> : "悬浮边"}
          </div>
          <div className="mem-content">{escapeHtml(getEdgeFact(activeEdge, nodeById))}</div>
          <div className="mem-meta">
            {escapeHtml(
              nodeById.get(
                typeof activeEdge.source === "object"
                  ? (activeEdge.source as { id: string }).id
                  : (activeEdge.source as string)
              )?.label ||
                activeEdge.source ||
                "?"
            )}{" "}
            --[{escapeHtml(activeEdge.relation || "?")}]--&gt;{" "}
            {escapeHtml(
              nodeById.get(
                typeof activeEdge.target === "object"
                  ? (activeEdge.target as { id: string }).id
                  : (activeEdge.target as string)
              )?.label ||
                activeEdge.target ||
                "?"
            )}
          </div>
          {/* {activeEdge.evidence && (
            <div className="mem-meta">
              证据: {escapeHtml(activeEdge.evidence.slice(0, 200))}
            </div>
          )} */}
          {(activeEdge.confidence != null || activeEdge.timestamp || activeEdge.source_session) && (
            <div className="mem-meta">
              {activeEdge.confidence != null &&
                `置信度: ${(activeEdge.confidence * 100).toFixed(0)}%`}
              {activeEdge.timestamp &&
                `${activeEdge.confidence != null ? " | " : ""}时间: ${activeEdge.timestamp}`}
              {activeEdge.source_session &&
                `${activeEdge.confidence != null || activeEdge.timestamp ? " | " : ""}会话: ${activeEdge.source_session}`}
            </div>
          )}
        </div>
      )}

      {/* Edge list */}
      <div className="memory-list">
        {graphEdges.length === 0 ? (
          <div className="empty-hint">暂无图谱事实边</div>
        ) : (
          graphEdges.map((e, i) => {
            const s = typeof e.source === "object" ? (e.source as { id: string }).id : e.source;
            const t = typeof e.target === "object" ? (e.target as { id: string }).id : e.target;
            const srcLabel = nodeById.get(s as string)?.label || s || "?";
            const tgtLabel = nodeById.get(t as string)?.label || t || "?";
            const key = edgeKey(e);
            const isActive = key === activeKey;

            return (
              <div
                key={i}
                className={`memory-item${isActive ? " edge-active" : ""}`}
                onMouseEnter={() => {
                  if (!selectedEdge) setHoveredEdge(e);
                }}
                onMouseLeave={() => {
                  if (!selectedEdge) setHoveredEdge(null);
                }}
                onClick={() => {
                  setSelectedEdge(e);
                  setHoveredEdge(e);
                }}
              >
                <div className="mem-label graph"><LinkOutlined /> 图谱事实</div>
                <div className="mem-content">
                  {escapeHtml(getEdgeFact(e, nodeById))}
                </div>
                <div className="mem-meta">
                  {escapeHtml(srcLabel as string)} --[{escapeHtml(e.relation || "?")}]--&gt;{" "}
                  {escapeHtml(tgtLabel as string)}
                </div>
                {/* {e.evidence && (
                  <div className="mem-meta">
                    证据: {escapeHtml(e.evidence.slice(0, 200))}
                  </div>
                )} */}
                {(e.confidence != null || e.timestamp) && (
                  <div className="mem-meta">
                    {e.confidence != null &&
                      `置信度: ${(e.confidence * 100).toFixed(0)}%`}
                    {e.confidence != null && e.timestamp ? " | " : ""}
                    {e.timestamp && `时间: ${escapeHtml(e.timestamp)}`}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </>
  );
}
