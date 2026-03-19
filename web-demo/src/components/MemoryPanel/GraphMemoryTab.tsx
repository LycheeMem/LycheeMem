import { DeleteOutlined, LinkOutlined, NodeIndexOutlined, PlusOutlined, PushpinOutlined, SearchOutlined } from "@ant-design/icons";
import { useCallback, useEffect, useState } from "react";
import { addGraphNode, deleteGraphNode, fetchGraphData, fetchGraphEdges, searchGraphNodes } from "../../api";
import { useStore } from "../../state";
import type { GraphEdge, GraphNode } from "../../types";
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

type SubTab = "nodes" | "edges";

export default function GraphMemoryTab() {
  const graphEdges = useStore((s) => s.graphEdges);
  const graphData = useStore((s) => s.graphData);
  const hoveredEdge = useStore((s) => s.hoveredEdge);
  const selectedEdge = useStore((s) => s.selectedEdge);
  const setGraphEdges = useStore((s) => s.setGraphEdges);
  const setGraphData = useStore((s) => s.setGraphData);
  const setHoveredEdge = useStore((s) => s.setHoveredEdge);
  const setSelectedEdge = useStore((s) => s.setSelectedEdge);

  const [subTab, setSubTab] = useState<SubTab>("nodes");

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<GraphNode[] | null>(null);
  const [searching, setSearching] = useState(false);

  // Add node form state
  const [showAddForm, setShowAddForm] = useState(false);
  const [newNodeId, setNewNodeId] = useState("");
  const [newNodeLabel, setNewNodeLabel] = useState("Entity");
  const [addError, setAddError] = useState("");

  useEffect(() => {
    fetchGraphEdges().then(setGraphEdges).catch(() => {});
    fetchGraphData().then(setGraphData).catch(() => {});
  }, [setGraphEdges, setGraphData]);

  const nodeById = new Map(
    (graphData.nodes || []).map((n) => [n.id, n])
  );
  const displayNodes = searchResults ?? graphData.nodes;

  const handleSearch = useCallback(async () => {
    const q = searchQuery.trim();
    if (!q) {
      setSearchResults(null);
      return;
    }
    setSearching(true);
    try {
      const data = await searchGraphNodes(q, 20);
      setSearchResults(data.nodes);
    } catch {
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  }, [searchQuery]);

  const handleSearchKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  const clearSearch = () => {
    setSearchQuery("");
    setSearchResults(null);
  };

  const reload = useCallback(async () => {
    try { setGraphData(await fetchGraphData()); } catch { /* */ }
    try { setGraphEdges(await fetchGraphEdges()); } catch { /* */ }
  }, [setGraphData, setGraphEdges]);

  const handleAddNode = async () => {
    const id = newNodeId.trim();
    if (!id) { setAddError("节点 ID 不能为空"); return; }
    setAddError("");
    try {
      await addGraphNode(id, newNodeLabel.trim() || "Entity");
      setNewNodeId("");
      setNewNodeLabel("Entity");
      setShowAddForm(false);
      await reload();
    } catch (err) {
      setAddError(err instanceof Error ? err.message : "添加失败");
    }
  };

  const handleDeleteNode = async (nodeId: string) => {
    if (!window.confirm(`确定删除节点「${nodeId}」及其所有关联边？此操作不可撤销。`)) return;
    try {
      await deleteGraphNode(nodeId);
      await reload();
    } catch { /* ignore */ }
  };

  const activeEdge = selectedEdge || hoveredEdge;
  const activeKey = activeEdge ? edgeKey(activeEdge) : null;

  return (
    <>
      {/* Sub-tabs: Nodes / Edges */}
      <div className="crud-subtabs">
        <button className={`crud-subtab${subTab === "nodes" ? " active" : ""}`} onClick={() => setSubTab("nodes")}>
          <NodeIndexOutlined /> 节点 ({graphData.nodes.length})
        </button>
        <button className={`crud-subtab${subTab === "edges" ? " active" : ""}`} onClick={() => setSubTab("edges")}>
          <LinkOutlined /> 边 ({graphEdges.length})
        </button>
      </div>

      {/* ── Nodes sub-tab ── */}
      {subTab === "nodes" && (
        <>
          {/* Search bar */}
          <div className="crud-toolbar">
            <div className="crud-search">
              <input
                className="crud-search-input"
                type="text"
                placeholder="搜索图谱节点…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={handleSearchKeyDown}
              />
              <button className="crud-btn crud-btn-sm" onClick={handleSearch} disabled={searching}>
                <SearchOutlined />
              </button>
              {searchResults && (
                <button className="crud-btn crud-btn-sm crud-btn-ghost" onClick={clearSearch}>清除</button>
              )}
            </div>
            <button
              className="crud-btn crud-btn-primary crud-btn-sm"
              onClick={() => setShowAddForm(!showAddForm)}
              title="添加节点"
            >
              <PlusOutlined /> 添加
            </button>
          </div>

          {/* Add node form */}
          {showAddForm && (
            <div className="crud-form">
              <input
                className="crud-input"
                type="text"
                placeholder="节点 ID（必填）"
                value={newNodeId}
                onChange={(e) => setNewNodeId(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAddNode()}
              />
              <input
                className="crud-input"
                type="text"
                placeholder="标签（默认 Entity）"
                value={newNodeLabel}
                onChange={(e) => setNewNodeLabel(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAddNode()}
              />
              <div className="crud-form-actions">
                <button className="crud-btn crud-btn-primary" onClick={handleAddNode}>确认添加</button>
                <button className="crud-btn crud-btn-ghost" onClick={() => setShowAddForm(false)}>取消</button>
              </div>
              {addError && <div className="crud-error">{addError}</div>}
            </div>
          )}

          {searchResults && (
            <div className="crud-search-hint">
              搜索结果：{searchResults.length} 个节点
            </div>
          )}

          {/* Node list */}
          <div className="memory-list">
            {displayNodes.length === 0 ? (
              <div className="empty-hint">{searchResults ? "未找到匹配节点" : "暂无图谱节点"}</div>
            ) : (
              displayNodes.map((n) => (
                <div key={n.id} className="memory-item">
                  <div className="mem-label graph mem-label-row">
                    <span>
                      <NodeIndexOutlined /> {escapeHtml(n.label || n.id)}
                    </span>
                    <button
                      className="crud-btn-icon crud-btn-danger"
                      title="删除节点"
                      onClick={() => handleDeleteNode(n.id)}
                    >
                      <DeleteOutlined />
                    </button>
                  </div>
                  {n.typeLabel && <div className="mem-meta">类型: {escapeHtml(n.typeLabel)}</div>}
                  <div className="mem-meta">ID: {escapeHtml(n.id)}</div>
                </div>
              ))
            )}
          </div>
        </>
      )}

      {/* ── Edges sub-tab ── */}
      {subTab === "edges" && (
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
                    onMouseEnter={() => { if (!selectedEdge) setHoveredEdge(e); }}
                    onMouseLeave={() => { if (!selectedEdge) setHoveredEdge(null); }}
                    onClick={() => { setSelectedEdge(e); setHoveredEdge(e); }}
                  >
                    <div className="mem-label graph"><LinkOutlined /> 图谱事实</div>
                    <div className="mem-content">
                      {escapeHtml(getEdgeFact(e, nodeById))}
                    </div>
                    <div className="mem-meta">
                      {escapeHtml(srcLabel as string)} --[{escapeHtml(e.relation || "?")}]--&gt;{" "}
                      {escapeHtml(tgtLabel as string)}
                    </div>
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
      )}
    </>
  );
}
