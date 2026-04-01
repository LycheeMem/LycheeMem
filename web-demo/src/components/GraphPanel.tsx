import {
    ApartmentOutlined,
    DownOutlined,
    ReloadOutlined,
    RightOutlined,
} from "@ant-design/icons";
import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchGraphData, fetchPipelineStatus } from "../api";
import { useStore } from "../state";
import type { GraphTreeNode } from "../types";

function collectCompositeIds(nodes: GraphTreeNode[]): string[] {
  const ids: string[] = [];
  const visit = (node: GraphTreeNode) => {
    if (node.nodeKind === "composite") {
      ids.push(node.id);
    }
    for (const child of node.children) {
      visit(child);
    }
  };
  for (const node of nodes) {
    visit(node);
  }
  return ids;
}

function fallbackTreeRoots(nodes: ReturnType<typeof useStore.getState>["graphData"]["nodes"]): GraphTreeNode[] {
  return nodes.map((node) => ({
    id: node.id,
    label: node.label,
    typeLabel: node.typeLabel,
    nodeKind: node.nodeKind || "record",
    properties: node.properties,
    children: [],
  }));
}

function TreeNodeCard({
  node,
  collapsedIds,
  onToggle,
}: {
  node: GraphTreeNode;
  collapsedIds: Set<string>;
  onToggle: (nodeId: string) => void;
}) {
  const isComposite = node.nodeKind === "composite";
  const hasChildren = node.children.length > 0;
  const collapsed = isComposite && collapsedIds.has(node.id);
  const semanticText = String(node.properties.semantic_text || "").trim();
  const sourceRecordCount = Number(node.properties.source_record_count || (isComposite ? 0 : 1));
  const childCompositeCount = Number(node.properties.child_composite_count || 0);
  const directRecordCount = Number(node.properties.direct_record_count || 0);

  return (
    <li className={`memory-tree-item ${isComposite ? "composite" : "record"}`}>
      <div className="memory-tree-row">
        {isComposite ? (
          <button
            type="button"
            className="memory-tree-toggle"
            onClick={() => onToggle(node.id)}
            title={collapsed ? "展开子节点" : "收起子节点"}
          >
            {collapsed ? <RightOutlined /> : <DownOutlined />}
          </button>
        ) : (
          <span className="memory-tree-leaf-dot" />
        )}

        <div className="memory-tree-card">
          <div className="memory-tree-card-header">
            <span className={`memory-tree-kind ${isComposite ? "composite" : "record"}`}>
              {isComposite ? "融合记忆" : "原子记录"}
            </span>
            {node.typeLabel && <span className="memory-tree-type">{node.typeLabel}</span>}
          </div>
          <div className="memory-tree-title">{node.label}</div>
          {semanticText && <div className="memory-tree-summary">{semanticText}</div>}
          <div className="memory-tree-meta">
            {isComposite ? (
              <>
                <span>{sourceRecordCount} 条底层记录</span>
                <span>{childCompositeCount} 个子记忆</span>
                <span>{directRecordCount} 个直接叶子</span>
              </>
            ) : (
              <>
                {node.properties.created_at && <span>{String(node.properties.created_at)}</span>}
                {node.properties.confidence != null && (
                  <span>置信度 {(Number(node.properties.confidence) * 100).toFixed(0)}%</span>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {hasChildren && !collapsed && (
        <ul className="memory-tree-children">
          {node.children.map((child) => (
            <TreeNodeCard
              key={child.id}
              node={child}
              collapsedIds={collapsedIds}
              onToggle={onToggle}
            />
          ))}
        </ul>
      )}
    </li>
  );
}

export default function GraphPanel() {
  const graphData = useStore((s) => s.graphData);
  const setGraphData = useStore((s) => s.setGraphData);

  // 定时检测图谱更新：保存上次检测的统计数据
  const [lastStatus, setLastStatus] = useState<{
    node_count: number;
    edge_count: number;
  } | null>(null);
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(new Set());

  const treeRoots = useMemo(
    () => (graphData.treeRoots.length > 0 ? graphData.treeRoots : fallbackTreeRoots(graphData.nodes)),
    [graphData.treeRoots, graphData.nodes]
  );
  const compositeIds = useMemo(() => collectCompositeIds(treeRoots), [treeRoots]);

  useEffect(() => {
    setCollapsedIds(new Set());
  }, [treeRoots]);

  const handleRefresh = useCallback(async () => {
    try {
      setGraphData(await fetchGraphData());
    } catch {
      /* ignore */
    }
  }, [setGraphData]);

  useEffect(() => {
    handleRefresh();
  }, [handleRefresh]);

  // 定时检测图谱更新（默认 3 秒轮询一次）
  useEffect(() => {
    const POLL_INTERVAL = 3000; // 3 秒

    const pollGraphStatus = async () => {
      try {
        const status = await fetchPipelineStatus();
        const currentStatus = {
          node_count: status.graph_node_count,
          edge_count: status.graph_edge_count,
        };

        // 对比：如果节点数或边数发生变化，刷新图谱
        if (
          lastStatus === null ||
          lastStatus.node_count !== currentStatus.node_count ||
          lastStatus.edge_count !== currentStatus.edge_count
        ) {
          setLastStatus(currentStatus);
          // 有更新：重新获取完整图谱数据
          if (lastStatus !== null) {
            // 排除第一次初始化，避免重复刷新
            setGraphData(await fetchGraphData());
          }
        }
      } catch {
        /* 轮询失败忽略，继续等待下次轮询 */
      }
    };

    // 启动定时轮询
    const pollTimer = setInterval(pollGraphStatus, POLL_INTERVAL);

    // 清理定时器
    return () => clearInterval(pollTimer);
  }, [lastStatus, setGraphData]);

  const hasNodes = treeRoots.length > 0;
  const allCollapsed = compositeIds.length > 0 && collapsedIds.size >= compositeIds.length;

  const toggleNode = useCallback((nodeId: string) => {
    setCollapsedIds((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, []);

  const toggleAll = useCallback(() => {
    if (compositeIds.length === 0) return;
    setCollapsedIds(allCollapsed ? new Set() : new Set(compositeIds));
  }, [allCollapsed, compositeIds]);

  return (
    <section id="panel-graph" className="panel">
      <div className="panel-header">
        <h2><ApartmentOutlined /> 记忆树</h2>
        <div className="panel-actions">
          <button
            className="icon-btn"
            title="刷新记忆树"
            onClick={handleRefresh}
          >
            <ReloadOutlined />
          </button>
          <button className="icon-btn" title={allCollapsed ? "展开全部" : "折叠全部"} onClick={toggleAll}>
            {allCollapsed ? <DownOutlined /> : <RightOutlined />}
          </button>
        </div>
      </div>
      <div className="graph-container graph-tree-container">
        {!hasNodes && (
          <div className="empty-state">
            <span>暂无树状记忆</span>
            <span className="sub">对话后长期记忆会自动固化并组织为层级树</span>
          </div>
        )}
        {hasNodes && (
          <div className="memory-tree-scroll">
            <ul className="memory-tree-list">
              {treeRoots.map((node) => (
                <TreeNodeCard
                  key={node.id}
                  node={node}
                  collapsedIds={collapsedIds}
                  onToggle={toggleNode}
                />
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}
