import {
  ApartmentOutlined,
  CompressOutlined,
  ExpandOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { hierarchy, tree } from "d3-hierarchy";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchGraphData, fetchPipelineStatus } from "../api";
import { useStore } from "../state";
import type { GraphTreeNode } from "../types";

// ── Tree layout constants ──────────────────────────────────────────────────

const NODE_W = 200;
const NODE_H = 90;
const H_GAP = 240;   // horizontal slot width per node (passed to d3 nodeSize)
const V_GAP = 140;   // vertical gap between levels (passed to d3 nodeSize)
const TREE_V_GAP = 16; // vertical gap between independently stacked root trees

interface HierarchyDatum extends GraphTreeNode {
  _virtual?: boolean;
}

// ── Collect all composite ids (even deep) ─────────────────────────────────

function collectCompositeIds(nodes: GraphTreeNode[]): string[] {
  const ids: string[] = [];
  const visit = (n: GraphTreeNode) => {
    if (n.nodeKind === "composite") ids.push(n.id);
    for (const c of n.children) visit(c);
  };
  for (const n of nodes) visit(n);
  return ids;
}

// ── Fallback when treeRoots not yet received ───────────────────────────────

function fallbackTreeRoots(
  nodes: ReturnType<typeof useStore.getState>["graphData"]["nodes"]
): GraphTreeNode[] {
  return nodes.map((n) => ({
    id: n.id,
    label: n.label,
    typeLabel: n.typeLabel,
    nodeKind: n.nodeKind || "record",
    properties: n.properties,
    children: [],
  }));
}

// ── Cubic bezier link (top-down: parent-bottom → child-top) ───────────────

function cubicLink(
  sx: number, sy: number,
  tx: number, ty: number,
): string {
  const mx = (sx + tx) / 2;
  const my = (sy + ty) / 2;
  // Use four-point cubic bezier for a graceful S-curve
  return `M ${sx} ${sy} C ${sx} ${my}, ${tx} ${my}, ${tx} ${ty}`;
}

// ── SVG node card (rendered via foreignObject) ─────────────────────────────

function TreeSvgNode({
  node,
  collapsed,
  onToggle,
}: {
  node: { x: number; y: number; data: HierarchyDatum; children?: unknown[] };
  collapsed: boolean;
  onToggle: (id: string) => void;
}) {
  const d = node.data;
  const isComposite = d.nodeKind === "composite";
  const isEpisode = d.nodeKind === "episode";
  const cardKind = isComposite ? "composite" : isEpisode ? "episode" : "record";
  const hasChildren = (d.children || []).length > 0;

  const title = isEpisode
    ? String(d.properties.content || d.label || "")
    : d.label || "";
  const sourceCount = isComposite
    ? Number(d.properties.source_record_count || 0)
    : null;
  const confidence = !isComposite && !isEpisode && d.properties.confidence != null
    ? `${(Number(d.properties.confidence) * 100).toFixed(0)}%`
    : null;
  const typeTag = !isEpisode ? d.typeLabel || null : null;
  const episodeRole = isEpisode ? String(d.properties.role || d.label || "对话") : "";
  const episodeTurn = isEpisode ? Number(d.properties.turn_index ?? -1) : -1;
  const episodeSession = isEpisode ? String(d.properties.session_id || "") : "";

  return (
    <foreignObject
      x={node.x - NODE_W / 2}
      y={node.y}
      width={NODE_W}
      height={NODE_H}
      style={{ overflow: "visible" }}
    >
      <div
        className={`tsv-card ${cardKind}${collapsed ? " collapsed" : ""}`}
        onClick={hasChildren ? () => onToggle(d.id) : undefined}
        style={{ cursor: hasChildren ? "pointer" : "default", width: NODE_W, height: NODE_H }}
      >
        <div className="tsv-header">
          <span className={`tsv-badge ${cardKind}`}>
            {isComposite ? "融合记忆" : isEpisode ? "原始对话" : "原子记录"}
          </span>
          {typeTag && <span className="tsv-type">{typeTag}</span>}
          {hasChildren && (
            <span className="tsv-chevron">{collapsed ? "▸" : "▾"}</span>
          )}
        </div>
        <div className="tsv-title" title={title}>{title}</div>
        <div className="tsv-meta">
          {isComposite && sourceCount !== null && sourceCount > 0 && (
            <span>{sourceCount} 条底层记录</span>
          )}
          {confidence && <span>置信度 {confidence}</span>}
          {isEpisode && (
            <span>
              {episodeRole}
              {episodeTurn >= 0 ? ` · t${episodeTurn}` : ""}
              {episodeSession ? ` · ${episodeSession.slice(0, 8)}` : ""}
            </span>
          )}
        </div>
      </div>
    </foreignObject>
  );
}

// ── SVG tree canvas ────────────────────────────────────────────────────────

function MemoryTreeSvg({
  roots,
  collapsedIds,
  onToggle,
}: {
  roots: GraphTreeNode[];
  collapsedIds: Set<string>;
  onToggle: (id: string) => void;
}) {
  // Build one independent D3 tree layout per root.
  // This ensures expanding root A only grows that tree downward,
  // and does NOT push sibling roots to the right.
  const trees = useMemo(() => {
    return roots.map((rootNode) => {
      const h = hierarchy<HierarchyDatum>(rootNode as HierarchyDatum, (d) => {
        if (collapsedIds.has(d.id)) return null;
        if (!d.children || d.children.length === 0) return null;
        return (d.children || []) as HierarchyDatum[];
      });
      tree<HierarchyDatum>().nodeSize([H_GAP, V_GAP])(h);
      const nodes = h.descendants();
      const links = h.links();
      const xs = nodes.map((n) => n.x ?? 0);
      const ys = nodes.map((n) => n.y ?? 0);
      return {
        id: rootNode.id,
        nodes,
        links,
        minX: Math.min(...xs) - NODE_W / 2 - 20,
        maxX: Math.max(...xs) + NODE_W / 2 + 20,
        minY: Math.min(...ys) - 4,
        maxY: Math.max(...ys) + NODE_H + 4,
      };
    });
  }, [roots, collapsedIds]);

  // SVG total width = max width across all trees (enables horizontal scroll only when needed)
  const svgW = useMemo(
    () => Math.max(400, ...trees.map((t) => t.maxX - t.minX)),
    [trees]
  );

  // Stack each tree's group vertically; center each horizontally within svgW
  const { positioned, svgH } = useMemo(() => {
    let y = 0;
    const positioned = trees.map((t) => {
      const treeW = t.maxX - t.minX;
      const treeH = t.maxY - t.minY;
      const xOffset = (svgW - treeW) / 2 - t.minX;
      const yOffset = y - t.minY;
      y += treeH + TREE_V_GAP;
      return { ...t, xOffset, yOffset };
    });
    const svgH = Math.max(200, y - TREE_V_GAP + 24);
    return { positioned, svgH };
  }, [trees, svgW]);

  return (
    <svg width={svgW} height={svgH} style={{ display: "block" }}>
      {positioned.map(({ id, nodes, links, xOffset, yOffset }) => (
        <g key={id} transform={`translate(${xOffset}, ${yOffset})`}>
          {/* Links */}
          <g>
            {links.map((link, i) => {
              const isCompositeTarget = link.target.data.nodeKind === "composite";
              const isEpisodeTarget = link.target.data.nodeKind === "episode";
              return (
                <path
                  key={i}
                  d={cubicLink(
                    link.source.x ?? 0, (link.source.y ?? 0) + NODE_H,
                    link.target.x ?? 0, link.target.y ?? 0
                  )}
                  fill="none"
                  stroke={isCompositeTarget
                    ? "rgba(99,102,241,0.4)"
                    : isEpisodeTarget
                      ? "rgba(245,158,11,0.45)"
                      : "rgba(20,184,166,0.4)"}
                  strokeWidth={1.5}
                  strokeDasharray={isCompositeTarget ? "none" : isEpisodeTarget ? "2 3" : "4 3"}
                />
              );
            })}
          </g>
          {/* Connector dots */}
          <g>
            {links.map((link, i) => (
              <circle
                key={i}
                cx={link.target.x ?? 0}
                cy={link.target.y ?? 0}
                r={3}
                fill={link.target.data.nodeKind === "composite"
                  ? "rgba(99,102,241,0.6)"
                  : link.target.data.nodeKind === "episode"
                    ? "rgba(245,158,11,0.65)"
                    : "rgba(20,184,166,0.6)"}
              />
            ))}
          </g>
          {/* Nodes */}
          <g>
            {nodes.map((node) => (
              <TreeSvgNode
                key={node.data.id}
                node={node as { x: number; y: number; data: HierarchyDatum; children?: unknown[] }}
                collapsed={collapsedIds.has(node.data.id)}
                onToggle={onToggle}
              />
            ))}
          </g>
        </g>
      ))}
    </svg>
  );
}

// ── Main panel ─────────────────────────────────────────────────────────────

export default function GraphPanel() {
  const graphData = useStore((s) => s.graphData);
  const setGraphData = useStore((s) => s.setGraphData);

  const scrollRef = useRef<HTMLDivElement>(null);

  const [lastStatus, setLastStatus] = useState<{
    node_count: number;
    edge_count: number;
  } | null>(null);
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(new Set());

  const treeRoots = useMemo(
    () =>
      graphData.treeRoots.length > 0
        ? graphData.treeRoots
        : fallbackTreeRoots(graphData.nodes),
    [graphData.treeRoots, graphData.nodes]
  );

  const compositeIds = useMemo(() => collectCompositeIds(treeRoots), [treeRoots]);

  // Reset collapse state when tree changes
  useEffect(() => {
    setCollapsedIds(new Set());
  }, [treeRoots]);

  const handleRefresh = useCallback(async () => {
    try {
      setGraphData(await fetchGraphData());
    } catch { /* ignore */ }
  }, [setGraphData]);

  useEffect(() => {
    handleRefresh();
  }, [handleRefresh]);

  // Poll for tree updates
  useEffect(() => {
    const POLL = 3000;
    const poll = async () => {
      try {
        const status = await fetchPipelineStatus();
        const cur = {
          node_count: status.graph_node_count,
          edge_count: status.graph_edge_count,
        };
        if (
          lastStatus === null ||
          lastStatus.node_count !== cur.node_count ||
          lastStatus.edge_count !== cur.edge_count
        ) {
          setLastStatus(cur);
          if (lastStatus !== null) {
            setGraphData(await fetchGraphData());
          }
        }
      } catch { /* ignore */ }
    };
    const t = setInterval(poll, POLL);
    return () => clearInterval(t);
  }, [lastStatus, setGraphData]);

  const hasNodes = treeRoots.length > 0;
  const allCollapsed =
    compositeIds.length > 0 && collapsedIds.size >= compositeIds.length;

  const toggleNode = useCallback((nodeId: string) => {
    setCollapsedIds((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
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
        <h2>
          <ApartmentOutlined /> 记忆树
        </h2>
        <div className="panel-actions">
          <button className="icon-btn" title="刷新记忆树" onClick={handleRefresh}>
            <ReloadOutlined />
          </button>
          <button
            className="icon-btn"
            title={allCollapsed ? "展开全部" : "折叠全部"}
            onClick={toggleAll}
          >
            {allCollapsed ? <ExpandOutlined /> : <CompressOutlined />}
          </button>
        </div>
      </div>

      <div className="graph-container graph-tree-container" ref={scrollRef}>
        {!hasNodes && (
          <div className="empty-state">
            <span>暂无树状记忆</span>
            <span className="sub">对话后长期记忆会自动固化并组织为层级树</span>
          </div>
        )}
        {hasNodes && (
          <div className="memory-tree-svg-wrap">
            <MemoryTreeSvg
              roots={treeRoots}
              collapsedIds={collapsedIds}
              onToggle={toggleNode}
            />
          </div>
        )}
      </div>
    </section>
  );
}
