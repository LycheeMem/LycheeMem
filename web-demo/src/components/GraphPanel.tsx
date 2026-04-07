import {
  ApartmentOutlined,
  ExpandOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchGraphData, fetchGraphEdges, fetchPipelineStatus } from "../api";
import { useGraphCanvas } from "../hooks/useGraphCanvas";
import { useStore } from "../state";

export default function GraphPanel() {
  const graphData = useStore((s) => s.graphData);
  const setGraphData = useStore((s) => s.setGraphData);
  const hoveredEdge = useStore((s) => s.hoveredEdge);
  const selectedEdge = useStore((s) => s.selectedEdge);
  const setGraphEdges = useStore((s) => s.setGraphEdges);
  const setHoveredEdge = useStore((s) => s.setHoveredEdge);
  const setSelectedEdge = useStore((s) => s.setSelectedEdge);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const [lastStatus, setLastStatus] = useState<{
    node_count: number;
    edge_count: number;
  } | null>(null);

  const loadGraph = useCallback(async () => {
    try {
      const [data, edges] = await Promise.all([fetchGraphData(), fetchGraphEdges()]);
      setGraphData(data);
      setGraphEdges(edges);
    } catch {
      /* ignore */
    }
  }, [setGraphData, setGraphEdges]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

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
            await loadGraph();
          }
        }
      } catch {
        /* ignore */
      }
    };
    const t = setInterval(poll, POLL);
    return () => clearInterval(t);
  }, [lastStatus, loadGraph]);

  const { fitGraph } = useGraphCanvas({
    canvasRef,
    containerRef,
    tooltipRef,
    nodes: graphData.nodes,
    edges: graphData.edges,
    hoveredEdge,
    selectedEdge,
    onEdgeHover: setHoveredEdge,
    onEdgeSelect: setSelectedEdge,
  });

  const handleRefresh = useCallback(async () => {
    setHoveredEdge(null);
    setSelectedEdge(null);
    await loadGraph();
  }, [loadGraph, setHoveredEdge, setSelectedEdge]);

  const hasNodes = graphData.nodes.length > 0;

  return (
    <section id="panel-graph" className="panel">
      <div className="panel-header">
        <h2>
          <ApartmentOutlined /> 记忆图谱
        </h2>
        <div className="panel-actions">
          <button className="icon-btn" title="刷新图谱" onClick={handleRefresh}>
            <ReloadOutlined />
          </button>
          <button className="icon-btn" title="适配画布" onClick={fitGraph}>
            <ExpandOutlined />
          </button>
        </div>
      </div>

      <div className="graph-container" ref={containerRef}>
        {!hasNodes && (
          <div className="empty-state">
            <span>暂无图谱记忆</span>
            <span className="sub">对话后语义记忆会自动固化为关系图</span>
          </div>
        )}
        {hasNodes && (
          <div className="memory-graph-toolbar">
            <div className="memory-graph-legend">
              <span><i className="kind-composite" /> 融合记忆</span>
              <span><i className="kind-record" /> 记录节点</span>
              <span><i className="kind-episode" /> 原始对话</span>
            </div>
            <span className="memory-graph-hint">拖拽节点、滚轮缩放、拖动画布，悬浮查看详情</span>
          </div>
        )}
        <canvas id="graph-canvas" ref={canvasRef} />
        <div ref={tooltipRef} className="graph-tooltip hidden" />
      </div>
    </section>
  );
}
