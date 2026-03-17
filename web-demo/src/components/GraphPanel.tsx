import { BorderOuterOutlined, LinkOutlined, ReloadOutlined } from "@ant-design/icons";
import { useCallback, useEffect, useRef } from "react";
import { fetchGraphData } from "../api";
import { useGraphCanvas } from "../hooks/useGraphCanvas";
import { useStore } from "../state";

export default function GraphPanel() {
  const graphData = useStore((s) => s.graphData);
  const hoveredEdge = useStore((s) => s.hoveredEdge);
  const selectedEdge = useStore((s) => s.selectedEdge);
  const setGraphData = useStore((s) => s.setGraphData);
  const setHoveredEdge = useStore((s) => s.setHoveredEdge);
  const setSelectedEdge = useStore((s) => s.setSelectedEdge);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

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
    try {
      setGraphData(await fetchGraphData());
    } catch {
      /* ignore */
    }
  }, [setGraphData]);

  useEffect(() => {
    handleRefresh();
  }, [handleRefresh]);

  const hasNodes = graphData.nodes.length > 0;

  return (
    <section id="panel-graph" className="panel">
      <div className="panel-header">
        <h2><LinkOutlined /> 记忆图谱</h2>
        <div className="panel-actions">
          <button
            className="icon-btn"
            title="刷新图谱"
            onClick={handleRefresh}
          >
            <ReloadOutlined />
          </button>
          <button className="icon-btn" title="适配视图" onClick={fitGraph}>
            <BorderOuterOutlined />
          </button>
        </div>
      </div>
      <div className="graph-container" ref={containerRef}>
        <canvas id="graph-canvas" ref={canvasRef} />
        <div
          className="graph-tooltip hidden"
          ref={tooltipRef}
        />
        {!hasNodes && (
          <div className="empty-state">
            <span>暂无图谱数据</span>
            <span className="sub">对话后知识实体将自动提取并展示</span>
          </div>
        )}
      </div>
    </section>
  );
}
