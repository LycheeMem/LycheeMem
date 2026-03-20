import { BorderOuterOutlined, LinkOutlined, ReloadOutlined } from "@ant-design/icons";
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchGraphData, fetchPipelineStatus } from "../api";
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

  // 定时检测图谱更新：保存上次检测的统计数据
  const [lastStatus, setLastStatus] = useState<{
    node_count: number;
    edge_count: number;
  } | null>(null);

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
