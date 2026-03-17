import type { Simulation, SimulationLinkDatum, SimulationNodeDatum } from "d3-force";
import {
    forceCenter,
    forceCollide,
    forceLink,
    forceManyBody,
    forceSimulation,
} from "d3-force";
import { useCallback, useEffect, useRef } from "react";
import type { GraphEdge, GraphNode, GraphTransform } from "../types";
import { escapeHtml } from "../utils";

interface SimNode extends SimulationNodeDatum {
  id: string;
  label: string;
  typeLabel: string;
  properties: Record<string, unknown>;
}

interface SimEdge extends SimulationLinkDatum<SimNode> {
  relation: string;
  confidence: number;
  fact: string;
  evidence: string;
  timestamp: string;
  source_session: string;
}

interface UseGraphCanvasOptions {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  containerRef: React.RefObject<HTMLDivElement | null>;
  tooltipRef: React.RefObject<HTMLDivElement | null>;
  nodes: GraphNode[];
  edges: GraphEdge[];
  hoveredEdge: GraphEdge | null;
  selectedEdge: GraphEdge | null;
  onEdgeHover?: (edge: GraphEdge | null) => void;
  onEdgeSelect?: (edge: GraphEdge | null) => void;
}

const COLORS = [
  "#6366f1", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b",
  "#ec4899", "#14b8a6", "#f43f5e", "#0ea5e9", "#a855f7",
];

function edgeColor(e: SimEdge): string {
  const c = e.confidence || 0.5;
  if (c >= 0.9) return "#06b6d4";
  if (c >= 0.7) return "#6366f1";
  return "#334155";
}

function sameEdgeKey(a: GraphEdge | SimEdge | null, b: GraphEdge | SimEdge | null): boolean {
  if (!a || !b) return false;
  const getKey = (e: GraphEdge | SimEdge) => {
    const s = typeof e.source === "object" ? (e.source as SimNode).id : e.source;
    const t = typeof e.target === "object" ? (e.target as SimNode).id : e.target;
    return `${s}::${e.relation}::${t}::${e.timestamp}`;
  };
  return getKey(a) === getKey(b);
}

function pointToSegmentDistSq(
  px: number, py: number,
  x1: number, y1: number,
  x2: number, y2: number
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  if (dx === 0 && dy === 0) {
    const ox = px - x1;
    const oy = py - y1;
    return ox * ox + oy * oy;
  }
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
  const cx = x1 + t * dx;
  const cy = y1 + t * dy;
  const ox = px - cx;
  const oy = py - cy;
  return ox * ox + oy * oy;
}

function findNodeAt(nodes: SimNode[], wx: number, wy: number, radius: number): SimNode | null {
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    const dx = (n.x ?? 0) - wx;
    const dy = (n.y ?? 0) - wy;
    if (dx * dx + dy * dy <= radius * radius) return n;
  }
  return null;
}

function findEdgeAt(edges: SimEdge[], wx: number, wy: number, threshold: number): SimEdge | null {
  const thrSq = threshold * threshold;
  for (let i = edges.length - 1; i >= 0; i--) {
    const e = edges[i];
    const s = typeof e.source === "object" ? (e.source as SimNode) : null;
    const t = typeof e.target === "object" ? (e.target as SimNode) : null;
    if (!s || !t) continue;
    const d2 = pointToSegmentDistSq(wx, wy, s.x ?? 0, s.y ?? 0, t.x ?? 0, t.y ?? 0);
    if (d2 <= thrSq) return e;
  }
  return null;
}

export function useGraphCanvas({
  canvasRef,
  containerRef,
  tooltipRef,
  nodes,
  edges,
  hoveredEdge,
  selectedEdge,
  onEdgeHover,
  onEdgeSelect,
}: UseGraphCanvasOptions) {
  const simRef = useRef<Simulation<SimNode, SimEdge> | null>(null);
  const transformRef = useRef<GraphTransform>({ x: 0, y: 0, k: 1 });
  const simNodesRef = useRef<SimNode[]>([]);
  const simEdgesRef = useRef<SimEdge[]>([]);
  const hoveredNodeRef = useRef<SimNode | null>(null);
  const dragNodeRef = useRef<SimNode | null>(null);
  const dragOffsetRef = useRef({ x: 0, y: 0 });
  const isPanningRef = useRef(false);
  const panStartRef = useRef({ x: 0, y: 0 });
  const rafRef = useRef<number | null>(null);
  const hoveredEdgeRef = useRef(hoveredEdge);
  const selectedEdgeRef = useRef(selectedEdge);

  hoveredEdgeRef.current = hoveredEdge;
  selectedEdgeRef.current = selectedEdge;

  const screenToWorld = useCallback((sx: number, sy: number): [number, number] => {
    const t = transformRef.current;
    return [(sx - t.x) / t.k, (sy - t.y) / t.k];
  }, []);

  const drawGrid = useCallback((ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const spacing = 40;
    const t = transformRef.current;
    const ox = ((t.x % spacing) + spacing) % spacing;
    const oy = ((t.y % spacing) + spacing) % spacing;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
    ctx.lineWidth = 1;
    for (let x = ox; x < w; x += spacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = oy; y < h; y += spacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  }, []);

  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const t = transformRef.current;
    const simNodes = simNodesRef.current;
    const simEdges = simEdgesRef.current;
    const hoveredNode = hoveredNodeRef.current;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);

    drawGrid(ctx, W, H);

    ctx.save();
    ctx.translate(t.x, t.y);
    ctx.scale(t.k, t.k);

    // Edges
    simEdges.forEach((e) => {
      const src = typeof e.source === "object" ? (e.source as SimNode) : null;
      const tgt = typeof e.target === "object" ? (e.target as SimNode) : null;
      if (!src || !tgt) return;

      const isActive = sameEdgeKey(e, hoveredEdgeRef.current) || sameEdgeKey(e, selectedEdgeRef.current);
      const hasAny = Boolean(hoveredEdgeRef.current || selectedEdgeRef.current || hoveredNode);

      ctx.beginPath();
      ctx.moveTo(src.x ?? 0, src.y ?? 0);
      ctx.lineTo(tgt.x ?? 0, tgt.y ?? 0);
      ctx.strokeStyle = edgeColor(e);
      ctx.lineWidth = isActive ? 2.6 : (e.confidence || 0.5) > 0.7 ? 1.8 : 1;
      ctx.globalAlpha = hasAny ? (isActive ? 1 : 0.12) : 0.6;
      ctx.stroke();
      ctx.globalAlpha = 1;

      if (e.relation) {
        const mx = ((src.x ?? 0) + (tgt.x ?? 0)) / 2;
        const my = ((src.y ?? 0) + (tgt.y ?? 0)) / 2;
        ctx.font = `${9 / Math.max(t.k, 0.5)}px Inter`;
        ctx.fillStyle = "#4b5563";
        ctx.globalAlpha = hasAny ? (isActive ? 0.95 : 0.06) : 0.5;
        ctx.textAlign = "center";
        ctx.fillText(e.relation, mx, my - 4);
        ctx.globalAlpha = 1;
      }
    });

    // Nodes
    simNodes.forEach((n, i) => {
      const isHovered = n === hoveredNode;
      const radius = isHovered ? 16 : 12;
      const color = COLORS[i % COLORS.length];

      if (isHovered) {
        ctx.beginPath();
        ctx.arc(n.x ?? 0, n.y ?? 0, radius + 8, 0, Math.PI * 2);
        const glow = ctx.createRadialGradient(
          n.x ?? 0, n.y ?? 0, radius,
          n.x ?? 0, n.y ?? 0, radius + 8
        );
        glow.addColorStop(0, color + "40");
        glow.addColorStop(1, "transparent");
        ctx.fillStyle = glow;
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(n.x ?? 0, n.y ?? 0, radius, 0, Math.PI * 2);
      ctx.fillStyle = hoveredNode && !isHovered ? color + "30" : color;
      ctx.fill();
      ctx.strokeStyle = isHovered ? "#ffffff" : color + "80";
      ctx.lineWidth = isHovered ? 2.5 : 1.5;
      ctx.stroke();

      const labelAlpha = hoveredNode ? (isHovered ? 1 : 0.15) : 0.9;
      ctx.globalAlpha = labelAlpha;
      ctx.font = `${isHovered ? 11 : 10}px Inter`;
      ctx.fillStyle = "#0f172a";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const txt = String(n.label || "?");
      const label = txt.length > 16 ? txt.slice(0, 14) + "\u2026" : txt;
      ctx.fillText(label, n.x ?? 0, (n.y ?? 0) + radius + 4);
      ctx.globalAlpha = 1;
    });

    ctx.restore();
  }, [canvasRef, containerRef, drawGrid]);

  const scheduleRender = useCallback(() => {
    if (rafRef.current !== null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      drawFrame();
    });
  }, [drawFrame]);

  // Build simulation when data changes
  useEffect(() => {
    if (simRef.current) simRef.current.stop();

    const nodeIds = new Set(nodes.map((n) => n.id));
    const validEdges = edges.filter(
      (e) =>
        nodeIds.has(typeof e.source === "object" ? (e.source as GraphNode).id : (e.source as string)) &&
        nodeIds.has(typeof e.target === "object" ? (e.target as GraphNode).id : (e.target as string))
    );

    const simNodes: SimNode[] = nodes.map((n) => ({ ...n }));
    const simEdges: SimEdge[] = validEdges.map((e) => ({
      source: typeof e.source === "object" ? (e.source as GraphNode).id : e.source,
      target: typeof e.target === "object" ? (e.target as GraphNode).id : e.target,
      relation: e.relation,
      confidence: e.confidence,
      fact: e.fact,
      evidence: e.evidence,
      timestamp: e.timestamp,
      source_session: e.source_session,
    }));

    simNodesRef.current = simNodes;
    simEdgesRef.current = simEdges;

    const container = containerRef.current;
    if (!container || !nodes.length) {
      scheduleRender();
      return;
    }

    const rect = container.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    const sim = forceSimulation<SimNode>(simNodes)
      .force(
        "link",
        forceLink<SimNode, SimEdge>(simEdges)
          .id((d) => d.id)
          .distance(100)
          .strength(0.4)
      )
      .force("charge", forceManyBody().strength(-300))
      .force("center", forceCenter(W / 2, H / 2))
      .force("collide", forceCollide(30))
      .alphaDecay(0.03);

    sim.on("tick", scheduleRender);
    simRef.current = sim;

    return () => {
      sim.stop();
    };
  }, [nodes, edges, containerRef, scheduleRender]);

  // Canvas event handlers
  useEffect(() => {
    const canvas = canvasRef.current;
    const tooltip = tooltipRef.current;
    if (!canvas) return;

    const nodeById = new Map(simNodesRef.current.map((n) => [n.id, n]));

    const getEdgeFact = (e: SimEdge): string => {
      const fact = (e.fact || "").toString().trim();
      if (fact) return fact;
      const s = typeof e.source === "object" ? (e.source as SimNode) : nodeById.get(e.source as string);
      const t = typeof e.target === "object" ? (e.target as SimNode) : nodeById.get(e.target as string);
      return `${s?.label || "?"} ${e.relation || "?"} ${t?.label || "?"}`;
    };

    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const t = transformRef.current;
      const scaleFactor = ev.deltaY > 0 ? 0.92 : 1.08;
      const mx = ev.offsetX;
      const my = ev.offsetY;
      t.x = mx - (mx - t.x) * scaleFactor;
      t.y = my - (my - t.y) * scaleFactor;
      t.k *= scaleFactor;
      t.k = Math.max(0.1, Math.min(5, t.k));
      scheduleRender();
    };

    const onMouseDown = (ev: MouseEvent) => {
      const [mx, my] = screenToWorld(ev.offsetX, ev.offsetY);
      const t = transformRef.current;
      const hit = findNodeAt(simNodesRef.current, mx, my, 20 / t.k);
      if (hit) {
        dragNodeRef.current = hit;
        dragOffsetRef.current = { x: hit.x! - mx, y: hit.y! - my };
        hit.fx = hit.x;
        hit.fy = hit.y;
        simRef.current?.alphaTarget(0.3).restart();
        onEdgeSelect?.(null);
        scheduleRender();
      } else {
        const edgeHit = findEdgeAt(simEdgesRef.current, mx, my, 10 / t.k);
        if (edgeHit) {
          onEdgeSelect?.(edgeHit as unknown as GraphEdge);
          onEdgeHover?.(edgeHit as unknown as GraphEdge);
          scheduleRender();
        } else {
          onEdgeSelect?.(null);
          onEdgeHover?.(null);
          isPanningRef.current = true;
          panStartRef.current = { x: ev.offsetX, y: ev.offsetY };
          scheduleRender();
        }
      }
    };

    const onMouseMove = (ev: MouseEvent) => {
      const t = transformRef.current;
      if (dragNodeRef.current) {
        const [mx, my] = screenToWorld(ev.offsetX, ev.offsetY);
        dragNodeRef.current.fx = mx + dragOffsetRef.current.x;
        dragNodeRef.current.fy = my + dragOffsetRef.current.y;
        scheduleRender();
      } else if (isPanningRef.current) {
        t.x += ev.offsetX - panStartRef.current.x;
        t.y += ev.offsetY - panStartRef.current.y;
        panStartRef.current = { x: ev.offsetX, y: ev.offsetY };
        scheduleRender();
      } else {
        const [mx, my] = screenToWorld(ev.offsetX, ev.offsetY);
        const node = findNodeAt(simNodesRef.current, mx, my, 20 / t.k);
        hoveredNodeRef.current = node;

        if (node) {
          canvas.style.cursor = "pointer";
          if (tooltip) {
            tooltip.classList.remove("hidden");
            tooltip.style.left = ev.offsetX + 12 + "px";
            tooltip.style.top = ev.offsetY - 20 + "px";
            let html = `<strong>${escapeHtml(node.label)}</strong>`;
            if (node.typeLabel) html += `<br>类型: ${escapeHtml(node.typeLabel)}`;
            if (node.properties) {
              const p = node.properties;
              if (p.type) html += `<br>type: ${escapeHtml(p.type as string)}`;
              if (p.description) html += `<br>${escapeHtml(String(p.description).slice(0, 100))}`;
            }
            tooltip.innerHTML = html;
          }
          onEdgeHover?.(null);
        } else {
          const edgeHit = findEdgeAt(simEdgesRef.current, mx, my, 10 / t.k);
          if (edgeHit) {
            canvas.style.cursor = "pointer";
            if (tooltip) {
              tooltip.classList.remove("hidden");
              tooltip.style.left = ev.offsetX + 12 + "px";
              tooltip.style.top = ev.offsetY - 20 + "px";
              const fact = getEdgeFact(edgeHit);
              const evidence = (edgeHit.evidence || "").toString().trim();
              tooltip.innerHTML = `
                <strong>关系</strong><br>
                ${escapeHtml(fact.slice(0, 180))}
                ${evidence ? `<br><span style="opacity:0.75">证据: ${escapeHtml(evidence.slice(0, 120))}</span>` : ""}
              `;
            }
            if (!selectedEdgeRef.current) onEdgeHover?.(edgeHit as unknown as GraphEdge);
          } else {
            canvas.style.cursor = "default";
            tooltip?.classList.add("hidden");
            if (!selectedEdgeRef.current) onEdgeHover?.(null);
          }
        }
        scheduleRender();
      }
    };

    const onMouseUp = () => {
      if (dragNodeRef.current) {
        dragNodeRef.current.fx = dragNodeRef.current.x;
        dragNodeRef.current.fy = dragNodeRef.current.y;
        simRef.current?.alphaTarget(0);
        dragNodeRef.current = null;
        scheduleRender();
      }
      isPanningRef.current = false;
    };

    const onMouseLeave = () => {
      isPanningRef.current = false;
      tooltip?.classList.add("hidden");
      if (!selectedEdgeRef.current) onEdgeHover?.(null);
      scheduleRender();
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("mouseleave", onMouseLeave);

    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("mouseup", onMouseUp);
      canvas.removeEventListener("mouseleave", onMouseLeave);
    };
  }, [canvasRef, tooltipRef, screenToWorld, scheduleRender, onEdgeHover, onEdgeSelect]);

  const fitGraph = useCallback(() => {
    const container = containerRef.current;
    if (!container || !nodes.length) return;
    const rect = container.getBoundingClientRect();
    transformRef.current = { x: rect.width / 4, y: rect.height / 4, k: 0.8 };
    simRef.current?.alpha(0.3).restart();
  }, [containerRef, nodes.length]);

  return { fitGraph };
}
