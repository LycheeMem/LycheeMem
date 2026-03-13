"use client"

import { useMemo, useRef, useState } from "react"
import type { GraphData } from "../lib/types"
import styles from "../styles/graph.module.css"

const WIDTH = 520
const HEIGHT = 340

type PositionedNode = {
  id: string
  label: string
  x: number
  y: number
  degree: number
}

function hashToSeed(value: string): number {
  let hash = 2166136261
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

function seededUnit(seed: number): number {
  let t = seed + 0x6d2b79f5
  t = Math.imul(t ^ (t >>> 15), t | 1)
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function buildFreeLayout(data: GraphData): PositionedNode[] {
  const nodes = data.nodes ?? []
  const edges = data.edges ?? []
  if (nodes.length === 0) return []

  const degrees = new Map<string, number>()
  nodes.forEach((node) => degrees.set(node.id, 0))
  edges.forEach((edge) => {
    degrees.set(edge.source, (degrees.get(edge.source) ?? 0) + 1)
    degrees.set(edge.target, (degrees.get(edge.target) ?? 0) + 1)
  })

  const centerX = WIDTH / 2
  const centerY = HEIGHT / 2
  const xMin = 40
  const xMax = WIDTH - 40
  const yMin = 36
  const yMax = HEIGHT - 36

  const positioned = nodes.map((node, index) => {
    const seed = hashToSeed(`${node.id}:${index}:${nodes.length}`)
    const angle = seededUnit(seed) * Math.PI * 2
    const radius = 60 + seededUnit(seed + 17) * Math.min(centerX, centerY) * 0.8
    return {
      id: node.id,
      label: node.name || node.label || node.id,
      x: clamp(centerX + Math.cos(angle) * radius, xMin, xMax),
      y: clamp(centerY + Math.sin(angle) * radius, yMin, yMax),
      degree: degrees.get(node.id) ?? 0,
    }
  })

  const byId = new Map<string, PositionedNode>()
  positioned.forEach((node) => byId.set(node.id, node))

  const iterations = Math.min(80, Math.max(30, nodes.length * 2))
  const repelStrength = 2200
  const springLength = 90
  const springStrength = 0.05

  for (let iter = 0; iter < iterations; iter += 1) {
    for (let i = 0; i < positioned.length; i += 1) {
      for (let j = i + 1; j < positioned.length; j += 1) {
        const a = positioned[i]
        const b = positioned[j]
        const dx = b.x - a.x
        const dy = b.y - a.y
        const distSq = dx * dx + dy * dy + 0.01
        const dist = Math.sqrt(distSq)
        const force = repelStrength / distSq
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force

        a.x = clamp(a.x - fx, xMin, xMax)
        a.y = clamp(a.y - fy, yMin, yMax)
        b.x = clamp(b.x + fx, xMin, xMax)
        b.y = clamp(b.y + fy, yMin, yMax)
      }
    }

    edges.forEach((edge) => {
      const source = byId.get(edge.source)
      const target = byId.get(edge.target)
      if (!source || !target) return
      const dx = target.x - source.x
      const dy = target.y - source.y
      const dist = Math.sqrt(dx * dx + dy * dy) || 1
      const stretch = dist - springLength
      const force = stretch * springStrength
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force

      source.x = clamp(source.x + fx, xMin, xMax)
      source.y = clamp(source.y + fy, yMin, yMax)
      target.x = clamp(target.x - fx, xMin, xMax)
      target.y = clamp(target.y - fy, yMin, yMax)
    })
  }

  return positioned
}

export function GraphView({ data }: { data: GraphData }) {
  const nodes = data.nodes ?? []
  const edges = data.edges ?? []
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [dragNodeId, setDragNodeId] = useState<string | null>(null)
  const [hoverNodeId, setHoverNodeId] = useState<string | null>(null)
  const [localPositions, setLocalPositions] = useState<Record<string, { x: number; y: number }>>({})
  const panState = useRef<{ startX: number; startY: number; baseX: number; baseY: number } | null>(null)

  const layout = useMemo(() => {
    return buildFreeLayout(data)
  }, [data])

  const displayLayout = useMemo(
    () =>
      layout.map((node) => {
        const override = localPositions[node.id]
        return override ? { ...node, ...override } : node
      }),
    [layout, localPositions],
  )

  const nodeMap = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>()
    displayLayout.forEach((n) => map.set(n.id, { x: n.x, y: n.y }))
    return map
  }, [displayLayout])

  const edgeSet = useMemo(() => {
    const set = new Set<string>()
    edges.forEach((edge) => {
      set.add(`${edge.source}|${edge.target}`)
      set.add(`${edge.target}|${edge.source}`)
    })
    return set
  }, [edges])

  function toGraphPoint(clientX: number, clientY: number) {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return null
    const x = ((clientX - rect.left) / rect.width) * WIDTH
    const y = ((clientY - rect.top) / rect.height) * HEIGHT
    return {
      x: (x - offset.x) / zoom,
      y: (y - offset.y) / zoom,
    }
  }

  function handleWheel(event: React.WheelEvent<SVGSVGElement>) {
    event.preventDefault()
    const delta = event.deltaY > 0 ? -0.08 : 0.08
    setZoom((current) => clamp(current + delta, 0.65, 1.9))
  }

  function handleBackgroundMouseDown(event: React.MouseEvent<SVGSVGElement>) {
    if (event.target !== event.currentTarget) return
    panState.current = {
      startX: event.clientX,
      startY: event.clientY,
      baseX: offset.x,
      baseY: offset.y,
    }
  }

  function handleMouseMove(event: React.MouseEvent<SVGSVGElement>) {
    if (dragNodeId) {
      const point = toGraphPoint(event.clientX, event.clientY)
      if (!point) return
      setLocalPositions((prev) => ({
        ...prev,
        [dragNodeId]: {
          x: clamp(point.x, 26, WIDTH - 26),
          y: clamp(point.y, 24, HEIGHT - 24),
        },
      }))
      return
    }

    if (!panState.current) return
    const dx = event.clientX - panState.current.startX
    const dy = event.clientY - panState.current.startY
    setOffset({
      x: panState.current.baseX + dx * 0.55,
      y: panState.current.baseY + dy * 0.55,
    })
  }

  function handleMouseUp() {
    panState.current = null
    setDragNodeId(null)
  }

  function resetView() {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }

  return (
    <section className={styles.panel}>
      <div className={styles.panelHeader}>
        <div>
          <h2>Memory Graph</h2>
          <p>Entities and relations (free layout, drag and zoom)</p>
        </div>
        <div className={styles.headerActions}>
          <span className={styles.counter}>{nodes.length} nodes</span>
          <button className={styles.resetButton} onClick={resetView} type="button">
            Reset View
          </button>
        </div>
      </div>

      {nodes.length === 0 ? (
        <div className={styles.empty}>Graph is empty.</div>
      ) : (
        <div className={styles.graphWrap}>
          <svg
            ref={svgRef}
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            className={styles.graph}
            onWheel={handleWheel}
            onMouseDown={handleBackgroundMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            <defs>
              <linearGradient id="edge" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#9ca3af" />
                <stop offset="100%" stopColor="#60a5fa" />
              </linearGradient>
              <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
                <path d="M 24 0 L 0 0 0 24" fill="none" stroke="rgba(148, 163, 184, 0.12)" strokeWidth="1" />
              </pattern>
            </defs>

            <rect x="0" y="0" width={WIDTH} height={HEIGHT} fill="url(#grid)" />
            <g transform={`translate(${offset.x}, ${offset.y}) scale(${zoom})`}>
              {edges.map((edge, idx) => {
                const source = nodeMap.get(edge.source)
                const target = nodeMap.get(edge.target)
                if (!source || !target) return null
                const highlight =
                  hoverNodeId != null && (edge.source === hoverNodeId || edge.target === hoverNodeId)

                return (
                  <line
                    key={`edge-${idx}`}
                    x1={source.x}
                    y1={source.y}
                    x2={target.x}
                    y2={target.y}
                    stroke="url(#edge)"
                    strokeWidth={highlight ? 2.1 : 1.3}
                    opacity={highlight ? 0.9 : 0.48}
                  />
                )
              })}

              {displayLayout.map((node) => {
                const connected =
                  hoverNodeId != null && edgeSet.has(`${hoverNodeId}|${node.id}`)
                const isActive = hoverNodeId === node.id || connected
                const radius = 9 + Math.min(node.degree, 7)

                return (
                  <g
                    key={node.id}
                    onMouseEnter={() => setHoverNodeId(node.id)}
                    onMouseLeave={() => setHoverNodeId(null)}
                    onMouseDown={(event) => {
                      event.stopPropagation()
                      setDragNodeId(node.id)
                    }}
                    className={styles.nodeGroup}
                  >
                    <circle cx={node.x} cy={node.y} r={radius + 10} className={styles.nodeHalo} opacity={isActive ? 0.25 : 0.1} />
                    <circle cx={node.x} cy={node.y} r={radius} className={styles.nodeCore} opacity={isActive ? 1 : 0.88} />
                    <text x={node.x} y={node.y + radius + 16} textAnchor="middle" className={styles.nodeLabel}>
                      {node.label}
                    </text>
                  </g>
                )
              })}
            </g>
          </svg>
          <div className={styles.tip}>Mouse wheel: zoom. Drag blank area: pan. Drag node: reposition.</div>
        </div>
      )}
    </section>
  )
}
