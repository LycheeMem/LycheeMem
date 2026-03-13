"use client"

import type { PipelineStatus } from "../lib/types"
import styles from "../styles/panels.module.css"

export function StatusCards({ status }: { status: PipelineStatus | null }) {
  const cards = [
    { label: "Sessions", value: status?.session_count ?? 0 },
    { label: "Graph Nodes", value: status?.graph_node_count ?? 0 },
    { label: "Graph Edges", value: status?.graph_edge_count ?? 0 },
    { label: "Skills", value: status?.skill_count ?? 0 },
    { label: "Sensory", value: status?.sensory_buffer_size ?? 0 },
  ]

  return (
    <section className={styles.statusGrid}>
      {cards.map((card) => (
        <div key={card.label} className={styles.statusCard}>
          <div className={styles.statusLabel}>{card.label}</div>
          <div className={styles.statusValue}>{card.value}</div>
        </div>
      ))}
    </section>
  )
}
