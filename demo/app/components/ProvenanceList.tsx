"use client"

import type { ProvenanceItem } from "../lib/types"
import styles from "../styles/panels.module.css"

export function ProvenanceList({ items }: { items: ProvenanceItem[] }) {
  return (
    <section className={styles.panel}>
      <div className={styles.panelHeader}>
        <div>
          <h2>Provenance</h2>
          <p>Scored fragments from synthesis</p>
        </div>
        <span className={styles.counter}>{items.length}</span>
      </div>
      <div className={styles.scrollArea}>
        {items.length === 0 && <p className={styles.empty}>No provenance yet.</p>}
        {items.map((item, idx) => (
          <div key={`${item.source}-${idx}`} className={styles.provenanceRow}>
            <div className={styles.provenanceHeader}>
              <span className={styles.tag}>{item.source}</span>
              <span className={styles.provenanceScore}>{item.relevance.toFixed(2)}</span>
            </div>
            <div className={styles.provenanceSummary}>{item.summary}</div>
          </div>
        ))}
      </div>
    </section>
  )
}
