"use client"

import type { AgentStep } from "../lib/types"
import styles from "../styles/panels.module.css"

type Props = {
  steps: AgentStep[]
}

export function AgentTimeline({ steps }: Props) {
  return (
    <section className={styles.panel}>
      <div className={styles.panelHeader}>
        <div>
          <h2>Agents</h2>
          <p>Execution flow and memory operations</p>
        </div>
      </div>
      <div className={styles.timeline}>
        {steps.length === 0 && <p className={styles.empty}>No agent activity yet.</p>}
        {steps.map((step) => (
          <div key={step.id} className={styles.timelineRow}>
            <span className={`${styles.badge} ${styles[`badge-${step.status}`]}`}>
              {step.status}
            </span>
            <div>
              <div className={styles.timelineLabel}>{step.label}</div>
              <div className={styles.timelineHint}>
                {step.status === "active" && "Running"}
                {step.status === "done" && "Completed"}
                {step.status === "idle" && "Waiting"}
                {step.status === "error" && "Issue"}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
