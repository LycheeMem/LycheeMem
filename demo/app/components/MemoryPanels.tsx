"use client"

import type { MemorySnapshot, SessionSummary } from "../lib/types"
import styles from "../styles/panels.module.css"

type Props = {
  snapshot: MemorySnapshot
  onSessionSelect: (session: SessionSummary) => void
}

export function MemoryPanels({ snapshot, onSessionSelect }: Props) {
  return (
    <section className={styles.memoryGrid}>
      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>Sessions</h2>
            <p>Recent activity and metadata</p>
          </div>
          <span className={styles.counter}>{snapshot.sessions.length}</span>
        </div>
        <div className={styles.scrollArea}>
          {snapshot.sessions.length === 0 && <p className={styles.empty}>No sessions yet.</p>}
          {snapshot.sessions.map((s) => (
            <button
              key={s.session_id}
              className={styles.sessionRow}
              onClick={() => onSessionSelect(s)}
            >
              <div>
                <div className={styles.sessionId}>{s.session_id}</div>
                <div className={styles.sessionMeta}>
                  {s.topic || "No topic"} · {s.turn_count} turns
                </div>
              </div>
              <div className={styles.tagRow}>
                {(s.tags ?? []).slice(0, 3).map((tag) => (
                  <span key={tag} className={styles.tag}>{tag}</span>
                ))}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>Skills</h2>
            <p>Reusable tool chains</p>
          </div>
          <span className={styles.counter}>{snapshot.skills.length}</span>
        </div>
        <div className={styles.scrollArea}>
          {snapshot.skills.length === 0 && <p className={styles.empty}>No skills extracted.</p>}
          {snapshot.skills.map((skill) => (
            <div key={skill.id} className={styles.skillCard}>
              <div className={styles.skillTitle}>{skill.intent}</div>
              <div className={styles.skillMeta}>
                Used: {skill.success_count ?? 0} · {skill.conditions || "General"}
              </div>
              <pre className={styles.skillChain}>{JSON.stringify(skill.tool_chain, null, 2)}</pre>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>Reuse Plan</h2>
            <p>Top skill matches from last query</p>
          </div>
          <span className={styles.counter}>{snapshot.reusePlan.length}</span>
        </div>
        <div className={styles.scrollArea}>
          {snapshot.reusePlan.length === 0 && <p className={styles.empty}>No reusable skills yet.</p>}
          {snapshot.reusePlan.map((item) => (
            <div key={item.id} className={styles.skillCard}>
              <div className={styles.skillTitle}>{item.intent}</div>
              <div className={styles.skillMeta}>
                Score: {item.score.toFixed(2)} · {item.conditions || "General"}
              </div>
              <pre className={styles.skillChain}>{JSON.stringify(item.tool_chain, null, 2)}</pre>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>Sensory Buffer</h2>
            <p>Latest sensory items</p>
          </div>
          <span className={styles.counter}>{snapshot.sensory.length}</span>
        </div>
        <div className={styles.scrollArea}>
          {snapshot.sensory.length === 0 && <p className={styles.empty}>No sensory data.</p>}
          {snapshot.sensory.map((item, idx) => (
            <div key={`${item.timestamp}-${idx}`} className={styles.sensoryRow}>
              <div className={styles.sensoryTop}>
                <span className={styles.sensoryModality}>{item.modality}</span>
                <span className={styles.sensoryTime}>{item.timestamp}</span>
              </div>
              <div className={styles.sensoryContent}>{item.content}</div>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <h2>Session Detail</h2>
            <p>Turns and summaries</p>
          </div>
        </div>
        <div className={styles.scrollArea}>
          {!snapshot.sessionDetail && <p className={styles.empty}>Select a session.</p>}
          {snapshot.sessionDetail && (
            <>
              <div className={styles.sessionDetailHeader}>
                <div className={styles.sessionId}>{snapshot.sessionDetail.session_id}</div>
                <div className={styles.sessionMeta}>{snapshot.sessionDetail.turn_count} turns</div>
              </div>
              <div className={styles.sessionDetailTurns}>
                {snapshot.sessionDetail.turns.map((turn, idx) => (
                  <div key={`turn-${idx}`} className={styles.turnRow}>
                    <span className={styles.turnRole}>{turn.role}</span>
                    <span>{turn.content}</span>
                  </div>
                ))}
              </div>
              {snapshot.sessionDetail.summaries.length > 0 && (
                <div className={styles.summaryBlock}>
                  <div className={styles.summaryTitle}>Summaries</div>
                  {snapshot.sessionDetail.summaries.map((sum, idx) => (
                    <div key={`sum-${idx}`} className={styles.summaryRow}>
                      <span className={styles.summaryIndex}>#{sum.boundary_index}</span>
                      <span>{sum.content}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  )
}
