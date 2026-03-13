"use client"

import { useMemo, useState } from "react"
import type { ChatMessage } from "../lib/types"
import styles from "../styles/chat.module.css"

type Props = {
  messages: ChatMessage[]
  onSend: (message: string) => void
  isBusy: boolean
  lastEvent: string
  error: string | null
}

export function ChatPanel({ messages, onSend, isBusy, lastEvent, error }: Props) {
  const [input, setInput] = useState("")

  const canSend = useMemo(() => input.trim().length > 0 && !isBusy, [input, isBusy])

  return (
    <section className={styles.panel}>
      <div className={styles.panelHeader}>
        <div>
          <h2>Conversation</h2>
          <p>Streaming chat with SSE telemetry</p>
        </div>
        <span className={styles.pill}>{isBusy ? `Live: ${lastEvent}` : "Idle"}</span>
      </div>

      <div className={styles.chatWindow}>
        {messages.length === 0 && (
          <div className={styles.emptyState}>
            Start a conversation to populate memory panels.
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={m.role === "user" ? styles.user : styles.assistant}>
            <div className={styles.messageRole}>{m.role}</div>
            <div className={styles.messageContent}>{m.content}</div>
          </div>
        ))}
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <form
        className={styles.composer}
        onSubmit={(e) => {
          e.preventDefault()
          if (!canSend) return
          onSend(input)
          setInput("")
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your memory graph, skills, or context..."
        />
        <button type="submit" disabled={!canSend}>
          {isBusy ? "Thinking..." : "Send"}
        </button>
      </form>
    </section>
  )
}
