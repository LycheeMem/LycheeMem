"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AgentTimeline } from "./components/AgentTimeline"
import { ChatPanel } from "./components/ChatPanel"
import { GraphView } from "./components/GraphView"
import { MemoryPanels } from "./components/MemoryPanels"
import { ProvenanceList } from "./components/ProvenanceList"
import { StatusCards } from "./components/StatusCards"
import { getApiBase } from "./lib/api"
import type {
    AgentStep,
    ChatMessage,
    GraphData,
    MemorySnapshot,
    PipelineStatus,
    ProvenanceItem,
    ReusePlanItem,
    SessionSummary,
} from "./lib/types"
import styles from "./styles/layout.module.css"

export default function HomePage() {
  const [sessionId, setSessionId] = useState("demo")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([])
  const [graph, setGraph] = useState<GraphData>({ nodes: [], edges: [] })
  const [memories, setMemories] = useState<MemorySnapshot>({
    skills: [],
    sensory: [],
    sessions: [],
    sessionDetail: null,
    reusePlan: [],
  })
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null)
  const [provenance, setProvenance] = useState<ProvenanceItem[]>([])
  const [isBusy, setIsBusy] = useState(false)
  const [lastEvent, setLastEvent] = useState("")
  const [error, setError] = useState<string | null>(null)
  const refreshTimer = useRef<NodeJS.Timeout | null>(null)

  const apiBase = useMemo(() => getApiBase(), [])

  const refreshAll = useCallback(async () => {
    try {
      const [graphRes, skillsRes, sensoryRes, sessionsRes, statusRes] = await Promise.all([
        fetch(`${apiBase}/memory/graph`),
        fetch(`${apiBase}/memory/skills`),
        fetch(`${apiBase}/memory/sensory`),
        fetch(`${apiBase}/sessions?offset=0&limit=20`),
        fetch(`${apiBase}/pipeline/status`),
      ])

      const graphJson = await graphRes.json()
      const skillsJson = await skillsRes.json()
      const sensoryJson = await sensoryRes.json()
      const sessionsJson = await sessionsRes.json()
      const statusJson = await statusRes.json()

      setGraph({ nodes: graphJson.nodes ?? [], edges: graphJson.edges ?? [] })
      setMemories((prev) => ({
        ...prev,
        skills: skillsJson.skills ?? [],
        sensory: sensoryJson.items ?? [],
        sessions: sessionsJson.sessions ?? [],
      }))
      setPipelineStatus(statusJson)
    } catch (err) {
      setError("Failed to refresh memory panels")
    }
  }, [apiBase])

  const refreshSession = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/memory/session/${sessionId}`)
      const data = await res.json()
      setMemories((prev) => ({
        ...prev,
        sessionDetail: data,
      }))
    } catch (err) {
      setError("Failed to refresh session detail")
    }
  }, [apiBase, sessionId])

  const refreshSearch = useCallback(
    async (query: string) => {
      try {
        const res = await fetch(`${apiBase}/memory/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, include_graph: true, include_skills: true, include_sensory: true }),
        })
        const data = await res.json()

        const provenanceItems: ProvenanceItem[] = []
        const reusePlan: ReusePlanItem[] = []

        ;(data.graph_results ?? []).forEach((item: Record<string, unknown>, idx: number) => {
          provenanceItems.push({
            source: "graph",
            index: idx,
            relevance: 0.6,
            summary: JSON.stringify(item).slice(0, 160),
          })
        })

        ;(data.skill_results ?? []).forEach((item: Record<string, unknown>, idx: number) => {
          const score = typeof item.score === "number" ? item.score : 0.0
          provenanceItems.push({
            source: "skill",
            index: idx,
            relevance: score,
            summary: String(item.intent || "skill").slice(0, 160),
          })
          if (score >= 0.85) {
            reusePlan.push({
              id: String(item.id || `skill-${idx}`),
              intent: String(item.intent || ""),
              tool_chain: Array.isArray(item.tool_chain) ? item.tool_chain : [],
              score,
              conditions: String(item.conditions || ""),
            })
          }
        })

        ;(data.sensory_results ?? []).forEach((item: Record<string, unknown>, idx: number) => {
          provenanceItems.push({
            source: "sensory",
            index: idx,
            relevance: 0.4,
            summary: String(item.content || "sensory").slice(0, 160),
          })
        })

        setProvenance(provenanceItems)
        setMemories((prev) => ({
          ...prev,
          reusePlan,
        }))
      } catch (err) {
        setError("Failed to refresh retrieval summary")
      }
    },
    [apiBase],
  )

  useEffect(() => {
    refreshAll()
    refreshSession()

    refreshTimer.current = setInterval(() => {
      refreshAll()
    }, 6000)

    return () => {
      if (refreshTimer.current) clearInterval(refreshTimer.current)
    }
  }, [refreshAll, refreshSession])

  const updateAgentStep = useCallback((id: string, status: AgentStep["status"]) => {
    setAgentSteps((prev) =>
      prev.map((step) => (step.id === id ? { ...step, status } : step)),
    )
  }, [])

  const handleNewMessage = useCallback(
    async (message: string) => {
      setError(null)
      setIsBusy(true)
      setLastEvent("sending")
      setProvenance([])
      setAgentSteps([
        { id: "wm", label: "WM Manager", status: "active" },
        { id: "router", label: "Router", status: "idle" },
        { id: "search", label: "Search", status: "idle" },
        { id: "synth", label: "Synthesize", status: "idle" },
        { id: "reason", label: "Reason", status: "idle" },
        { id: "consolidate", label: "Consolidate", status: "idle" },
      ])

      setMessages((prev) => [
        ...prev,
        { id: `${Date.now()}-user`, role: "user", content: message },
      ])

      try {
        const resp = await fetch(`${apiBase}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message }),
        })

        if (!resp.body) {
          throw new Error("No stream")
        }

        updateAgentStep("router", "active")

        const reader = resp.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        let finalAnswer = ""

        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n")
          buffer = lines.pop() ?? ""

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue
            const payload = JSON.parse(line.slice(6))
            setLastEvent(payload.type)

            if (payload.type === "status") {
              if (payload.content === "processing") {
                updateAgentStep("router", "active")
                updateAgentStep("search", "active")
              }
              if (payload.content === "retrieved") {
                updateAgentStep("router", "done")
                updateAgentStep("search", "done")
                updateAgentStep("synth", "active")
              }
            }

            if (payload.type === "answer") {
              finalAnswer = payload.content
              updateAgentStep("synth", "done")
              updateAgentStep("reason", "active")
            }

            if (payload.type === "done") {
              updateAgentStep("reason", "done")
              updateAgentStep("consolidate", "active")
              setMessages((prev) => [
                ...prev,
                { id: `${Date.now()}-assistant`, role: "assistant", content: finalAnswer },
              ])
            }
          }
        }

        updateAgentStep("consolidate", "done")
        await refreshAll()
        await refreshSession()
        await refreshSearch(message)
      } catch (err) {
        setError("Chat request failed. Is the API running?")
      } finally {
        setIsBusy(false)
        setLastEvent("idle")
      }
    },
    [apiBase, refreshAll, refreshSession, refreshSearch, sessionId, updateAgentStep],
  )

  const handleSessionChange = useCallback((next: SessionSummary) => {
    setSessionId(next.session_id)
  }, [])

  useEffect(() => {
    refreshSession()
  }, [refreshSession, sessionId])

  return (
    <main className={styles.shell}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>A-Frame Memory Console</p>
          <h1 className={styles.title}>Cognitive Memory Demo</h1>
          <p className={styles.subtitle}>
            Live telemetry for working memory, graph, skills, sensory buffer, and agents.
          </p>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.sessionBox}>
            <label className={styles.label}>Session</label>
            <input
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              className={styles.sessionInput}
              placeholder="demo"
            />
          </div>
          <button
            className={styles.refreshButton}
            onClick={() => {
              refreshAll()
              refreshSession()
            }}
          >
            Refresh
          </button>
        </div>
      </header>

      <section className={styles.grid}>
        <section className={styles.leftColumn}>
          <ChatPanel
            messages={messages}
            onSend={handleNewMessage}
            isBusy={isBusy}
            lastEvent={lastEvent}
            error={error}
          />
          <AgentTimeline steps={agentSteps} />
          <ProvenanceList items={provenance} />
        </section>

        <section className={styles.rightColumn}>
          <StatusCards status={pipelineStatus} />
          <GraphView data={graph} />
          <MemoryPanels
            snapshot={memories}
            onSessionSelect={handleSessionChange}
          />
        </section>
      </section>
    </main>
  )
}
