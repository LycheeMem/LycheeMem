export type ChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
}

export type AgentStep = {
  id: string
  label: string
  status: "idle" | "active" | "done" | "error"
}

export type GraphNode = {
  id: string
  label?: string
  name?: string
  created_at?: string
}

export type GraphEdge = {
  source: string
  target: string
  relation?: string
  timestamp?: string
  confidence?: number
}

export type GraphData = {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export type SkillItem = {
  id: string
  intent: string
  tool_chain: Array<Record<string, unknown>>
  metadata?: Record<string, unknown>
  success_count?: number
  last_used?: string | null
  conditions?: string
}

export type ReusePlanItem = {
  id: string
  intent: string
  tool_chain: Array<Record<string, unknown>>
  score: number
  conditions?: string
}

export type SensoryItem = {
  content: string
  modality: string
  timestamp: string
}

export type SessionSummary = {
  session_id: string
  turn_count: number
  last_message: string
  topic?: string
  tags?: string[]
  created_at?: string
  updated_at?: string
}

export type SessionDetail = {
  session_id: string
  turns: Array<{ role: string; content: string }>
  turn_count: number
  summaries: Array<{ boundary_index: number; content: string }>
}

export type MemorySnapshot = {
  skills: SkillItem[]
  sensory: SensoryItem[]
  sessions: SessionSummary[]
  sessionDetail: SessionDetail | null
  reusePlan: ReusePlanItem[]
}

export type PipelineStatus = {
  session_count: number
  graph_node_count: number
  graph_edge_count: number
  skill_count: number
  sensory_buffer_size: number
}

export type ProvenanceItem = {
  source: string
  index: number
  relevance: number
  summary: string
}
