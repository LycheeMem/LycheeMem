// ── Message ──
export interface MessageMeta {
  memories_retrieved: number;
  wm_token_usage: number;
  trace?: PipelineTrace | null;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  meta: MessageMeta | null;
}

// ── Graph ──
export interface GraphNode {
  id: string;
  label: string;
  typeLabel: string;
  properties: Record<string, unknown>;
  // simulation coordinates (set by d3-force)
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  relation: string;
  confidence: number;
  fact: string;
  evidence: string;
  timestamp: string;
  source_session: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// ── Agent ──
export type AgentStatusValue = "idle" | "running" | "done";

export const AGENT_NAMES = [
  "wm_manager",
  "search_coordinator",
  "synthesizer",
  "reasoner",
  "consolidator",
] as const;

export type AgentName = (typeof AGENT_NAMES)[number];

export interface AgentInfo {
  key: AgentName;
  icon: string;
  label: string;
}

export const AGENTS: AgentInfo[] = [
  { key: "wm_manager", icon: "\u{1F9E9}", label: "WM Manager" },
  { key: "search_coordinator", icon: "\u{1F50D}", label: "Search" },
  { key: "synthesizer", icon: "\u{1F52C}", label: "Synthesizer" },
  { key: "reasoner", icon: "\u{1F4A1}", label: "Reasoner" },
  { key: "consolidator", icon: "\u{1F4E6}", label: "Consolidator" },
];

// ── Pipeline Status ──
export interface PipelineStatus {
  session_count: number;
  graph_node_count: number;
  skill_count: number;
}

// ── Session ──
export interface SessionInfo {
  session_id: string;
  topic?: string;
}

// ── Pipeline Steps (Timeline) ──
export const PIPELINE_STEPS = [
  "wm_manager",
  "search_coordinator",
  "synthesizer",
  "reasoner",
] as const;

export const STEP_LABELS: Record<string, string> = {
  wm_manager: "\u5DE5\u4F5C\u8BB0\u5FC6",
  search_coordinator: "\u68C0\u7D22",
  synthesizer: "\u5408\u6210",
  reasoner: "\u63A8\u7406",
};

// ── Skill ──
export interface SkillItem {
  intent?: string;
  name?: string;
  skill_id?: string;
  id?: string;
  doc_markdown?: string;
  doc?: string;
  markdown?: string;
  conditions?: string;
  success_count?: number;
  score?: number;
  last_used?: string;
}

// ── Turn (from session history) ──
export interface Turn {
  role: string;
  content: string;
}

// ── Graph Transform ──
export interface GraphTransform {
  x: number;
  y: number;
  k: number;
}

// ── Pipeline Trace ──
export interface WMManagerTrace {
  wm_token_usage: number;
  compressed_turn_count: number;
  raw_recent_turn_count: number;
  compression_happened: boolean;
}

export interface GraphMemoryHit {
  node_id: string;
  name: string;
  label: string;
  score: number;
  neighbor_count: number;
}

export interface SkillHit {
  skill_id: string;
  intent: string;
  score: number;
  reusable: boolean;
}

export interface SearchCoordinatorTrace {
  graph_memories: GraphMemoryHit[];
  skills: SkillHit[];
  total_retrieved: number;
}

export interface ProvenanceItem {
  source: string;
  index: number;
  relevance: number;
  summary: string;
}

export interface SynthesizerTrace {
  background_context: string;
  provenance: ProvenanceItem[];
  skill_reuse_plan: Record<string, unknown>[];
  kept_count: number;
  dropped_count: number;
}

export interface ReasonerTrace {
  response_length: number;
}

export interface ConsolidatorTrace {
  status: "pending" | "done" | "skipped";
  entities_added: number;
  skills_added: number;
}

export interface PipelineTrace {
  wm_manager: WMManagerTrace;
  search_coordinator: SearchCoordinatorTrace;
  synthesizer: SynthesizerTrace;
  reasoner: ReasonerTrace;
  consolidator: ConsolidatorTrace;
}
