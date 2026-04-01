import { create } from "zustand";
import { clearAuth, getStoredUser, storeAuth } from "./api";
import type {
    AgentName,
    AgentStatusValue,
    AuthUser,
    GraphData,
    GraphEdge,
    Message,
    PipelineStatus,
    PipelineTrace,
    SessionInfo,
    SkillItem,
    Turn,
} from "./types";
import { AGENT_NAMES } from "./types";

export interface AppState {
  // Auth
  user: AuthUser | null;

  // Session
  sessionId: string;
  sessions: SessionInfo[];

  // Chat
  messages: Message[];
  isStreaming: boolean;
  isTyping: boolean;

  // Graph
  graphData: GraphData;

  // Agents
  agents: Record<AgentName, AgentStatusValue>;
  showTimeline: boolean;

  // Working Memory
  wmTokenUsage: number;
  wmMaxTokens: number;
  wmTurns: Turn[];
  wmSummaries: Array<{ boundary_index: number; content: string; token_count?: number }>;
  wmBoundaryIndex: number;  // Latest compression boundary (-1 if no compression)

  // Memory panels
  graphEdges: GraphEdge[];
  skills: SkillItem[];

  // Pipeline status
  pipelineStatus: PipelineStatus;

  // Active memory tab
  activeTab: string;

  // Graph edge interaction
  hoveredEdge: GraphEdge | null;
  selectedEdge: GraphEdge | null;

  // Pipeline trace
  currentTrace: PipelineTrace | null;

  // Partial trace fragments accumulated during streaming (step name → trace piece)
  partialTrace: Partial<PipelineTrace> | null;

  // Streaming step progress ("wm_manager" | "search" | "synthesize" | "reason")
  completedSteps: string[];

  // 流式输出中的临时文本（token by token 累积）
  streamingContent: string;

  // Actions
  setSessionId: (id: string) => void;
  setSessions: (sessions: SessionInfo[]) => void;
  addMessage: (msg: Message) => void;
  setMessages: (msgs: Message[]) => void;
  setIsStreaming: (v: boolean) => void;
  setIsTyping: (v: boolean) => void;
  setGraphData: (data: GraphData) => void;
  setAgentStatus: (name: AgentName, status: AgentStatusValue) => void;
  resetAgents: () => void;
  setShowTimeline: (v: boolean) => void;
  setWmTokenUsage: (v: number) => void;
  setWmMaxTokens: (v: number) => void;
  setWmTurns: (turns: Turn[]) => void;
  setWmSummaries: (summaries: Array<{ boundary_index: number; content: string; token_count?: number }>) => void;
  setWmBoundaryIndex: (index: number) => void;
  setGraphEdges: (edges: GraphEdge[]) => void;
  setSkills: (skills: SkillItem[]) => void;
  setPipelineStatus: (status: PipelineStatus) => void;
  setActiveTab: (tab: string) => void;
  setHoveredEdge: (edge: GraphEdge | null) => void;
  setSelectedEdge: (edge: GraphEdge | null) => void;
  setCurrentTrace: (trace: PipelineTrace | null) => void;
  setPartialTrace: (trace: Partial<PipelineTrace> | null) => void;
  mergePartialTrace: (fragment: Record<string, unknown>) => void;
  setCompletedSteps: (steps: string[]) => void;
  setStreamingContent: (content: string) => void;
  appendStreamingContent: (token: string) => void;
  newSession: () => void;
  login: (user: AuthUser) => void;
  logout: () => void;
}

function makeInitialAgents(): Record<AgentName, AgentStatusValue> {
  const agents = {} as Record<AgentName, AgentStatusValue>;
  for (const name of AGENT_NAMES) {
    agents[name] = "idle";
  }
  return agents;
}

function generateSessionId(): string {
  return "session_" + Date.now().toString(36);
}

export const useStore = create<AppState>((set) => ({
  user: getStoredUser(),
  sessionId: generateSessionId(),
  sessions: [],
  messages: [],
  isStreaming: false,
  isTyping: false,
  graphData: { nodes: [], edges: [], treeRoots: [] },
  agents: makeInitialAgents(),
  showTimeline: false,
  wmTokenUsage: 0,
  wmMaxTokens: 128000,
  wmTurns: [],
  wmSummaries: [],
  wmBoundaryIndex: -1,
  graphEdges: [],
  skills: [],
  pipelineStatus: {
    session_count: 0,
    graph_node_count: 0,
    graph_edge_count: 0,
    skill_count: 0,
  },
  activeTab: "tab-agents",
  hoveredEdge: null,
  selectedEdge: null,
  currentTrace: null,
  partialTrace: null,
  completedSteps: [],
  streamingContent: "",

  setSessionId: (id) => set({ sessionId: id }),
  setSessions: (sessions) => set({ sessions }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setMessages: (msgs) => set({ messages: msgs }),
  setIsStreaming: (v) => set({ isStreaming: v }),
  setIsTyping: (v) => set({ isTyping: v }),
  setGraphData: (data) => set({ graphData: data }),
  setAgentStatus: (name, status) =>
    set((s) => ({ agents: { ...s.agents, [name]: status } })),
  resetAgents: () => set({ agents: makeInitialAgents() }),
  setShowTimeline: (v) => set({ showTimeline: v }),
  setWmTokenUsage: (v) => set({ wmTokenUsage: v }),
  setWmMaxTokens: (v) => set({ wmMaxTokens: v }),
  setWmTurns: (turns) => set({ wmTurns: turns }),
  setWmSummaries: (summaries) => set({ wmSummaries: summaries }),
  setWmBoundaryIndex: (index) => set({ wmBoundaryIndex: index }),
  setGraphEdges: (edges) => set({ graphEdges: edges }),
  setSkills: (skills) => set({ skills }),
  setPipelineStatus: (status) => set({ pipelineStatus: status }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setHoveredEdge: (edge) => set({ hoveredEdge: edge }),
  setSelectedEdge: (edge) => set({ selectedEdge: edge }),
  setCurrentTrace: (trace) => set({ currentTrace: trace }),
  setPartialTrace: (trace) => set({ partialTrace: trace }),
  mergePartialTrace: (fragment) =>
    set((s) => ({
      partialTrace: { ...(s.partialTrace ?? {}), ...fragment } as Partial<PipelineTrace>,
    })),
  setCompletedSteps: (steps) => set({ completedSteps: steps }),
  setStreamingContent: (content) => set({ streamingContent: content }),
  appendStreamingContent: (token) => set((s) => ({ streamingContent: s.streamingContent + token })),
  newSession: () =>
    set({
      sessionId: generateSessionId(),
      messages: [],
      agents: makeInitialAgents(),
      showTimeline: false,
      wmTurns: [],
      wmSummaries: [],
      wmBoundaryIndex: -1,
      hoveredEdge: null,
      selectedEdge: null,
      currentTrace: null,
      partialTrace: null,
      completedSteps: [],
      streamingContent: "",
    }),
  login: (user) => {
    storeAuth(user);
    set({ user });
  },
  logout: () => {
    clearAuth();
    set({
      user: null,
      sessionId: generateSessionId(),
      sessions: [],
      messages: [],
      agents: makeInitialAgents(),
      graphData: { nodes: [], edges: [], treeRoots: [] },
      graphEdges: [],
      skills: [],
      wmTurns: [],
      wmSummaries: [],
      wmBoundaryIndex: -1,
      currentTrace: null,
      partialTrace: null,
      completedSteps: [],
    });
  },
}));
