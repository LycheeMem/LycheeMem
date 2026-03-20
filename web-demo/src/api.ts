import type {
    AuthUser,
    ConsolidatorTrace,
    GraphData,
    GraphEdge,
    GraphNode,
    PipelineStatus,
    PipelineTrace,
    SessionInfo,
    SkillItem,
    Turn,
} from "./types";

const API = "";

// ── Auth helpers ──

const TOKEN_KEY = "lychee_token";
const USER_KEY = "lychee_user";

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function getStoredUser(): AuthUser | null {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw) as AuthUser; } catch { return null; }
}

export function storeAuth(user: AuthUser): void {
  localStorage.setItem(TOKEN_KEY, user.token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function clearAuth(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getStoredToken();
  if (token) return { Authorization: `Bearer ${token}` };
  return {};
}

export async function apiRegister(username: string, password: string, displayName?: string): Promise<AuthUser> {
  const r = await fetch(`${API}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, display_name: displayName || username }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data as AuthUser;
}

export async function apiLogin(username: string, password: string): Promise<AuthUser> {
  const r = await fetch(`${API}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data as AuthUser;
}

export async function fetchSessions(): Promise<SessionInfo[]> {
  const r = await fetch(`${API}/sessions?limit=100`, { headers: authHeaders() });
  const data = await r.json();
  return data.sessions || [];
}

export async function updateSessionMeta(sessionId: string, topic: string): Promise<void> {
  await fetch(`${API}/memory/session/${encodeURIComponent(sessionId)}/meta`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ topic }),
  });
}

export async function fetchSessionTurns(
  sessionId: string
): Promise<{
  turns: Turn[];
  summaries: Array<{ boundary_index: number; content: string }>;
  boundary_index: number;
  wm_current_tokens: number;
  wm_max_tokens: number;
}> {
  const r = await fetch(
    `${API}/memory/session/${encodeURIComponent(sessionId)}`,
    { headers: authHeaders() }
  );
  const data = await r.json();
  const turns: Turn[] = data.turns || [];
  const summaries: Array<{ boundary_index: number; content: string; token_count?: number }> =
    (data.summaries || []).sort(
      (a: { boundary_index: number }, b: { boundary_index: number }) =>
        a.boundary_index - b.boundary_index
    );

  // 获取最新的 boundary_index（如果有压缩）
  const boundaryIndex = summaries.length > 0 ? summaries[summaries.length - 1].boundary_index : -1;

  return {
    turns,  // 返回完整的 turns（包括被软删除的原始对话）
    summaries,
    boundary_index: boundaryIndex,
    wm_current_tokens: data.wm_current_tokens ?? 0,
    wm_max_tokens: data.wm_max_tokens ?? 128000,
  };
}



export async function sendChatMessage(
  sessionId: string,
  message: string
): Promise<{
  response: string;
  wm_token_usage?: number;
  memories_retrieved?: number;
  trace?: PipelineTrace | null;
}> {
  const r = await fetch(`${API}/chat/complete`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail =
      (data as Record<string, string>)?.detail ||
      (data as Record<string, string>)?.message ||
      `HTTP ${r.status}`;
    throw new Error(detail);
  }
  return data;
}

export interface StreamStepEvent {
  type: "step";
  step: string;
  status: "done";
  wm_token_usage?: number;
}

export interface StreamAnswerEvent {
  type: "answer";
  content: string;
}

export interface StreamDoneEvent {
  type: "done";
  session_id: string;
  memories_retrieved: number;
  wm_token_usage: number;
  trace: PipelineTrace;
}

export type StreamEvent = StreamStepEvent | StreamAnswerEvent | StreamDoneEvent;

/**
 * SSE 流式对话。通过回调实时接收每个 pipeline 步骤的进度。
 * 后端依次发送 step 事件（wm_manager / search / synthesize / reason），
 * 最后发送 answer 和 done 事件。
 */
export async function streamChatMessage(
  sessionId: string,
  message: string,
  callbacks: {
    onStep?: (step: string, data: Record<string, unknown>) => void;
    onAnswer?: (answer: string) => void;
    onDone?: (data: StreamDoneEvent) => void;
  }
): Promise<void> {
  const r = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ session_id: sessionId, message }),
  });

  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    const detail =
      (data as Record<string, string>)?.detail ||
      (data as Record<string, string>)?.message ||
      `HTTP ${r.status}`;
    throw new Error(detail);
  }

  const reader = r.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by double newlines
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data: ")) continue;
      const jsonStr = line.slice(6).trim();
      if (!jsonStr) continue;

      let evt: Record<string, unknown>;
      try {
        evt = JSON.parse(jsonStr);
      } catch {
        continue;
      }

      if (evt.type === "step" && callbacks.onStep) {
        callbacks.onStep(evt.step as string, evt);
      } else if (evt.type === "answer" && callbacks.onAnswer) {
        callbacks.onAnswer(evt.content as string);
      } else if (evt.type === "done" && callbacks.onDone) {
        callbacks.onDone(evt as unknown as StreamDoneEvent);
      }
    }
  }
}

function getNodeDisplayText(n: Record<string, unknown>): string {
  const name =
    (n?.name as string) ||
    ((n?.properties as Record<string, unknown>)?.name as string);
  const id = (n?.node_id as string) || (n?.id as string);
  return String(name || id || "?");
}

function getNodeTypeLabel(n: Record<string, unknown>): string {
  return String(
    (n?.label as string) ||
      ((n?.properties as Record<string, unknown>)?.label as string) ||
      ""
  );
}

export async function fetchGraphData(): Promise<GraphData> {
  const r = await fetch(`${API}/memory/graph`, { headers: authHeaders() });
  const data = await r.json();
  const nodes: GraphNode[] = ((data.nodes || []) as Record<string, unknown>[]).map(
    (n) => ({
      id: (n.node_id as string) || (n.id as string),
      label: getNodeDisplayText(n),
      properties: (n.properties as Record<string, unknown>) || n,
      typeLabel: getNodeTypeLabel(n),
    })
  );
  const edges: GraphEdge[] = ((data.edges || []) as Record<string, unknown>[]).map(
    (e) => ({
      source: e.source as string,
      target: e.target as string,
      relation: (e.relation as string) || "",
      confidence: (e.confidence as number) || 0.5,
      fact: (e.fact as string) || "",
      evidence: (e.evidence as string) || "",
      timestamp: (e.timestamp as string) || "",
      source_session: (e.source_session as string) || "",
    })
  );
  return { nodes, edges };
}

export async function fetchGraphEdges(): Promise<GraphEdge[]> {
  const r = await fetch(`${API}/memory/graph`, { headers: authHeaders() });
  const data = await r.json();
  const edges: GraphEdge[] = ((data.edges || []) as Record<string, unknown>[]).map(
    (e) => ({
      source: e.source as string,
      target: e.target as string,
      relation: (e.relation as string) || "",
      confidence: (e.confidence as number) || 0.5,
      fact: (e.fact as string) || "",
      evidence: (e.evidence as string) || "",
      timestamp: (e.timestamp as string) || "",
      source_session: (e.source_session as string) || "",
    })
  );
  edges.sort((a, b) =>
    String(b.timestamp || "").localeCompare(String(a.timestamp || ""))
  );
  return edges.slice(0, 80);
}

export async function fetchSkills(): Promise<SkillItem[]> {
  const r = await fetch(`${API}/memory/skills`, { headers: authHeaders() });
  const data = await r.json();
  return data.skills || [];
}

export async function fetchPipelineStatus(): Promise<PipelineStatus> {
  const r = await fetch(`${API}/pipeline/status`, { headers: authHeaders() });
  const data = await r.json();
  return {
    session_count: data.session_count || 0,
    graph_node_count: data.graph_node_count || 0,
    skill_count: data.skill_count || 0,
  };
}

export async function fetchConsolidationResult(): Promise<ConsolidatorTrace> {
  const r = await fetch(`${API}/pipeline/last-consolidation`, { headers: authHeaders() });
  const data = await r.json();
  const status =
    data.status === "done" ? "done" : data.status === "skipped" ? "skipped" : "pending";
  return {
    status,
    entities_added: data.entities_added || 0,
    skills_added: data.skills_added || 0,
    facts_added: data.facts_added || 0,
    has_novelty: data.has_novelty,
    skipped_reason: data.skipped_reason,
    steps: Array.isArray(data.steps) ? data.steps : [],
  };
}

// ── Session CRUD ──

export async function deleteSession(sessionId: string): Promise<void> {
  const r = await fetch(`${API}/memory/session/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error((data as Record<string, string>).detail || `HTTP ${r.status}`);
  }
}

// ── Graph CRUD ──

export async function searchGraphNodes(
  q: string,
  topK = 10
): Promise<GraphData> {
  const r = await fetch(
    `${API}/memory/graph/search?q=${encodeURIComponent(q)}&top_k=${topK}`,
    { headers: authHeaders() }
  );
  const data = await r.json();
  const nodes: GraphNode[] = ((data.nodes || []) as Record<string, unknown>[]).map(
    (n) => ({
      id: (n.node_id as string) || (n.id as string),
      label: getNodeDisplayText(n),
      properties: (n.properties as Record<string, unknown>) || n,
      typeLabel: getNodeTypeLabel(n),
    })
  );
  const edges: GraphEdge[] = ((data.edges || []) as Record<string, unknown>[]).map(
    (e) => ({
      source: e.source as string,
      target: e.target as string,
      relation: (e.relation as string) || "",
      confidence: (e.confidence as number) || 0.5,
      fact: (e.fact as string) || "",
      evidence: (e.evidence as string) || "",
      timestamp: (e.timestamp as string) || "",
      source_session: (e.source_session as string) || "",
    })
  );
  return { nodes, edges };
}

export async function addGraphNode(
  id: string,
  label: string,
  properties?: Record<string, unknown>
): Promise<void> {
  const r = await fetch(`${API}/memory/graph/nodes`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ id, label, properties: properties || {} }),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error((data as Record<string, string>).detail || `HTTP ${r.status}`);
  }
}

export async function deleteGraphNode(nodeId: string): Promise<void> {
  const r = await fetch(`${API}/memory/graph/nodes/${encodeURIComponent(nodeId)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error((data as Record<string, string>).detail || `HTTP ${r.status}`);
  }
}

// ── Skill CRUD ──

export async function deleteSkill(skillId: string): Promise<void> {
  const r = await fetch(`${API}/memory/skills/${encodeURIComponent(skillId)}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error((data as Record<string, string>).detail || `HTTP ${r.status}`);
  }
}

export async function clearGraphMemory(): Promise<void> {
  const r = await fetch(`${API}/memory/graph/clear`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error((data as Record<string, string>).detail || `HTTP ${r.status}`);
  }
}

export async function clearSkillMemory(): Promise<void> {
  const r = await fetch(`${API}/memory/skills/clear`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error((data as Record<string, string>).detail || `HTTP ${r.status}`);
  }
}
