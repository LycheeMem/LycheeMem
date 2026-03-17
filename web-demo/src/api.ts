import type {
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

export async function fetchSessions(): Promise<SessionInfo[]> {
  const r = await fetch(`${API}/sessions?limit=50`);
  const data = await r.json();
  return data.sessions || [];
}

export async function fetchSessionTurns(
  sessionId: string
): Promise<Turn[]> {
  const r = await fetch(
    `${API}/memory/session/${encodeURIComponent(sessionId)}`
  );
  const data = await r.json();
  return data.turns || [];
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
    headers: { "Content-Type": "application/json" },
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
    headers: { "Content-Type": "application/json" },
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
  const r = await fetch(`${API}/memory/graph`);
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
  const r = await fetch(`${API}/memory/graph`);
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
  const r = await fetch(`${API}/memory/skills`);
  const data = await r.json();
  return data.skills || [];
}

export async function fetchPipelineStatus(): Promise<PipelineStatus> {
  const r = await fetch(`${API}/pipeline/status`);
  const data = await r.json();
  return {
    session_count: data.session_count || 0,
    graph_node_count: data.graph_node_count || 0,
    skill_count: data.skill_count || 0,
  };
}

export async function fetchConsolidationResult(): Promise<ConsolidatorTrace> {
  const r = await fetch(`${API}/pipeline/last-consolidation`);
  const data = await r.json();
  return {
    status: data.status === "done" ? "done" : "pending",
    entities_added: data.entities_added || 0,
    skills_added: data.skills_added || 0,
  };
}
