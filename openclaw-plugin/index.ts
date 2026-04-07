const PLUGIN_ID = "lycheemem-tools";

type Json = Record<string, unknown>;
type ToolResult = {
  content: Array<{ type: "text"; text: string }>;
  structuredContent?: Json;
  isError?: boolean;
};

type PluginConfig = {
  baseUrl: string;
  transport: "mcp" | "http";
  timeout: number;
  apiToken: string;
  enableHostLifecycle: boolean;
  enablePromptPresence: boolean;
  enableAutoAppendTurns: boolean;
  enableBoundaryConsolidation: boolean;
  enableProactiveConsolidation: boolean;
  proactiveConsolidationCooldownSeconds: number;
};

type LifecycleSessionState = {
  generation: number;
  lycheeSessionId: string;
  openclawSessionId?: string;
  channelId?: string;
  conversationId?: string;
  lastRetrievedContext: string;
  lastBoundaryAt?: number;
  lastBoundaryKind?: string;
  pendingProactiveConsolidation?: boolean;
  lastConsolidatedAt?: number;
  forceSkillReminderOnNextPrompt?: boolean;
};

type LifecycleState = {
  sessions: Record<string, LifecycleSessionState>;
  sessionAliases: Record<string, string>;
  latestConversationByChannel: Record<string, string>;
};

const SEARCH_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    query: { type: "string", description: "Natural language recall request." },
    top_k: {
      type: "integer",
      default: 5,
      description: "Maximum results per source before synthesis."
    },
    include_graph: {
      type: "boolean",
      default: true,
      description: "Whether to search graph memories."
    },
    include_skills: {
      type: "boolean",
      default: true,
      description: "Whether to search skill memories."
    }
  },
  required: ["query"]
} as const;

const SMART_SEARCH_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    query: { type: "string", description: "Natural language recall request." },
    top_k: {
      type: "integer",
      default: 5,
      description: "Maximum results per source before optional synthesis."
    },
    include_graph: {
      type: "boolean",
      default: true,
      description: "Whether to search graph memories."
    },
    include_skills: {
      type: "boolean",
      default: true,
      description: "Whether to search skill memories."
    },
    synthesize: {
      type: "boolean",
      default: true,
      description: "Whether to automatically synthesize a compressed background_context."
    },
    include_provenance: {
      type: "boolean",
      default: false,
      description: "Whether to include verbose provenance details in the tool result."
    },
    mode: {
      type: "string",
      default: "full",
      description: "Return mode: full is the recommended default for agents, raw returns only retrieval results, and compact returns only synthesized output."
    }
  },
  required: ["query"]
} as const;

const APPEND_TURN_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    session_id: {
      type: "string",
      description: "LycheeMem session id used to accumulate mirrored host turns."
    },
    role: {
      type: "string",
      description: "Conversation role, typically user or assistant."
    },
    content: {
      type: "string",
      description: "Raw turn text to append into LycheeMem's session store."
    },
    token_count: {
      type: "integer",
      default: 0,
      description: "Optional token count if the host already knows it."
    }
  },
  required: ["session_id", "role", "content"]
} as const;

const SYNTHESIZE_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    user_query: {
      type: "string",
      description: "The current user request used for relevance scoring."
    },
    graph_results: {
      type: "array",
      description: "graph_results returned by lychee_memory_search."
    },
    skill_results: {
      type: "array",
      description: "skill_results returned by lychee_memory_search."
    }
  },
  required: ["user_query", "graph_results", "skill_results"]
} as const;

const CONSOLIDATE_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    session_id: {
      type: "string",
      description: "LycheeMem session id."
    },
    retrieved_context: {
      type: "string",
      default: "",
      description: "Compressed background_context from the current turn for novelty checks."
    },
    background: {
      type: "boolean",
      default: true,
      description: "Whether consolidation should run asynchronously."
    }
  },
  required: ["session_id"]
} as const;

function normalizeBaseUrl(url: string): string {
  return (url || "http://127.0.0.1:8000").replace(/\/+$/, "");
}

function asString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) {
      return true;
    }
    if (["0", "false", "no", "off"].includes(normalized)) {
      return false;
    }
  }
  return fallback;
}

function asNumber(value: unknown, fallback: number): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function sanitizeSegment(value: string): string {
  const sanitized = value.replace(/[^a-zA-Z0-9._:-]+/g, "_").replace(/^_+|_+$/g, "");
  return sanitized || "default";
}

function getPluginConfig(api: any): PluginConfig {
  const raw = api?.config?.plugins?.entries?.[PLUGIN_ID]?.config ?? {};

  return {
    baseUrl: normalizeBaseUrl(String(raw.baseUrl ?? "http://127.0.0.1:8000")),
    transport: String(raw.transport ?? "mcp").trim().toLowerCase() === "http" ? "http" : "mcp",
    timeout: Number(raw.timeout ?? 120),
    apiToken: String(raw.apiToken ?? ""),
    enableHostLifecycle: asBoolean(
      raw.enableHostLifecycle,
      true
    ),
    enablePromptPresence: asBoolean(
      raw.enablePromptPresence,
      true
    ),
    enableAutoAppendTurns: asBoolean(
      raw.enableAutoAppendTurns,
      true
    ),
    enableBoundaryConsolidation: asBoolean(
      raw.enableBoundaryConsolidation,
      true
    ),
    enableProactiveConsolidation: asBoolean(
      raw.enableProactiveConsolidation,
      true
    ),
    proactiveConsolidationCooldownSeconds: Math.max(
      15,
      Math.floor(
        asNumber(
          raw.proactiveConsolidationCooldownSeconds,
          180
        )
      )
    )
  };
}

function createAbortSignal(timeoutSeconds: number): AbortSignal | undefined {
  const timeoutMs = Math.max(1_000, Math.floor(timeoutSeconds * 1000));
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
    return AbortSignal.timeout(timeoutMs);
  }
  return undefined;
}

function toHeaders(cfg: PluginConfig, extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...extra
  };
  // Keep optional Bearer support for older secured deployments, but the
  // current merged LycheeMem backend no longer requires authentication.
  if (cfg.apiToken) {
    headers.Authorization = `Bearer ${cfg.apiToken}`;
  }
  return headers;
}

function toToolResult(payload: unknown): ToolResult {
  const text = JSON.stringify(payload, null, 2);
  return {
    content: [{ type: "text", text }],
    structuredContent: isObject(payload) ? payload : undefined,
    isError: false
  };
}

function isObject(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function parseJsonResponse(response: Response, context: string): Promise<unknown> {
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${context}: HTTP ${response.status} ${body}`.trim());
  }
  return response.json();
}

function getMcpState(api: any): { sessionId?: string } {
  const holder = api.__lycheeMemState ?? {};
  api.__lycheeMemState = holder;
  return holder;
}

function getLifecycleState(api: any): LifecycleState {
  const holder = api.__lycheeLifecycleState ?? {
    sessions: {},
    sessionAliases: {},
    latestConversationByChannel: {}
  };
  api.__lycheeLifecycleState = holder;
  return holder as LifecycleState;
}

async function ensureMcpInitialized(api: any, cfg: PluginConfig): Promise<string> {
  const state = getMcpState(api);
  if (state.sessionId) {
    return state.sessionId;
  }

  const initResponse = await fetch(`${cfg.baseUrl}/mcp`, {
    method: "POST",
    headers: toHeaders(cfg),
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: "init",
      method: "initialize",
      params: {}
    }),
    signal: createAbortSignal(cfg.timeout)
  });

  await parseJsonResponse(initResponse, "LycheeMem MCP initialize failed");

  const sessionId = initResponse.headers.get("Mcp-Session-Id") ?? initResponse.headers.get("mcp-session-id");
  if (!sessionId) {
    throw new Error("LycheeMem MCP initialize did not return Mcp-Session-Id");
  }

  const confirmResponse = await fetch(`${cfg.baseUrl}/mcp`, {
    method: "POST",
    headers: toHeaders(cfg, { "Mcp-Session-Id": sessionId }),
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "initialized",
      params: {}
    }),
    signal: createAbortSignal(cfg.timeout)
  });

  if (!confirmResponse.ok) {
    const body = await confirmResponse.text();
    throw new Error(`LycheeMem MCP initialization confirmation failed: HTTP ${confirmResponse.status} ${body}`.trim());
  }

  state.sessionId = sessionId;
  return sessionId;
}

async function callMcpTool(api: any, cfg: PluginConfig, name: string, arguments_: Json): Promise<ToolResult> {
  const sessionId = await ensureMcpInitialized(api, cfg);
  const response = await fetch(`${cfg.baseUrl}/mcp`, {
    method: "POST",
    headers: toHeaders(cfg, { "Mcp-Session-Id": sessionId }),
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: `${name}-${Date.now()}`,
      method: "tools/call",
      params: {
        name,
        arguments: arguments_
      }
    }),
    signal: createAbortSignal(cfg.timeout)
  });

  const body = await parseJsonResponse(response, `LycheeMem MCP tool failed: ${name}`);
  if (!isObject(body)) {
    throw new Error(`LycheeMem MCP tool returned invalid payload: ${name}`);
  }
  if (isObject(body.error)) {
    throw new Error(String(body.error.message ?? JSON.stringify(body.error)));
  }

  const result = body.result;
  if (isObject(result) && Array.isArray(result.content)) {
    return {
      content: result.content as Array<{ type: "text"; text: string }>,
      structuredContent: isObject(result.structuredContent) ? result.structuredContent : undefined,
      isError: Boolean(result.isError)
    };
  }

  return toToolResult(result);
}

async function callHttpEndpoint(cfg: PluginConfig, path: string, payload: Json, context: string): Promise<ToolResult> {
  const response = await fetch(`${cfg.baseUrl}${path}`, {
    method: "POST",
    headers: toHeaders(cfg),
    body: JSON.stringify(payload),
    signal: createAbortSignal(cfg.timeout)
  });
  const body = await parseJsonResponse(response, context);
  return toToolResult(body);
}

function registerTool(api: any, spec: {
  name: string;
  description: string;
  parameters: Json;
  execute: (_id: string, params: Json) => Promise<ToolResult>;
}): void {
  api.registerTool({
    name: spec.name,
    description: spec.description,
    parameters: spec.parameters,
    async execute(id: string, params: Json) {
      try {
        return await spec.execute(id, params);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        api?.logger?.error?.(`[${PLUGIN_ID}] ${spec.name} failed: ${message}`);
        return {
          content: [{ type: "text", text: message }],
          isError: true
        };
      }
    }
  });
}

function extractStructuredPayload(result: unknown): Json | undefined {
  if (!isObject(result)) {
    return undefined;
  }

  if (isObject(result.structuredContent)) {
    return result.structuredContent;
  }

  if (isObject(result.result) && isObject(result.result.structuredContent)) {
    return result.result.structuredContent;
  }

  return result;
}

function extractBackgroundContext(result: unknown): string {
  const payload = extractStructuredPayload(result);
  const direct = payload ? asString(payload.background_context) : undefined;
  if (direct) {
    return direct;
  }

  if (isObject(result) && Array.isArray(result.content)) {
    for (const item of result.content) {
      if (!isObject(item) || typeof item.text !== "string") {
        continue;
      }
      try {
        const parsed = JSON.parse(item.text);
        if (isObject(parsed)) {
          const nested = asString(parsed.background_context);
          if (nested) {
            return nested;
          }
        }
      } catch {
        // Ignore non-JSON tool text output.
      }
    }
  }

  return "";
}

function extractMessageText(message: unknown): string {
  if (!isObject(message)) {
    return "";
  }

  if (typeof message.text === "string") {
    return message.text.trim();
  }

  const content = message.content;
  if (typeof content === "string") {
    return content.trim();
  }

  if (!Array.isArray(content)) {
    return "";
  }

  const parts: string[] = [];
  for (const item of content) {
    if (!isObject(item) || item.type !== "text" || typeof item.text !== "string") {
      continue;
    }
    const trimmed = item.text.trim();
    if (trimmed) {
      parts.push(trimmed);
    }
  }
  return parts.join("\n").trim();
}

function extractMessageRole(message: unknown): string | undefined {
  if (!isObject(message)) {
    return undefined;
  }
  return asString(message.role)?.toLowerCase();
}

function looksLikeLongTermMemorySignal(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) {
    return false;
  }

  const patterns = [
    /\bremember\b/i,
    /\bkeep in mind\b/i,
    /\bdefault\b/i,
    /\bpreference\b/i,
    /\bworkflow\b/i,
    /\bpolicy\b/i,
    /\brequirement\b/i,
    /\bstandard\b/i,
    /\bprefer\b/i,
    /\balways\b/i,
    /\bnever\b/i,
    /\bmust\b/i,
    /(?:\u8bb0\u4f4f|\u8bb0\u4e0b|\u9ed8\u8ba4|\u4ee5\u540e|\u89c4\u8303|\u89c4\u5219|\u8981\u6c42|\u504f\u597d|\u4f18\u5148|\u5fc5\u987b|\u6d41\u7a0b|\u6807\u51c6|\u90e8\u7f72\u524d|\u6587\u6863)/
  ];

  return patterns.some((pattern) => pattern.test(trimmed));
}

class OpenClawAdapter {
  api: any;
  cfg: PluginConfig;

  constructor(api: any) {
    this.api = api;
    this.cfg = getPluginConfig(api);
  }

  register(): void {
    if (!this.cfg.enableHostLifecycle) {
      this.api?.logger?.info?.(`[${PLUGIN_ID}] host lifecycle integration disabled`);
      return;
    }

    this.api.on("session_start", (event: any, ctx: any) => {
      this.rememberSession({
        sessionKey: event?.sessionKey ?? ctx?.sessionKey,
        sessionId: event?.sessionId ?? ctx?.sessionId
      });
    });

    this.api.on("before_prompt_build", (_event: any, ctx: any) => {
      const session = this.ensureSession({
        sessionKey: ctx?.sessionKey,
        sessionId: ctx?.sessionId,
        channelId: ctx?.channelId
      });

      if (!this.cfg.enablePromptPresence) {
        return;
      }

      const promptParts = [this.buildPromptGuidance()];
      if (session?.forceSkillReminderOnNextPrompt) {
        promptParts.push(this.buildResetReminderGuidance());
        session.forceSkillReminderOnNextPrompt = false;
      }

      return {
        prependSystemContext: promptParts.join("\n\n")
      };
    });

    this.api.on("message_received", async (event: any, ctx: any) => {
      if (!this.cfg.enableAutoAppendTurns) {
        return;
      }
      await this.appendTurn({
        channelId: asString(ctx?.channelId),
        conversationId: asString(ctx?.conversationId),
        role: "user",
        content: asString(event?.content) ?? ""
      });
    });

    this.api.on("before_message_write", (event: any, ctx: any) => {
      if (!this.cfg.enableAutoAppendTurns) {
        return;
      }
      const role = extractMessageRole(event?.message);
      if (role !== "assistant") {
        return;
      }
      const content = extractMessageText(event?.message);
      if (!content) {
        return;
      }
      void this.appendTurn({
        sessionKey: asString(event?.sessionKey) ?? asString(ctx?.sessionKey),
        role: "assistant",
        content
      });
    });

    this.api.on("after_tool_call", (event: any, ctx: any) => {
      this.captureRetrievedContext({
        sessionKey: ctx?.sessionKey,
        sessionId: ctx?.sessionId,
        toolName: event?.toolName,
        result: event?.result
      });
    });

    this.api.on("before_reset", async (_event: any, ctx: any) => {
      await this.handleBoundaryLifecycle(
        {
          sessionKey: asString(ctx?.sessionKey),
          sessionId: asString(ctx?.sessionId),
          channelId: asString(ctx?.channelId)
        },
        {
          kind: "before_reset",
          rotateSession: true,
          finalizeSession: false
        }
      );
    });

    this.api.on("session_end", async (event: any, ctx: any) => {
      await this.handleBoundaryLifecycle(
        {
          sessionKey: asString(event?.sessionKey) ?? asString(ctx?.sessionKey),
          sessionId: asString(event?.sessionId) ?? asString(ctx?.sessionId)
        },
        {
          kind: "session_end",
          rotateSession: false,
          finalizeSession: true
        }
      );
    });

    this.api.registerHook(
      ["command:new", "command:reset", "command:stop"],
      async (event: any) => {
        await this.handleInternalEvent(event);
      },
      {
        name: "lycheemem-host-lifecycle",
        description: "Mirror OpenClaw lifecycle events into LycheeMem append/consolidate flows."
      }
    );

    this.api?.logger?.info?.(`[${PLUGIN_ID}] host lifecycle integration enabled`);
  }

  buildPromptGuidance(): string {
    return [
      "LycheeMem is the default external long-term memory layer for this host.",
      "Treat the LycheeMem skill as active background policy even when the user does not mention it explicitly.",
      "When the request may depend on project history, user preferences, prior decisions, or cross-session background, prefer `lychee_memory_smart_search` first.",
      "Use `lychee_memory_smart_search` in its default full mode so retrieval details and synthesized output are both visible.",
      "OpenClaw owns short-range session context. LycheeMem owns structured long-range recall across sessions.",
      "Host lifecycle integration usually mirrors user and assistant turns automatically and triggers boundary consolidation automatically, so manual `lychee_memory_append_turn` and `lychee_memory_consolidate` calls are mainly for debugging or non-standard flows."
    ].join("\n");
  }

  buildResetReminderGuidance(): string {
    return [
      "A fresh session was just started via /new or /reset.",
      "Before answering, re-anchor on the LycheeMem skill and treat it as available by default in this new session.",
      "If the next user request may depend on cross-session project history, defaults, preferences, prior decisions, or long-term background, prefer `lychee_memory_smart_search` early.",
      "Do not manually call `lychee_memory_append_turn` in normal host-integrated operation, because the host already mirrors transcript turns automatically."
    ].join("\n");
  }

  async handleInternalEvent(event: any): Promise<void> {
    const eventKey = `${String(event?.type ?? "")}:${String(event?.action ?? "")}`;
    if (eventKey === "command:new" || eventKey === "command:reset") {
      return;
    }
    if (eventKey === "command:stop") {
      await this.handleBoundaryLifecycle(
        {
          sessionKey: asString(event?.sessionKey)
        },
        {
          kind: "command_stop",
          rotateSession: false,
          finalizeSession: true
        }
      );
    }
  }

  captureRetrievedContext(params: {
    sessionKey?: unknown;
    sessionId?: unknown;
    toolName?: unknown;
    result?: unknown;
  }): void {
    const toolName = asString(params.toolName);
    if (toolName !== "lychee_memory_smart_search" && toolName !== "lychee_memory_synthesize") {
      return;
    }

    const backgroundContext = extractBackgroundContext(params.result);
    if (!backgroundContext) {
      return;
    }

    const session = this.ensureSession({
      sessionKey: asString(params.sessionKey),
      sessionId: asString(params.sessionId)
    });
    if (!session) {
      return;
    }

    session.lastRetrievedContext = backgroundContext;
  }

  resolveSessionRecord(params: {
    sessionKey?: string;
    sessionId?: string;
    channelId?: string;
  }): { state: LifecycleState; resolvedKey?: string; session?: LifecycleSessionState } {
    const state = getLifecycleState(this.api);
    const candidates = [
      params.sessionKey,
      params.sessionId ? `session:${params.sessionId}` : undefined,
      params.channelId ? state.latestConversationByChannel[params.channelId] : undefined,
      params.channelId ? `channel:${params.channelId}` : undefined
    ].filter((value): value is string => Boolean(value));

    for (const candidate of candidates) {
      const resolvedKey = state.sessionAliases[candidate] ?? candidate;
      const session = state.sessions[resolvedKey];
      if (session) {
        return { state, resolvedKey, session };
      }
    }

    return { state };
  }

  shouldSkipBoundary(session: LifecycleSessionState, kind: string): boolean {
    const now = Date.now();
    if (session.lastBoundaryKind === kind && session.lastBoundaryAt && now - session.lastBoundaryAt < 5_000) {
      return true;
    }
    if (
      kind === "session_end" &&
      session.lastBoundaryKind === "before_reset" &&
      session.lastBoundaryAt &&
      now - session.lastBoundaryAt < 5_000
    ) {
      return true;
    }
    session.lastBoundaryKind = kind;
    session.lastBoundaryAt = now;
    return false;
  }

  async handleBoundaryLifecycle(
    identity: {
      sessionKey?: string;
      sessionId?: string;
      channelId?: string;
    },
    behavior: {
      kind: string;
      rotateSession: boolean;
      finalizeSession: boolean;
    }
  ): Promise<void> {
    const { state, resolvedKey, session } = this.resolveSessionRecord(identity);
    if (!resolvedKey || !session) {
      return;
    }
    if (this.shouldSkipBoundary(session, behavior.kind)) {
      return;
    }

    const now = Date.now();
    const recentlyConsolidated =
      session.lastConsolidatedAt !== undefined &&
      now - session.lastConsolidatedAt < 15_000 &&
      !session.pendingProactiveConsolidation;

    if (this.cfg.enableBoundaryConsolidation && !recentlyConsolidated) {
      const consolidated = await this.callLifecycleTool(
        "lychee_memory_consolidate",
        "/memory/consolidate",
        {
          session_id: session.lycheeSessionId,
          retrieved_context: session.lastRetrievedContext,
          background: true
        },
        "boundary consolidate"
      );
      if (consolidated) {
        session.lastConsolidatedAt = Date.now();
        session.pendingProactiveConsolidation = false;
      }
    }

    if (behavior.finalizeSession) {
      delete state.sessions[resolvedKey];
      if (identity.sessionKey) {
        delete state.sessionAliases[identity.sessionKey];
      }
      return;
    }

    if (behavior.rotateSession) {
      session.generation += 1;
      session.lycheeSessionId = this.buildLycheeSessionId(resolvedKey, session.generation);
      session.lastRetrievedContext = "";
      session.openclawSessionId = undefined;
      session.forceSkillReminderOnNextPrompt = true;
    }
  }

  async appendTurn(params: {
    sessionKey?: string;
    sessionId?: string;
    channelId?: string;
    conversationId?: string;
    role: "user" | "assistant";
    content: string;
  }): Promise<void> {
    const normalized = this.normalizeTurnContent(params.role, params.content);
    if (!normalized) {
      return;
    }

    const session = this.ensureSession(params);
    if (!session) {
      return;
    }

    await this.callLifecycleTool(
      "lychee_memory_append_turn",
      "/memory/append-turn",
      {
        session_id: session.lycheeSessionId,
        role: params.role,
        content: normalized,
        token_count: 0
      },
      `append ${params.role} turn`
    );

    if (params.role === "user" && this.cfg.enableProactiveConsolidation && looksLikeLongTermMemorySignal(normalized)) {
      session.pendingProactiveConsolidation = true;
    }

    if (params.role === "assistant") {
      await this.maybeRunProactiveConsolidation(session);
    }
  }

  async maybeRunProactiveConsolidation(session: LifecycleSessionState): Promise<void> {
    if (!this.cfg.enableProactiveConsolidation || !session.pendingProactiveConsolidation) {
      return;
    }

    const now = Date.now();
    const cooldownMs = this.cfg.proactiveConsolidationCooldownSeconds * 1000;
    if (session.lastConsolidatedAt !== undefined && now - session.lastConsolidatedAt < cooldownMs) {
      return;
    }

    const consolidated = await this.callLifecycleTool(
      "lychee_memory_consolidate",
      "/memory/consolidate",
      {
        session_id: session.lycheeSessionId,
        retrieved_context: session.lastRetrievedContext,
        background: true
      },
      "proactive consolidate"
    );

    if (consolidated) {
      session.pendingProactiveConsolidation = false;
      session.lastConsolidatedAt = Date.now();
    }
  }

  normalizeTurnContent(role: "user" | "assistant", content: string): string {
    const trimmed = content.trim();
    if (!trimmed) {
      return "";
    }

    if (role === "user" && trimmed.startsWith("/")) {
      return "";
    }

    return trimmed;
  }

  rememberSession(params: {
    sessionKey?: unknown;
    sessionId?: unknown;
    channelId?: unknown;
    conversationId?: unknown;
  }): void {
    const session = this.ensureSession({
      sessionKey: asString(params.sessionKey),
      sessionId: asString(params.sessionId),
      channelId: asString(params.channelId),
      conversationId: asString(params.conversationId)
    });
    if (!session) {
      return;
    }

    const sessionKey = asString(params.sessionKey);
    const channelId = asString(params.channelId);
    const state = getLifecycleState(this.api);
    if (sessionKey) {
      const conversationKey =
        channelId && state.latestConversationByChannel[channelId]
          ? state.latestConversationByChannel[channelId]
          : undefined;
      if (conversationKey && state.sessions[conversationKey] === session) {
        state.sessionAliases[sessionKey] = conversationKey;
      } else {
        state.sessionAliases[sessionKey] = sessionKey;
      }
    }
  }

  ensureSession(params: {
    sessionKey?: string;
    sessionId?: string;
    channelId?: string;
    conversationId?: string;
  }): LifecycleSessionState | undefined {
    const stateKey = this.buildStateKey(params);
    if (!stateKey) {
      return undefined;
    }

    const state = getLifecycleState(this.api);
    const resolvedKey = state.sessionAliases[stateKey] ?? stateKey;
    let session = state.sessions[resolvedKey];
    if (!session) {
      session = {
        generation: 0,
        lycheeSessionId: this.buildLycheeSessionId(resolvedKey, 0),
        openclawSessionId: params.sessionId,
        channelId: params.channelId,
        conversationId: params.conversationId,
        lastRetrievedContext: ""
      };
      state.sessions[resolvedKey] = session;
    }

    if (params.sessionId) {
      session.openclawSessionId = params.sessionId;
    }
    if (params.channelId) {
      session.channelId = params.channelId;
    }
    if (params.conversationId) {
      session.conversationId = params.conversationId;
    }
    if (params.channelId && params.conversationId) {
      state.latestConversationByChannel[params.channelId] = resolvedKey;
    }
    if (resolvedKey !== stateKey) {
      state.sessionAliases[stateKey] = resolvedKey;
    }

    return session;
  }

  buildStateKey(params: {
    sessionKey?: string;
    sessionId?: string;
    channelId?: string;
    conversationId?: string;
  }): string | undefined {
    if (params.sessionKey) {
      return params.sessionKey;
    }
    if (params.sessionId) {
      return `session:${params.sessionId}`;
    }
    if (params.channelId && params.conversationId) {
      return `conversation:${params.channelId}:${params.conversationId}`;
    }
    if (params.channelId) {
      return `channel:${params.channelId}`;
    }
    return undefined;
  }

  buildLycheeSessionId(stateKey: string, generation: number): string {
    return `openclaw:${sanitizeSegment(stateKey)}:${generation}`;
  }

  async callLifecycleTool(name: string, path: string, payload: Json, context: string): Promise<boolean> {
    try {
      if (this.cfg.transport === "http") {
        await callHttpEndpoint(this.cfg, path, payload, `LycheeMem ${context} failed`);
        return true;
      }
      await callMcpTool(this.api, this.cfg, name, payload);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.api?.logger?.warn?.(`[${PLUGIN_ID}] ${context} failed: ${message}`);
      return false;
    }
  }
}

export default {
  id: PLUGIN_ID,
  name: "LycheeMem Tools",
  description: "Thin OpenClaw adapter for LycheeMem structured memory tools.",
  register(api: any) {
    const adapter = new OpenClawAdapter(api);
    adapter.register();

    registerTool(api, {
      name: "lychee_memory_smart_search",
      description:
        "Primary LycheeMem recall path for OpenClaw. Uses full mode by default so agents receive both retrieval details and synthesized background_context in one call.",
      parameters: SMART_SEARCH_SCHEMA,
      async execute(_id, params) {
        const cfg = getPluginConfig(api);
        const payload = {
          mode: "full",
          ...params
        };
        if (cfg.transport === "http") {
          return callHttpEndpoint(cfg, "/memory/smart-search", payload, "LycheeMem smart_search failed");
        }
        return callMcpTool(api, cfg, "lychee_memory_smart_search", payload);
      }
    });

    registerTool(api, {
      name: "lychee_memory_search",
      description:
        "Developer-facing raw LycheeMem retrieval tool. Use it when you explicitly want to inspect graph_results and skill_results before synthesis.",
      parameters: SEARCH_SCHEMA,
      async execute(_id, params) {
        const cfg = getPluginConfig(api);
        if (cfg.transport === "http") {
          return callHttpEndpoint(cfg, "/memory/search", params, "LycheeMem search failed");
        }
        return callMcpTool(api, cfg, "lychee_memory_search", params);
      }
    });

    registerTool(api, {
      name: "lychee_memory_append_turn",
      description:
        "Mirror a single host conversation turn into LycheeMem's session store so later consolidation can see the transcript. In OpenClaw with host lifecycle integration enabled, models should normally not call this manually because the host already mirrors user and assistant turns automatically. Keep manual use for debugging or non-host environments.",
      parameters: APPEND_TURN_SCHEMA,
      async execute(_id, params) {
        const cfg = getPluginConfig(api);
        if (cfg.transport === "http") {
          return callHttpEndpoint(cfg, "/memory/append-turn", params, "LycheeMem append_turn failed");
        }
        return callMcpTool(api, cfg, "lychee_memory_append_turn", params);
      }
    });

    registerTool(api, {
      name: "lychee_memory_synthesize",
      description:
        "Developer-facing synthesis tool for cases where you intentionally want to inspect search and synthesis as separate stages.",
      parameters: SYNTHESIZE_SCHEMA,
      async execute(_id, params) {
        const cfg = getPluginConfig(api);
        if (cfg.transport === "http") {
          return callHttpEndpoint(cfg, "/memory/synthesize", params, "LycheeMem synthesize failed");
        }
        return callMcpTool(api, cfg, "lychee_memory_synthesize", params);
      }
    });

    registerTool(api, {
      name: "lychee_memory_consolidate",
      description:
        "Persist new long-term knowledge into LycheeMem after a conversation. Prefer background=true for normal agent operation. In OpenClaw, host lifecycle integration may trigger this automatically at reset/new/stop boundaries, while models may still call consolidate when they intentionally want to persist important new knowledge early.",
      parameters: CONSOLIDATE_SCHEMA,
      async execute(_id, params) {
        const cfg = getPluginConfig(api);
        if (cfg.transport === "http") {
          return callHttpEndpoint(cfg, "/memory/consolidate", params, "LycheeMem consolidate failed");
        }
        return callMcpTool(api, cfg, "lychee_memory_consolidate", params);
      }
    });
  }
};
