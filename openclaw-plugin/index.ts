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

function getPluginConfig(api: any): PluginConfig {
  const raw = api?.config?.plugins?.entries?.[PLUGIN_ID]?.config ?? {};
  const env = typeof process !== "undefined" ? process.env ?? {} : {};

  return {
    baseUrl: normalizeBaseUrl(String(raw.baseUrl ?? env.LYCHEEMEM_BASE_URL ?? "http://127.0.0.1:8100")),
    transport: String(raw.transport ?? env.LYCHEEMEM_TRANSPORT ?? "mcp").trim().toLowerCase() === "http" ? "http" : "mcp",
    timeout: Number(raw.timeout ?? env.LYCHEEMEM_TIMEOUT ?? 120),
    apiToken: String(raw.apiToken ?? env.LYCHEEMEM_API_TOKEN ?? "")
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

export default {
  id: PLUGIN_ID,
  name: "LycheeMem Tools",
  description: "Thin OpenClaw adapter for LycheeMem structured memory tools.",
  register(api: any) {
    registerTool(api, {
      name: "lychee_memory_search",
      description:
        "Retrieve structured long-term memory from LycheeMem. Use it for historical facts, entity relationships, long-running project context, and procedural memory recall.",
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
      name: "lychee_memory_synthesize",
      description:
        "Compress and fuse the structured retrieval results from lychee_memory_search into a shorter background_context for downstream reasoning.",
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
        "Persist new long-term knowledge into LycheeMem after a conversation. Prefer background=true for normal agent operation.",
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
