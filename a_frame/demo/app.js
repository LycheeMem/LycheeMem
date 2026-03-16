/* ═══════════════════════════════════════════════════
   A-Frame Console — Frontend Application
   ═══════════════════════════════════════════════════ */

const API = window.location.origin;

// ── State ──
const state = {
  sessionId: null,
  messages: [],          // { role, content, meta }
  graphData: { nodes: [], edges: [] },
  isStreaming: false,
  agents: {},            // agent_name -> "idle" | "running" | "done"
  wmTokenUsage: 0,
  wmMaxTokens: 128000,
};

// ═══ INIT ═══
document.addEventListener("DOMContentLoaded", () => {
  initSession();
  initTabs();
  initChatForm();
  initGraphControls();
  loadPipelineStatus();
  loadSessions();
});

// ═══ SESSION ═══
function initSession() {
  state.sessionId = "session_" + Date.now().toString(36);
  updateSessionLabel();
}

function updateSessionLabel() {
  const el = document.getElementById("current-session-label");
  el.textContent = state.sessionId;
}

async function loadSessions() {
  try {
    const r = await fetch(`${API}/sessions?limit=50`);
    const data = await r.json();
    const sel = document.getElementById("session-select");
    sel.innerHTML = '<option value="">+ 新建会话</option>';
    (data.sessions || []).forEach(s => {
      const opt = document.createElement("option");
      opt.value = s.session_id;
      opt.textContent = (s.topic || s.session_id).slice(0, 40);
      sel.appendChild(opt);
    });
  } catch (_) { /* ignore */ }
}

document.getElementById("session-select").addEventListener("change", async (e) => {
  if (e.target.value) {
    state.sessionId = e.target.value;
    updateSessionLabel();
    state.messages = [];
    renderMessages();
    await loadSessionTurns(state.sessionId);
    resetAgents();
    loadGraphData();
    loadAllMemory();
  }
});

document.getElementById("btn-new-session").addEventListener("click", () => {
  initSession();
  state.messages = [];
  renderMessages();
  resetAgents();
  document.getElementById("session-select").value = "";
});

async function loadSessionTurns(sid) {
  try {
    const r = await fetch(`${API}/memory/session/${encodeURIComponent(sid)}`);
    const data = await r.json();
    state.messages = (data.turns || []).map(t => ({
      role: t.role, content: t.content, meta: null,
    }));
    renderMessages();
    const tab = document.getElementById("tab-working");
    renderWorkingMemoryTurns(data.turns || []);
  } catch (_) { /* ignore */ }
}

// ═══ TAB SWITCHING ═══
function initTabs() {
  document.querySelectorAll(".memory-tabs .tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".memory-tabs .tab").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

// ═══ CHAT ═══
function initChatForm() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");

  // Auto-grow textarea
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });

  // Enter to send, Shift+Enter for newline
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event("submit"));
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || state.isStreaming) return;
    sendMessage(text);
    input.value = "";
    input.style.height = "auto";
  });
}

async function sendMessage(text) {
  // Add user message
  state.messages.push({ role: "user", content: text, meta: null });
  renderMessages();

  // Show typing & timeline
  showTyping();
  resetAgents();
  showTimeline();
  state.isStreaming = true;
  document.getElementById("btn-send").disabled = true;

  // Best-effort status UI (API is non-streaming here)
  setAgentStatus("wm_manager", "running");
  updateTimeline("wm_manager", "running");
  setAgentStatus("router", "running");
  updateTimeline("router", "running");
  setAgentStatus("reasoner", "running");
  updateTimeline("reasoner", "running");

  try {
    const response = await fetch(`${API}/chat/complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message: text }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data?.detail || data?.message || `HTTP ${response.status}`;
      throw new Error(detail);
    }

    hideTyping();

    // Route + WM token usage
    if (data.route) {
      showRouteInfo(data.route);
    }
    state.wmTokenUsage = data.wm_token_usage || 0;
    updateWMProgress();

    // Assistant message
    state.messages.push({
      role: "assistant",
      content: data.response || "",
      meta: {
        memories_retrieved: data.memories_retrieved || 0,
        wm_token_usage: data.wm_token_usage || 0,
      },
    });
    renderMessages();

    // Mark pipeline steps as done
    setAgentStatus("wm_manager", "done");
    updateTimeline("wm_manager", "done");
    setAgentStatus("router", "done");
    updateTimeline("router", "done");
    const needRetrieval = Boolean(data?.route?.need_graph || data?.route?.need_skills || data?.route?.need_sensory);
    if (needRetrieval) {
      setAgentStatus("search_coordinator", "done");
      updateTimeline("search_coordinator", "done");
      setAgentStatus("synthesizer", "done");
      updateTimeline("synthesizer", "done");
    }
    setAgentStatus("reasoner", "done");
    updateTimeline("reasoner", "done");
    setTimeout(() => setAgentStatus("consolidator", "done"), 1000);
  } catch (err) {
    hideTyping();
    state.messages.push({ role: "assistant", content: "⚠️ 连接错误: " + err.message, meta: null });
    renderMessages();
  }

  state.isStreaming = false;
  document.getElementById("btn-send").disabled = false;

  // Post-chat: refresh everything
  setTimeout(() => {
    loadGraphData();
    loadAllMemory();
    loadPipelineStatus();
    loadSessions();
  }, 500);
}

function handleSSEEvent(evt) {
  switch (evt.type) {
    case "agent":
      setAgentStatus(evt.agent, evt.status);
      updateTimeline(evt.agent, evt.status);
      if (evt.agent === "router" && evt.data) {
        showRouteInfo(evt.data);
      }
      if (evt.agent === "wm_manager" && evt.data) {
        state.wmTokenUsage = evt.data.wm_token_usage || 0;
        updateWMProgress();
      }
      break;

    case "memories":
      showRetrievedMemories(evt);
      break;

    case "answer":
      hideTyping();
      state.messages.push({
        role: "assistant",
        content: evt.content,
        meta: null,
      });
      renderMessages();
      break;

    case "done":
      const lastMsg = state.messages[state.messages.length - 1];
      if (lastMsg && lastMsg.role === "assistant") {
        lastMsg.meta = {
          memories_retrieved: evt.memories_retrieved,
          wm_token_usage: evt.wm_token_usage,
        };
        renderMessages();
      }
      state.wmTokenUsage = evt.wm_token_usage || 0;
      updateWMProgress();
      // Mark consolidator as done (background)
      setTimeout(() => setAgentStatus("consolidator", "done"), 1000);
      break;

    case "status":
      // Legacy status events
      break;
  }
}

// ═══ MESSAGE RENDERING ═══
function renderMessages() {
  const container = document.getElementById("chat-messages");
  container.innerHTML = "";

  state.messages.forEach(msg => {
    const el = document.createElement("div");
    el.className = `msg msg-${msg.role}`;
    el.innerHTML = formatContent(msg.content);

    if (msg.role === "assistant" && msg.meta) {
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      meta.innerHTML = `
        <span>🧠 ${msg.meta.memories_retrieved} 条记忆</span>
        <span>📊 ${(msg.meta.wm_token_usage || 0).toLocaleString()} tokens</span>
      `;
      el.appendChild(meta);
    }
    container.appendChild(el);
  });

  container.scrollTop = container.scrollHeight;
}

function formatContent(text) {
  if (!text) return "";
  // Simple markdown: bold, italic, code, newlines
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, '<code style="background:rgba(99,102,241,0.15);padding:1px 4px;border-radius:3px;">$1</code>')
    .replace(/\n/g, "<br>");
}

function showTyping() {
  const container = document.getElementById("chat-messages");
  let el = document.getElementById("typing-msg");
  if (!el) {
    el = document.createElement("div");
    el.id = "typing-msg";
    el.className = "msg msg-assistant";
    el.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    container.appendChild(el);
  }
  container.scrollTop = container.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById("typing-msg");
  if (el) el.remove();
}

// ═══ AGENT STATUS ═══
function setAgentStatus(name, status) {
  state.agents[name] = status;
  const card = document.querySelector(`.agent-card[data-agent="${name}"]`);
  if (!card) return;

  card.className = `agent-card ${status}`;
  const statusEl = card.querySelector(".agent-status");

  const labels = { idle: "待机", running: "运行中", done: "完成" };
  statusEl.textContent = labels[status] || status;
  statusEl.className = `agent-status ${status}`;
}

function resetAgents() {
  ["wm_manager", "router", "search_coordinator", "synthesizer", "reasoner", "consolidator"].forEach(a => {
    setAgentStatus(a, "idle");
  });
  document.getElementById("route-info").classList.add("hidden");
}

function showRouteInfo(routeData) {
  const el = document.getElementById("route-info");
  el.classList.remove("hidden");

  const detail = document.getElementById("route-detail");
  const reasoning = routeData.reasoning || "";
  const flags = [
    { key: "need_graph", label: "📊 图谱", on: routeData.need_graph },
    { key: "need_skills", label: "⚡ 技能", on: routeData.need_skills },
    { key: "need_sensory", label: "👁 感觉", on: routeData.need_sensory },
  ];

  detail.innerHTML = `
    ${reasoning ? `<div style="margin-bottom:6px;">${escapeHtml(reasoning)}</div>` : ""}
    <div class="route-flags">
      ${flags.map(f => `<span class="route-flag ${f.on ? 'on' : 'off'}">${f.label}</span>`).join("")}
    </div>
  `;
}

// ═══ TIMELINE ═══
const PIPELINE_STEPS = ["wm_manager", "router", "search_coordinator", "synthesizer", "reasoner"];
const STEP_LABELS = {
  wm_manager: "工作记忆",
  router: "路由",
  search_coordinator: "检索",
  synthesizer: "合成",
  reasoner: "推理",
};

function showTimeline() {
  const tl = document.getElementById("agent-timeline");
  tl.classList.remove("hidden");
  tl.innerHTML = PIPELINE_STEPS.map((step, i) => {
    const arrow = i < PIPELINE_STEPS.length - 1 ? '<span class="tl-arrow">→</span>' : "";
    return `<span class="tl-step" id="tl-${step}">${STEP_LABELS[step]}</span>${arrow}`;
  }).join("");
}

function updateTimeline(agent, status) {
  const el = document.getElementById(`tl-${agent}`);
  if (el) {
    el.className = `tl-step ${status}`;
    if (status === "running") {
      el.innerHTML = `<span class="spinner"></span> ${STEP_LABELS[agent] || agent}`;
    } else if (status === "done") {
      el.innerHTML = `✓ ${STEP_LABELS[agent] || agent}`;
    }
  }
}

// ═══ WORKING MEMORY ═══
function updateWMProgress() {
  const pct = Math.min(100, (state.wmTokenUsage / state.wmMaxTokens) * 100);
  const fill = document.getElementById("wm-progress");
  fill.style.width = pct + "%";
  fill.className = "progress-fill" + (pct > 70 ? " warn" : "");
  document.getElementById("wm-tokens").textContent =
    `${state.wmTokenUsage.toLocaleString()} / ${state.wmMaxTokens.toLocaleString()}`;
}

function renderWorkingMemoryTurns(turns) {
  const container = document.getElementById("wm-turns");
  container.innerHTML = "";
  turns.slice(-20).forEach(t => {
    const item = document.createElement("div");
    item.className = "memory-item";
    item.innerHTML = `
      <div class="mem-label turn">${t.role === "user" ? "👤 USER" : "🤖 ASSISTANT"}</div>
      <div class="mem-content">${escapeHtml((t.content || "").slice(0, 200))}${t.content && t.content.length > 200 ? "…" : ""}</div>
    `;
    container.appendChild(item);
  });
}

// ═══ RETRIEVED MEMORIES ═══
function showRetrievedMemories(evt) {
  // Switch to relevant tab and show
  if (evt.graph && evt.graph.length > 0) {
    renderGraphMemory(evt.graph, true);
  }
  if (evt.skills && evt.skills.length > 0) {
    renderSkills(evt.skills, true);
  }
  if (evt.sensory && evt.sensory.length > 0) {
    renderSensory(evt.sensory, true);
  }
}

// ═══ MEMORY PANELS ═══
async function loadAllMemory() {
  await Promise.all([loadGraphMemory(), loadSkills(), loadSensory()]);
}

async function loadGraphMemory() {
  try {
    const r = await fetch(`${API}/memory/graph`);
    const data = await r.json();
    const nodes = data.nodes || [];
    renderGraphMemory(nodes.slice(0, 50));
  } catch (_) { /* ignore */ }
}

function renderGraphMemory(nodes, retrieved = false) {
  const container = document.getElementById("graph-mem-list");
  container.innerHTML = "";
  if (!nodes.length) {
    container.innerHTML = '<div class="empty-hint">暂无图谱记忆</div>';
    return;
  }
  nodes.forEach(n => {
    const item = document.createElement("div");
    item.className = "memory-item" + (retrieved ? " retrieved" : "");
    const label = getNodeDisplayText(n) || "entity";
    const props = n.properties ? Object.entries(n.properties).map(([k,v]) => `${k}: ${v}`).join(", ") : "";
    item.innerHTML = `
      <div class="mem-label graph">🔵 ${escapeHtml(label)}</div>
      <div class="mem-content">${escapeHtml(n.node_id || n.id || "")}</div>
      ${props ? `<div class="mem-meta">${escapeHtml(props.slice(0, 120))}</div>` : ""}
    `;
    container.appendChild(item);
  });
}

async function loadSkills() {
  try {
    const r = await fetch(`${API}/memory/skills`);
    const data = await r.json();
    renderSkills(data.skills || []);
  } catch (_) { /* ignore */ }
}

function renderSkills(skills, retrieved = false) {
  const container = document.getElementById("skills-list");
  container.innerHTML = "";
  if (!skills.length) {
    container.innerHTML = '<div class="empty-hint">暂无技能记忆</div>';
    return;
  }
  skills.forEach(s => {
    const item = document.createElement("div");
    item.className = "memory-item" + (retrieved ? " retrieved" : "");

    const title = s.intent || s.name || s.skill_id || s.id || "skill";
    const doc = (s.doc_markdown || s.doc || s.markdown || "").toString();
    const desc = doc || s.conditions || "";

    item.innerHTML = `
      <div class="mem-label skill">⚡ ${escapeHtml(title)}</div>
      <div class="mem-content">${formatContent(String(desc).slice(0, 800))}</div>
      ${s.success_count !== undefined || s.score !== undefined || s.last_used ? `
        <div class="mem-meta">
          ${s.success_count !== undefined ? `成功次数: ${s.success_count}` : ""}
          ${s.score !== undefined ? ` | 评分: ${(s.score || 0).toFixed(2)}` : ""}
          ${s.last_used ? ` | 最近使用: ${escapeHtml(String(s.last_used))}` : ""}
        </div>
      ` : ""}
    `;
    container.appendChild(item);
  });
}

async function loadSensory() {
  try {
    const r = await fetch(`${API}/memory/sensory`);
    const data = await r.json();
    renderSensory(data.items || []);
  } catch (_) { /* ignore */ }
}

function renderSensory(items, retrieved = false) {
  const container = document.getElementById("sensory-list");
  container.innerHTML = "";
  if (!items.length) {
    container.innerHTML = '<div class="empty-hint">感觉缓冲区为空</div>';
    return;
  }
  items.forEach(item => {
    const el = document.createElement("div");
    el.className = "memory-item" + (retrieved ? " retrieved" : "");
    el.innerHTML = `
      <div class="mem-label sensory">👁 ${escapeHtml(item.modality || "text")}</div>
      <div class="mem-content">${escapeHtml((item.content || "").slice(0, 200))}</div>
      ${item.timestamp ? `<div class="mem-meta">${item.timestamp}</div>` : ""}
    `;
    container.appendChild(el);
  });
}

// ═══ PIPELINE STATUS ═══
async function loadPipelineStatus() {
  try {
    const r = await fetch(`${API}/pipeline/status`);
    const data = await r.json();
    document.getElementById("chip-sessions").textContent = `📋 ${data.session_count || 0}`;
    document.getElementById("chip-graph").textContent = `🔵 ${data.graph_node_count || 0}`;
    document.getElementById("chip-skills").textContent = `⚡ ${data.skill_count || 0}`;
    document.getElementById("chip-sensory").textContent = `👁 ${data.sensory_buffer_size || 0}`;
  } catch (_) { /* ignore */ }
}

// ═══ GRAPH VISUALIZATION ═══
let graphSim = null;
let graphTransform = { x: 0, y: 0, k: 1 };

function getNodeDisplayText(n) {
  // Prefer human-friendly names; fallback to id.
  // Backend often stores entity type in `label` (e.g. Person/Place), which we do NOT want as the main text.
  const name = n?.name || n?.properties?.name;
  const id = n?.node_id || n?.id;
  return String(name || id || "?");
}

function getNodeTypeLabel(n) {
  // Entity type label (Person/Place/...) may exist as top-level `label` or inside properties.
  return String(n?.label || n?.properties?.label || "");
}

function initGraphControls() {
  document.getElementById("btn-graph-refresh").addEventListener("click", loadGraphData);
  document.getElementById("btn-graph-fit").addEventListener("click", fitGraph);
}

async function loadGraphData() {
  try {
    const r = await fetch(`${API}/memory/graph`);
    const data = await r.json();
    state.graphData = {
      nodes: (data.nodes || []).map(n => ({
        id: n.node_id || n.id,
        // Show `name` (or `id`) on the node, NOT the entity type label.
        label: getNodeDisplayText(n),
        // Keep raw data for tooltip, etc.
        properties: n.properties || n,
        typeLabel: getNodeTypeLabel(n),
      })),
      edges: (data.edges || []).map(e => ({
        source: e.source,
        target: e.target,
        relation: e.relation || "",
        confidence: e.confidence || 0.5,
      })),
    };
    renderGraph();
  } catch (_) {
    // Ignore errors, may not have data yet
  }
}

function renderGraph() {
  const container = document.getElementById("graph-container");
  const canvas = document.getElementById("graph-canvas");
  const emptyEl = document.getElementById("graph-empty");
  const tooltip = document.getElementById("graph-tooltip");

  const { nodes, edges } = state.graphData;

  if (!nodes.length) {
    emptyEl.style.display = "flex";
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  emptyEl.style.display = "none";

  const dpr = window.devicePixelRatio || 1;
  const rect = container.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";

  const ctx = canvas.getContext("2d");
  // Reset transform each render to avoid cumulative scaling.
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const W = rect.width;
  const H = rect.height;

  // Build valid edges (only those where both source & target exist)
  const nodeIds = new Set(nodes.map(n => n.id));
  const validEdges = edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));

  // Create simulation
  if (graphSim) graphSim.stop();

  const simNodes = nodes.map(n => ({ ...n }));
  const simEdges = validEdges.map(e => ({ ...e }));

  const nodeById = new Map(simNodes.map(n => [n.id, n]));

  graphSim = d3.forceSimulation(simNodes)
    .force("link", d3.forceLink(simEdges).id(d => d.id).distance(100).strength(0.4))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide(30))
    .alphaDecay(0.03);

  // Color palette for nodes
  const colors = [
    "#6366f1", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b",
    "#ec4899", "#14b8a6", "#f43f5e", "#0ea5e9", "#a855f7",
  ];
  const nodeColor = (i) => colors[i % colors.length];

  // Edge color by confidence
  const edgeColor = (e) => {
    const c = e.confidence || 0.5;
    if (c >= 0.9) return "#06b6d4";
    if (c >= 0.7) return "#6366f1";
    return "#334155";
  };

  let hoveredNode = null;
  let dragNode = null;
  let dragOffsetX = 0, dragOffsetY = 0;

  // Render throttling (avoid drawing multiple times per frame)
  let rafId = null;
  function scheduleRender() {
    if (rafId !== null) return;
    rafId = requestAnimationFrame(() => {
      rafId = null;
      drawFrame();
    });
  }

  function resolveNode(ref) {
    if (!ref) return null;
    if (typeof ref === "object") return ref;
    return nodeById.get(ref) || null;
  }

  function drawFrame() {
    // Keep base transform stable in case other code changed it.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = "#060a10";
    ctx.fillRect(0, 0, W, H);

    // Draw grid (screen-space spacing for performance)
    drawGrid(ctx, W, H);

    ctx.save();
    ctx.translate(graphTransform.x, graphTransform.y);
    ctx.scale(graphTransform.k, graphTransform.k);

    // Draw edges
    simEdges.forEach(e => {
      const src = resolveNode(e.source);
      const tgt = resolveNode(e.target);
      if (!src || !tgt) return;

      ctx.beginPath();
      ctx.moveTo(src.x, src.y);
      ctx.lineTo(tgt.x, tgt.y);
      ctx.strokeStyle = edgeColor(e);
      ctx.lineWidth = (e.confidence || 0.5) > 0.7 ? 1.8 : 1;
      ctx.globalAlpha = hoveredNode ?
        (hoveredNode === src || hoveredNode === tgt ? 1 : 0.15) : 0.6;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Relation label on edge
      if (e.relation) {
        const mx = (src.x + tgt.x) / 2;
        const my = (src.y + tgt.y) / 2;
        ctx.font = `${9 / Math.max(graphTransform.k, 0.5)}px Inter`;
        ctx.fillStyle = "#64748b";
        ctx.globalAlpha = hoveredNode ?
          (hoveredNode === src || hoveredNode === tgt ? 0.9 : 0.08) : 0.5;
        ctx.textAlign = "center";
        ctx.fillText(e.relation, mx, my - 4);
        ctx.globalAlpha = 1;
      }
    });

    // Draw nodes
    simNodes.forEach((n, i) => {
      const isHovered = n === hoveredNode;
      const radius = isHovered ? 16 : 12;

      // Glow
      if (isHovered) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, radius + 8, 0, Math.PI * 2);
        const glow = ctx.createRadialGradient(n.x, n.y, radius, n.x, n.y, radius + 8);
        glow.addColorStop(0, nodeColor(i) + "40");
        glow.addColorStop(1, "transparent");
        ctx.fillStyle = glow;
        ctx.fill();
      }

      // Node body
      ctx.beginPath();
      ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = hoveredNode && !isHovered ? nodeColor(i) + "30" : nodeColor(i);
      ctx.fill();
      ctx.strokeStyle = isHovered ? "#ffffff" : nodeColor(i) + "80";
      ctx.lineWidth = isHovered ? 2.5 : 1.5;
      ctx.stroke();

      // Label
      const labelAlpha = hoveredNode ? (isHovered ? 1 : 0.15) : 0.9;
      ctx.globalAlpha = labelAlpha;
      ctx.font = `${isHovered ? 11 : 10}px Inter`;
      ctx.fillStyle = "#e2e8f0";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const txt = String(n.label || "?");
      const label = txt.length > 16 ? txt.slice(0, 14) + "…" : txt;
      ctx.fillText(label, n.x, n.y + radius + 4);
      ctx.globalAlpha = 1;
    });

    ctx.restore();
  }

  // Zoom/pan
  // NOTE: renderGraph() can be called many times; use `on*` handlers to avoid stacking listeners.
  canvas.onwheel = (e) => {
    e.preventDefault();
    const scaleFactor = e.deltaY > 0 ? 0.92 : 1.08;
    const mx = e.offsetX, my = e.offsetY;
    graphTransform.x = mx - (mx - graphTransform.x) * scaleFactor;
    graphTransform.y = my - (my - graphTransform.y) * scaleFactor;
    graphTransform.k *= scaleFactor;
    graphTransform.k = Math.max(0.1, Math.min(5, graphTransform.k));
    scheduleRender();
  };

  let isPanning = false;
  let panStartX, panStartY;

  canvas.onmousedown = (e) => {
    const [mx, my] = screenToWorld(e.offsetX, e.offsetY);
    // Keep hit testing consistent in screen-space even when zoomed.
    const hit = findNodeAt(simNodes, mx, my, 20 / graphTransform.k);
    if (hit) {
      dragNode = hit;
      dragOffsetX = hit.x - mx;
      dragOffsetY = hit.y - my;
      hit.fx = hit.x;
      hit.fy = hit.y;
      graphSim.alphaTarget(0.3).restart();
      scheduleRender();
    } else {
      isPanning = true;
      panStartX = e.offsetX;
      panStartY = e.offsetY;
      scheduleRender();
    }
  };

  canvas.onmousemove = (e) => {
    if (dragNode) {
      const [mx, my] = screenToWorld(e.offsetX, e.offsetY);
      dragNode.fx = mx + dragOffsetX;
      dragNode.fy = my + dragOffsetY;
      scheduleRender();
    } else if (isPanning) {
      graphTransform.x += e.offsetX - panStartX;
      graphTransform.y += e.offsetY - panStartY;
      panStartX = e.offsetX;
      panStartY = e.offsetY;
      scheduleRender();
    } else {
      const [mx, my] = screenToWorld(e.offsetX, e.offsetY);
      hoveredNode = findNodeAt(simNodes, mx, my, 20 / graphTransform.k);
      if (hoveredNode) {
        canvas.style.cursor = "pointer";
        tooltip.classList.remove("hidden");
        tooltip.style.left = (e.offsetX + 12) + "px";
        tooltip.style.top = (e.offsetY - 20) + "px";
        let tooltipText = `<strong>${escapeHtml(hoveredNode.label)}</strong>`;
        if (hoveredNode.typeLabel) {
          tooltipText += `<br>类型: ${escapeHtml(hoveredNode.typeLabel)}`;
        }
        if (hoveredNode.properties) {
          const p = hoveredNode.properties;
          if (p.type) tooltipText += `<br>type: ${escapeHtml(p.type)}`;
          if (p.description) tooltipText += `<br>${escapeHtml(String(p.description).slice(0, 100))}`;
        }
        tooltip.innerHTML = tooltipText;
      } else {
        canvas.style.cursor = "default";
        tooltip.classList.add("hidden");
      }
      scheduleRender();
    }
  };

  canvas.onmouseup = () => {
    if (dragNode) {
      // Keep the node where the user dropped it (more intuitive than snapping back).
      dragNode.fx = dragNode.x;
      dragNode.fy = dragNode.y;
      graphSim.alphaTarget(0);
      dragNode = null;
      scheduleRender();
    }
    isPanning = false;
  };

  canvas.onmouseleave = () => {
    isPanning = false;
    tooltip.classList.add("hidden");
    scheduleRender();
  };

  function screenToWorld(sx, sy) {
    return [(sx - graphTransform.x) / graphTransform.k, (sy - graphTransform.y) / graphTransform.k];
  }

  function worldToScreen(wx, wy) {
    return [wx * graphTransform.k + graphTransform.x, wy * graphTransform.k + graphTransform.y];
  }

  // Render loop (throttled)
  graphSim.on("tick", scheduleRender);

  // Initial paint
  scheduleRender();
}

function drawGrid(ctx, w, h) {
  // Keep grid spacing constant in screen pixels.
  const spacing = 40;
  const ox = ((graphTransform.x % spacing) + spacing) % spacing;
  const oy = ((graphTransform.y % spacing) + spacing) % spacing;

  ctx.strokeStyle = "rgba(148, 163, 184, 0.04)";
  ctx.lineWidth = 1;

  for (let x = ox; x < w; x += spacing) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = oy; y < h; y += spacing) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
}

function findNodeAt(nodes, wx, wy, radius) {
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    const dx = n.x - wx;
    const dy = n.y - wy;
    if (dx * dx + dy * dy <= radius * radius) return n;
  }
  return null;
}

function fitGraph() {
  if (!state.graphData.nodes.length) return;
  const container = document.getElementById("graph-container");
  const rect = container.getBoundingClientRect();

  // Reset to center
  graphTransform = { x: rect.width / 4, y: rect.height / 4, k: 0.8 };

  if (graphSim) {
    graphSim.alpha(0.3).restart();
  }
}

// ═══ UTILS ═══
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
