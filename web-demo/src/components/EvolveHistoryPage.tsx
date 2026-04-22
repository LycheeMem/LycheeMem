import { PlayCircleOutlined, ReloadOutlined } from "@ant-design/icons";
import { useEffect, useMemo, useState } from "react";
import { fetchEvolveEvents, postEvolveOptimize } from "../api";
import MarkdownRenderer from "./MarkdownRenderer";

/** 与后端 registry 注册的 prompt 名一致，便于一键选择 */
const COMMON_PROMPTS = [
  "reasoning",
  "synthesis",
  "search_coordinator",
  "retrieval_planning",
  "retrieval_adequacy_check",
  "compact_encoding",
  "consolidation",
  "wm_compression",
  "composite_filter",
  "retrieval_additional_queries",
] as const;

type EvolveEvent = {
  id: number;
  created_at: string;
  event_type: string;
  prompt_name: string;
  from_version: number | null;
  to_version: number | null;
  summary: string;
  payload: unknown;
};

function formatTime(ts: string): string {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function JsonBlock({ value }: { value: unknown }) {
  const text = useMemo(() => {
    try {
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }, [value]);
  return (
    <div className="md-code-block">
      <pre>
        <code>{text}</code>
      </pre>
    </div>
  );
}

function eventLabel(t: string): string {
  const map: Record<string, string> = {
    optimize_diagnosis: "诊断",
    candidate_created: "生成候选",
    candidate_review: "评审",
    promote_probation: "提升进入 probation",
    probation_pass: "probation 通过",
    probation_fail: "probation 失败",
    rollback: "回滚",
  };
  return map[t] || t;
}

function eventTone(t: string): "good" | "warn" | "bad" | "info" {
  if (t === "probation_pass") return "good";
  if (t === "probation_fail" || t === "rollback") return "bad";
  if (t === "candidate_review") return "warn";
  return "info";
}

function toneColor(tone: "good" | "warn" | "bad" | "info"): string {
  if (tone === "good") return "#22c55e";
  if (tone === "warn") return "#f59e0b";
  if (tone === "bad") return "#ef4444";
  return "#60a5fa";
}

function safeObj(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

function pickString(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function pickNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function MiniKv({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="trace-kv">
      <span>{k}</span>
      <span>{v}</span>
    </div>
  );
}

function EventPreview({ ev }: { ev: EvolveEvent }) {
  const payload = safeObj(ev.payload);
  const activePromptDiag = pickString(payload.active_prompt);

  // Common: health report (when available)
  const health = safeObj(payload.health);
  const healthScore = pickNumber(health.health_score);
  const samples = pickNumber(health.sample_count);
  const failures = pickNumber(health.failure_count);

  // 完整 prompt：新事件用 original_prompt / candidate_prompt；旧库可能仅有 *_excerpt
  const origFull =
    pickString(payload.original_prompt) || pickString(payload.original_prompt_excerpt);
  const candFull =
    pickString(payload.candidate_prompt) || pickString(payload.candidate_prompt_excerpt);
  const changes = Array.isArray(payload.changes) ? (payload.changes as unknown[]) : [];

  // Probation numbers
  const preHealth = pickNumber(payload.pre_health);
  const newSamples = pickNumber(payload.new_samples);
  const threshold = pickNumber(payload.improvement_threshold);
  const currentReport = safeObj(payload.current_report);
  const curHealth = pickNumber(currentReport.health_score);

  return (
    <div className="trace-detail">
      {activePromptDiag && (
        <>
          <div className="trace-section-title">诊断时 Active Prompt（完整）</div>
          <div className="evolve-prompt-md">
            <MarkdownRenderer content={activePromptDiag} />
          </div>
        </>
      )}
      {(healthScore != null || samples != null || failures != null) && (
        <>
          {healthScore != null && (
            <MiniKv k="健康分" v={<span className="chip"> {healthScore.toFixed(3)} </span>} />
          )}
          {samples != null && <MiniKv k="样本数" v={<span className="chip">{samples}</span>} />}
          {failures != null && <MiniKv k="失败数" v={<span className="chip">{failures}</span>} />}
        </>
      )}

      {(preHealth != null || curHealth != null || newSamples != null) && (
        <>
          {preHealth != null && curHealth != null && (
            <MiniKv
              k="probation 健康变化"
              v={
                <span className="chip">
                  {preHealth.toFixed(3)} → {curHealth.toFixed(3)}
                </span>
              }
            />
          )}
          {newSamples != null && <MiniKv k="新样本增量" v={<span className="chip">{newSamples}</span>} />}
          {threshold != null && <MiniKv k="阈值" v={<span className="chip">{threshold.toFixed(3)}</span>} />}
        </>
      )}

      {changes.length > 0 && (
        <>
          <div className="trace-section-title">变更要点（全部）</div>
          {(changes as Record<string, unknown>[]).map((c, idx) => (
            <div key={idx} className="trace-item">
              <span className="trace-item-name">{pickString(c.reason) || `Change#${idx + 1}`}</span>
              {pickString(c.type) && <span className="trace-tag">{pickString(c.type)}</span>}
            </div>
          ))}
        </>
      )}

      {(origFull || candFull) && (
        <>
          <div className="trace-section-title">Prompt 对比（完整）</div>
          <div className="trace-detail" style={{ gap: 10 }}>
            {origFull && (
              <div>
                <div className="trace-meta" style={{ marginBottom: 6 }}>原始</div>
                <div className="evolve-prompt-md">
                  <MarkdownRenderer content={origFull} />
                </div>
              </div>
            )}
            {candFull && (
              <div>
                <div className="trace-meta" style={{ marginBottom: 6 }}>候选</div>
                <div className="evolve-prompt-md">
                  <MarkdownRenderer content={candFull} />
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default function EvolveHistoryPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [events, setEvents] = useState<EvolveEvent[]>([]);
  const [promptFilter, setPromptFilter] = useState<string>("");
  const [typeFilter, setTypeFilter] = useState<string>("");
  const [limit, setLimit] = useState<number>(200);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  const [optimizePrompt, setOptimizePrompt] = useState("");
  const [optimizing, setOptimizing] = useState(false);
  const [optimizeHint, setOptimizeHint] = useState<string>("");

  async function load() {
    setLoading(true);
    setError("");
    try {
      const data = await fetchEvolveEvents({
        limit: Math.min(500, Math.max(1, limit || 200)),
        prompt_name: promptFilter.trim() || undefined,
        event_type: typeFilter.trim() || undefined,
      });
      const list = (data as Record<string, unknown>).events;
      setEvents((Array.isArray(list) ? list : []) as EvolveEvent[]);
    } catch (e) {
      setError((e as Error)?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const promptOptions = useMemo(() => {
    const set = new Set<string>();
    for (const e of events) {
      if (e.prompt_name) set.add(e.prompt_name);
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [events]);

  const typeOptions = useMemo(() => {
    const set = new Set<string>();
    for (const e of events) {
      if (e.event_type) set.add(e.event_type);
    }
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [events]);

  function togglePrompt(p: string) {
    setPromptFilter((cur) => (cur === p ? "" : p));
  }

  function toggleType(t: string) {
    setTypeFilter((cur) => (cur === t ? "" : t));
  }

  async function runOptimize() {
    setOptimizing(true);
    setOptimizeHint("");
    setError("");
    try {
      const name = optimizePrompt.trim();
      const res = await postEvolveOptimize(name || undefined);
      const enabled = (res as Record<string, unknown>).enabled;
      if (enabled === false) {
        setOptimizeHint("当前未启用 evolve（后端未注入 evolve_loop）。");
        return;
      }
      const results = (res as Record<string, unknown>).results;
      if (Array.isArray(results) && results.length > 0) {
        const lines = results.map((r: Record<string, unknown>) => {
          const pn = String(r.prompt_name ?? "");
          const ok = r.success === true ? "成功" : "未成功";
          const cv = r.candidate_version != null ? ` 候选 v${r.candidate_version}` : "";
          const msg = r.message != null ? ` — ${String(r.message)}` : "";
          return `${pn || "?"}: ${ok}${cv}${msg}`;
        });
        setOptimizeHint(lines.join("\n"));
      } else {
        setOptimizeHint(String((res as Record<string, unknown>).error || "已请求优化，请查看下方事件列表。"));
      }
      await load();
    } catch (e) {
      setError((e as Error)?.message || String(e));
    } finally {
      setOptimizing(false);
    }
  }

  return (
    <div className="evolve-history-outer" style={{ padding: 14, width: "100%" }}>
      <div className="panel evolve-history-panel" style={{ width: "100%" }}>
        <div className="panel-header">
          <h2>🧬 自进化历史</h2>
          <div className="panel-actions" style={{ marginLeft: "auto", alignItems: "center" }}>
            <button
              className="crud-btn crud-btn-primary crud-btn-sm"
              onClick={() => void load()}
              disabled={loading}
              title="刷新事件列表"
            >
              <ReloadOutlined /> 刷新
            </button>
            <button
              className="crud-btn crud-btn-ghost crud-btn-sm"
              onClick={() => setShowAdvanced((v) => !v)}
              disabled={loading}
            >
              {showAdvanced ? "收起筛选" : "筛选"}
            </button>
          </div>
        </div>

        {error && <div className="trace-empty" style={{ color: "#ff4d4f" }}>{error}</div>}

        <div
          style={{
            margin: "12px 12px 0",
            padding: "12px 14px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border)",
            background: "var(--bg-card)",
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          <div className="trace-section-title" style={{ margin: 0 }}>
            手动优化
          </div>
          <div className="trace-meta">
            留空则由后端按 health 自动选择；或点选下方快捷名，再点「运行优化」。
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
            <input
              value={optimizePrompt}
              onChange={(e) => setOptimizePrompt(e.target.value)}
              placeholder="优化模块（可空）"
              style={{
                minWidth: 220,
                flex: "1 1 200px",
                padding: "6px 10px",
                borderRadius: "var(--radius-xs)",
                border: "1px solid var(--border-light)",
                background: "var(--bg-surface)",
                color: "var(--text-primary)",
              }}
            />
            <button
              type="button"
              className="crud-btn crud-btn-primary crud-btn-sm"
              onClick={() => void runOptimize()}
              disabled={optimizing || loading}
            >
              <PlayCircleOutlined /> {optimizing ? "运行中…" : "运行优化"}
            </button>
          </div>
          <div className="status-chips" style={{ flexWrap: "wrap" }}>
            <span className="chip" style={{ opacity: 0.85 }}>快捷：</span>
            {COMMON_PROMPTS.map((p) => (
              <span
                key={p}
                className="chip"
                style={{
                  cursor: "pointer",
                  borderColor: optimizePrompt === p ? "rgba(99,102,241,0.8)" : undefined,
                  color: optimizePrompt === p ? "#c7d2fe" : undefined,
                }}
                onClick={() => setOptimizePrompt((cur) => (cur === p ? "" : p))}
                title="填入 / 取消"
              >
                {p}
              </span>
            ))}
          </div>
          {optimizeHint && (
            <div className="md-code-block" style={{ margin: 0 }}>
              <pre style={{ margin: 0, padding: "10px 12px" }}>
                <code style={{ whiteSpace: "pre-wrap" }}>{optimizeHint}</code>
              </pre>
            </div>
          )}
        </div>

        {/* Quick filters (chips) */}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", padding: "10px 12px 0" }}>
          <span className="chip" style={{ opacity: 0.9 }}>
            显示：{Math.min(limit, 500)} 条
          </span>
          {promptFilter && (
            <span className="chip" style={{ borderColor: "rgba(99,102,241,0.5)", color: "#c7d2fe" }}>
              prompt: {promptFilter}
            </span>
          )}
          {typeFilter && (
            <span className="chip" style={{ borderColor: "rgba(34,197,94,0.4)", color: "#bbf7d0" }}>
              type: {eventLabel(typeFilter)}
            </span>
          )}
        </div>

        {showAdvanced && (
          <div style={{ padding: "10px 12px 0" }}>
            <div className="trace-section-title">Prompt 快速选择</div>
            <div className="status-chips" style={{ flexWrap: "wrap" }}>
              {promptOptions.length === 0 && <span className="chip">暂无</span>}
              {promptOptions.map((p) => (
                <span
                  key={p}
                  className="chip"
                  style={{
                    cursor: "pointer",
                    borderColor: promptFilter === p ? "rgba(99,102,241,0.8)" : undefined,
                    color: promptFilter === p ? "#c7d2fe" : undefined,
                  }}
                  onClick={() => togglePrompt(p)}
                  title="点击筛选/取消"
                >
                  {p}
                </span>
              ))}
            </div>

            <div className="trace-section-title" style={{ marginTop: 10 }}>事件类型</div>
            <div className="status-chips" style={{ flexWrap: "wrap" }}>
              {typeOptions.length === 0 && <span className="chip">暂无</span>}
              {typeOptions.map((t) => (
                <span
                  key={t}
                  className="chip"
                  style={{
                    cursor: "pointer",
                    borderColor: typeFilter === t ? "rgba(34,197,94,0.6)" : undefined,
                    color: typeFilter === t ? "#bbf7d0" : undefined,
                  }}
                  onClick={() => toggleType(t)}
                  title="点击筛选/取消"
                >
                  {eventLabel(t)}
                </span>
              ))}
            </div>

            <div className="trace-section-title" style={{ marginTop: 10 }}>显示条数</div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              {[50, 100, 200, 500].map((n) => (
                <span
                  key={n}
                  className="chip"
                  style={{
                    cursor: "pointer",
                    borderColor: limit === n ? "rgba(99,102,241,0.8)" : undefined,
                    color: limit === n ? "#c7d2fe" : undefined,
                  }}
                  onClick={() => setLimit(n)}
                >
                  {n}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="trace-container" style={{ marginTop: 8 }}>
          {events.length === 0 && !loading && (
            <div className="trace-empty">暂无事件（先触发几次 optimize / 等待 probation 判断）</div>
          )}

          {events.map((ev) => (
            <details
              key={ev.id}
              className="trace-step done"
              style={{
                paddingBottom: 6,
                borderLeft: `3px solid ${toneColor(eventTone(ev.event_type))}`,
              }}
            >
              <summary
                className="trace-step-header"
                style={{ cursor: "pointer", listStyle: "none" as never }}
              >
                <span className="trace-step-icon">🧬</span>
                <span className="trace-step-label">
                  {eventLabel(ev.event_type)}
                  {ev.prompt_name ? ` / ${ev.prompt_name}` : ""}
                </span>
                <span className="trace-step-summary">
                  {ev.summary || ""}
                </span>
                <span className="trace-meta" style={{ marginLeft: "auto" }}>
                  {formatTime(ev.created_at)} #{ev.id}
                  {ev.from_version != null || ev.to_version != null
                    ? `  v${ev.from_version ?? "?"}→v${ev.to_version ?? "?"}`
                    : ""}
                </span>
              </summary>
              <div className="trace-step-body" style={{ gap: 10 }}>
                <EventPreview ev={ev} />
                <div className="trace-section-title">完整事件数据（JSON）</div>
                <JsonBlock value={ev.payload} />
              </div>
            </details>
          ))}
        </div>
      </div>
    </div>
  );
}

