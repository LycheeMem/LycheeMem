import { ReloadOutlined } from "@ant-design/icons";
import { useEffect, useMemo, useState } from "react";
import { fetchEvolveEvents } from "../api";

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

  // Common: health report (when available)
  const health = safeObj(payload.health);
  const healthScore = pickNumber(health.health_score);
  const samples = pickNumber(health.sample_count);
  const failures = pickNumber(health.failure_count);

  // Candidate excerpts
  const origExcerpt = pickString(payload.original_prompt_excerpt);
  const candExcerpt = pickString(payload.candidate_prompt_excerpt);
  const changes = Array.isArray(payload.changes) ? (payload.changes as unknown[]) : [];

  // Probation numbers
  const preHealth = pickNumber(payload.pre_health);
  const newSamples = pickNumber(payload.new_samples);
  const threshold = pickNumber(payload.improvement_threshold);
  const currentReport = safeObj(payload.current_report);
  const curHealth = pickNumber(currentReport.health_score);

  return (
    <div className="trace-detail">
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
          <div className="trace-section-title">变更要点</div>
          {(changes.slice(0, 5) as Record<string, unknown>[]).map((c, idx) => (
            <div key={idx} className="trace-item">
              <span className="trace-item-name">{pickString(c.reason) || `Change#${idx + 1}`}</span>
              {pickString(c.type) && <span className="trace-tag">{pickString(c.type)}</span>}
            </div>
          ))}
          {changes.length > 5 && <div className="trace-meta">… 还有 {changes.length - 5} 条</div>}
        </>
      )}

      {(origExcerpt || candExcerpt) && (
        <>
          <div className="trace-section-title">Prompt 对比（节选）</div>
          <div className="trace-detail" style={{ gap: 10 }}>
            {origExcerpt && (
              <div>
                <div className="trace-meta" style={{ marginBottom: 6 }}>原始（节选）</div>
                <JsonBlock value={origExcerpt} />
              </div>
            )}
            {candExcerpt && (
              <div>
                <div className="trace-meta" style={{ marginBottom: 6 }}>候选（节选）</div>
                <JsonBlock value={candExcerpt} />
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

  return (
    <div style={{ padding: 14, width: "100%" }}>
      <div className="panel" style={{ width: "100%" }}>
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
                <div className="trace-section-title">原始细节（展开可用于排查）</div>
                <JsonBlock value={ev.payload} />
              </div>
            </details>
          ))}
        </div>
      </div>
    </div>
  );
}

