import {
  ApiOutlined,
  BulbOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  DownOutlined,
  ExperimentOutlined,
  InboxOutlined,
  RightOutlined,
  SearchOutlined,
  SyncOutlined,
} from "@ant-design/icons";
import { useEffect, useState } from "react";
import { fetchConsolidationResult } from "../../api";
import { useStore } from "../../state";
import type { PipelineTrace } from "../../types";

interface TraceStepProps {
  icon: React.ReactNode;
  label: string;
  summary: string;
  status: "idle" | "running" | "done" | "pending";
  children: React.ReactNode;
  defaultOpen?: boolean;
}

function TraceStep({
  icon,
  label,
  summary,
  status,
  children,
  defaultOpen = false,
}: TraceStepProps) {
  const [open, setOpen] = useState(defaultOpen);

  // When a step transitions from running → done, defaultOpen may flip to true.
  useEffect(() => {
    if (defaultOpen) setOpen(true);
  }, [defaultOpen]);

  return (
    <div className={`trace-step ${status}`}>
      <div className="trace-step-header" onClick={() => setOpen(!open)}>
        <span className="trace-step-toggle">
          {open ? <DownOutlined /> : <RightOutlined />}
        </span>
        <span className="trace-step-icon">{icon}</span>
        <span className="trace-step-label">{label}</span>
        <span className="trace-step-status">
          {status === "running" && <SyncOutlined spin />}
          {status === "done" && <CheckCircleOutlined />}
          {status === "pending" && <ClockCircleOutlined />}
        </span>
        <span className="trace-step-summary">{summary}</span>
      </div>
      {open && <div className="trace-step-body">{children}</div>}
    </div>
  );
}

// ── Per-step content components ────────────────────────────────────────────

function WmContent({ wm }: { wm: PipelineTrace["wm_manager"] }) {
  return (
    <div className="trace-detail">
      <div className="trace-kv">
        <span>Token 用量</span>
        <span>{wm.wm_token_usage.toLocaleString()}</span>
      </div>
      <div className="trace-kv">
        <span>压缩后轮数</span>
        <span>{wm.compressed_turn_count}</span>
      </div>
      <div className="trace-kv">
        <span>近期原始轮数</span>
        <span>{wm.raw_recent_turn_count}</span>
      </div>
      <div className="trace-kv">
        <span>压缩触发</span>
        <span>{wm.compression_happened ? "是" : "否"}</span>
      </div>
    </div>
  );
}

function SearchContent({ search }: { search: PipelineTrace["search_coordinator"] }) {
  return (
    <div className="trace-detail">
      {search.graph_memories.length > 0 && (
        <>
          <div className="trace-section-title">图谱记忆</div>
          {search.graph_memories.map((gm, i) => (
            <div key={i} className="trace-item">
              <span className="trace-item-name">{gm.name || gm.node_id}</span>
              {gm.label && <span className="trace-tag">{gm.label}</span>}
              {gm.score > 0 && (
                <span className="trace-score">{(gm.score * 100).toFixed(0)}%</span>
              )}
              <span className="trace-meta">{gm.neighbor_count} neighbors</span>
            </div>
          ))}
        </>
      )}
      {search.skills.length > 0 && (
        <>
          <div className="trace-section-title">技能</div>
          {search.skills.map((sk, i) => (
            <div key={i} className="trace-item">
              <span className="trace-item-name">{sk.intent || sk.skill_id}</span>
              {sk.score > 0 && (
                <span className="trace-score">{(sk.score * 100).toFixed(0)}%</span>
              )}
              {sk.reusable && <span className="trace-tag reusable">可复用</span>}
            </div>
          ))}
        </>
      )}
      {search.total_retrieved === 0 && (
        <div className="trace-empty">未检索到记忆</div>
      )}
    </div>
  );
}

function SynthContent({ synth }: { synth: PipelineTrace["synthesizer"] }) {
  return (
    <div className="trace-detail">
      {synth.provenance.length > 0 && (
        <>
          <div className="trace-section-title">溯源 (Provenance)</div>
          {synth.provenance.map((p, i) => (
            <div key={i} className="trace-prov-item">
              <div className="trace-prov-header">
                <span className={`trace-tag ${p.source}`}>{p.source}</span>
                <span className="trace-relevance-bar">
                  <span style={{ width: `${p.relevance * 100}%` }} />
                </span>
                <span className="trace-score">{(p.relevance * 100).toFixed(0)}%</span>
              </div>
              {p.summary && <div className="trace-prov-summary">{p.summary}</div>}
            </div>
          ))}
        </>
      )}
      {synth.background_context && (
        <>
          <div className="trace-section-title">背景上下文</div>
          <div className="trace-context-preview">{synth.background_context}</div>
        </>
      )}
      {synth.provenance.length === 0 && !synth.background_context && (
        <div className="trace-empty">无合成内容</div>
      )}
    </div>
  );
}

function ReasonerContent({ reasoner }: { reasoner: PipelineTrace["reasoner"] }) {
  return (
    <div className="trace-detail">
      <div className="trace-kv">
        <span>响应长度</span>
        <span>{reasoner.response_length} 字符</span>
      </div>
    </div>
  );
}

// ── Full trace (after streaming completes) ─────────────────────────────────

function TraceContent({ trace }: { trace: PipelineTrace }) {
  const {
    wm_manager: wm,
    search_coordinator: search,
    synthesizer: synth,
    reasoner,
    consolidator,
  } = trace;

  return (
    <>
      {/* WM Manager */}
      <TraceStep
        icon={<ApiOutlined />}
        label="工作记忆"
        summary={`${wm.wm_token_usage.toLocaleString()} tokens | ${wm.raw_recent_turn_count} 轮${wm.compression_happened ? " | 已压缩" : ""}`}
        status="done"
      >
        <WmContent wm={wm} />
      </TraceStep>

      {/* Search Coordinator */}
      <TraceStep
        icon={<SearchOutlined />}
        label="检索"
        summary={`${search.graph_memories.length} 图谱 | ${search.skills.length} 技能`}
        status="done"
        defaultOpen={search.total_retrieved > 0}
      >
        <SearchContent search={search} />
      </TraceStep>

      {/* Synthesizer */}
      <TraceStep
        icon={<ExperimentOutlined />}
        label="合成"
        summary={`${synth.kept_count} 条保留${synth.skill_reuse_plan.length > 0 ? ` | ${synth.skill_reuse_plan.length} 技能计划` : ""}`}
        status="done"
        defaultOpen={synth.provenance.length > 0}
      >
        <SynthContent synth={synth} />
      </TraceStep>

      {/* Reasoner */}
      <TraceStep
        icon={<BulbOutlined />}
        label="推理"
        summary={`${reasoner.response_length} 字符`}
        status="done"
      >
        <ReasonerContent reasoner={reasoner} />
      </TraceStep>

      {/* Consolidator */}
      <TraceStep
        icon={<InboxOutlined />}
        label="固化"
        summary={
          consolidator.status === "pending"
            ? "后台处理中..."
            : `${consolidator.entities_added} 实体 | ${consolidator.skills_added} 技能`
        }
        status={consolidator.status === "done" ? "done" : "pending"}
      >
        <div className="trace-detail">
          <div className="trace-kv">
            <span>状态</span>
            <span>{consolidator.status === "pending" ? "处理中" : "完成"}</span>
          </div>
          {consolidator.status === "done" && (
            <>
              <div className="trace-kv">
                <span>新增实体</span>
                <span>{consolidator.entities_added}</span>
              </div>
              <div className="trace-kv">
                <span>新增技能</span>
                <span>{consolidator.skills_added}</span>
              </div>
            </>
          )}
        </div>
      </TraceStep>
    </>
  );
}

// ── Step key → trace fragment key mapping ─────────────────────────────────

const RUNNING_STEPS = [
  { key: "wm_manager",  traceKey: "wm_manager",        icon: <ApiOutlined />,        label: "工作记忆" },
  { key: "search",      traceKey: "search_coordinator", icon: <SearchOutlined />,     label: "检索" },
  { key: "synthesize",  traceKey: "synthesizer",        icon: <ExperimentOutlined />, label: "合成" },
  { key: "reason",      traceKey: "reasoner",           icon: <BulbOutlined />,       label: "推理" },
];

export default function AgentsTab() {
  const currentTrace = useStore((s) => s.currentTrace);
  const isStreaming = useStore((s) => s.isStreaming);
  const completedSteps = useStore((s) => s.completedSteps);
  const partialTrace = useStore((s) => s.partialTrace);
  const setCurrentTrace = useStore((s) => s.setCurrentTrace);

  // Poll for consolidator result ~3s after response
  useEffect(() => {
    if (!currentTrace || currentTrace.consolidator.status !== "pending") return;
    const timer = setTimeout(async () => {
      try {
        const result = await fetchConsolidationResult();
        if (result.status === "done") {
          setCurrentTrace({
            ...currentTrace,
            consolidator: result,
          });
        }
      } catch {
        /* ignore */
      }
    }, 3000);
    return () => clearTimeout(timer);
  }, [currentTrace, setCurrentTrace]);

  if (!currentTrace) {
    if (!isStreaming) {
      return (
        <div className="trace-container">
          <div className="trace-empty">发送消息后查看 Pipeline 追踪</div>
        </div>
      );
    }

    // Determine the index of the currently running step (first not yet completed).
    const runningIdx = RUNNING_STEPS.findIndex((s) => !completedSteps.includes(s.key));

    return (
      <div className="trace-container">
        {RUNNING_STEPS.map((step, i) => {
          const isDone = completedSteps.includes(step.key);
          const isRunning = i === runningIdx;
          const status: "done" | "running" | "idle" = isDone
            ? "done"
            : isRunning
            ? "running"
            : "idle";

          // Retrieve the per-step trace fragment already available from the SSE event.
          const stepData = partialTrace
            ? (partialTrace as Record<string, unknown>)[step.traceKey]
            : null;

          let summary = isDone ? "完成" : isRunning ? "运行中..." : "等待中";
          let content: React.ReactNode = null;
          let autoOpen = false;

          if (isDone && stepData) {
            if (step.traceKey === "wm_manager") {
              const wm = stepData as PipelineTrace["wm_manager"];
              summary = `${wm.wm_token_usage.toLocaleString()} tokens | ${wm.raw_recent_turn_count} 轮${wm.compression_happened ? " | 已压缩" : ""}`;
              content = <WmContent wm={wm} />;
              autoOpen = true;
            } else if (step.traceKey === "search_coordinator") {
              const search = stepData as PipelineTrace["search_coordinator"];
              summary = `${search.graph_memories.length} 图谱 | ${search.skills.length} 技能`;
              content = <SearchContent search={search} />;
              autoOpen = search.total_retrieved > 0;
            } else if (step.traceKey === "synthesizer") {
              const synth = stepData as PipelineTrace["synthesizer"];
              summary = `${synth.kept_count} 条保留${synth.skill_reuse_plan.length > 0 ? ` | ${synth.skill_reuse_plan.length} 技能计划` : ""}`;
              content = <SynthContent synth={synth} />;
              autoOpen = synth.provenance.length > 0;
            } else if (step.traceKey === "reasoner") {
              const reasoner = stepData as PipelineTrace["reasoner"];
              summary = `${reasoner.response_length} 字符`;
              content = <ReasonerContent reasoner={reasoner} />;
              autoOpen = true;
            }
          }

          return (
            <TraceStep
              key={step.key}
              icon={step.icon}
              label={step.label}
              summary={summary}
              status={status}
              defaultOpen={autoOpen}
            >
              {content}
            </TraceStep>
          );
        })}
      </div>
    );
  }

  return (
    <div className="trace-container">
      <TraceContent trace={currentTrace} />
    </div>
  );
}

