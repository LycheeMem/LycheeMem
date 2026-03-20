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
import { useEffect, useRef, useState } from "react";
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

// ── Consolidator step labels & content ────────────────────────────────────

const STEP_NAME_LABELS: Record<string, string> = {
  novelty_check: "新颖性检查",
  episode_ingest: "Episode 写入",
  semantic_build: "语义构建",
  community_refresh: "社区刷新",
  llm_analysis: "LLM 分析",
  entity_extraction: "实体抽取",
  skill_extraction: "技能提取",
};

function ConsolidatorContent({
  consolidator,
}: {
  consolidator: PipelineTrace["consolidator"];
}) {
  if (consolidator.status === "pending") {
    return (
      <div className="trace-detail">
        <div className="trace-empty">
          <SyncOutlined spin style={{ marginRight: 8 }} />
          后台固化中，请稍候...
        </div>
      </div>
    );
  }

  return (
    <div className="trace-detail">
      <div className="trace-kv">
        <span>状态</span>
        <span>{consolidator.status === "skipped" ? "已跳过" : "完成"}</span>
      </div>
      {consolidator.status === "skipped" && (
        <div className="trace-kv">
          <span>原因</span>
          <span>
            {consolidator.skipped_reason === "no_novelty"
              ? "未检测到新信息"
              : (consolidator.skipped_reason ?? "未知")}
          </span>
        </div>
      )}
      {consolidator.status === "done" && (
        <>
          <div className="trace-kv">
            <span>新增实体</span>
            <span>{consolidator.entities_added}</span>
          </div>
          {consolidator.facts_added > 0 && (
            <div className="trace-kv">
              <span>写入事实</span>
              <span>{consolidator.facts_added}</span>
            </div>
          )}
          <div className="trace-kv">
            <span>新增技能</span>
            <span>{consolidator.skills_added}</span>
          </div>
        </>
      )}
      {consolidator.steps.length > 0 && (
        <>
          <div className="trace-section-title">执行步骤</div>
          {consolidator.steps.map((step, i) => (
            <div key={i} className="trace-item">
              <CheckCircleOutlined
                style={{ color: "#52c41a", marginRight: 6, flexShrink: 0 }}
              />
              <span className="trace-item-name">
                {STEP_NAME_LABELS[step.name] ?? step.name}
              </span>
              {step.detail && <span className="trace-meta">{step.detail}</span>}
            </div>
          ))}
        </>
      )}
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
            : consolidator.status === "skipped"
            ? "已跳过（无新信息）"
            : `${consolidator.entities_added} 实体${
                consolidator.facts_added > 0 ? ` | ${consolidator.facts_added} 事实` : ""
              } | ${consolidator.skills_added} 技能`
        }
        status={
          consolidator.status === "done" || consolidator.status === "skipped"
            ? "done"
            : "pending"
        }
        defaultOpen={consolidator.status !== "pending"}
      >
        <ConsolidatorContent consolidator={consolidator} />
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

  const currentTraceRef = useRef(currentTrace);
  useEffect(() => {
    currentTraceRef.current = currentTrace;
  }, [currentTrace]);

  const [consolidatorPoll, setConsolidatorPoll] = useState(0);

  // 当新 pending trace 到达时重置轮询计数器（确保每次新对话完成都从 0 重新开始）
  useEffect(() => {
    if (currentTrace?.consolidator.status === "pending") {
      // 若当前已经是 0 也要强制触发轮询 effect，所以先设 -1 再设 0
      setConsolidatorPoll(-1);
    }
  }, [currentTrace]);

  // 固化轮询：逐步加大间隔（3s→10s），之后以 10s 固定间隔持续轮询直到完成
  useEffect(() => {
    if (consolidatorPoll < 0) {
      // -1 是"重置信号"，立即跳到 0 触发真正的首次轮询
      setConsolidatorPoll(0);
      return;
    }
    const trace = currentTraceRef.current;
    if (!trace || trace.consolidator.status !== "pending") return;
    const delay = consolidatorPoll < 8 ? Math.min(3000 + consolidatorPoll * 1000, 10000) : 10000;
    const timer = setTimeout(async () => {
      try {
        const result = await fetchConsolidationResult();
        const latestTrace = currentTraceRef.current;
        if (!latestTrace) return;
        if (result.status !== "pending") {
          setCurrentTrace({ ...latestTrace, consolidator: result });
        } else {
          setConsolidatorPoll((p) => p + 1);
        }
      } catch {
        setConsolidatorPoll((p) => p + 1);
      }
    }, delay);
    return () => clearTimeout(timer);
  }, [consolidatorPoll, setCurrentTrace]);

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

