import { MessageOutlined } from "@ant-design/icons";
import { useCallback, useEffect, useRef } from "react";
import { fetchGraphData, fetchGraphEdges, fetchPipelineStatus, fetchSessionTurns, fetchSessions, fetchSkills, streamChatMessage } from "../api";
import { useStore } from "../state";
import MarkdownRenderer from "./MarkdownRenderer";

export default function ChatPanel() {
  const sessionId = useStore((s) => s.sessionId);
  const messages = useStore((s) => s.messages);
  const isStreaming = useStore((s) => s.isStreaming);
  const isTyping = useStore((s) => s.isTyping);
  const streamingContent = useStore((s) => s.streamingContent);

  const addMessage = useStore((s) => s.addMessage);
  const setIsStreaming = useStore((s) => s.setIsStreaming);
  const setIsTyping = useStore((s) => s.setIsTyping);
  const setWmTokenUsage = useStore((s) => s.setWmTokenUsage);
  const setWmMaxTokens = useStore((s) => s.setWmMaxTokens);
  const setWmTurns = useStore((s) => s.setWmTurns);
  const setWmSummaries = useStore((s) => s.setWmSummaries);
  const setWmBoundaryIndex = useStore((s) => s.setWmBoundaryIndex);
  const setGraphData = useStore((s) => s.setGraphData);
  const setGraphEdges = useStore((s) => s.setGraphEdges);
  const setSkills = useStore((s) => s.setSkills);
  const setPipelineStatus = useStore((s) => s.setPipelineStatus);
  const setSessions = useStore((s) => s.setSessions);
  const setCurrentTrace = useStore((s) => s.setCurrentTrace);
  const setPartialTrace = useStore((s) => s.setPartialTrace);
  const mergePartialTrace = useStore((s) => s.mergePartialTrace);
  const setCompletedSteps = useStore((s) => s.setCompletedSteps);
  const setStreamingContent = useStore((s) => s.setStreamingContent);
  const appendStreamingContent = useStore((s) => s.appendStreamingContent);

  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // rAF token batching：将高频 token 事件缓冲至每帧最多一次 state 更新
  const tokenBufferRef = useRef<string>("");
  const rafRef = useRef<number | null>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages, isTyping, streamingContent]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const input = inputRef.current;
      if (!input) return;
      const text = input.value.trim();
      if (!text || isStreaming) return;

      input.value = "";
      input.style.height = "auto";

      addMessage({ role: "user", content: text, meta: null });
      setIsTyping(true);
      setIsStreaming(true);
      setStreamingContent("");
      setCurrentTrace(null);
      setPartialTrace(null);
      setCompletedSteps([]);

      // Track accumulated state across SSE callbacks
      const doneSteps: string[] = [];

      try {
        await streamChatMessage(sessionId, text, {
          onStep(step, data) {
            doneSteps.push(step);
            setCompletedSteps([...doneSteps]);
            const fragment = data.trace_fragment;
            if (fragment && typeof fragment === "object") {
              mergePartialTrace(fragment as Record<string, unknown>);
            }
          },
          onToken(token) {
            // 首个 token 到达时关闭 typing 指示器，改为真实流式渲染
            setIsTyping(false);
            // 缓冲 token，通过 rAF 每帧批量刷新一次，避免逐字符 re-render 导致的生硬闪烁
            tokenBufferRef.current += token;
            if (rafRef.current === null) {
              rafRef.current = requestAnimationFrame(() => {
                appendStreamingContent(tokenBufferRef.current);
                tokenBufferRef.current = "";
                rafRef.current = null;
              });
            }
          },
          onAnswer(_answer) {
            // answer 事件是对所有 token 的汇总确认，前端已通过 token 事件累积，忽略即可
            setIsTyping(false);
          },
          onDone(data) {
            // 取消待处理的 rAF，同步刷新缓冲区，确保最后几个 token 不丢失
            if (rafRef.current !== null) {
              cancelAnimationFrame(rafRef.current);
              rafRef.current = null;
            }
            if (tokenBufferRef.current) {
              appendStreamingContent(tokenBufferRef.current);
              tokenBufferRef.current = "";
            }
            setWmTokenUsage(data.wm_token_usage || 0);
            addMessage({
              role: "assistant",
              content: useStore.getState().streamingContent,
              meta: {
                memories_retrieved: data.memories_retrieved || 0,
                wm_token_usage: data.wm_token_usage || 0,
                turn_input_tokens: data.turn_input_tokens,
                turn_output_tokens: data.turn_output_tokens,
                trace: data.trace || null,
              },
            });
            if (data.trace) {
              setCurrentTrace(data.trace);
            }
            setStreamingContent("");
          },
        });
      } catch (err) {
        if (rafRef.current !== null) {
          cancelAnimationFrame(rafRef.current);
          rafRef.current = null;
        }
        tokenBufferRef.current = "";
        setIsTyping(false);
        setStreamingContent("");
        addMessage({
          role: "assistant",
          content:
            "⚠️ 连接错误: " +
            (err instanceof Error ? err.message : String(err)),
          meta: null,
        });
      }

      setIsStreaming(false);

      // Post-chat refresh
      setTimeout(async () => {
        try { setGraphData(await fetchGraphData()); } catch { /* */ }
        try { setGraphEdges(await fetchGraphEdges()); } catch { /* */ }
        try { setSkills(await fetchSkills()); } catch { /* */ }
        try { setPipelineStatus(await fetchPipelineStatus()); } catch { /* */ }
        try { setSessions(await fetchSessions()); } catch { /* */ }
        try {
          const { turns, summaries, boundary_index, wm_current_tokens, wm_max_tokens } = await fetchSessionTurns(sessionId);
          setWmTurns(turns);
          setWmSummaries(summaries);
          setWmBoundaryIndex(boundary_index);
          setWmTokenUsage(wm_current_tokens);
          setWmMaxTokens(wm_max_tokens);
        } catch { /* */ }
      }, 500);
    },
    [
      isStreaming,
      sessionId,
      addMessage,
      setIsTyping,
      setIsStreaming,
      setStreamingContent,
      appendStreamingContent,
      setWmTokenUsage,
      setWmMaxTokens,
      setWmTurns,
      setWmSummaries,
      setWmBoundaryIndex,
      setCurrentTrace,
      setPartialTrace,
      mergePartialTrace,
      setCompletedSteps,
      setGraphData,
      setGraphEdges,
      setSkills,
      setPipelineStatus,
      setSessions,
    ]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const form = (e.target as HTMLElement).closest("form");
      if (form) form.dispatchEvent(new Event("submit", { bubbles: true }));
    }
  };

  const handleInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const el = e.currentTarget;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  };

  return (
    <section id="panel-chat" className="panel">
      <div className="panel-header">
        <h2><MessageOutlined /> 对话</h2>
      </div>

      <div id="chat-messages" className="chat-messages" ref={messagesRef}>
        {messages.map((msg, i) => (
          <div key={i}>
            <div className={`msg msg-${msg.role}`}>
              {msg.role === "user" ? (
                <span className="msg-user-text">{msg.content}</span>
              ) : (
                <MarkdownRenderer content={msg.content} />
              )}
            </div>
            {/* 显示 token 统计（仅 assistant 消息） */}
            {/* {msg.role === "assistant" && msg.meta && (
              <div className="msg-meta">
                {msg.meta.turn_input_tokens !== undefined && msg.meta.turn_output_tokens !== undefined && (
                  <span className="token-stat">
                    📊 输入: {msg.meta.turn_input_tokens} | 输出: {msg.meta.turn_output_tokens}
                  </span>
                )}
                {msg.meta.memories_retrieved > 0 && (
                  <span className="memory-stat">🧠 检索: {msg.meta.memories_retrieved}</span>
                )}
                {msg.meta.wm_token_usage > 0 && (
                  <span className="wm-stat">💾 WM占用: {msg.meta.wm_token_usage}</span>
                )}
              </div>
            )} */}
          </div>
        ))}

        {/* 流式输出中的 assistant 气泡（token by token 实时渲染） */}
        {isStreaming && streamingContent && (
          <div className="msg msg-assistant">
            <MarkdownRenderer content={streamingContent} streaming />
          </div>
        )}

        {/* 等待首个 token 前的打字指示器 */}
        {isTyping && (
          <div className="msg msg-assistant">
            <div className="typing-indicator">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
      </div>

      <form className="chat-input-area" onSubmit={handleSubmit}>
        <textarea
          id="chat-input"
          ref={inputRef}
          placeholder="输入消息… (Enter 发送, Shift+Enter 换行)"
          rows={1}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
        />
        <button
          type="submit"
          className="send-btn"
          title="发送"
          disabled={isStreaming}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path
              d="M22 2L11 13"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            />
            <path
              d="M22 2L15 22L11 13L2 9L22 2Z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </form>
    </section>
  );
}