import { AppstoreOutlined, BarChartOutlined, MessageOutlined } from "@ant-design/icons";
import { useCallback, useEffect, useRef } from "react";
import { fetchGraphData, fetchGraphEdges, fetchPipelineStatus, fetchSessionTurns, fetchSessions, fetchSkills, streamChatMessage } from "../api";
import { useStore } from "../state";
import { formatContent } from "../utils";

export default function ChatPanel() {
  const sessionId = useStore((s) => s.sessionId);
  const messages = useStore((s) => s.messages);
  const isStreaming = useStore((s) => s.isStreaming);
  const isTyping = useStore((s) => s.isTyping);

  const addMessage = useStore((s) => s.addMessage);
  const setIsStreaming = useStore((s) => s.setIsStreaming);
  const setIsTyping = useStore((s) => s.setIsTyping);
  const setWmTokenUsage = useStore((s) => s.setWmTokenUsage);
  const setWmMaxTokens = useStore((s) => s.setWmMaxTokens);
  const setWmTurns = useStore((s) => s.setWmTurns);
  const setGraphData = useStore((s) => s.setGraphData);
  const setGraphEdges = useStore((s) => s.setGraphEdges);
  const setSkills = useStore((s) => s.setSkills);
  const setPipelineStatus = useStore((s) => s.setPipelineStatus);
  const setSessions = useStore((s) => s.setSessions);
  const setCurrentTrace = useStore((s) => s.setCurrentTrace);
  const setPartialTrace = useStore((s) => s.setPartialTrace);
  const mergePartialTrace = useStore((s) => s.mergePartialTrace);
  const setCompletedSteps = useStore((s) => s.setCompletedSteps);

  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

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
      setCurrentTrace(null);
      setPartialTrace(null);
      setCompletedSteps([]);

      // Track accumulated state across SSE callbacks
      let responseText = "";
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
          onAnswer(answer) {
            responseText = answer;
            setIsTyping(false);
          },
          onDone(data) {
            setWmTokenUsage(data.wm_token_usage || 0);
            addMessage({
              role: "assistant",
              content: responseText,
              meta: {
                memories_retrieved: data.memories_retrieved || 0,
                wm_token_usage: data.wm_token_usage || 0,
                trace: data.trace || null,
              },
            });
            if (data.trace) {
              setCurrentTrace(data.trace);
            }
          },
        });
      } catch (err) {
        setIsTyping(false);
        addMessage({
          role: "assistant",
          content:
            "\u26A0\uFE0F 连接错误: " +
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
          const { turns, wm_max_tokens } = await fetchSessionTurns(sessionId);
          setWmTurns(turns);
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
      setWmTokenUsage,
      setWmMaxTokens,
      setWmTurns,
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
        <span className="session-id-label">{sessionId}</span>
      </div>

      <div id="chat-messages" className="chat-messages" ref={messagesRef}>
        {messages.map((msg, i) => (
          <div key={i} className={`msg msg-${msg.role}`}>
            <span
              dangerouslySetInnerHTML={{ __html: formatContent(msg.content) }}
            />
            {msg.role === "assistant" && msg.meta && (
              <div className="msg-meta">
                <span><AppstoreOutlined /> {msg.meta.memories_retrieved} 条记忆</span>
                <span>
                  <BarChartOutlined /> {(msg.meta.wm_token_usage || 0).toLocaleString()} tokens
                </span>
              </div>
            )}
          </div>
        ))}
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
