import { MessageOutlined, PictureOutlined, CloseCircleOutlined } from "@ant-design/icons";
import { useCallback, useEffect, useRef, useState } from "react";
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

  const [pendingImages, setPendingImages] = useState<{ base64: string; preview: string; mimeType: string }[]>([]);
  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const tokenBufferRef = useRef<string>("");
  const rafRef = useRef<number | null>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages, isTyping, streamingContent]);

  // 处理图片选择 - 使用 Promise 确保异步正确性
  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const promises: Promise<{ base64: string; preview: string; mimeType: string }>[] = [];
    Array.from(files).forEach((file) => {
      if (!file.type.startsWith("image/")) return;
      if (file.size > 10 * 1024 * 1024) return;

      const promise = new Promise<{ base64: string; preview: string; mimeType: string }>((resolve) => {
        const reader = new FileReader();
        reader.onload = (evt) => {
          const result = evt.target?.result as string;
          const base64 = result.split(",")[1] || result;
          resolve({ base64, preview: result, mimeType: file.type });
        };
        reader.readAsDataURL(file);
      });
      promises.push(promise);
    });

    Promise.all(promises).then((results) => {
      if (results.length > 0) {
        setPendingImages((prev) => [...prev, ...results]);
      }
    });

    // 重置 input
    e.target.value = "";
  }, []);

  const removePendingImage = useCallback((index: number) => {
    setPendingImages((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const input = inputRef.current;
      if (!input) return;
      const text = input.value.trim();
      if ((!text && pendingImages.length === 0) || isStreaming) return;

      input.value = "";
      input.style.height = "auto";

      // 保存当前图片
      const imagesToSend = [...pendingImages];
      setPendingImages([]);

      // 用户消息内容
      let messageContent = text;
      if (imagesToSend.length > 0 && !text) {
        messageContent = "[图片]";
      }

      addMessage({
        role: "user",
        content: messageContent,
        meta: null,
        images: imagesToSend.map((img) => img.preview),
      });
      setIsTyping(true);
      setIsStreaming(true);
      setStreamingContent("");
      setCurrentTrace(null);
      setPartialTrace(null);
      setCompletedSteps([]);

      const doneSteps: string[] = [];

      try {
        console.log("[ChatPanel] Sending message with images:", imagesToSend.length);
        await streamChatMessage(
          sessionId,
          text || "[图片]",
          {
            onStep(step, data) {
              doneSteps.push(step);
              setCompletedSteps([...doneSteps]);
              const fragment = data.trace_fragment;
              if (fragment && typeof fragment === "object") {
                mergePartialTrace(fragment as Record<string, unknown>);
              }
            },
            onToken(token) {
              setIsTyping(false);
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
              setIsTyping(false);
            },
            onDone(data) {
              if (rafRef.current !== null) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = null;
              }
              if (tokenBufferRef.current) {
                appendStreamingContent(tokenBufferRef.current);
                tokenBufferRef.current = "";
              }
              setWmTokenUsage(data.wm_token_usage || 0);
              console.log("[ChatPanel] Done, memories retrieved:", data.memories_retrieved);
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
          },
          imagesToSend.map((img) => img.base64),
          imagesToSend.map((img) => img.mimeType)  // 传递 MIME 类型
        );
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
          content: "⚠️ 连接错误: " + (err instanceof Error ? err.message : String(err)),
          meta: null,
        });
      }

      setIsStreaming(false);

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
      isStreaming, sessionId, pendingImages,
      addMessage, setIsTyping, setIsStreaming, setStreamingContent,
      appendStreamingContent, setWmTokenUsage, setWmMaxTokens, setWmTurns,
      setWmSummaries, setWmBoundaryIndex, setCurrentTrace, setPartialTrace,
      mergePartialTrace, setCompletedSteps, setGraphData, setGraphEdges,
      setSkills, setPipelineStatus, setSessions,
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
            {/* 显示用户消息中的图片 */}
            {msg.role === "user" && msg.images && msg.images.length > 0 && (
              <div className="msg-images">
                {msg.images.map((img: string, idx: number) => (
                  <img key={idx} src={img} alt={`图片 ${idx + 1}`} className="msg-image-thumb" />
                ))}
              </div>
            )}
          </div>
        ))}

        {isStreaming && streamingContent && (
          <div className="msg msg-assistant">
            <MarkdownRenderer content={streamingContent} streaming />
          </div>
        )}

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

      {/* 待发送图片预览 */}
      {pendingImages.length > 0 && (
        <div className="image-preview-bar">
          {pendingImages.map((img, idx) => (
            <div key={idx} className="image-preview-item">
              <img src={img.preview} alt={`预览 ${idx + 1}`} />
              <CloseCircleOutlined
                className="image-preview-remove"
                onClick={() => removePendingImage(idx)}
              />
            </div>
          ))}
        </div>
      )}

      <form className="chat-input-area" onSubmit={handleSubmit}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          style={{ display: "none" }}
          onChange={handleImageSelect}
        />
        <button
          type="button"
          className="image-upload-btn"
          title="上传图片"
          disabled={isStreaming}
          onClick={() => fileInputRef.current?.click()}
        >
          <PictureOutlined />
        </button>
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
