import { DeleteOutlined, EyeOutlined, SearchOutlined } from "@ant-design/icons";
import { useCallback, useEffect, useState } from "react";
import {
  deleteVisualMemory,
  fetchVisualMemories,
  fetchVisualMemoryImage,
  searchVisualMemories,
  type VisualMemoryItem,
} from "../../api";
import { useStore } from "../../state";

export default function VisualMemoryTab() {
  const sessionId = useStore((s) => s.sessionId);
  const [memories, setMemories] = useState<VisualMemoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedMemory, setSelectedMemory] = useState<VisualMemoryItem | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [sortBy, setSortBy] = useState<"time" | "importance">("time");

  const loadMemories = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchVisualMemories(sessionId, 100);
      const sorted = [...data].sort((a, b) => {
        if (sortBy === "time") {
          return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
        }
        return b.importance_score - a.importance_score;
      });
      setMemories(sorted);
    } catch (e) {
      console.error("Failed to load visual memories:", e);
    } finally {
      setLoading(false);
    }
  }, [sessionId, sortBy]);

  useEffect(() => {
    loadMemories();
  }, [loadMemories]);

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) {
      loadMemories();
      return;
    }
    setLoading(true);
    try {
      const results = await searchVisualMemories(searchQuery, 50);
      setMemories(results);
    } catch (e) {
      console.error("Search failed:", e);
    } finally {
      setLoading(false);
    }
  }, [searchQuery, loadMemories]);

  const handleDelete = useCallback(async (recordId: string) => {
    if (!confirm("确定要删除这条视觉记忆吗？")) return;
    try {
      await deleteVisualMemory(recordId);
      setMemories((prev) => prev.filter((m) => m.record_id !== recordId));
      if (selectedMemory?.record_id === recordId) {
        setSelectedMemory(null);
        setImageUrl("");
      }
    } catch (e) {
      console.error("Delete failed:", e);
    }
  }, [selectedMemory]);

  const handleViewImage = useCallback((memory: VisualMemoryItem) => {
    setSelectedMemory(memory);
    const url = fetchVisualMemoryImage(memory.record_id);
    setImageUrl(url);
  }, []);

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return "刚刚";
    if (minutes < 60) return `${minutes} 分钟前`;
    if (hours < 24) return `${hours} 小时前`;
    if (days < 7) return `${days} 天前`;
    return date.toLocaleDateString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const sceneTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
      screenshot: "截图",
      chart: "图表",
      photo: "照片",
      document: "文档",
      ui: "界面",
      other: "其他",
    };
    return labels[type] || type;
  };

  return (
    <div className="visual-memory-tab">
      {/* 搜索栏 */}
      <div className="visual-memory-search">
        <input
          type="text"
          placeholder="搜索图片内容（描述、场景类型）..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          className="visual-memory-search-input"
        />
        <button onClick={handleSearch} className="visual-memory-search-btn">
          <SearchOutlined />
        </button>
        <button
          onClick={() => {
            setSearchQuery("");
            loadMemories();
          }}
          className="visual-memory-reset-btn"
          title="重置"
        >
          重置
        </button>
      </div>

      {/* 排序选项 */}
      <div className="visual-memory-sort">
        <span>排序：</span>
        <button
          className={sortBy === "time" ? "active" : ""}
          onClick={() => setSortBy("time")}
        >
          上传时间
        </button>
        <button
          className={sortBy === "importance" ? "active" : ""}
          onClick={() => setSortBy("importance")}
        >
          重要性
        </button>
        <span className="visual-memory-count">共 {memories.length} 张图片</span>
      </div>

      {loading && memories.length === 0 ? (
        <div className="visual-memory-loading">加载中...</div>
      ) : memories.length === 0 ? (
        <div className="visual-memory-empty">
          {searchQuery ? "未找到匹配的图片" : "暂无视觉记忆。在对话中发送图片即可自动创建视觉记忆。"}
        </div>
      ) : (
        <>
          {/* 记忆网格 */}
          <div className="visual-memory-grid">
            {memories.map((mem) => (
              <div
                key={mem.record_id}
                className={`visual-memory-card ${selectedMemory?.record_id === mem.record_id ? "selected" : ""}`}
                onClick={() => handleViewImage(mem)}
              >
                <div className="visual-memory-thumb">
                  <img src={fetchVisualMemoryImage(mem.record_id)} alt={mem.caption} />
                </div>
                <div className="visual-memory-info">
                  <div className="visual-memory-caption">{mem.caption.slice(0, 80)}...</div>
                  <div className="visual-memory-meta">
                    <span className="visual-memory-scene">
                      {sceneTypeLabel(mem.scene_type)}
                    </span>
                    <span className="visual-memory-time">
                      🕒 {formatTime(mem.timestamp)}
                    </span>
                  </div>
                  <div className="visual-memory-meta-secondary">
                    <span className="visual-memory-importance">
                      重要性 {mem.importance_score.toFixed(2)}
                    </span>
                    <span className="visual-memory-retrievals">
                      🔍 {mem.retrieval_count}
                    </span>
                  </div>
                </div>
                <div className="visual-memory-actions">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleViewImage(mem);
                    }}
                    title="查看"
                  >
                    <EyeOutlined />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(mem.record_id);
                    }}
                    title="删除"
                  >
                    <DeleteOutlined />
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* 详情面板 */}
          {selectedMemory && (
            <div className="visual-memory-detail" onClick={() => { setSelectedMemory(null); setImageUrl(""); }}>
              <div className="visual-memory-detail-content" onClick={(e) => e.stopPropagation()}>
                <div className="visual-memory-detail-header">
                  <h3>视觉记忆详情</h3>
                  <button onClick={() => { setSelectedMemory(null); setImageUrl(""); }}>
                    关闭
                  </button>
                </div>
                <div className="visual-memory-detail-body">
                  <div className="visual-memory-detail-left">
                    <img
                      src={imageUrl}
                      alt={selectedMemory.caption}
                      className="visual-memory-detail-image"
                    />
                  </div>
                  <div className="visual-memory-detail-right">
                    <div className="visual-memory-detail-info">
                      <div className="visual-memory-detail-info-row">
                        <strong>图片描述:</strong>
                        {selectedMemory.caption.startsWith("[") && selectedMemory.caption.includes("解析失败") ? (
                          <span className="caption-error">
                            ⚠️ {selectedMemory.caption}
                            <span className="caption-error-hint">
                              VLM 未能正确解析图片，可能是图片格式问题或模型响应超时。
                            </span>
                          </span>
                        ) : (
                          <span>{selectedMemory.caption}</span>
                        )}
                      </div>
                      <div className="visual-memory-detail-info-row">
                        <strong>场景类型:</strong> {sceneTypeLabel(selectedMemory.scene_type)}
                      </div>
                      <div className="visual-memory-detail-info-row">
                        <strong>上传时间:</strong> {formatTime(selectedMemory.timestamp)}
                      </div>
                      <div className="visual-memory-detail-info-row">
                        <strong>完整时间:</strong> {new Date(selectedMemory.timestamp).toLocaleString("zh-CN")}
                      </div>
                      <div className="visual-memory-detail-info-row">
                        <strong>重要性评分:</strong> {selectedMemory.importance_score.toFixed(3)}
                      </div>
                      <div className="visual-memory-detail-info-row">
                        <strong>检索次数:</strong> {selectedMemory.retrieval_count}
                      </div>
                      <div className="visual-memory-detail-info-row">
                        <strong>来源会话:</strong> {selectedMemory.session_id}
                      </div>
                      {selectedMemory.entities.length > 0 && (
                        <div className="visual-memory-detail-info-row">
                          <strong>识别到的实体:</strong>
                          <ul className="visual-memory-entities-list">
                            {selectedMemory.entities.map((ent: any, i: number) => (
                              <li key={i}>{ent.type}: {ent.name} (置信度：{ent.confidence?.toFixed(2)})</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
