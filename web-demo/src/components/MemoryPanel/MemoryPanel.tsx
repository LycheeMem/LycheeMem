import { AppstoreOutlined, PictureOutlined } from "@ant-design/icons";
import { useStore } from "../../state";
import AgentsTab from "./AgentsTab";
import GraphMemoryTab from "./GraphMemoryTab";
import SkillsTab from "./SkillsTab";
import VisualMemoryTab from "./VisualMemoryTab";
import WorkingMemoryTab from "./WorkingMemoryTab";

const TABS = [
  { id: "tab-agents", label: "Pipeline" },
  { id: "tab-working", label: "工作记忆" },
  { id: "tab-graph-mem", label: "语义记忆" },
  { id: "tab-visual", label: "视觉记忆", icon: <PictureOutlined /> },
  { id: "tab-skills", label: "技能记忆" },
];

export default function MemoryPanel() {
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);

  return (
    <section id="panel-memory" className="panel">
      <div className="panel-header">
        <h2><AppstoreOutlined /> 记忆状态</h2>
      </div>
      <div className="memory-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab${activeTab === tab.id ? " active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div
        className={`tab-content${activeTab === "tab-agents" ? " active" : ""}`}
      >
        {activeTab === "tab-agents" && <AgentsTab />}
      </div>
      <div
        className={`tab-content${activeTab === "tab-working" ? " active" : ""}`}
      >
        {activeTab === "tab-working" && <WorkingMemoryTab />}
      </div>
      <div
        className={`tab-content${activeTab === "tab-graph-mem" ? " active" : ""}`}
      >
        {activeTab === "tab-graph-mem" && <GraphMemoryTab />}
      </div>
      <div
        className={`tab-content${activeTab === "tab-skills" ? " active" : ""}`}
      >
        {activeTab === "tab-skills" && <SkillsTab />}
      </div>
      <div
        className={`tab-content${activeTab === "tab-visual" ? " active" : ""}`}
      >
        {activeTab === "tab-visual" && <VisualMemoryTab />}
      </div>
    </section>
  );
}
