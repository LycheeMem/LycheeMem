import ChatPanel from "./components/ChatPanel";
import EvolveHistoryPage from "./components/EvolveHistoryPage";
import GraphPanel from "./components/GraphPanel";
import Header from "./components/Header";
import MemoryPanel from "./components/MemoryPanel/MemoryPanel";
import SessionSidebar from "./components/SessionSidebar";
import { useStore } from "./state";

export default function App() {
  const activePage = useStore((s) => s.activePage);
  return (
    <>
      <Header />
      {activePage === "evolve-history" ? (
        <div id="app-body">
          <main id="app-main" className="app-main-single">
            <EvolveHistoryPage />
          </main>
        </div>
      ) : (
        <div id="app-body">
          <SessionSidebar />
          <main id="app-main">
            <ChatPanel />
            <GraphPanel />
            <MemoryPanel />
          </main>
        </div>
      )}
    </>
  );
}
