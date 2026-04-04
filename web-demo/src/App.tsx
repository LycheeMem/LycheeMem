import ChatPanel from "./components/ChatPanel";
import GraphPanel from "./components/GraphPanel";
import Header from "./components/Header";
import MemoryPanel from "./components/MemoryPanel/MemoryPanel";
import SessionSidebar from "./components/SessionSidebar";

export default function App() {
  return (
    <>
      <Header />
      <div id="app-body">
        <SessionSidebar />
        <main id="app-main">
          <ChatPanel />
          <GraphPanel />
          <MemoryPanel />
        </main>
      </div>
    </>
  );
}
