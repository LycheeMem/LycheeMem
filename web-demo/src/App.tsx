import ChatPanel from "./components/ChatPanel";
import GraphPanel from "./components/GraphPanel";
import Header from "./components/Header";
import MemoryPanel from "./components/MemoryPanel/MemoryPanel";

export default function App() {
  return (
    <>
      <Header />
      <main id="app-main">
        <ChatPanel />
        <GraphPanel />
        <MemoryPanel />
      </main>
    </>
  );
}
