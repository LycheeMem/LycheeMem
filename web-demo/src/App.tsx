import AuthPage from "./components/AuthPage";
import ChatPanel from "./components/ChatPanel";
import GraphPanel from "./components/GraphPanel";
import Header from "./components/Header";
import MemoryPanel from "./components/MemoryPanel/MemoryPanel";
import SessionSidebar from "./components/SessionSidebar";
import { useStore } from "./state";

export default function App() {
  const user = useStore((s) => s.user);

  if (!user) {
    return <AuthPage />;
  }

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
