import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/auth": "http://localhost:8000",
      "/chat": "http://localhost:8000",
      "/sessions": "http://localhost:8000",
      "/memory": "http://localhost:8000",
      "/pipeline": "http://localhost:8000",
      "/visual": "http://localhost:8000",
    },
  },
});
