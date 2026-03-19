import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 55555,
    host: "0.0.0.0",
    proxy: {
      "/chat": "http://localhost:8000",
      "/sessions": "http://localhost:8000",
      "/memory": "http://localhost:8000",
      "/pipeline": "http://localhost:8000",
    },
  },
});
