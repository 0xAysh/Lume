import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Tauri expects a fixed dev port and serves the static `dist/` build in
// production — there is no Node server in the shipped app (DESIGN §3).
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    watch: {
      // Don't watch the Rust/Python sides from the Vite dev server.
      ignored: ["**/src-tauri/**", "**/crates/**", "**/sidecar/**"],
    },
  },
});
