import { defineConfig } from "vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  base: "/static/chat/",
  build: {
    outDir: fileURLToPath(new URL("../static/chat", import.meta.url)),
    emptyOutDir: true,
  },
});
