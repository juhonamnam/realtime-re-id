import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

import pkg from "./package.json"

let base = "/"

try {
  const url = new URL(pkg.homepage)
  base = url.pathname
} catch {
  base = pkg.homepage || "/"
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  assetsInclude: ["**/*.onnx"],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  base,
})
