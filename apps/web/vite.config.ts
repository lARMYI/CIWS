import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies to the Python backend so the UI runs on :5173 with hot
// reload while every /api call still hits the real hub on :8787.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8787', changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
})
