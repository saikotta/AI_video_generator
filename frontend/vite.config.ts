import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  appType: 'spa',
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      // Proxy all /auth and /projects API calls to the FastAPI backend
      '/auth': { target: 'http://127.0.0.1:8001', changeOrigin: true },
      '/projects': { target: 'http://127.0.0.1:8001', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8001', changeOrigin: true },
    },
  },
  preview: {
    port: 5174,
    strictPort: true,
  },
})
