import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // คำขอที่ขึ้นต้นด้วย /data ให้ไปที่ Flask แทน
      '/data': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
