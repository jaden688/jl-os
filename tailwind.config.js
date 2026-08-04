module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"Fira Code"', 'monospace'],
        display: ['Orbitron', 'sans-serif'],
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        franken: {
          neon: '#00ff66',
          cyan: '#00f3ff',
          pink: '#ff0055',
          purple: '#9d00ff',
          dark: '#080c14',
          panel: '#0d1527',
          border: '#1e2d4a'
        }
      },
      animation: {
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(0,243,255,0.4)' },
          '100%': { boxShadow: '0 0 20px rgba(0,243,255,0.9), 0 0 30px rgba(157,0,255,0.6)' },
        }
      }
    },
  },
  plugins: [],
}
