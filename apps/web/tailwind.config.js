/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        void: '#05070a',
        hull: '#0a0e13',
        panel: '#0d1219',
        raised: '#111823',
        line: '#1a2432',
        line2: '#243244',
        ink: '#dbe6f3',
        dim: '#8296ad',
        faint: '#55677d',
        cyan: '#22d3ee',
        amber: '#fbbf24',
        jade: '#34d399',
        rose: '#fb7185',
        violet: '#a78bfa',
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'JetBrains Mono', 'Menlo', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['10px', '14px'],
        xs: ['11px', '16px'],
        sm: ['12px', '18px'],
        base: ['13px', '20px'],
        md: ['14px', '22px'],
      },
      animation: {
        'pulse-slow': 'pulse 2.6s cubic-bezier(0.4,0,0.6,1) infinite',
        sweep: 'sweep 2.2s linear infinite',
        'fade-in': 'fadeIn .18s ease-out',
      },
      keyframes: {
        sweep: { '0%': { transform: 'translateX(-100%)' }, '100%': { transform: 'translateX(100%)' } },
        fadeIn: { from: { opacity: '0', transform: 'translateY(3px)' }, to: { opacity: '1', transform: 'none' } },
      },
    },
  },
  plugins: [],
}
