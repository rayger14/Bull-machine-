export const colors = {
  bg: {
    primary: '#030712',
    secondary: '#0f172a',
    tertiary: '#1e293b',
  },
  glass: {
    bg: 'rgba(255, 255, 255, 0.03)',
    bgHover: 'rgba(255, 255, 255, 0.06)',
    border: 'rgba(255, 255, 255, 0.08)',
    borderHover: 'rgba(255, 255, 255, 0.15)',
  },
  accent: {
    cyan: '#06b6d4',
    emerald: '#10b981',
    rose: '#f43f5e',
    amber: '#f59e0b',
    violet: '#8b5cf6',
    blue: '#3b82f6',
  },
  text: {
    primary: '#f1f5f9',
    secondary: '#94a3b8',
    muted: '#475569',
  },
  chart: {
    equity: '#4da6ff',
    btc: '#fbbf24',
    riskTemp: '#34d399',
    instability: '#f87171',
    crisisProb: '#fb923c',
    threshold: 'rgba(168,85,247,0.6)',
    volume: 'rgba(77,166,255,0.3)',
  },
  regime: {
    bull: '#34d399',
    neutral: '#fbbf24',
    bear: '#f87171',
    crisis: '#f87171',
    stagflation: '#fb923c',
  },
} as const;
